import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from src.config.enums import INGESTED_STATUSES, INGESTED_STUDY_TYPES
from src.models.clinical_trial import ClinicalTrial
from src.models.ingestion import IngestionResult, SyncResult
from src.services.clinical_trials_api import ClinicalTrialsApiService
from src.services.embedding import EmbeddingService, get_embedding_service
from src.services.vector_store import VectorStoreService, get_vector_store_service

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = BACKEND_ROOT / "data" / "clinical_trials.parquet"
EMBED_BATCH_SIZE = 64
DEFAULT_SYNC_LOOKBACK_DAYS = 7

logger = logging.getLogger(__name__)


class IngestionController:
    def __init__(
        self,
        api_service: ClinicalTrialsApiService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
    ):
        self.api_service = api_service or ClinicalTrialsApiService()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store_service()

    def _upsert_trials(self, trials: list[ClinicalTrial]) -> IngestionResult:
        result = IngestionResult(processed=len(trials))
        for i in range(0, len(trials), EMBED_BATCH_SIZE):
            batch = trials[i : i + EMBED_BATCH_SIZE]
            documents = [trial.to_embedding_documents() for trial in batch]
            try:
                self.vector_store.add_documents(documents)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - never abort the batch for one failure
                logger.warning(
                    "Ingestion failed for a batch of %s trials: %s", len(batch), exc
                )
                result.failed += len(batch)
                for trial in batch:
                    result.errors[trial.nct_id] = str(exc)
                continue
            result.succeeded += len(batch)
        return result

    def ingest_single(self, nct_id: str) -> IngestionResult:
        raw = self.api_service.get_study(nct_id)
        trial = ClinicalTrial.from_raw_api_response(raw)
        return self._upsert_trials([trial])

    @staticmethod
    def _should_ingest(row: dict) -> bool:
        """Whether a corpus row belongs in the vector index."""
        return (
            row.get("overall_status") in INGESTED_STATUSES
            and row.get("study_type") in INGESTED_STUDY_TYPES
        )

    @staticmethod
    def _row_groups_for(
        parquet: pq.ParquetFile, start: int, end: int
    ) -> tuple[list[int], int]:
        """Row groups overlapping ``[start, end)``, and the skip within the first."""
        metadata = parquet.metadata
        groups: list[int] = []
        offset_in_first = 0
        row_offset = 0
        for index in range(metadata.num_row_groups):
            group_rows = metadata.row_group(index).num_rows
            if row_offset + group_rows > start and row_offset < end:
                if not groups:
                    offset_in_first = start - row_offset
                groups.append(index)
            row_offset += group_rows
        return groups, offset_in_first

    def ingest_bulk_from_parquet(
        self,
        limit: int | None = None,
        start: int = 0,
        apply_filter: bool = True,
    ) -> IngestionResult:
        """Embed and upsert a range of corpus rows."""
        parquet = pq.ParquetFile(PARQUET_PATH)
        corpus_rows = parquet.metadata.num_rows
        start = max(0, min(start, corpus_rows))
        end = corpus_rows if limit is None else min(start + limit, corpus_rows)

        total = IngestionResult(row_start=start, row_end=end)
        if start >= end:
            return total

        groups, offset_in_first = self._row_groups_for(parquet, start, end)
        scanned = 0  # raw rows consumed from the range

        with tqdm(total=end - start, unit="row", desc="Ingesting") as progress:
            embeddings = self.embedding_service.embeddings
            embeddings.on_progress = progress.update
            try:
                for record_batch in parquet.iter_batches(
                    batch_size=EMBED_BATCH_SIZE, row_groups=groups
                ):
                    rows = record_batch.to_pylist()
                    if offset_in_first:
                        if len(rows) <= offset_in_first:
                            offset_in_first -= len(rows)
                            continue
                        rows = rows[offset_in_first:]
                        offset_in_first = 0
                    rows = rows[: (end - start) - scanned]
                    if not rows:
                        break
                    scanned += len(rows)

                    keep = [
                        r for r in rows if not apply_filter or self._should_ingest(r)
                    ]
                    skipped = len(rows) - len(keep)
                    total.skipped += skipped
                    if skipped:  # filtered rows never reach the embedder
                        progress.update(skipped)

                    if keep:
                        trials = [ClinicalTrial.model_validate(row) for row in keep]
                        partial = self._upsert_trials(trials)
                        total.processed += partial.processed
                        total.succeeded += partial.succeeded
                        total.failed += partial.failed
                        total.errors.update(partial.errors)
                        # A failed batch embeds nothing, so nudge the bar past it.
                        if partial.failed:
                            progress.update(partial.failed)

                    if scanned >= end - start:
                        break
            finally:
                embeddings.on_progress = None
        return total

    def update_single(self, nct_id: str) -> IngestionResult:
        return self.ingest_single(nct_id)

    def update_bulk(self, nct_ids: list[str]) -> IngestionResult:
        result = IngestionResult()
        trials: list[ClinicalTrial] = []
        for nct_id in nct_ids:
            try:
                raw = self.api_service.get_study(nct_id)
                trials.append(ClinicalTrial.from_raw_api_response(raw))
            except Exception as exc:  # noqa: BLE001 - keep processing remaining IDs
                result.processed += 1
                result.failed += 1
                result.errors[nct_id] = str(exc)

        partial = self._upsert_trials(trials)
        result.processed += partial.processed
        result.succeeded += partial.succeeded
        result.errors.update(partial.errors)
        return result

    def delete_single(self, nct_id: str) -> None:
        self.vector_store.delete([nct_id])

    def delete_bulk(self, nct_ids: list[str]) -> None:
        self.vector_store.delete(nct_ids)

    def sync(self, since: str | None = None) -> SyncResult:
        since = since or (
            datetime.now(timezone.utc) - timedelta(days=DEFAULT_SYNC_LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")

        changed_raw = list(self.api_service.iter_studies_updated_since(since))
        trials = [ClinicalTrial.from_raw_api_response(r) for r in changed_raw]
        checked = len(trials)

        nct_ids = [t.nct_id for t in trials]
        stored_docs = self.vector_store.get_by_ids(nct_ids)
        stored_dates = {
            doc.metadata.get("nct_id"): doc.metadata.get("last_update_post_date")
            for doc in stored_docs
        }

        def _last_update_post_date(trial: ClinicalTrial) -> str | None:
            # Must match how to_embedding_documents() derives this metadata field.
            if trial.last_update_post_date and trial.last_update_post_date.parsed_date:
                return trial.last_update_post_date.parsed_date.isoformat()
            return None

        to_upsert = [
            t for t in trials if stored_dates.get(t.nct_id) != _last_update_post_date(t)
        ]

        partial = self._upsert_trials(to_upsert)
        return SyncResult(
            processed=partial.processed,
            succeeded=partial.succeeded,
            failed=partial.failed,
            errors=partial.errors,
            checked=checked,
            unchanged=checked - len(to_upsert),
        )


@lru_cache(maxsize=1)
def get_ingestion_controller() -> IngestionController:
    return IngestionController()

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.models.clinical_trial import CLINICAL_TRIAL_ARROW_SCHEMA, ClinicalTrial
from src.services.clinical_trials_api import ClinicalTrialsApiService

DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "data" / "clinical_trials.parquet"
DEFAULT_CHECKPOINT_PATH = BACKEND_ROOT / "data" / ".fetch_checkpoint.json"
DEFAULT_PARTS_DIR = BACKEND_ROOT / "data" / ".fetch_parts"
DEFAULT_SKIPPED_LOG_PATH = BACKEND_ROOT / "data" / ".fetch_skipped.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_trials_data")


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Checkpoint at %s is unreadable (%s); ignoring it.", path, exc)
        return None


def save_checkpoint(path: Path, state: dict) -> None:
    state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    os.replace(tmp_path, path)


def flush_chunk(records: list[dict], chunk_path: Path, schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(records, schema=schema)
    with pq.ParquetWriter(chunk_path, schema) as writer:
        writer.write_table(table)


def merge_chunks(chunk_paths: list[Path], output_path: Path, schema: pa.Schema) -> int:
    tmp_output = output_path.with_suffix(".parquet.tmp")
    total_rows = 0
    with pq.ParquetWriter(tmp_output, schema) as writer:
        for chunk_path in chunk_paths:
            table = pq.read_table(chunk_path)
            writer.write_table(table)
            total_rows += table.num_rows
    os.replace(tmp_output, output_path)
    return total_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the full ClinicalTrials.gov v2 dataset into one parquet file."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    parser.add_argument(
        "--skipped-log-path", type=Path, default=DEFAULT_SKIPPED_LOG_PATH
    )
    parser.add_argument(
        "--page-size", type=int, default=1000, help="API page size (max 1000)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Records per row-group / part file.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop after N pages (for smoke testing).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any existing checkpoint/parts and start over.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.restart:
        for part in args.parts_dir.glob("chunk_*.parquet"):
            part.unlink()
        if args.checkpoint_path.exists():
            args.checkpoint_path.unlink()
        checkpoint = None
    else:
        checkpoint = load_checkpoint(args.checkpoint_path)

    if checkpoint and checkpoint.get("finished"):
        logger.info("Checkpoint already marked finished; re-running merge only.")
        chunk_paths = [args.parts_dir / name for name in checkpoint["chunk_files"]]
        total_rows = merge_chunks(
            chunk_paths, args.output_path, CLINICAL_TRIAL_ARROW_SCHEMA
        )
        logger.info("Wrote %s records to %s", total_rows, args.output_path)
        return

    if checkpoint:
        for name in checkpoint["chunk_files"]:
            if not (args.parts_dir / name).exists():
                raise SystemExit(
                    f"Checkpoint references missing part file {name!r}. "
                    "Re-run with --restart to start over safely."
                )
        chunk_files: list[str] = list(checkpoint["chunk_files"])
        next_page_token = checkpoint["next_page_token"]
        total_records_written: int = checkpoint["total_records_written"]
        chunk_index = len(chunk_files)
        started_at = checkpoint["started_at"]
        logger.info(
            "Resuming from checkpoint: %s records already written across %s chunks.",
            total_records_written,
            chunk_index,
        )
    else:
        chunk_files = []
        next_page_token = None
        total_records_written = 0
        chunk_index = 0
        started_at = datetime.now(timezone.utc).isoformat()

    api_service = ClinicalTrialsApiService()
    total_count = api_service.get_total_count()
    logger.info("ClinicalTrials.gov reports %s total studies.", total_count)

    buffer: list[dict] = []
    skipped_count = 0

    def flush_buffer() -> None:
        nonlocal buffer, chunk_index, total_records_written
        if not buffer:
            return
        chunk_path = args.parts_dir / f"chunk_{chunk_index:05d}.parquet"
        flush_chunk(buffer, chunk_path, CLINICAL_TRIAL_ARROW_SCHEMA)
        chunk_files.append(chunk_path.name)
        total_records_written += len(buffer)
        chunk_index += 1
        buffer = []
        save_checkpoint(
            args.checkpoint_path,
            {
                "version": 1,
                "next_page_token": next_page_token,
                "completed_chunks": chunk_index,
                "total_records_written": total_records_written,
                "chunk_files": chunk_files,
                "page_size": args.page_size,
                "chunk_size": args.chunk_size,
                "source_total_count": total_count,
                "finished": False,
                "started_at": started_at,
            },
        )

    with tqdm(
        total=total_count,
        initial=total_records_written,
        unit="study",
        desc="Fetching studies",
    ) as progress:
        for page in api_service.iter_pages(
            page_size=args.page_size,
            page_token=next_page_token,
            max_pages=args.max_pages,
        ):
            for raw_record in page.get("studies", []):
                try:
                    trial = ClinicalTrial.from_raw_api_response(raw_record)
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - never abort the run for one bad record
                    skipped_count += 1
                    nct_id = (
                        (raw_record.get("protocolSection") or {})
                        .get("identificationModule", {})
                        .get("nctId")
                    )
                    logger.warning("Skipping %s: %s", nct_id, exc)
                    with args.skipped_log_path.open("a") as f:
                        f.write(
                            json.dumps({"nct_id": nct_id, "error": str(exc)}) + "\n"
                        )
                    continue
                buffer.append(trial.model_dump(mode="python"))
                progress.update(1)

            next_page_token = page.get("nextPageToken")

            if len(buffer) >= args.chunk_size:
                flush_buffer()

    flush_buffer()  # flush the final partial chunk

    logger.info(
        "Fetch complete: %s records written, %s skipped. Merging %s chunks into %s...",
        total_records_written,
        skipped_count,
        len(chunk_files),
        args.output_path,
    )

    chunk_paths = [args.parts_dir / name for name in chunk_files]
    merged_rows = merge_chunks(
        chunk_paths, args.output_path, CLINICAL_TRIAL_ARROW_SCHEMA
    )

    if merged_rows != total_records_written:
        raise SystemExit(
            f"Merged row count ({merged_rows}) does not match checkpoint total "
            f"({total_records_written}); leaving part files in place for inspection."
        )

    save_checkpoint(
        args.checkpoint_path,
        {
            "version": 1,
            "next_page_token": next_page_token,
            "completed_chunks": chunk_index,
            "total_records_written": total_records_written,
            "chunk_files": chunk_files,
            "page_size": args.page_size,
            "chunk_size": args.chunk_size,
            "source_total_count": total_count,
            "finished": True,
            "started_at": started_at,
        },
    )

    for chunk_path in chunk_paths:
        chunk_path.unlink()

    logger.info(
        "Done. %s records in %s (%s skipped; see %s).",
        merged_rows,
        args.output_path,
        skipped_count,
        args.skipped_log_path,
    )


if __name__ == "__main__":
    main()

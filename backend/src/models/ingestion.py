from pydantic import BaseModel, Field

# Ingestion runs as a batch job (tools/ingest.py), not over HTTP, so these are
# result types only -- the request models the old router needed are gone.


class IngestionResult(BaseModel):
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    #: Rows read but not embedded because they failed the ingestion filter.
    skipped: int = 0
    #: Raw file rows covered, as [start, end). Lets a caller resume a bulk run
    #: without guessing which part of the corpus it already did.
    row_start: int = 0
    row_end: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


class SyncResult(IngestionResult):
    checked: int = 0
    unchanged: int = 0

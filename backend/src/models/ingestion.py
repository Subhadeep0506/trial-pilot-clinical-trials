from pydantic import BaseModel, Field


class IngestionResult(BaseModel):
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    row_start: int = 0
    row_end: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


class SyncResult(IngestionResult):
    checked: int = 0
    unchanged: int = 0

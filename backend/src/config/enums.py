"""Shared vocabulary for ClinicalTrials.gov field values."""

from enum import StrEnum


class OverallStatus(StrEnum):
    RECRUITING = "RECRUITING"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


class StudyType(StrEnum):
    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPANDED_ACCESS = "EXPANDED_ACCESS"


# The slice worth embedding: a trial a patient could still be enrolled into.
# Everything else -- completed, terminated, withdrawn, status unknown -- is
# retrievable from the parquet but has no place in the vector index.
# Roughly 70-75k of the corpus's 597,691 rows.
INGESTED_STATUSES = frozenset(
    {
        OverallStatus.RECRUITING,
        OverallStatus.ACTIVE_NOT_RECRUITING,
        OverallStatus.ENROLLING_BY_INVITATION,
    }
)

INGESTED_STUDY_TYPES = frozenset({StudyType.INTERVENTIONAL})

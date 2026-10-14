from collections.abc import Iterator
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_TIMEOUT = 30
DEFAULT_FIELDS = "ProtocolSection,DerivedSection"


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class ClinicalTrialsApiService:
    """Thin wrapper around the /studies list and /studies/{nctId} endpoints."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self._session = _build_session()

    def get_total_count(self) -> int:
        response = self._session.get(
            self.base_url,
            params={"pageSize": 1, "countTotal": "true"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["totalCount"]

    def iter_pages(
        self,
        page_size: int = 1000,
        page_token: str | None = None,
        max_pages: int | None = None,
        fields: str | None = DEFAULT_FIELDS,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw list-endpoint response payloads, following nextPageToken."""
        pages_fetched = 0
        while True:
            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token
            if fields:
                params["fields"] = fields
            response = self._session.get(
                self.base_url, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            yield payload
            pages_fetched += 1
            page_token = payload.get("nextPageToken")
            if not page_token or (max_pages is not None and pages_fetched >= max_pages):
                return

    def get_study(
        self, nct_id: str, fields: str | None = DEFAULT_FIELDS
    ) -> dict[str, Any]:
        """Fetch a single study by NCT ID (used by single ingest/update)."""
        url = f"{self.base_url}/{nct_id}"
        params = {"fields": fields} if fields else None
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def iter_studies_updated_since(
        self,
        since_date: str,
        page_size: int = 1000,
        fields: str | None = DEFAULT_FIELDS,
    ) -> Iterator[dict[str, Any]]:
        """Yield individual raw study dicts with lastUpdatePostDate >= since_date (YYYY-MM-DD).

        Used for delta sync instead of re-scanning the entire registry.
        """
        page_token: str | None = None
        advanced_filter = f"AREA[LastUpdatePostDate]RANGE[{since_date},MAX]"
        while True:
            params: dict[str, Any] = {
                "pageSize": page_size,
                "filter.advanced": advanced_filter,
            }
            if page_token:
                params["pageToken"] = page_token
            if fields:
                params["fields"] = fields
            response = self._session.get(
                self.base_url, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            yield from payload.get("studies", [])
            page_token = payload.get("nextPageToken")
            if not page_token:
                return

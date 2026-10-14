import logging
import uuid
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "clinical_trials"
EMBED_DIM = 768

_CONTENT_KEY = "page_content"
_METADATA_KEY = "metadata"

INDEXED_FIELDS = (
    "nct_id",
    "overall_status",
    "study_type",
    "phases",
    "lead_sponsor_name",
)


class VectorStoreService:
    """Thin wrapper around the LangChain/Qdrant vector store holding trial embeddings."""

    def __init__(
        self, embedding_function: Embeddings, collection_name: str = COLLECTION_NAME
    ):
        self._collection_name = collection_name

        Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=settings.QDRANT_PATH)
        self._ensure_collection()
        
        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=embedding_function,
            distance=models.Distance.COSINE,
        )

    def _ensure_collection(self) -> None:
        """Create the collection (full-precision 768-dim, cosine) and payload indexes if missing."""
        if not self._client.collection_exists(self._collection_name):
            logger.info("Creating Qdrant collection %s", self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=EMBED_DIM, distance=models.Distance.COSINE
                ),
            )
        for field in INDEXED_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=f"{_METADATA_KEY}.{field}",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.debug("Payload index on %s not (re)created: %s", field, exc)

    @staticmethod
    def _point_id(nct_id: str) -> str:
        """Deterministic uuid5 so re-ingesting the same trial upserts instead of duplicating."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, nct_id))

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Each document's metadata must include "nct_id"; its uuid5 is the Qdrant point id."""
        ids = [self._point_id(doc.metadata["nct_id"]) for doc in documents]
        return self._store.add_documents(documents, ids=ids)

    def delete(self, nct_ids: list[str]) -> None:
        if nct_ids:
            self._store.delete(ids=[self._point_id(nct_id) for nct_id in nct_ids])

    def get_by_ids(self, nct_ids: list[str]) -> list[Document]:
        if not nct_ids:
            return []
        records = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[self._point_id(nct_id) for nct_id in nct_ids],
            with_payload=True,
            with_vectors=False,
        )
        documents: list[Document] = []
        for record in records:
            payload = record.payload or {}
            documents.append(
                Document(
                    page_content=payload.get(_CONTENT_KEY, ""),
                    metadata=payload.get(_METADATA_KEY, {}),
                )
            )
        return documents


@lru_cache(maxsize=1)
def get_vector_store_service() -> VectorStoreService:
    from src.services.embedding import get_embedding_service

    return VectorStoreService(embedding_function=get_embedding_service().embeddings)

from fastapi import APIRouter, Depends

from src.models.inference import EmbedRequest, EmbedResponse
from src.router.common import run
from src.services.embedding import EmbeddingService, get_embedding_service

router = APIRouter(prefix="/embed", tags=["embed"])


@router.post("/predict", response_model=EmbedResponse)
def predict(
    request: EmbedRequest,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbedResponse:
    """Embed a batch of texts. The first call loads the GGUF and is slow."""
    vectors = run(service.embed_texts, request.texts)
    return EmbedResponse(
        embeddings=vectors,
        dim=len(vectors[0]) if vectors else 0,
        count=len(vectors),
        model=service.model_path,
    )

from fastapi import APIRouter, Depends

from src.models.inference import LLMRequest, LLMResponse
from src.router.common import run
from src.services.llm import LLMService, get_llm_service

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/predict", response_model=LLMResponse)
def predict(
    request: LLMRequest,
    service: LLMService = Depends(get_llm_service),
) -> LLMResponse:
    """Answer a single prompt with Azure OpenAI."""
    if request.role:
        service = LLMService.for_role(request.role)

    result = run(
        service.complete,
        request.prompt,
        system=request.system,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return LLMResponse(**result)

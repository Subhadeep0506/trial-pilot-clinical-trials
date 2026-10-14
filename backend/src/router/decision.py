from fastapi import APIRouter, Depends

from src.models.inference import DecisionRequest, DecisionResponse
from src.router.common import run
from src.services.decision_lm import DecisionLMService, get_decision_lm_service

router = APIRouter(prefix="/decision", tags=["decision"])


@router.post("/predict", response_model=DecisionResponse)
def predict(
    request: DecisionRequest,
    service: DecisionLMService = Depends(get_decision_lm_service),
) -> DecisionResponse:
    """Answer every question against the state in one forward pass.

    The first call builds the ONNX session and takes roughly 13 seconds.
    """
    answers = run(service.decide, request.state, request.questions)
    return DecisionResponse(answers=answers, model=service.model_name)

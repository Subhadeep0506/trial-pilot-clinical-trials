"""Request and response models for the direct model-inference routes."""

from typing import Any

from pydantic import BaseModel, Field

from src.services.decision_lm import Answer, Question


class EmbedRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        description="Texts to embed, returned in the same order.",
        examples=[["Absolute neutrophil count 2.4 x10^9/L", "ECOG performance status 1"]],
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int
    model: str


class LLMRequest(BaseModel):
    prompt: str = Field(min_length=1, examples=["Reply with the single word OK."])
    system: str | None = Field(
        default=None, description="Optional system instruction."
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    role: str | None = Field(
        default=None,
        description=(
            "Pick a per-role deployment (extract, parse, verify, verdict, "
            "clarify). Falls back to the default deployment when unset or when "
            "that role has no deployment configured."
        ),
    )


class LLMResponse(BaseModel):
    text: str
    model: str | None
    usage: dict[str, Any] = Field(default_factory=dict)


class DecisionRequest(BaseModel):
    """Mirrors Laya's own call shape: one state, many typed questions.

    All questions are answered in a single forward pass, so asking several at
    once costs no more than asking one.
    """

    state: str | dict[str, Any] = Field(
        description="The evidence to decide over: raw text or a JSON object.",
        examples=["Absolute neutrophil count 2.4 x10^9/L measured 2026-05-28."],
    )
    questions: dict[str, Question] = Field(
        min_length=1,
        description="Caller-chosen name -> typed question.",
        examples=[
            {
                "path": {
                    "type": "choice",
                    "instructions": "How should this eligibility criterion be verified?",
                    "criteria": {
                        "deterministic": "direct comparison against a structured field",
                        "predicate": "a numeric threshold or date window evaluated in code",
                        "narrative": "requires reading free text",
                    },
                },
                "stated": {
                    "type": "noul",
                    "instructions": "Is a neutrophil count stated?",
                },
            }
        ],
    )


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    model: str

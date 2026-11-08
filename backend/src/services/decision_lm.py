from __future__ import annotations

import json
import logging
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from pydantic import BaseModel, Field
from src.config import settings

if TYPE_CHECKING:
    from laya.onnx_agent import ONNXAgent

logger = logging.getLogger(__name__)


class QuestionType(StrEnum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class Question(BaseModel):
    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    def to_laya(self) -> dict[str, Any]:
        """Render into the dict shape Laya's own API expects."""
        payload: dict[str, Any] = {
            "type": self.type.value,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


class Answer(BaseModel):
    type: QuestionType
    noul: float | None = None
    choice: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    score: float | None = None
    distribution: list[float] = Field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False


def noul(instructions: str) -> Question:
    """Build a yes/no question."""
    return Question(type=QuestionType.NOUL, instructions=instructions)


def choice(instructions: str, criteria: dict[str, str]) -> Question:
    """Build a single-select question over a bounded set of options."""
    return Question(
        type=QuestionType.CHOICE, instructions=instructions, criteria=criteria
    )


def score(instructions: str, criteria: list[str]) -> Question:
    """Build an ordinal-score question over an ordered rubric (lowest to highest)."""
    return Question(
        type=QuestionType.SCORE, instructions=instructions, criteria=criteria
    )


class DecisionLMService:
    def __init__(
        self,
        onnx_path: str | None = None,
        checkpoint_id: str | None = None,
        min_confidence: dict[str, float] | None = None,
        batch_size: int | None = None,
    ):
        self._onnx_path = onnx_path or settings.DECISION_ONNX_PATH
        self._checkpoint_id = checkpoint_id or settings.DECISION_CHECKPOINT_ID
        self._min_confidence = (
            settings.DECISION_MIN_CONFIDENCE
            if min_confidence is None
            else min_confidence
        )
        self._batch_size = batch_size or settings.DECISION_BATCH_SIZE
        self._agent: ONNXAgent | None = None

    @property
    def model_name(self) -> str:
        """The graph backing this service, for echoing back in responses."""
        return self._onnx_path

    def _ensure_agent(self) -> ONNXAgent:
        """Build the ONNX session on first use; it is expensive, so build it once."""
        if self._agent is None:
            from laya.onnx_agent import ONNXAgent

            logger.info("Loading Laya decision model (ONNX) from %s", self._onnx_path)
            self._agent = ONNXAgent(
                model_id_or_path=self._checkpoint_id,
                onnx_path=self._onnx_path,
            )
            self._check_graph(self._agent)
        return self._agent

    @staticmethod
    def _check_graph(agent: ONNXAgent) -> None:
        session = getattr(agent, "session", None)
        if session is None:
            return
        marker = next((i for i in session.get_inputs() if i.name == "marker_pos"), None)
        if marker is None or len(marker.shape) < 2:
            return
        width = marker.shape[1]
        if isinstance(width, int):
            raise ValueError(
                f"Laya ONNX graph has a fixed option axis (marker_pos is [*, {width}]), so it cannot answer questions with a different number of options. Re-export with laya's scripts/export_onnx.py, which declares marker_pos dim 1 as the dynamic 'num_markers'."
            )

    def decide(
        self, state: dict[str, Any] | str, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        agent = self._ensure_agent()
        result = agent.predict(
            self._state_text(state),
            {name: q.to_laya() for name, q in questions.items()},
            **self._gate(),
        )
        return self._answers_from(result, questions)

    def decide_batch(
        self, states: list[dict[str, Any] | str], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        """Answer the same ``questions`` against many ``states``."""
        if not states:
            return []
        agent = self._ensure_agent()
        results = agent.predict_batch(
            [self._state_text(state) for state in states],
            {name: q.to_laya() for name, q in questions.items()},
            batch_size=self._batch_size,
            sort_by_length=self._batch_size > 1,
            **self._gate(),
        )
        return [self._answers_from(result, questions) for result in results]

    def _gate(self) -> dict[str, Any]:
        """Abstention thresholds, omitted entirely when none are configured."""
        return {"min_confidence": self._min_confidence} if self._min_confidence else {}

    @staticmethod
    def _state_text(state: dict[str, Any] | str) -> str:
        return (
            state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        )

    def _answers_from(
        self, result: dict[str, Any], questions: dict[str, Question]
    ) -> dict[str, Answer]:
        answers = result.get("answers", result)
        return {
            name: self._answer_from(question, answers.get(name) or {})
            for name, question in questions.items()
        }

    @staticmethod
    def _answer_from(question: Question, raw: dict[str, Any]) -> Answer:
        answer = Answer(
            type=question.type,
            confidence=float(raw.get("answer_confidence") or 0.0),
            abstained=raw.get("abstention") == "abstained",
        )

        if question.type is QuestionType.NOUL:
            answer.noul = raw.get("noul")
            return answer

        if question.type is QuestionType.CHOICE:
            probabilities = raw.get("probabilities") or {}
            answer.probabilities = {k: float(v) for k, v in probabilities.items()}
            answer.choice = raw.get("choice")
            if answer.choice is None and answer.probabilities:
                answer.choice = max(answer.probabilities, key=answer.probabilities.get)
            return answer

        probabilities = raw.get("probabilities") or {}
        answer.score = raw.get("score")
        answer.probabilities = {k: float(v) for k, v in probabilities.items()}
        try:
            answer.distribution = [
                float(probabilities[k]) for k in sorted(probabilities, key=int)
            ]
        except ValueError:  # non-numeric level keys; leave the ordered view empty
            logger.debug("Score levels are not integer-keyed: %s", list(probabilities))
        return answer


@lru_cache(maxsize=1)
def get_decision_lm_service() -> DecisionLMService:
    """Cached singleton -- the ONNX session is expensive to build, so build it once."""
    return DecisionLMService()

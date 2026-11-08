from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from src.config import settings

if TYPE_CHECKING:
    from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

_REQUIRED = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
)


class LLMConfigError(RuntimeError):
    """Azure OpenAI is not configured. Raised instead of failing inside the SDK."""


class LLMService:
    """Chat completions from Azure OpenAI, via langchain-openai."""

    def __init__(self, deployment: str | None = None, role: str | None = None):
        self._role = role
        self._deployment = deployment or settings.deployment_for_role(role)
        self._client: AzureChatOpenAI | None = None

    @property
    def deployment(self) -> str | None:
        return self._deployment

    @staticmethod
    def for_role(role: str) -> "LLMService":
        """A service bound to ``role``'s deployment, or the default if unset."""
        return LLMService(role=role)

    def _ensure_client(self) -> AzureChatOpenAI:
        """Build the client on first use, after checking the config is complete."""
        if self._client is None:
            missing = [key for key in _REQUIRED if not getattr(settings, key)]
            if not self._deployment:
                missing.append("AZURE_OPENAI_DEPLOYMENT")
            if missing:
                raise LLMConfigError(
                    "Azure OpenAI is not configured; set "
                    + ", ".join(missing)
                    + " in backend/.env"
                )

            from langchain_openai import AzureChatOpenAI

            logger.info("Creating Azure OpenAI client for %s", self._deployment)
            self._client = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_deployment=self._deployment,
            )
        return self._client

    @property
    def client(self) -> AzureChatOpenAI:
        """The LangChain chat model, for wiring into deepagents or a chain."""
        return self._ensure_client()

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Answer a single prompt, returning the text plus token usage."""
        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))
        overrides: dict[str, Any] = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens

        response = self._ensure_client().invoke(messages, **overrides)
        return {
            "text": (
                response.text()
                if callable(getattr(response, "text", None))
                else str(response.content)
            ),
            "model": self._deployment,
            "usage": response.usage_metadata or {},
        }


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Cached singleton bound to the default deployment."""
    return LLMService()

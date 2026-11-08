from fastapi import HTTPException
from src.services.llm import LLMConfigError


def run(action, *args, **kwargs):
    """Call a service, mapping its failures onto HTTP status codes."""
    try:
        return action(*args, **kwargs)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:  # a model file named in .env is absent
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface service errors as 502s
        raise HTTPException(status_code=502, detail=str(exc)) from exc

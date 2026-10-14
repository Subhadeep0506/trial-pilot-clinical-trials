import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")
MODELS_DIR = BACKEND_ROOT / "models"


def _env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, treating blank as unset."""
    value = os.environ.get(name)
    return value.strip() or default if value is not None else default


def _model_path(name: str, default: str) -> str:
    """Resolve a model setting that may be a bare name or a path."""
    value = _env(name, default) or default
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    local = MODELS_DIR / value
    return str(local) if local.exists() else value


QDRANT_URL = _env("QDRANT_URL")
QDRANT_API_KEY = _env("QDRANT_API_KEY")
QDRANT_PATH = _env("QDRANT_PATH", str(BACKEND_ROOT / "data" / "qdrant"))

# Embedding model, run in-process through llama-cpp-python.
EMBEDDING_MODEL = _model_path(
    "EMBEDDING_MODEL", "jina-embeddings-v5-omni-nano-retrieval-Q8_0.gguf"
)
EMBEDDING_N_CTX = int(_env("EMBEDDING_N_CTX", "2048"))
EMBEDDING_N_BATCH = int(_env("EMBEDDING_N_BATCH", "2048"))
# 2, not 8: this CPU is 2 performance cores plus 8 efficiency cores, and spilling
# the encoder onto the E-cores costs more than the parallelism wins. Measured per
# document on real trial text: 2 threads 1.68s, 4 1.79s, 6 2.03s, 8 2.07s, 10 2.13s.
EMBEDDING_N_THREADS = int(_env("EMBEDDING_N_THREADS", "2"))
AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT")
LLM_ROLES = ("extract", "parse", "verify", "verdict", "clarify")


def deployment_for_role(role: str | None = None) -> str | None:
    """The deployment for ``role``, falling back to the default deployment."""
    if not role:
        return AZURE_OPENAI_DEPLOYMENT
    return _env(f"AZURE_OPENAI_DEPLOYMENT_{role.upper()}") or AZURE_OPENAI_DEPLOYMENT


# Laya decision model (multilingual checkpoint), served through ONNX Runtime.
DECISION_ONNX_PATH = _model_path(
    "DECISION_ONNX_PATH", "laya-multilingual-onnx/model.onnx"
)
DECISION_CHECKPOINT_ID = _model_path("DECISION_CHECKPOINT_ID", "laya-ml-checkpoint")
DECISION_BATCH_SIZE = int(_env("DECISION_BATCH_SIZE", "1"))


def _load_min_confidence() -> dict[str, float]:
    """Per-bucket abstention thresholds, as a JSON object in the environment.

    Buckets are the cross product of answer type and option count band --
    ``"choice:3-5"``, ``"choice:6-10"``, ``"noul:2"`` and so on -- plus
    ``"default"``. An empty map leaves every bucket ungated at 0.0.

    These are a policy to be fitted, not a model property: Laya ships
    over-confident, and ``answer_confidence`` is not comparable across option
    counts. Treat any value here as a placeholder until fitted with
    ``laya.calibrate.fit_abstention_thresholds`` on real routing decisions.
    """
    raw = os.environ.get("DECISION_MIN_CONFIDENCE", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("DECISION_MIN_CONFIDENCE is not valid JSON; gating disabled")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("DECISION_MIN_CONFIDENCE must be a JSON object; gating disabled")
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


DECISION_MIN_CONFIDENCE = _load_min_confidence()

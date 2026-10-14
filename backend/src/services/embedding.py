from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from llama_cpp import Llama

from src.config import settings


class LlamaEmbeddings(Embeddings):
    """A LangChain ``Embeddings`` backed directly by ``llama_cpp.Llama``.

    Deliberately not langchain-community's ``LlamaCppEmbeddings``: that wrapper
    has no ``n_ubatch`` field and forbids extra kwargs, so it cannot raise the
    micro-batch size. The embedding models here are bidirectional encoders, which
    require the whole sequence in one micro-batch -- past llama.cpp's 512-token
    default the process dies on
    ``GGML_ASSERT(cparams.n_ubatch >= n_tokens)``, an abort rather than an
    exception. Trial documents routinely exceed that. langchain-community is also
    sunset upstream.

    ``Llama.embed`` truncates each input to ``n_batch`` tokens, so inputs longer
    than the configured window are cut rather than crashing.
    """

    def __init__(self, model: Llama, on_progress: Callable[[int], None] | None = None):
        self._model = model
        #: Called with 1 after each document is embedded. Embedding dominates
        #: ingestion at ~2s per document, so a caller that reports progress only
        #: per upsert batch appears frozen for minutes at a time.
        self.on_progress = on_progress

    @staticmethod
    def _flatten(vector) -> list[float]:
        """Collapse a per-token result to one vector.

        With a pooling type set the model returns one vector per input; with
        pooling NONE it returns one per token. Mean-pool that case so callers
        always get a single embedding.
        """
        if vector and isinstance(vector[0], list):
            width = len(vector[0])
            return [sum(row[i] for row in vector) / len(vector) for i in range(width)]
        return [float(x) for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one document per decode.

        Handing ``Llama.embed`` the whole list packs several sequences into one
        batch, and the KV cache is only ``n_ctx`` tokens in total -- shared across
        sequences, not per sequence. A full-length document therefore leaves no
        room for the next one and llama_decode fails with -2. One document per
        call gives each the whole window.
        """
        vectors = []
        for text in texts:
            vectors.append(self._flatten(self._model.embed(text)))
            if self.on_progress is not None:
                self.on_progress(1)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._flatten(self._model.embed(text))


class EmbeddingService:
    """Generates dense vector embeddings for clinical trial text."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        n_ctx: int | None = None,
        n_batch: int | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        seed: int = 42,
        verbose: bool = False,
    ):
        self._model_path = str(model_path or settings.EMBEDDING_MODEL)
        self._n_ctx = n_ctx if n_ctx is not None else settings.EMBEDDING_N_CTX
        self._n_batch = n_batch if n_batch is not None else settings.EMBEDDING_N_BATCH
        self._n_threads = (
            n_threads if n_threads is not None else settings.EMBEDDING_N_THREADS
        )
        self._n_gpu_layers = n_gpu_layers
        self._seed = seed
        self._verbose = verbose
        self._embeddings: LlamaEmbeddings | None = None
        self._dim: int | None = None

    @property
    def model_path(self) -> str:
        return self._model_path

    def _ensure_model(self) -> LlamaEmbeddings:
        """Load the GGUF on first use; it is expensive, so load it once.

        Deferred rather than loaded in __init__ so that importing this module, or
        constructing the service at app start, does not block for seconds.
        """
        if self._embeddings is None:
            model = Llama(
                model_path=self._model_path,
                embedding=True,
                n_ctx=self._n_ctx,
                n_batch=self._n_batch,
                # Must match n_batch: a bidirectional encoder needs the whole
                # sequence in one micro-batch.
                n_ubatch=self._n_batch,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                seed=self._seed,
                verbose=self._verbose,
            )
            self._embeddings = LlamaEmbeddings(model)
        return self._embeddings

    @property
    def embeddings(self) -> LlamaEmbeddings:
        """The underlying LangChain Embeddings object, for wiring into a VectorStore."""
        return self._ensure_model()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents, in the order given."""
        vectors = self._ensure_model().embed_documents(texts)
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        vector = self._ensure_model().embed_query(text)
        if self._dim is None:
            self._dim = len(vector)
        return vector

    @property
    def dim(self) -> int:
        """Vector width, learned from the first embedding and cached."""
        if self._dim is None:
            self.embed_query("dimension probe")
        return self._dim  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Cached singleton — the GGUF model is expensive to load, so load it once."""
    return EmbeddingService()

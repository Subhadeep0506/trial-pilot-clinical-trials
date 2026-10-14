"""FastAPI application entrypoint."""

from fastapi import FastAPI

from src.router.decision import router as decision_router
from src.router.embed import router as embed_router
from src.router.llm import router as llm_router

app = FastAPI(title="Clinical Trials Backend", version="0.1.0")

# Ingestion is deliberately not exposed over HTTP -- it is a long-running batch
# job. Run it with tools/ingest.py.
app.include_router(embed_router)
app.include_router(llm_router)
app.include_router(decision_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

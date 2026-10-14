# TrialPilot

A neurosymbolic clinical-trial matching copilot. You describe a patient in plain
text; it retrieves candidate trials from ClinicalTrials.gov, evaluates each
eligibility criterion, and returns a ranked shortlist with an evidence trail for
every verdict.

The design principle throughout: **verdicts, costs and ranking are computed by rule,
never written by a language model.** The LLM writes prose explanations; it does not
decide whether a patient is eligible.

> TrialPilot is a research and educational screening aid. It is not clinical advice
> and must not be used as the sole basis for treatment decisions.

## Status

Honest state as of 14 October 2026. This is a work in progress, not a running system.

| Area | State |
|---|---|
| Frontend | Complete, but **entirely mocked** — no backend calls yet |
| Trial ingestion | Working. 597,691 trials in parquet; local on-disk Qdrant |
| Embedding / decision / LLM services | Working, behind dev harness routes |
| Decision-model training data | **In progress** — synthetic corpus being generated in batches |
| Decision-model fine-tune | Not started; blocked on the corpus and on where to train |
| Matching pipeline | **Not built.** `src/controller/recommend.py` is empty |
| Tests / CI | None, by decision |

There is no git history, so [TECHNICAL_REQUIREMENTS.md](TECHNICAL_REQUIREMENTS.md) is
the record of what was decided and why. Read it before changing anything — it also
lists which requirements from the original BRD were deliberately dropped.

## Layout

```
backend/
  src/
    config/      settings, env loading
    controller/  ingestion; recommend.py (the matching pipeline — empty)
    models/      pydantic request/response shapes
    router/      FastAPI routes, incl. the /embed /decision /llm dev harness
    services/    embedding, decision_lm (Laya), llm (Azure), vector_store (Qdrant)
  tools/
    ingest.py              batch trial ingestion (a CLI, not an API)
    fetch_trials_data.py   corpus download
    data/
      parser_train_data.py synthetic training-data generator for the decision model
  models/        local model files (GGUF + ONNX); not Hub ids, see "Gotchas"
  data/          parquet corpus, Qdrant store, synthetic training corpus
frontend/        TanStack Start + React
```

## Setup

Python **3.14**, managed with `uv`. Behind the corporate TLS proxy, `--native-tls` is
not optional:

```bash
cd backend
uv sync --native-tls
```

Secrets go in `backend/.env`. Note that `load_dotenv` will **not** override a
variable already exported in your real environment — if a setting seems to be
ignored, check `env` before suspecting the code.

## Common commands

Run everything from `backend/`.

```bash
# Dev API (harness routes: /embed/predict, /decision/predict, /llm/predict)
.venv/Scripts/python.exe main.py

# Ingest trials — a long batch job, not an HTTP endpoint
.venv/Scripts/python.exe tools/ingest.py bulk --limit 1000
.venv/Scripts/python.exe tools/ingest.py single NCT05129748
.venv/Scripts/python.exe tools/ingest.py sync --since 2026-10-01

# Synthetic training data for the decision model
.venv/Scripts/python.exe tools/data/parser_train_data.py dry-run --count 200
.venv/Scripts/python.exe tools/data/parser_train_data.py check
.venv/Scripts/python.exe tools/data/parser_train_data.py generate --count 500
.venv/Scripts/python.exe tools/data/parser_train_data.py stats
```

`generate` always appends and skips work already done, so it is safe to re-run — the
corpus is built up in batches over time. `dry-run` makes no API calls at all.

## Why the decision model is fine-tuned

Laya is a non-generative encoder: it scores options declared in advance. It cannot
emit a drug name, a lab value, a date, or an arbitrary number — so it cannot produce
a patient profile on its own, and must not be asked to. It answers bounded fields
(sex, age *band*, ECOG, smoking, prior lines); the LLM fills everything else.

The stock checkpoint is not good enough for even that. Measured on a real narrative,
it answered ECOG 0 at 0.836 confidence for a patient clearly described as ECOG 1 —
wrong, plausible, and confident enough to pass any sane threshold. Its
"is this explicitly stated?" head returns the same value whether a fact is present or
absent. Both are label problems, so both are fixable with training data, which is
what `tools/data/parser_train_data.py` produces.

That generator works backwards: it samples a ground-truth patient in code, then asks
an LLM to write prose expressing exactly those facts. The LLM only writes; it never
decides an answer. That is what keeps its own ECOG mistakes out of the training
targets. See §4.4 of the technical requirements.

## Gotchas

These each cost real time to diagnose. Full list in §9 of the technical requirements.

- **`truststore` breaks all outbound HTTPS** on Python 3.14, failing with
  `RecursionError: Stack overflow (used ~1952 kB)` — an error that looks like a bug in
  whatever code you were writing. It is injected into every process by
  `pip_system_certs.pth`, so you did not opt in. Fix:
  `pip._vendor.truststore.extract_from_ssl()` at startup.
- **Model checkpoints are local directories**, never Hub ids — a consequence of the above.
- **Batching and threads make things slower here**, not faster. The dev box is a 15 W
  ultrabook: 2 performance cores and often ~2.7 GB free. Laya is fastest at batch 1;
  embedding is fastest at 2 threads. Check measured numbers before optimising.
- **Anything touching the corpus must stream.** Loading the full parquet starves the
  embedding model's allocator and causes hard aborts, not slowdowns.

## Where to read next

1. [TECHNICAL_REQUIREMENTS.md](TECHNICAL_REQUIREMENTS.md) — the agreed spec, and the
   authority when sources conflict.
2. `app-design-idea.md` — the product vision.
3. `TrialMatch_BRD_v1.docx` — background only; it describes a materially different
   system and much of it is formally dropped.

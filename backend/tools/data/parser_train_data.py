"""Generate synthetic training data for fine-tuning the Laya decision model.

Run it from the backend root:

    .venv/Scripts/python.exe tools/data/parser_train_data.py dry-run --count 200
    .venv/Scripts/python.exe tools/data/parser_train_data.py check
    .venv/Scripts/python.exe tools/data/parser_train_data.py generate --count 500
    .venv/Scripts/python.exe tools/data/parser_train_data.py stats
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = BACKEND_ROOT / "data" / "synthetic"
DATASET = "intake_v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("parser_train_data")


# MIRRORED CONTRACTS
#
# Duplicated on purpose; this file imports nothing from src/.
#
#   Question dict  {"type": "choice"|"noul"|"score", "instructions": str,
#                   "criteria": dict|list|None}
#                  -- the shape src/services/decision_lm.py:Question.to_laya() emits.
#                     `choice` takes {option: description}, `score` an ordered list
#                     (lowest first), `noul` no criteria at all.
#
#   Training row   {"state": str, "questions": {...}, "expected"|"gold": {...}}
#                  -- read by laya/train.py:items_from_rows().
#
#   Label literals -- laya/train.py:target_from_expected() is strict about these:
#                     choice -> the option key, as a string
#                     noul   -> a JSON boolean
#                     score  -> the ZERO-BASED level index, not the level's label.
#                               Fractional values interpolate across adjacent levels,
#                               which is how "two or three prior lines" is expressed.
#                     gold   -> {"probabilities": {...}} keyed by option label, by
#                               "false"/"true" for noul, or by level index for score.
#
# Getting a literal wrong does not raise: train.py counts the item under
# `invalid_target` and silently drops it. The `laya-train --dry-run` verification step
# exists to catch exactly that.


def noul(instructions: str) -> dict[str, Any]:
    """A yes/no question."""
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    """A single-select question over a bounded option set."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
    """An ordinal question over an ordered rubric, lowest level first."""
    return {"type": "score", "instructions": instructions, "criteria": criteria}


SEX = {
    "female": "the patient is female",
    "male": "the patient is male",
    "unknown": "the patient's sex is not stated or cannot be determined",
}

AGE_BAND = {
    "under_18": "younger than 18 years",
    "18_39": "between 18 and 39 years",
    "40_64": "between 40 and 64 years",
    "65_74": "between 65 and 74 years",
    "75_plus": "75 years or older",
    "unknown": "age is not stated",
}

SMOKING = {
    "never": "has never smoked",
    "former": "smoked in the past but has quit",
    "current": "currently smokes",
    "unknown": "smoking status is not stated",
}

MENOPAUSAL = {
    "pregnant": "currently pregnant",
    "premenopausal": "premenopausal and not pregnant",
    "postmenopausal": "postmenopausal",
    "not_applicable": "not applicable to this patient",
    "unknown": "not stated",
}

ECOG = {
    "ecog_0": "fully active, able to carry on all pre-disease activity without restriction",
    "ecog_1": "restricted in strenuous activity but ambulatory and able to do light work",
    "ecog_2": "ambulatory and self-caring but unable to work; up more than 50% of waking hours",
    "ecog_3": "capable of only limited self-care; confined to bed or chair more than 50% of waking hours",
    "ecog_4": "completely disabled; totally confined to bed or chair",
    "unknown": "performance status is not stated or implied",
}

PRIOR_LINE_LEVELS = ["0", "1", "2", "3", "4+"]

UNIVERSAL_FIELDS = {
    "sex": choice("What is the biological sex of the patient described?", SEX),
    "age_band": choice("Which age band does the patient fall into?", AGE_BAND),
    "smoking_status": choice("What is the patient's smoking status?", SMOKING),
    "pregnancy_menopausal_status": choice(
        "What is the patient's pregnancy or menopausal status?", MENOPAUSAL
    ),
    "performance_status": choice(
        "What is the patient's ECOG performance status?", ECOG
    ),
}

STATED_FIELDS = {
    "sex_stated": ("sex", "Is the patient's sex explicitly stated in the text?"),
    "age_stated": ("age_band", "Is the patient's age explicitly stated in the text?"),
    "smoking_stated": (
        "smoking_status",
        "Is the patient's smoking history explicitly stated in the text?",
    ),
    "menopausal_stated": (
        "pregnancy_menopausal_status",
        "Is the patient's menopausal status explicitly stated in the text?",
    ),
    "performance_status_stated": (
        "performance_status",
        "Is a formal performance status score explicitly stated in the text?",
    ),
}

ONCOLOGY_FIELDS = {
    "prior_lines": score(
        "How many prior lines of systemic therapy has the patient received?",
        PRIOR_LINE_LEVELS,
    ),
}


def questions_for(domain: str) -> dict[str, dict[str, Any]]:
    """The catalogue asked of one row. Keyed by condition, as section 6.1 specifies."""
    questions: dict[str, dict[str, Any]] = dict(UNIVERSAL_FIELDS)
    for name, (_, instructions) in STATED_FIELDS.items():
        questions[name] = noul(instructions)
    if domain == "oncology":
        questions.update(ONCOLOGY_FIELDS)
    return questions


VAGUE_OK = {"age_band", "smoking_status", "performance_status", "prior_lines"}
DOMAIN_MIX = {"oncology": 0.70, "non_oncology": 0.20, "off_topic": 0.10}
PRESENTATION_MIX = {
    "clinician": 0.30,
    "caregiver": 0.30,
    "vague": 0.20,
    "partial": 0.15,
    "adversarial": 0.05,
}

DISCLOSURE_WEIGHTS = {
    "clinician": {"explicit": 0.80, "implied": 0.08, "vague": 0.02, "omit": 0.10},
    "caregiver": {"explicit": 0.40, "implied": 0.35, "vague": 0.10, "omit": 0.15},
    "vague": {"explicit": 0.10, "implied": 0.20, "vague": 0.55, "omit": 0.15},
    "partial": {"explicit": 0.35, "implied": 0.10, "vague": 0.05, "omit": 0.50},
    "adversarial": {"explicit": 0.45, "implied": 0.25, "vague": 0.15, "omit": 0.15},
}

ADVERSARIAL_TWISTS = [
    "Mention a different family member's unrelated cancer history, then make clear the "
    "subject of the request is the patient themselves.",
    "State one fact and then correct it mid-sentence to the true value.",
    "Use a negated phrasing for at least one fact (for example 'never been a smoker' "
    "rather than naming a status).",
    "Include a distracting second person -- a spouse or sibling -- whose details must "
    "not be confused with the patient's.",
    "Mention a lab value in unusual units alongside the correct clinical facts.",
]

ONCOLOGY_CASES = [
    (
        "HER2-positive metastatic breast cancer",
        [
            "trastuzumab + pertuzumab + docetaxel",
            "T-DM1",
            "tucatinib + capecitabine",
            "trastuzumab deruxtecan",
        ],
        "female",
    ),
    (
        "EGFR exon 19 deletion non-small cell lung cancer",
        ["osimertinib", "carboplatin + pemetrexed", "docetaxel"],
        None,
    ),
    (
        "MSS metastatic colorectal cancer",
        ["FOLFOX", "FOLFIRI + bevacizumab", "regorafenib", "trifluridine/tipiracil"],
        None,
    ),
    (
        "castration-resistant prostate cancer",
        ["abiraterone", "enzalutamide", "docetaxel", "cabazitaxel"],
        "male",
    ),
    (
        "high-grade serous ovarian cancer",
        ["carboplatin + paclitaxel", "niraparib", "pegylated liposomal doxorubicin"],
        "female",
    ),
    (
        "relapsed diffuse large B-cell lymphoma",
        ["R-CHOP", "R-ICE", "polatuzumab vedotin", "CAR-T"],
        None,
    ),
    (
        "BRAF V600E melanoma",
        ["dabrafenib + trametinib", "pembrolizumab", "ipilimumab + nivolumab"],
        None,
    ),
    ("pancreatic adenocarcinoma", ["FOLFIRINOX", "gemcitabine + nab-paclitaxel"], None),
]

NON_ONCOLOGY_CASES = [
    (
        "heart failure with reduced ejection fraction",
        ["sacubitril/valsartan", "carvedilol", "dapagliflozin"],
        None,
    ),
    ("rheumatoid arthritis", ["methotrexate", "adalimumab", "tofacitinib"], None),
    (
        "type 2 diabetes with diabetic nephropathy",
        ["metformin", "semaglutide", "empagliflozin"],
        None,
    ),
    (
        "relapsing-remitting multiple sclerosis",
        ["ocrelizumab", "dimethyl fumarate"],
        None,
    ),
    (
        "moderate-to-severe Crohn's disease",
        ["infliximab", "ustekinumab", "azathioprine"],
        None,
    ),
    ("idiopathic pulmonary fibrosis", ["pirfenidone", "nintedanib"], None),
    (
        "chronic obstructive pulmonary disease",
        ["tiotropium", "salmeterol/fluticasone"],
        None,
    ),
]

OFF_TOPIC_PROMPTS = [
    "what's the weather in Boston tomorrow?",
    "can you explain how clinical trial phases work?",
    "how much does a knee replacement cost in the US?",
    "I'm writing a paper about oncology drug pricing, can you help?",
    "what are the side effects of ibuprofen?",
    "hi",
    "is this service free?",
    "my laptop won't connect to wifi",
    "what does ECOG stand for?",
    "find me a cardiologist near Chicago",
    "thanks, that was helpful!",
    "summarise the latest guidelines on breast cancer screening",
]


@dataclass
class Record:
    """One row's ground truth. Everything a label needs is decided here, before any
    text exists -- which is what makes the labels exact."""

    seed: int
    domain: str
    presentation: str
    profile: dict[str, Any] = field(default_factory=dict)
    disclosure: dict[str, str] = field(default_factory=dict)
    twist: str | None = None
    narrative: str | None = None
    generator: str | None = None


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def _age_band(age: int) -> str:
    if age < 18:
        return "under_18"
    if age < 40:
        return "18_39"
    if age < 65:
        return "40_64"
    if age < 75:
        return "65_74"
    return "75_plus"


def _sample_age(rng: random.Random) -> int:
    """Trial-age adults mostly, with a small paediatric slice."""
    bucket = rng.choices(
        ["paediatric", "young", "middle", "older", "elderly"],
        weights=[0.04, 0.12, 0.42, 0.27, 0.15],
        k=1,
    )[0]
    return {
        "paediatric": lambda: rng.randint(3, 17),
        "young": lambda: rng.randint(18, 39),
        "middle": lambda: rng.randint(40, 64),
        "older": lambda: rng.randint(65, 74),
        "elderly": lambda: rng.randint(75, 91),
    }[bucket]()


def sample_record(seed: int) -> Record:
    """Invent a patient and a disclosure plan from `seed`, deterministically."""
    rng = random.Random(seed)
    domain = _weighted(rng, DOMAIN_MIX)
    presentation = _weighted(rng, PRESENTATION_MIX)
    record = Record(seed=seed, domain=domain, presentation=presentation)

    if domain == "off_topic":
        record.profile = {"off_topic_prompt": rng.choice(OFF_TOPIC_PROMPTS)}
        record.disclosure = {name: "omit" for name in UNIVERSAL_FIELDS}
        return record

    cases = ONCOLOGY_CASES if domain == "oncology" else NON_ONCOLOGY_CASES
    condition, therapies, sex_constraint = rng.choice(cases)

    sex = sex_constraint or rng.choice(["female", "male"])
    age = _sample_age(rng)

    if sex == "male":
        menopausal = "not_applicable"
    elif age < 40 and rng.random() < 0.10:
        menopausal = "pregnant"
    elif age < 51:
        menopausal = "premenopausal"
    else:
        menopausal = "postmenopausal"

    ecog = rng.choices([0, 1, 2, 3, 4], weights=[0.26, 0.40, 0.21, 0.09, 0.04], k=1)[0]
    if age < 18:
        smoking = "never"
    else:
        smoking = rng.choices(
            ["never", "former", "current"], weights=[0.45, 0.37, 0.18], k=1
        )[0]

    profile: dict[str, Any] = {
        "condition": condition,
        "sex": sex,
        "age": age,
        "age_band": _age_band(age),
        "smoking_status": smoking,
        "pregnancy_menopausal_status": menopausal,
        "performance_status": f"ecog_{ecog}",
        "ecog_int": ecog,
    }

    if domain == "oncology":
        n_lines = rng.choices(
            [0, 1, 2, 3, 4], weights=[0.18, 0.27, 0.26, 0.18, 0.11], k=1
        )[0]
        profile["prior_lines"] = n_lines
        profile["therapies"] = therapies[:n_lines]
    else:
        profile["current_medications"] = rng.sample(therapies, k=min(2, len(therapies)))

    weights = DISCLOSURE_WEIGHTS[presentation]
    disclosure: dict[str, str] = {}
    for name in UNIVERSAL_FIELDS:
        mode = _weighted(rng, weights)
        if mode == "vague" and name not in VAGUE_OK:
            mode = "implied"
        disclosure[name] = mode
    if menopausal == "not_applicable":
        disclosure["pregnancy_menopausal_status"] = "omit"

    if sex_constraint and disclosure["sex"] == "omit":
        disclosure["sex"] = "implied"

    if domain == "oncology":
        mode = _weighted(rng, weights)
        disclosure["prior_lines"] = mode

    if presentation == "adversarial":
        record.twist = rng.choice(ADVERSARIAL_TWISTS)

    record.profile = profile
    record.disclosure = disclosure
    return record


def _spread(
    options: list[str], true_value: str, mass: float = 0.55
) -> dict[str, float]:
    """A soft distribution centred on the truth, for a fact the text only hedges at."""
    ordered = [o for o in options if o != "unknown"]
    if true_value not in ordered:
        return {true_value: 1.0}
    idx = ordered.index(true_value)
    probs: dict[str, float] = {true_value: mass}
    neighbours = [ordered[i] for i in (idx - 1, idx + 1) if 0 <= i < len(ordered)]
    if neighbours:
        share = (1.0 - mass - 0.05) / len(neighbours)
        for name in neighbours:
            probs[name] = share
    probs["unknown"] = 0.05
    return probs


def project_labels(record: Record) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return `(expected, gold)` for one record."""
    expected: dict[str, Any] = {}
    gold: dict[str, Any] = {}

    if record.domain == "off_topic":
        for name in UNIVERSAL_FIELDS:
            expected[name] = "unknown"
        for stated in STATED_FIELDS:
            expected[stated] = False
        return expected, gold

    sex_known = record.disclosure.get("sex") != "omit"

    for name, question in UNIVERSAL_FIELDS.items():
        mode = record.disclosure.get(name, "omit")
        truth = record.profile[name]
        if mode == "omit":
            if name == "pregnancy_menopausal_status" and truth == "not_applicable":
                expected[name] = "not_applicable" if sex_known else "unknown"
            else:
                expected[name] = "unknown"
        elif mode == "vague":
            gold[name] = {"probabilities": _spread(list(question["criteria"]), truth)}
        else:
            expected[name] = truth

    for stated, (base, _) in STATED_FIELDS.items():
        expected[stated] = record.disclosure.get(base) == "explicit"

    if record.domain == "oncology":
        mode = record.disclosure.get("prior_lines", "omit")
        lines = record.profile["prior_lines"]
        if mode == "vague":
            expected["prior_lines"] = lines - 0.5 if lines else 0.25
        elif mode != "omit":
            expected["prior_lines"] = lines
    return expected, gold


def build_row(record: Record) -> dict[str, Any]:
    """The JSONL row laya/train.py reads."""
    expected, gold = project_labels(record)
    row: dict[str, Any] = {
        "state": record.narrative,
        "questions": questions_for(record.domain),
    }
    if expected:
        row["expected"] = expected
    if gold:
        row["gold"] = gold
    row["seed"] = record.seed
    row["generator"] = record.generator
    row["domain"] = record.domain
    row["presentation"] = record.presentation
    return row


SYSTEM_PROMPT = """You write short, realistic free-text messages that someone would actually send to a clinical-trial matching service. You are given a patient's details, each tagged with how openly to express it. Your only job is to write the message.

Each fact carries one of three modes. Follow them exactly:

- explicit: state the value plainly and accurately, in the message's own register.
- implied: convey it WITHOUT naming the value. Describe what the person can or cannot do, what they have been through, how the writer refers to them. A reader should be able to infer it; the words naming it must not appear.
- vague: refer to it only approximately, with hedging ("getting on a bit", "a couple of treatments back", "she's slowed down a lot this year"). Never give the exact value.

Hard rules:
1. Never invent, round or alter a value marked explicit.
2. Add no clinical detail that is not in the list. If a fact is not listed, it is not known, leave it out entirely rather than inventing something plausible.
3. Write only the message. No JSON, no field names, no mode tags, no preamble, no commentary, and no quotation marks wrapping the whole thing.
4. Vary wording, length and tone between messages. Real people do not write to a template.
5. Keep it between one and six sentences.
"""

STYLE_BRIEFS = {
    "clinician": (
        "Register: a clinician's terse intake shorthand. Abbreviations, slashes and sentence fragments are expected. Example of the register only, do not reuse its content: '58F with HER2+ metastatic breast cancer, progressed on trastuzumab + pertuzumab, brain mets treated with SRS 8 weeks ago. ECOG 1.'"
    ),
    "caregiver": (
        "Register: a family member writing in plain, non-clinical words about someone they care for. Warm, slightly rambling, everyday descriptions rather than medical scores."
    ),
    "vague": (
        "Register: someone unsure of the details who hedges almost everything. Approximations, 'I think', 'a while back', 'a couple of'. They are trying to help but do not have the records in front of them."
    ),
    "partial": (
        "Register: a brief, incomplete enquiry. The person mentions only a few things and does not think to include the rest."
    ),
    "adversarial": (
        "Register: realistic but awkward for an automated reader, in the specific way described below."
    ),
}


def build_user_prompt(record: Record) -> str:
    """The per-row instruction. Stable text lives in SYSTEM_PROMPT so it can cache."""
    if record.domain == "off_topic":
        return (
            "Write a single short message to a clinical-trial matching service that "
            "contains NO patient details at all -- no age, sex, condition, smoking "
            "history or functional status, about anyone. It should read naturally as an "
            "off-topic or general enquiry.\n\n"
            f"Use this as the gist, reworded in your own way: "
            f"{record.profile['off_topic_prompt']}\n\n"
            "Write only the message."
        )

    profile = record.profile
    facts: list[str] = []

    def add(field_name: str, description: str) -> None:
        mode = record.disclosure.get(field_name)
        if mode and mode != "omit":
            facts.append(f"- {description} [{mode}]")

    add("sex", f"Sex: {profile['sex']}")
    add("age_band", f"Age: {profile['age']} years old")
    add("smoking_status", f"Smoking history: {SMOKING[profile['smoking_status']]}")
    add(
        "pregnancy_menopausal_status",
        f"Reproductive status: {MENOPAUSAL[profile['pregnancy_menopausal_status']]}",
    )
    add(
        "performance_status",
        f"Functional status: ECOG {profile['ecog_int']} "
        f"-- {ECOG['ecog_' + str(profile['ecog_int'])]}",
    )

    if record.domain == "oncology":
        therapies = profile.get("therapies") or []
        detail = ", ".join(therapies) if therapies else "no prior systemic therapy"
        add(
            "prior_lines",
            f"Prior lines of systemic therapy: {profile['prior_lines']} ({detail})",
        )
    else:
        meds = ", ".join(profile.get("current_medications", []))
        facts.append(f"- Current medications: {meds} [explicit]")

    brief = STYLE_BRIEFS[record.presentation]
    twist = f"\n\nAdditional requirement: {record.twist}" if record.twist else ""
    listed = "\n".join(facts) if facts else "- (no further details are known)"

    return (
        f"{brief}{twist}\n\n"
        f"Condition (always state this plainly): {profile['condition']}\n\n"
        f"Facts and their modes:\n{listed}\n\n"
        "Write only the message."
    )


class Writer:
    """One generator model behind a uniform call. Clients are built lazily and reused."""

    def __init__(self, spec: str):
        if ":" not in spec:
            raise ValueError(
                f"model spec {spec!r} must look like 'foundry:claude-opus-5' or "
                f"'azure:<deployment-name>'"
            )
        self.provider, self.model = spec.split(":", 1)
        if self.provider not in ("foundry", "azure"):
            raise ValueError(f"unknown provider {self.provider!r} in {spec!r}")
        self.spec = spec
        self._client: Any = None
        self._lock = threading.Lock()
        # Resolved on the first Azure call; see write(). Newer deployments reject
        # `max_tokens`, older ones reject `max_completion_tokens`, and nothing in the
        # deployment name says which, so it is settled once by asking.
        self._token_param: str | None = None

    def _ensure_client(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
            if self.provider == "foundry":
                from anthropic import AnthropicFoundry

                api_key = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
                base_url = os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL")
                if not api_key or not base_url:
                    raise RuntimeError(
                        "ANTHROPIC_FOUNDRY_API_KEY and ANTHROPIC_FOUNDRY_BASE_URL must "
                        "both be set to use a foundry: model"
                    )
                self._client = AnthropicFoundry(
                    api_key=api_key, base_url=base_url, max_retries=5
                )
            else:
                from openai import AzureOpenAI

                missing = [
                    key
                    for key in (
                        "AZURE_OPENAI_ENDPOINT",
                        "AZURE_OPENAI_API_KEY",
                        "AZURE_OPENAI_API_VERSION",
                    )
                    if not os.environ.get(key)
                ]
                if missing:
                    raise RuntimeError(
                        "Azure OpenAI is not configured; set "
                        + ", ".join(missing)
                        + " in backend/.env"
                    )
                # No `azure_deployment` here on purpose: setting it puts that
                # deployment in the URL path, which silently wins over the `model`
                # passed per call -- so `azure:gpt-5.5` would quietly hit whatever
                # AZURE_OPENAI_DEPLOYMENT happens to be. The spec stays authoritative;
                # `azure:` with no name falls back to the env default.
                self._client = AzureOpenAI(
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    max_retries=5,
                )
                if not self.model:
                    self.model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
                    self.spec = f"azure:{self.model}"
                if not self.model:
                    raise RuntimeError(
                        "no Azure deployment: pass azure:<deployment-name> or set "
                        "AZURE_OPENAI_DEPLOYMENT in backend/.env"
                    )
            return self._client

    def write(self, record: Record) -> str:
        prompt = build_user_prompt(record)
        client = self._ensure_client()
        if self.provider == "foundry":
            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Reasoning-era deployments (gpt-5.x) reject `max_tokens` outright and want
        # `max_completion_tokens`; older ones reject the new name. Try the modern name
        # first, fall back once, then remember which one this deployment accepts. The
        # budget is generous because on a reasoning model it also has to cover the
        # hidden reasoning tokens, and a budget spent entirely on those returns "".
        order = [self._token_param] if self._token_param else ["max_completion_tokens", "max_tokens"]
        last: Exception | None = None
        for name in order:
            try:
                response = client.chat.completions.create(
                    model=self.model, messages=messages, **{name: 4000}
                )
            except Exception as exc:  # noqa: BLE001 - only a parameter-name 400 retries
                last = exc
                if "max_tokens" in str(exc) or "max_completion_tokens" in str(exc):
                    continue
                raise
            self._token_param = name
            return (response.choices[0].message.content or "").strip()
        raise last if last else RuntimeError("no Azure response")


class StubWriter:
    """Deterministic fake narratives, so stages A/C/D can be exercised with no API calls."""

    spec = "stub:none"

    def write(self, record: Record) -> str:
        if record.domain == "off_topic":
            return record.profile["off_topic_prompt"]
        shown = [
            f"{name}={record.profile.get(name)}"
            for name, mode in sorted(record.disclosure.items())
            if mode != "omit"
        ]
        return f"[{record.presentation}] {record.profile['condition']}; " + "; ".join(
            shown
        )


def _paths(out_dir: Path) -> dict[str, Path]:
    return {
        "train": out_dir / f"{DATASET}.train.jsonl",
        "eval": out_dir / f"{DATASET}.eval.jsonl",
        "manifest": out_dir / f"{DATASET}.manifest.jsonl",
    }


def completed_seeds(paths: dict[str, Path]) -> set[int]:
    """Every seed already on disk, across all three files."""
    seeds: set[int] = set()
    for path in paths.values():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    seeds.add(int(json.loads(line)["seed"]))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
    return seeds


def next_seeds(done: set[int], base_seed: int, count: int, target: int) -> list[int]:
    """The next `count` unused seeds, never exceeding `target` rows in total."""
    room = max(0, target - len(done)) if target else count
    wanted = min(count, room)
    seeds: list[int] = []
    seed = base_seed
    while len(seeds) < wanted:
        if seed not in done:
            seeds.append(seed)
        seed += 1
    return seeds


def _archive(paths: dict[str, Path]) -> None:
    """Move an existing corpus aside rather than deleting it."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in paths.values():
        if path.exists():
            backup = path.with_suffix(path.suffix + f".{stamp}.bak")
            path.rename(backup)
            logger.info("archived %s -> %s", path.name, backup.name)


def generate(
    count: int,
    writers: list[Any],
    out_dir: Path,
    concurrency: int,
    base_seed: int,
    eval_every: int,
    target: int,
    max_failures: int,
    restart: bool,
) -> int:
    """Add one batch of rows to the corpus, appending to whatever is already there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(out_dir)

    if restart:
        _archive(paths)

    done = completed_seeds(paths)
    if done:
        logger.info("corpus already holds %s rows; appending", len(done))

    seeds = next_seeds(done, base_seed, count, target)
    if not seeds:
        logger.info(
            "nothing to do: %s rows already present, target is %s", len(done), target
        )
        return 0

    logger.info(
        "batch: %s new rows (seeds %s..%s), corpus will reach %s/%s",
        len(seeds),
        seeds[0],
        seeds[-1],
        len(done) + len(seeds),
        target,
    )

    handles = {name: path.open("a", encoding="utf-8") for name, path in paths.items()}
    write_lock = threading.Lock()
    stop = threading.Event()
    failures = 0
    written = 0
    last_error: Exception | None = None

    def work(seed: int) -> tuple[Record, Exception | None]:
        record = sample_record(seed)
        if stop.is_set():
            return record, None
        writer = writers[seed % len(writers)]
        try:
            record.narrative = writer.write(record)
            record.generator = writer.spec
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the batch
            return record, exc
        return record, None

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(work, seed) for seed in seeds]
            try:
                for future in as_completed(futures):
                    record, error = future.result()
                    if error is not None or not record.narrative:
                        if stop.is_set() and error is None:
                            continue  # drained, not failed
                        failures += 1
                        last_error = error
                        logger.warning("seed %s failed: %s", record.seed, error)
                        if (
                            max_failures
                            and failures >= max_failures
                            and not stop.is_set()
                        ):
                            logger.error(
                                "stopping after %s failures (--max-failures=%s); "
                                "draining in-flight requests",
                                failures,
                                max_failures,
                            )
                            stop.set()
                        continue

                    row = build_row(record)
                    split = "eval" if record.seed % eval_every == 0 else "train"
                    manifest_row = {
                        "seed": record.seed,
                        "split": split,
                        "domain": record.domain,
                        "presentation": record.presentation,
                        "generator": record.generator,
                        "disclosure": record.disclosure,
                        "profile": record.profile,
                        "twist": record.twist,
                        "n_expected": len(row.get("expected") or {}),
                        "n_gold": len(row.get("gold") or {}),
                    }
                    with write_lock:
                        handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                        handles["manifest"].write(
                            json.dumps(manifest_row, ensure_ascii=False) + "\n"
                        )
                        for handle in handles.values():
                            handle.flush()
                        written += 1
                        if written % 50 == 0:
                            logger.info(
                                "written=%s failed=%s (%s/%s total)",
                                written,
                                failures,
                                len(done) + written,
                                target,
                            )
            except KeyboardInterrupt:
                logger.warning(
                    "interrupted -- draining in-flight requests, keeping %s rows",
                    written,
                )
                stop.set()
                raise
    except KeyboardInterrupt:
        pass
    finally:
        for handle in handles.values():
            handle.flush()
            handle.close()

    total = len(done) + written
    logger.info(
        "batch done: written=%s failed=%s | corpus=%s/%s -> %s",
        written,
        failures,
        total,
        target,
        out_dir,
    )
    if total < target:
        logger.info("run the same command again to add the next batch")

    if written == 0 and failures:
        logger.error("no rows written. Last error: %s", last_error)
        return 1
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Stages A/C/D with a stub writer: no API calls, no cost."""
    counts: dict[str, Counter] = defaultdict(Counter)
    labelled: Counter = Counter()
    stub = StubWriter()
    rows = []

    for seed in range(args.base_seed, args.base_seed + args.count):
        record = sample_record(seed)
        record.narrative = stub.write(record)
        record.generator = stub.spec
        row = build_row(record)
        rows.append(row)
        counts["domain"][record.domain] += 1
        counts["presentation"][record.presentation] += 1
        for name, value in (row.get("expected") or {}).items():
            labelled[name] += 1
            counts[f"label:{name}"][str(value)] += 1
        for name in row.get("gold") or {}:
            labelled[name] += 1
            counts[f"label:{name}"]["<gold>"] += 1

    print(f"sampled {args.count} records with no API calls\n")
    for axis in ("domain", "presentation"):
        print(f"{axis}:")
        for key, n in counts[axis].most_common():
            print(f"  {key:<16} {n:>6}  {n / args.count:6.1%}")
        print()

    print("labelled questions per row:")
    for name, n in labelled.most_common():
        print(f"  {name:<30} {n:>6}  {n / args.count:6.1%}")

    print("\nlabel distributions:")
    for name in sorted(k for k in counts if k.startswith("label:")):
        spread = "  ".join(f"{k}={v}" for k, v in sorted(counts[name].items()))
        print(f"  {name[6:]:<30} {spread}")

    if args.write:
        args.out.mkdir(parents=True, exist_ok=True)
        stub_path = args.out / f"{DATASET}.stub.jsonl"
        with stub_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(rows)} stub rows to {stub_path}")
        print("validate the format with:")
        print(
            f"  .venv/Scripts/python.exe -m laya.train_cli --dry-run "
            f"--data {stub_path} --base models/laya-ml-checkpoint"
        )

    print("\nexample row:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:1200])
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """One cheap call per configured model, to prove credentials before a bulk run."""
    failed = 0
    for spec in args.models:
        try:
            writer = Writer(spec)
            record = sample_record(args.base_seed)
            text = writer.write(record)
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            logger.error("%s FAILED: %s", spec, exc)
            failed += 1
            continue
        logger.info("%s OK", spec)
        print(f"\n--- {spec} ---\n{text}\n")
    return 1 if failed else 0


def cmd_generate(args: argparse.Namespace) -> int:
    writers = [Writer(spec) for spec in args.models]
    logger.info("models: %s", ", ".join(args.models))
    return generate(
        count=args.count,
        writers=writers,
        out_dir=args.out,
        concurrency=args.concurrency,
        base_seed=args.base_seed,
        eval_every=args.eval_every,
        target=args.target,
        max_failures=args.max_failures,
        restart=args.restart,
    )


def cmd_stats(args: argparse.Namespace) -> int:
    manifest = _paths(args.out)["manifest"]
    if not manifest.exists():
        logger.error("no manifest at %s -- run generate first", manifest)
        return 1

    axes: dict[str, Counter] = defaultdict(Counter)
    total = 0
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total += 1
            for axis in ("domain", "presentation", "generator", "split"):
                axes[axis][entry.get(axis)] += 1
            for name, mode in (entry.get("disclosure") or {}).items():
                axes[f"disclosure:{name}"][mode] += 1

    if not total:
        logger.error("manifest at %s is empty", manifest)
        return 1

    print(f"{total} rows in {manifest}\n")
    for axis in ("split", "domain", "presentation", "generator"):
        print(f"{axis}:")
        for key, n in axes[axis].most_common():
            print(f"  {str(key):<26} {n:>7}  {n / total:6.1%}")
        print()
    print("disclosure modes:")
    for axis in sorted(k for k in axes if k.startswith("disclosure:")):
        spread = "  ".join(f"{k}={v}" for k, v in sorted(axes[axis].items()))
        print(f"  {axis[11:]:<30} {spread}")

    paths = _paths(args.out)
    print("\nper-split generator (read from the data files themselves):")
    for split in ("train", "eval"):
        path = paths[split]
        if not path.exists():
            continue
        per_model: Counter = Counter()
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    per_model[json.loads(line).get("generator")] += 1
                    rows += 1
                except json.JSONDecodeError:
                    continue
        spread = "  ".join(f"{k}={v}" for k, v in sorted(per_model.items(), key=str))
        print(f"  {split:<6} {rows:>7} rows   {spread}")

    done = completed_seeds(paths)
    if done:
        nxt = next_seeds(done, 0, 1, 0)
        print(
            f"\nseeds used: {len(done)} (highest {max(done)}); next free seed: {nxt[0]}"
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0, help="first seed (default: 0)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="sample records with no API calls")
    dry.add_argument("--count", type=int, default=200)
    dry.add_argument(
        "--write",
        action="store_true",
        help="also write the stub rows to <out>/intake_v1.stub.jsonl for laya-train --dry-run",
    )
    dry.set_defaults(func=cmd_dry_run)

    check = sub.add_parser(
        "check", help="one live call per model, to verify credentials"
    )
    check.add_argument("--models", nargs="+", default=["foundry:claude-opus-5"])
    check.set_defaults(func=cmd_check)

    gen = sub.add_parser(
        "generate",
        help="add a batch of rows to the corpus (always appends; safe to re-run)",
    )
    gen.add_argument(
        "--count",
        type=int,
        default=500,
        help="how many NEW rows to add in this batch (default: 500)",
    )
    gen.add_argument(
        "--target",
        type=int,
        default=10000,
        help="total corpus size to stop at across all batches (default: 10000)",
    )
    gen.add_argument(
        "--models",
        nargs="+",
        default=["foundry:claude-opus-5", "foundry:claude-sonnet-5"],
        help="round-robin roster, e.g. foundry:claude-opus-5 azure:gpt-4o",
    )
    gen.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="parallel requests; Foundry has no Batches API, so this is the throughput lever",
    )
    gen.add_argument(
        "--eval-every", type=int, default=10, help="1 in N rows goes to the eval split"
    )
    gen.add_argument(
        "--max-failures",
        type=int,
        default=25,
        help="stop the batch after this many failed rows, keeping what was written "
        "(0 disables; default: 25)",
    )
    gen.add_argument(
        "--restart",
        action="store_true",
        help="start a fresh corpus, archiving any existing files to .bak first",
    )
    gen.set_defaults(func=cmd_generate)

    stats = sub.add_parser("stats", help="coverage report from the manifest")
    stats.set_defaults(func=cmd_stats)

    return parser.parse_args(argv)


def _disable_broken_truststore() -> None:
    """Undo truststore's SSL injection on Python 3.14, where it recurses forever.

    `pip_system_certs.pth` injects truststore into every process started from this
    venv, so this is not something the script opts into. On 3.14 its `wrap_socket`
    recurses until the C stack is exhausted and every HTTPS call dies with
    `RecursionError: Stack overflow (used ~1952 kB)` -- which reads like a bug in
    this script rather than in the TLS layer, and bit the Azure path first only
    because that connection happened to be opened first.

    Removing it falls back to certifi's CA bundle. Both the Foundry and Azure
    endpoints verify fine that way; if some future host needs a corporate root CA,
    this is the line that will have hidden it.
    """
    if sys.version_info < (3, 14):
        return
    import ssl

    if "truststore" not in getattr(ssl.SSLContext, "__module__", ""):
        return
    for name in ("pip._vendor.truststore", "truststore"):
        try:
            module = __import__(name, fromlist=["extract_from_ssl"])
            module.extract_from_ssl()
        except Exception:  # noqa: BLE001 - try the next candidate module
            continue
        if "truststore" not in getattr(ssl.SSLContext, "__module__", ""):
            logger.info(
                "removed truststore's SSL injection (%s); it breaks HTTPS on "
                "Python %s.%s. Certificates now come from certifi.",
                name, *sys.version_info[:2],
            )
            return


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(BACKEND_ROOT / ".env")
    except ImportError:
        logger.warning("python-dotenv not installed; relying on real environment vars")

    _disable_broken_truststore()

    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

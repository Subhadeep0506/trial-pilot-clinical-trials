import type {
  PatientProfile,
  TrialPilot,
  AgentLogEntry,
  Criterion,
  VerificationPath,
  ClarificationQuestion,
  CoverageStatement,
} from "./types";

export const SAMPLE_NARRATIVES: { id: string; label: string; text: string }[] = [
  {
    id: "her2-mbc",
    label: "58F · HER2+ metastatic breast cancer",
    text:
      "58F with HER2+ metastatic breast cancer, progressed on trastuzumab + pertuzumab, brain mets treated with SRS 8 weeks ago. ECOG 1. LVEF 58%. No prior trastuzumab deruxtecan. Last CT: stable extracranial disease, new 6mm cerebellar lesion.",
  },
  {
    id: "nsclc",
    label: "67M · EGFR-mutant NSCLC, post-osimertinib",
    text:
      "67M with stage IV EGFR exon 19 deletion NSCLC. Progressed on osimertinib after 14 months. ECOG 1. No brain mets. Adequate organ function. Prior platinum doublet declined.",
  },
  {
    id: "crc",
    label: "44F · MSS colorectal, 3rd line",
    text:
      "44F with MSS metastatic colorectal cancer, KRAS G12V, BRAF wild-type. Progressed on FOLFOX, FOLFIRI + bevacizumab. ECOG 0. Liver-dominant disease. Adequate marrow and hepatic function.",
  },
];

export const SAMPLE_PROFILE: PatientProfile = {
  demographics: { age: 58, sex: "F", weight_kg: 64, height_cm: 165 },
  conditions: [
    { name: "HER2+ metastatic breast cancer", snomed: "254837009", confidence: "high" },
    { name: "Brain metastasis", snomed: "94225004", confidence: "high" },
    { name: "Hypertension", snomed: "38341003", confidence: "low" },
  ],
  medications: [
    { name: "Trastuzumab", rxnorm: "224905", start: "2023-04-11", stop: "2025-01-08" },
    { name: "Pertuzumab", rxnorm: "1298952", start: "2023-04-11", stop: "2025-01-08" },
    { name: "Letrozole", rxnorm: "6011", start: "2022-09-02" },
  ],
  labs: [
    { name: "ANC", value: "2.4", unit: "10^9/L", date: "2026-05-28" },
    { name: "Platelets", value: "186", unit: "10^9/L", date: "2026-05-28" },
    { name: "Hemoglobin", value: "11.2", unit: "g/dL", date: "2026-05-28" },
    { name: "ALT", value: "28", unit: "U/L", date: "2026-05-28" },
    { name: "Creatinine", value: "0.84", unit: "mg/dL", date: "2026-05-28" },
    { name: "LVEF", value: "58", unit: "%", date: "2026-04-15" },
  ],
  performance: { name: "ECOG", value: "1" },
  cancer: {
    stage: "IV",
    biomarkers: [
      { name: "HER2", value: "IHC 3+" },
      { name: "ER", value: "Positive (90%)" },
      { name: "PR", value: "Positive (45%)" },
    ],
    prior_lines: 2,
  },
  summary: "58F · HER2+ MBC · ECOG 1",
};

const mk = (
  id: string,
  type: "inclusion" | "exclusion",
  text: string,
  verdict: "met" | "unmet" | "unknown",
  cls: Criterion["semantic_class"],
  path: VerificationPath,
  evidence?: Criterion["evidence"],
  missing?: string,
  cost_min?: number,
  clarification_options?: string[],
): Criterion => ({
  id, type, text, verdict, semantic_class: cls, path,
  evidence, missing_data: missing, cost_min, clarification_options,
});

export const MOCK_TRIALS: TrialPilot[] = [
  {
    nct_id: "NCT05129748",
    title:
      "Tucatinib plus trastuzumab and capecitabine in HER2-positive metastatic breast cancer with brain metastases",
    phase: "II",
    status: "Recruiting",
    sponsor: "Seagen / Pfizer",
    locations: [
      { city: "Cambridge", state: "MA", country: "USA", distance_mi: 8, status: "Recruiting" },
      { city: "Providence", state: "RI", country: "USA", distance_mi: 52, status: "Recruiting" },
    ],
    verdict: "eligible",
    tier: "high",
    verification_cost_min: 0,
    calibrated_confidence: 0.94,
    explanation:
      "All 8 criteria resolved deterministically or by direct-source narrative. No clinician verification required beyond a site call.",
    criteria: [
      mk("t1", "inclusion", "HER2-positive metastatic breast cancer", "met", "biomarker", "narrative",
        { source: "Pathology 2023-03-30", value: "HER2 IHC 3+", date: "2023-03-30" }),
      mk("t2", "inclusion", "Brain metastases (treated or untreated)", "met", "condition", "narrative",
        { source: "MRI brain 2026-04-01", value: "Solitary cerebellar lesion, 6 mm", date: "2026-04-01" }),
      mk("t3", "inclusion", "Prior trastuzumab", "met", "treatment", "deterministic",
        { source: "Medication history", value: "Trastuzumab 2023-04 to 2025-01", date: "2025-01-08" }),
      mk("t4", "exclusion", "Prior tucatinib, neratinib, or lapatinib", "met", "treatment", "deterministic",
        { source: "Medication history", value: "No prior tucatinib, neratinib, or lapatinib", date: "2026-05-28" }),
      mk("t5", "inclusion", "ECOG 0–2", "met", "performance", "deterministic",
        { source: "Oncology note 2026-05-20", value: "ECOG 1", date: "2026-05-20" }),
      mk("t6", "inclusion", "Adequate hepatic function: ALT ≤ 2.5× ULN", "met", "lab", "deterministic",
        { source: "LFTs 2026-05-28", value: "ALT 28 U/L (ULN 40)", date: "2026-05-28" }),
      mk("t7", "exclusion", "Known DPD deficiency", "met", "biomarker", "deterministic",
        { source: "Genomic panel 2024-06-10", value: "No pathogenic DPYD variant", date: "2024-06-10" }),
      mk("t8", "inclusion", "Creatinine clearance ≥ 50 mL/min", "met", "lab", "deterministic",
        { source: "Chem 2026-05-28", value: "eGFR 88 mL/min/1.73m²", date: "2026-05-28" }),
    ],
  },
  {
    nct_id: "NCT05893030",
    title:
      "Trastuzumab deruxtecan vs investigator's choice in HER2-positive metastatic breast cancer after prior anti-HER2 therapy",
    phase: "III",
    status: "Recruiting",
    sponsor: "Daiichi Sankyo",
    locations: [
      { city: "Boston", state: "MA", country: "USA", distance_mi: 12, status: "Recruiting" },
      { city: "New Haven", state: "CT", country: "USA", distance_mi: 98, status: "Recruiting" },
      { city: "New York", state: "NY", country: "USA", distance_mi: 184, status: "Active, not recruiting" },
    ],
    verdict: "eligible_pending_unknown",
    tier: "quick_check",
    verification_cost_min: 5,
    calibrated_confidence: 0.86,
    explanation:
      "Disease, biomarker, prior-therapy and organ-function criteria all resolved. One quick chart lookup remains: recent corticosteroid course for SRS edema, which gates the ILD exclusion.",
    criteria: [
      mk("c1", "inclusion", "Histologically confirmed HER2-positive metastatic breast cancer (IHC 3+ or ISH-amplified)", "met", "biomarker", "narrative",
        { source: "Pathology report 2023-03-30", value: "HER2 IHC 3+, ISH ratio 4.1", date: "2023-03-30" }),
      mk("c2", "inclusion", "Prior treatment with trastuzumab and pertuzumab in metastatic setting", "met", "treatment", "deterministic",
        { source: "Medication history", value: "Trastuzumab + pertuzumab, 2023-04-11 to 2025-01-08", date: "2025-01-08" }),
      mk("c3", "inclusion", "ECOG performance status 0–1", "met", "performance", "deterministic",
        { source: "Oncology note 2026-05-20", value: "ECOG 1", date: "2026-05-20" }),
      mk("c4", "inclusion", "LVEF ≥ 50% within 28 days of enrollment", "met", "lab", "deterministic",
        { source: "Echocardiogram 2026-04-15", value: "LVEF 58%", date: "2026-04-15" }),
      mk("c5", "inclusion", "Adequate hematologic function: ANC ≥ 1.5 × 10^9/L, platelets ≥ 100 × 10^9/L", "met", "lab", "deterministic",
        { source: "CBC 2026-05-28", value: "ANC 2.4; platelets 186", date: "2026-05-28" }),
      mk("c6", "exclusion", "Prior treatment with any anti-HER2 antibody-drug conjugate", "met", "treatment", "deterministic",
        { source: "Medication history", value: "No prior T-DXd, T-DM1, or other ADC", date: "2026-05-28" }),
      mk("c7", "exclusion", "Untreated CNS metastases", "met", "condition", "narrative",
        { source: "Radiation oncology note", value: "Cerebellar lesion treated with SRS on 2026-04-02", date: "2026-04-02" }),
      mk("c8", "exclusion", "History of non-infectious ILD/pneumonitis requiring steroids", "unknown", "condition", "predicate",
        undefined,
        "Recent SRS edema management not documented; corticosteroid course unclear.",
        5,
        [
          "No corticosteroids given for SRS edema",
          "Short taper only (<14 days, <20 mg/day equivalent)",
          "Prolonged or high-dose steroid course",
          "Not documented — cannot determine",
        ]),
      mk("c9", "inclusion", "Measurable disease per RECIST 1.1", "met", "condition", "narrative",
        { source: "CT chest/abdomen 2026-05-10", value: "Two measurable hepatic lesions, 18 mm and 22 mm", date: "2026-05-10" }),
      mk("c10", "exclusion", "Pregnancy or breastfeeding", "met", "demographic", "deterministic",
        { source: "Demographics", value: "Postmenopausal, age 58", date: "2026-06-01" }),
      mk("c11", "inclusion", "Life expectancy ≥ 12 weeks", "met", "other", "narrative",
        { source: "Oncology assessment", value: "Estimated > 6 months", date: "2026-05-20" }),
      mk("c12", "inclusion", "Signed informed consent", "met", "other", "deterministic",
        { source: "Workflow", value: "Pending site visit", date: "2026-06-01" }),
    ],
  },
  {
    nct_id: "NCT05382299",
    title:
      "Sacituzumab govitecan in pretreated HER2-low and HER2-positive metastatic breast cancer",
    phase: "II",
    status: "Recruiting",
    sponsor: "Gilead",
    locations: [
      { city: "Boston", state: "MA", country: "USA", distance_mi: 12, status: "Recruiting" },
    ],
    verdict: "eligible_pending_unknown",
    tier: "manual_review",
    verification_cost_min: 45,
    calibrated_confidence: 0.62,
    explanation:
      "Disease and prior-therapy criteria met. Two open items require external work — a central Trop-2 IHC on archival tissue, plus a second-malignancy workup.",
    criteria: [
      mk("s1", "inclusion", "Metastatic breast cancer with prior anti-HER2 therapy", "met", "treatment", "deterministic",
        { source: "Medication history", value: "Trastuzumab + pertuzumab", date: "2025-01-08" }),
      mk("s2", "inclusion", "Trop-2 expression ≥ 1+ by central IHC", "unknown", "biomarker", "narrative",
        undefined,
        "No Trop-2 IHC on file. Archival tissue would need to be submitted to central lab (~30 min coordinator work + turnaround).",
        30,
        [
          "Archival tissue available — will submit",
          "Archival tissue exhausted — new biopsy needed",
          "Not pursuing central IHC",
        ]),
      mk("s3", "inclusion", "ECOG 0–1", "met", "performance", "deterministic",
        { source: "Oncology note", value: "ECOG 1", date: "2026-05-20" }),
      mk("s4", "exclusion", "Prior treatment with sacituzumab govitecan or topoisomerase-1 inhibitor ADC", "met", "treatment", "deterministic",
        { source: "Medication history", value: "No prior topo-1 ADC", date: "2026-05-28" }),
      mk("s5", "inclusion", "Adequate organ function", "met", "lab", "deterministic",
        { source: "Labs 2026-05-28", value: "All values within protocol limits", date: "2026-05-28" }),
      mk("s6", "exclusion", "Active second malignancy", "unknown", "condition", "narrative",
        undefined,
        "Family history noted; no patient-level workup documented. Requires review of recent imaging + PMH.",
        15,
        [
          "No active second malignancy",
          "History of prior cancer, in remission > 5 years",
          "Active second malignancy present",
        ]),
    ],
  },
  {
    nct_id: "NCT05715944",
    title:
      "Investigational PI3K-alpha inhibitor in PIK3CA-mutant HR-positive HER2-negative breast cancer",
    phase: "I",
    status: "Recruiting",
    sponsor: "Academic consortium",
    locations: [
      { city: "Boston", state: "MA", country: "USA", distance_mi: 12, status: "Recruiting" },
    ],
    verdict: "ineligible",
    tier: "ineligible",
    verification_cost_min: 0,
    calibrated_confidence: 0.99,
    explanation:
      "Trial requires HER2-negative disease; patient is HER2 IHC 3+. Deterministic exclusion — no clinician verification needed.",
    criteria: [
      mk("p1", "inclusion", "HER2-negative breast cancer (IHC 0–1+ or ISH non-amplified)", "unmet", "biomarker", "deterministic",
        { source: "Pathology 2023-03-30", value: "HER2 IHC 3+ (positive)", date: "2023-03-30" }),
      mk("p2", "inclusion", "Documented PIK3CA mutation", "unknown", "biomarker", "deterministic",
        undefined, "PIK3CA status not in current genomic panel.", 20),
      mk("p3", "inclusion", "HR-positive disease", "met", "biomarker", "deterministic",
        { source: "Pathology", value: "ER 90%, PR 45%", date: "2023-03-30" }),
      mk("p4", "exclusion", "Diabetes requiring insulin", "met", "condition", "deterministic",
        { source: "Problem list", value: "No diabetes diagnosis", date: "2026-05-28" }),
    ],
  },
];

export const COVERAGE: CoverageStatement = {
  probability: 0.9,
  confidence: 0.95,
  calibration_n: 412,
};

/** Structured, bounded clarification questions surfaced after retrieval + gap analysis. */
export const MOCK_CLARIFICATIONS: ClarificationQuestion[] = [
  {
    id: "q1",
    trial_id: "NCT05893030",
    criterion_id: "c8",
    criterion_text: "Exclusion: history of non-infectious ILD/pneumonitis requiring steroids",
    current_data: "Radiation oncology note documents SRS 2026-04-02 for cerebellar lesion. No corticosteroid orders found in med rec for the surrounding window.",
    gap: "Corticosteroid course for SRS-related edema is not in the record. This gates the ILD exclusion for T-DXd.",
    options: [
      "No corticosteroids given for SRS edema",
      "Short taper only (<14 days, <20 mg/day equivalent)",
      "Prolonged or high-dose steroid course",
      "Not documented — cannot determine",
    ],
    round: 1,
  },
  {
    id: "q2",
    trial_id: "NCT05382299",
    criterion_id: "s2",
    criterion_text: "Inclusion: Trop-2 expression ≥ 1+ by central IHC",
    current_data: "Pathology on file: HER2 IHC 3+, ER 90%, PR 45%. No Trop-2 stain performed.",
    gap: "Trial requires central Trop-2 IHC. Archival tissue availability determines whether this is a quick submission or requires re-biopsy.",
    options: [
      "Archival tissue available — will submit",
      "Archival tissue exhausted — new biopsy needed",
      "Not pursuing central IHC",
    ],
    round: 1,
  },
  {
    id: "q3",
    trial_id: "NCT05382299",
    criterion_id: "s6",
    criterion_text: "Exclusion: active second malignancy",
    current_data: "Family history of colorectal cancer noted. No patient-level second-malignancy workup in the last 12 months of encounters.",
    gap: "Trial excludes active second malignancy. Recent imaging and problem list need explicit sign-off.",
    options: [
      "No active second malignancy",
      "History of prior cancer, in remission > 5 years",
      "Active second malignancy present",
    ],
    round: 1,
  },
];

export const MOCK_LOG: AgentLogEntry[] = [
  { id: "l1", agent: "Stage 1 · Intake", action: "Ingested narrative + demographics. Tagged provenance for 6 source snippets.", ts: "00:00.2", status: "done" },
  { id: "l2", agent: "Stage 2 · Deterministic parser", action: "Mapped 6 lab values, 3 medications, 2 ICD-coded conditions from structured fields.", ts: "00:00.8", status: "done" },
  { id: "l3", agent: "Stage 2 · NLP extraction", action: "Extracted 4 narrative claims (SRS date, ECOG, LVEF context, HER2 status).", ts: "00:01.4", status: "done" },
  { id: "l4", agent: "Stage 3 · Retrieval", action: "Hybrid BM25 + dense search on ClinicalTrials.gov. Recall-optimized: 87 candidates.", ts: "00:02.6", status: "done" },
  { id: "l5", agent: "Stage 3 · Gap analysis", action: "Diffed 412 criteria against profile. 3 gaps flagged for clarification.", ts: "00:03.4", status: "done" },
  { id: "l6", agent: "Stage 4 · Clarification loop", action: "Generated 3 structured questions (bounded, 2–4 options each). Round 1 of 3.", ts: "00:04.1", status: "in_progress" },
  { id: "l7", agent: "Stage 5 · Deterministic verifier", action: "Evaluated 24 numeric/date predicates. All exact — no LLM calls.", ts: "00:05.5", status: "done" },
  { id: "l8", agent: "Stage 5 · Narrative entailment", action: "LLM entailment on 8 qualitative criteria with sentence-level citations.", ts: "00:07.2", status: "done" },
  { id: "l9", agent: "Stage 6 · Conformal calibration", action: "Split conformal (n=412). Coverage: ≥90% inclusion at 95% confidence.", ts: "00:08.3", status: "done" },
  { id: "l10", agent: "Stage 6 · Verification-cost ranking", action: "Ranked 4 trials ascending by clinician minutes-to-verify.", ts: "00:09.1", status: "done" },
];

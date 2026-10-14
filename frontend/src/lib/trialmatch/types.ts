export type Verdict = "met" | "unmet" | "unknown";
export type TrialVerdict = "eligible" | "eligible_pending_unknown" | "ineligible";
export type ConfidenceTier = "high" | "quick_check" | "manual_review" | "ineligible";
export type VerificationPath = "deterministic" | "narrative" | "predicate";

export type Criterion = {
  id: string;
  type: "inclusion" | "exclusion";
  text: string;
  semantic_class:
    | "demographic" | "condition" | "biomarker" | "lab"
    | "treatment" | "temporal" | "performance" | "other";
  verdict: Verdict;
  /** How this criterion was evaluated. */
  path: VerificationPath;
  /** Estimated minutes for a clinician to resolve if unknown. */
  cost_min?: number;
  /** True once clarification loop has exhausted retries. */
  unverifiable?: boolean;
  evidence?: { source: string; value: string; date?: string };
  missing_data?: string;
  /** Structured, bounded clarification options (2-4). */
  clarification_options?: string[];
};

export type TrialLocation = {
  city: string;
  state?: string;
  country: string;
  distance_mi?: number;
  status: string;
};

export type TrialPilot = {
  nct_id: string;
  title: string;
  phase: "I" | "II" | "III" | "IV" | "N/A";
  status: string;
  sponsor: string;
  locations: TrialLocation[];
  verdict: TrialVerdict;
  tier: ConfidenceTier;
  /** Total minutes of clinician work to fully verify. Lower = ranked higher. */
  verification_cost_min: number;
  /** Calibrated shortlist inclusion probability (0-1). */
  calibrated_confidence: number;
  criteria: Criterion[];
  explanation: string;
};

export type ClarificationQuestion = {
  id: string;
  trial_id: string;
  criterion_id: string;
  criterion_text: string;
  current_data: string;
  gap: string;
  options: string[];
  round: number;
  resolved?: string | "unverifiable";
};

export type AgentLogEntry = {
  id: string;
  agent: string;
  action: string;
  ts: string;
  status: "done" | "in_progress" | "reflection";
};

export type PatientProfile = {
  demographics: { age: number; sex: "F" | "M"; weight_kg?: number; height_cm?: number };
  conditions: { name: string; snomed?: string; confidence?: "low" | "high" }[];
  medications: { name: string; rxnorm?: string; start?: string; stop?: string }[];
  labs: { name: string; value: string; unit: string; date: string }[];
  performance?: { name: string; value: string };
  cancer?: {
    stage?: string;
    biomarkers?: { name: string; value: string }[];
    prior_lines?: number;
  };
  summary: string;
};

export type CoverageStatement = {
  /** e.g. 0.9 → "≥90% probability" */
  probability: number;
  /** e.g. 0.95 → "at 95% confidence" */
  confidence: number;
  /** Split conformal was calibrated on this many labeled cases. */
  calibration_n: number;
};

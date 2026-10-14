# UI/UX prompt — TrialPilot (Clinical Trial Matching Copilot) — v2

Single copy-pasteable prompt for an LLM that generates UI code. This version supersedes the original TrialMatch prompt and folds in every change requested since. **UI only. No backend. All agent behaviour is mocked on the client.**

---

## 1. What you are building

A web interface for **TrialPilot**, a neurosymbolic clinical trial matching copilot. It takes a patient's clinical profile and returns a shortlist of recruiting trials, **ranked by verification cost** (how many clinician minutes are needed to confirm eligibility), with calibrated coverage and a per-criterion evidence breakdown. You are building only the UI that renders the pipeline's progress and output for a clinician, trial coordinator, or motivated patient. Backend is simulated with timers and mock data.

**Critical positioning:** screening aid for research and educational use, not clinical advice. Every claim must trace to source data. Trust is the design goal.

---

## 2. Tech stack

- TanStack Start (React 19 + TypeScript), file routes under `src/routes`, Vite, Tailwind CSS v4 configured in `src/styles.css`
- `shadcn/ui` primitives (button, dialog, tabs, accordion, badge, card, sheet, tooltip, popover, radio-group, input, textarea, scroll-area)
- `lucide-react` outline icons only
- Flow state held in React state (a step state machine in `src/routes/index.tsx`)
- No CSS-in-JS, no Material UI. Tailwind utilities only.
- **Every scrollable region must use the Radix/shadcn `ScrollArea`**, never raw `overflow-auto`. ScrollArea needs an explicit height (e.g. `h-[calc(100vh-360px)]`), not just `max-h`, or it will not scroll.

---

## 3. Design principles

1. **Evidence over assertion.** Every verdict links to its source patient fact.
2. **Honest unknowns.** "Unknown" is a first-class state, visually distinct from Met and Unmet.
3. **Calm density.** Dense where clinicians scan, spacious where parsing matters.
4. **Streaming-first / honest effort.** Heavy work is shown per stage with cycling substeps, never one opaque spinner. Heavy stages take visibly longer than light ones.
5. **Trust through restraint.** No gradients (except the subtle animated grid on the landing hero), no glows, no AI sparkles, no emoji, no decorative imagery.
6. **Sentence case everywhere.**
7. **Bounded effort.** Clarification is limited, structured and non-skippable where required.

---

## 4. Visual system

Warm, clinician-friendly palette defined as OKLCH semantic tokens in `src/styles.css`. **Light and dark themes are both required**, with a theme toggle in the top bar. Theme is applied via a `dark` class on the root and persisted in `localStorage`. Never hardcode colors in components; use tokens only.

| Role               | Light     | Dark      |
| ------------------ | --------- | --------- |
| Page               | `#FAFAF7` | `#1A1A19` |
| Surface            | `#FFFFFF` | `#252523` |
| Border             | `#E8E6E0` | `#3A3A37` |
| Text primary       | `#1F1F1D` | `#F4F2EC` |
| Text secondary     | `#5E5D58` | `#A8A6A0` |
| Text tertiary      | `#8C8A82` | `#6E6C66` |
| Brand              | `#2E5C8A` | `#7BA7D9` |
| Met / eligible     | `#2F6E4E` | `#7DBF9B` |
| Unmet / ineligible | `#9B3232` | `#D88A8A` |
| Unknown / pending  | `#A36A1E` | `#E0A75C` |

Semantic colors appear only on verdict pills, criterion icons and verdict bars. Dark mode must keep high contrast for clinical data.

Typography: Inter/system body 14–15px; headings weight 500; IDs, lab values, timestamps and substeps in mono; sizes no smaller than 11–12px; two weights (400, 500). Radius: `rounded-md` for buttons/pills, `rounded-lg` for cards. Borders over shadows; at most `shadow-sm`. 8px spacing unit.

---

## 5. Information architecture and flow

Linear flow with a persistent top bar (brand "TrialPilot", current step label, patient summary chip once known, "View activity", "Start over", theme toggle). Clicking the logo returns to the landing page. No left navigation.

```
landing → intake → extracting (≈15s) → missing details form → re-analysing (≈6s)
→ confirm profile (editable) → screening pipeline (≈45s) → clarification (gaps)
→ re-run pipeline (≈20s) → results
```

A persistent disclaimer banner is visible on every screen.

---

## 6. Screen specs

### 6.0 Landing (interactive)

- Hero with subtle animated grid background, product name, one-line value proposition, primary CTA "Start screening".
- **No usage or availability stats.** Instead:
  - **"What it does"**: six feature cards with icons (structured intake, agentic matching, neurosymbolic verification, traceable evidence, bounded clarification, calibrated coverage). Cards react on hover.
  - **"How it works"**: visual steps matching the real pipeline: Intake, Extract, Retrieve, Gap analysis, Classify, Verify, Calibrate, Rank.

### 6.1 Intake

Centered, max-width 720px. Tabs: **Narrative** (textarea with realistic placeholder), **FHIR bundle** (file picker, filename and mock parse status), **Sample patient** (dropdown of 3–5 synthetic patients that fills the narrative). Collapsed "Optional filters" accordion (location + radius, recruitment status, phase, intervention type chips). Optional free-text focus field. CTA "Find matching trials", disabled until input exists. Disclaimer in footer.

### 6.2 Extraction processing (new)

After submit, do **not** jump to the profile. Show a `ProcessingStream`: a progress bar plus a scrolling/cycling list of status lines, each with its own duration, total ≈15s:
Reading patient narrative → Segmenting clinical sections → Extracting demographics → Extracting medications and exposure windows → Extracting conditions and problem list → Finding most recent diagnosis and staging → Parsing labs and vitals → Normalising to RxNorm / SNOMED / LOINC → Checking for missing protocol-critical facts.
Completed lines get a check, the active line a spinner. Respect `prefers-reduced-motion`.

### 6.3 Missing details form (new, required)

If protocol-critical facts were not found, ask for them in a structured form. Fixed clinical set: ECOG performance status, LVEF value and date, number of prior lines of therapy, smoking status, pregnancy / menopausal status, relevant comorbidities. **All required; no skip.** Submit is disabled until complete. Back returns to intake.

### 6.4 Re-analysis processing (new)

Same `ProcessingStream`, ≈6s: Re-analysing details with your answers → Merging clinician-supplied facts → Re-running normalisation → Rebuilding the structured profile.

### 6.5 Profile confirmation (editable)

Header "Confirm extracted patient profile". Two-column cards: demographics, conditions (SNOMED on hover), medications (RxNorm on hover, start/stop), recent labs (mono), performance status, cancer-specific (stage, biomarkers, prior lines). **Every card is editable**: an Edit button opens a per-section `Dialog` (changed from the original "inline, not modal" rule) with fields, and "Add row" / "Remove" for lists. Low-confidence facts are highlighted with an amber dot and an inline alert. Bottom bar: Back, "Continue to trial search".

### 6.6 Screening pipeline (heavy, ≈45s)

Two columns: left a stage list, right an "Agent activity log" in a `ScrollArea`. Stages with realistic uneven durations; each active stage cycles substeps:

1. Intake (fast) · 2. Deterministic + NLP extraction · 3. Retrieval (fast, hybrid search, 87 candidates) · 4. **Gap analysis (slow)** · 5. **Criterion classifier (slow)** · 6. **Neurosymbolic verifier (slowest)**: predicates in code, LLM entailment with citations · 7. Conformal calibration · 8. Verification-cost ranking.
   Show a progress bar, "n / total" stage counter, remaining-time estimate, and live log entries (agent name, action, mono timestamp, status dot: pulsing in-progress, solid done, amber reflection).

### 6.7 Clarification loop (bounded, active)

Shown after screening. Structured questions only (2–4 radio options each, plus "can't determine / unverifiable"). Each question shows the trial criterion, the current record data, the gap, and "pick one" options. Bounded rounds. Question list sits in a `ScrollArea` with explicit height; action bar ("Back", "Rank remaining trials") always visible. All questions must be answered before continuing.

### 6.8 Re-run pipeline (≈20s)

Same `LiveProcessing` component, `rerun` variant: Merging clinician answers → Neurosymbolic re-verification (only affected criteria) → Re-calibration → Re-ranking shortlist.

### 6.9 Results dashboard

- **Conformal coverage banner:** "At least 90% probability the shortlist contains every eligible trial, at 95% confidence (calibrated on n = 412)".
- Trials **grouped by confidence tier** and sorted by **verification cost** (lowest clinician minutes first): High confidence, Quick check, Manual review, Ineligible.
- Three regions: left filters + activity log (`ScrollArea`), centre ranked cards, right detail panel. Layout `xl:grid-cols-[240px_minmax(0,1fr)_480px]`, `lg:grid-cols-[220px_minmax(0,1fr)_380px]`. Below `lg`, the detail panel is a `Sheet`, mounted only on non-desktop (use a `useIsDesktop` hook so no stray overlay darkens the screen).
- Card: NCT ID mono, phase, status, distance, 2-line title, verdict pill (wraps; no fixed height), tier pill, verification-cost chip ("≈ 6 min to verify"), neurosymbolic path badge (Deterministic / Narrative / Predicate), calibrated confidence bar, top 2 criteria, locations. Selected card has a 2px brand left border. Click toggles the panel.
- Sort: Best match, Verification cost, Most criteria met, Fewest unknowns, Distance, Phase. Verdict filter. Export menu. Detail panel and activity log are sticky.

### 6.10 Trial detail panel

Header (title, NCT ID, phase, status, sponsor, ClinicalTrials.gov link) · verdict summary bar (met/unmet/unknown segments + counts) · tier and verification cost · "Why this verdict" · **criteria breakdown** (inclusion, then exclusion, unknowns pulled to the bottom; columns criterion / verdict / evidence; verification path badge per row; evidence popover with source, stored value, date) · for unknowns, estimated clinician minutes and missing data · locations list · next-step actions. Whole panel inside a `ScrollArea`.

### 6.11 Empty, error and edge states

No matches, all ineligible (top blockers), agent failure with trace ID and retry, clarification retries exhausted (criterion marked unverifiable), reflection cap banner, non-blocking network toast.

---

## 7. Interaction and streaming

- Card click opens/closes detail. Evidence click opens popover. Verdict pill hover shows rationale; criterion icon hover shows parsed predicate.
- Keyboard: ↑/↓ cards, Enter open, Esc close, `/` focus search, `?` shortcuts.
- Log entries slide in (translateY 4px, 150ms); progress animates smoothly; `aria-live="polite"` on logs; reduced-motion disables animations.
- All timings are mocked with timers; each stage has its own duration.

---

## 8. Microcopy and accessibility

- Verdicts: `Met`, `Unmet`, `Unknown`. Trial verdicts: `Eligible`, `Eligible pending`, `Ineligible`. Never "match / no match".
- Loading copy informative, not cute. Errors say what happened and what to do.
- Disclaimer ("TrialPilot is a research and educational screening aid. It is not clinical advice and must not be used as the sole basis for treatment decisions.") stays visible on every screen and in exports.
- Colour is never the only signal: icon shape plus text label. WCAG AA body, AAA verdicts. Visible 2px brand focus rings. Icons `aria-hidden` or labelled.

---

## 9. Responsiveness

Desktop ≥1100px full three-region results. Tablet 768–1099px detail becomes a sheet. Mobile <768px single column, log collapses to a button, filters in a bottom sheet.

---

## 10. Data shapes

```ts
type Verdict = "met" | "unmet" | "unknown";
type TrialVerdict = "eligible" | "eligible_pending_unknown" | "ineligible";
type ConfidenceTier = "high" | "quick_check" | "manual_review" | "ineligible";
type VerificationPath = "deterministic" | "narrative" | "predicate";

type Criterion = {
  id: string;
  type: "inclusion" | "exclusion";
  text: string;
  semantic_class:
    | "demographic"
    | "condition"
    | "biomarker"
    | "lab"
    | "treatment"
    | "temporal"
    | "performance"
    | "other";
  verdict: Verdict;
  path: VerificationPath;
  cost_min?: number;
  unverifiable?: boolean;
  evidence?: { source: string; value: string; date?: string };
  missing_data?: string;
  clarification_options?: string[];
};

type TrialPilot = {
  nct_id: string;
  title: string;
  phase: "I" | "II" | "III" | "IV" | "N/A";
  status: string;
  sponsor: string;
  locations: {
    city: string;
    state?: string;
    country: string;
    distance_mi?: number;
    status: string;
  }[];
  verdict: TrialVerdict;
  tier: ConfidenceTier;
  verification_cost_min: number; // lower ranks higher
  calibrated_confidence: number; // 0–1
  criteria: Criterion[];
  explanation: string;
};

type ClarificationQuestion = {
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

type CoverageStatement = {
  probability: number;
  confidence: number;
  calibration_n: number;
};
type AgentLogEntry = {
  id: string;
  agent: string;
  action: string;
  ts: string;
  status: "done" | "in_progress" | "reflection";
};
type PatientProfile = {
  /* demographics, conditions, medications, labs, performance, cancer, summary */
};
```

---

## 11. Anti-patterns

No traffic-light dots without text, no KPI dashboards on results, no usage/availability stats on the landing page, no AI sparkles or mascots, no confetti or "Great match!" toasts, no collapsing the disclaimer, no raw `overflow-auto` scroll regions, no hardcoded colors, no backend calls.

---

## 12. Deliverable

A working React + TypeScript + Tailwind project, modular components named after screens: `LandingScreen`, `IntakeScreen`, `ProcessingStream`, `MissingDetailsForm`, `ProfileConfirmation`, `LiveProcessing`, `ClarificationLoop`, `ResultsDashboard`, `TrialDetailPanel`, `TopBar`, `ThemeToggle`, `DisclaimerBanner`. Mock data in `src/lib/trialmatch/`. If forced to choose between fidelity and polish, choose fidelity: the criteria breakdown, evidence citations and verification-cost ranking are the heart of the product.

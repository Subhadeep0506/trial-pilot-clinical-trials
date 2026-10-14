import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { IntakeScreen } from "@/components/trialmatch/IntakeScreen";
import { ProfileConfirmation } from "@/components/trialmatch/ProfileConfirmation";
import { LandingScreen } from "@/components/trialmatch/LandingScreen";
import { LiveProcessing } from "@/components/trialmatch/LiveProcessing";
import { ClarificationLoop } from "@/components/trialmatch/ClarificationLoop";
import { ResultsDashboard } from "@/components/trialmatch/ResultsDashboard";
import { ProcessingStream } from "@/components/trialmatch/ProcessingStream";
import { MissingDetailsForm } from "@/components/trialmatch/MissingDetailsForm";
import { TopBar } from "@/components/trialmatch/TopBar";
import { DisclaimerBanner } from "@/components/trialmatch/DisclaimerBanner";
import { SAMPLE_PROFILE } from "@/lib/trialmatch/mockData";
import type { PatientProfile } from "@/lib/trialmatch/types";

type Step =
  | "landing"
  | "intake"
  | "extracting"
  | "missing"
  | "reanalysing"
  | "confirm"
  | "processing"
  | "clarify"
  | "reprocessing"
  | "results";

const EXTRACT_STEPS = [
  { label: "Reading patient narrative", ms: 1600 },
  { label: "Segmenting clinical sections", ms: 1400 },
  { label: "Extracting demographics", ms: 1300 },
  { label: "Extracting medications and exposure windows", ms: 2000 },
  { label: "Extracting conditions and problem list", ms: 1900 },
  { label: "Finding most recent diagnosis and staging", ms: 1800 },
  { label: "Parsing labs and vitals", ms: 1600 },
  { label: "Normalising to RxNorm / SNOMED / LOINC", ms: 1700 },
  { label: "Checking for missing protocol-critical facts", ms: 1700 },
];

const REANALYSE_STEPS = [
  { label: "Re-analysing details with your answers", ms: 1800 },
  { label: "Merging clinician-supplied facts", ms: 1500 },
  { label: "Re-running normalisation", ms: 1400 },
  { label: "Rebuilding the structured profile", ms: 1600 },
];

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TrialPilot — clinical trial matching copilot" },
      {
        name: "description",
        content:
          "A shortlist of recruiting trials ranked by verification cost, with calibrated coverage and per-criterion evidence.",
      },
      { property: "og:title", content: "TrialPilot — clinical trial matching copilot" },
      {
        property: "og:description",
        content:
          "Neurosymbolic patient-to-trial matching with bounded clarification and conformal coverage. Research and educational use only.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: TrialPilotApp,
});

function TrialPilotApp() {
  const [step, setStep] = useState<Step>("landing");
  const [profile, setProfile] = useState<PatientProfile>(SAMPLE_PROFILE);

  const showSummary = step !== "landing" && step !== "intake";

  const topBarStep: "landing" | "intake" | "confirm" | "processing" | "clarify" | "results" =
    step === "extracting" || step === "reanalysing"
      ? "processing"
      : step === "missing"
        ? "confirm"
        : step === "reprocessing"
          ? "processing"
          : step;

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <TopBar
        step={topBarStep}
        patientSummary={showSummary ? profile.summary : undefined}
        onLogoClick={() => setStep("landing")}
        onStartOver={
          step !== "landing" && step !== "intake" ? () => setStep("intake") : undefined
        }
      />

      <main className="flex-1">
        {step === "landing" && <LandingScreen onStart={() => setStep("intake")} />}

        {step === "intake" && <IntakeScreen onSubmit={() => setStep("extracting")} />}

        {step === "extracting" && (
          <ProcessingStream
            eyebrow="Stage 1–2 · Intake and extraction"
            title="Processing the patient record"
            subtitle="Deterministic parsers plus NLP extraction over the narrative you supplied."
            steps={EXTRACT_STEPS}
            onDone={() => setStep("missing")}
          />
        )}

        {step === "missing" && (
          <MissingDetailsForm
            onBack={() => setStep("intake")}
            onSubmit={() => setStep("reanalysing")}
          />
        )}

        {step === "reanalysing" && (
          <ProcessingStream
            eyebrow="Stage 2 · Re-analysis"
            title="Re-analysing details"
            subtitle="Folding your answers back into the structured profile."
            steps={REANALYSE_STEPS}
            onDone={() => setStep("confirm")}
          />
        )}

        {step === "confirm" && (
          <ProfileConfirmation
            profile={profile}
            onChange={setProfile}
            onBack={() => setStep("intake")}
            onContinue={() => setStep("processing")}
          />
        )}

        {step === "processing" && (
          <LiveProcessing variant="full" onDone={() => setStep("clarify")} />
        )}

        {step === "clarify" && (
          <ClarificationLoop
            onBack={() => setStep("confirm")}
            onContinue={() => setStep("reprocessing")}
          />
        )}

        {step === "reprocessing" && (
          <LiveProcessing variant="rerun" onDone={() => setStep("results")} />
        )}

        {step === "results" && <ResultsDashboard onStartOver={() => setStep("intake")} />}
      </main>

      <DisclaimerBanner />
    </div>
  );
}

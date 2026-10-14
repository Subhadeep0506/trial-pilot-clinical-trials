import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Clock,
  Split,
  MessageSquareQuote,
  ShieldCheck,
  FileSearch,
  Sparkles,
  ArrowRight,
  Stethoscope,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Props = { onStart: () => void };

const DIFFERENTIATORS = [
  {
    icon: Clock,
    title: "Ranked by verification cost",
    body: "Instead of a raw eligibility score, trials sort by how few minutes of clinician chart work stand between the patient and enrollment.",
  },
  {
    icon: Split,
    title: "Neurosymbolic verdicts",
    body: "Numeric and date predicates run in code — exact, cheap, auditable. LLM entailment is reserved for qualitative narrative with sentence-level citations.",
  },
  {
    icon: MessageSquareQuote,
    title: "Bounded clarification",
    body: "Gaps become structured questions with 2–4 options, capped at three rounds. No open-ended chat, no hallucinated facts to catch.",
  },
  {
    icon: ShieldCheck,
    title: "Calibrated coverage",
    body: "Split conformal prediction states plainly: this shortlist contains the true-eligible trial with ≥90% probability at 95% confidence.",
  },
  {
    icon: FileSearch,
    title: "Every verdict cites its source",
    body: "Med list, path report, lab value, radiology note — click any criterion to see the exact snippet that decided it.",
  },
  {
    icon: Sparkles,
    title: "Tiered, not scored",
    body: "Trials are grouped into high confidence, needs a quick check, and needs manual review — so the clinician's next action is obvious.",
  },
];

const STAGES = [
  { n: "01", title: "Intake", body: "Narrative, FHIR, or sample patient with source provenance." },
  { n: "02", title: "Extract", body: "Structured parser + NLP for narrative claims." },
  { n: "03", title: "Retrieve", body: "Hybrid search over recruiting trials. Recall-first." },
  { n: "04", title: "Clarify", body: "Bounded, structured Q&A. Max three rounds." },
  { n: "05", title: "Verify", body: "Deterministic predicates + LLM entailment." },
  { n: "06", title: "Calibrate", body: "Split conformal on labeled cases." },
  { n: "07", title: "Rank", body: "Ascending by verification cost, tiered for triage." },
];

export function LandingScreen({ onStart }: Props) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div className="relative">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-border">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 -z-10"
          style={{
            backgroundImage:
              "radial-gradient(800px 400px at 15% -10%, color-mix(in oklab, var(--brand) 12%, transparent), transparent 60%), radial-gradient(600px 320px at 100% 0%, color-mix(in oklab, var(--met) 10%, transparent), transparent 60%)",
          }}
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 -z-10 opacity-[0.35]"
          style={{
            backgroundImage:
              "linear-gradient(to right, color-mix(in oklab, var(--border) 60%, transparent) 1px, transparent 1px), linear-gradient(to bottom, color-mix(in oklab, var(--border) 60%, transparent) 1px, transparent 1px)",
            backgroundSize: "44px 44px",
            maskImage: "radial-gradient(900px 500px at 50% 0%, black, transparent 75%)",
          }}
        />

        <div className="mx-auto max-w-7xl px-6 pt-20 pb-24 lg:pt-28 lg:pb-32">
          <div
            className={cn(
              "max-w-3xl transition-all duration-700 ease-out",
              mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3",
            )}
          >
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/70 px-3 h-7 text-[11px] font-mono-tight tracking-wide text-muted-foreground">
              <Stethoscope className="size-3 text-brand" aria-hidden="true" />
              Clinical trial matching copilot · research use
            </div>
            <h1 className="mt-5 text-[44px] sm:text-[56px] leading-[1.02] font-medium tracking-tight">
              A shortlist ranked by
              <span className="block text-muted-foreground">
                the least verification work.
              </span>
            </h1>
            <p className="mt-5 text-[15px] sm:text-[16px] text-muted-foreground max-w-2xl leading-relaxed">
              TrialPilot turns a patient narrative into a tiered shortlist of recruiting trials —
              with a plain coverage statement, per-criterion evidence, and a bounded clarification
              loop that closes gaps without inventing facts.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Button
                size="lg"
                onClick={onStart}
                className="bg-brand hover:bg-brand/90 text-primary-foreground group"
              >
                Start a match
                <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
              </Button>
              <button
                type="button"
                onClick={onStart}
                className="text-[13px] text-muted-foreground hover:text-foreground transition-colors underline-offset-4 hover:underline"
              >
                or try a sample patient →
              </button>
            </div>

            {/* Coverage statement teaser */}
            <div className="mt-10 inline-flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-brand/25 bg-brand-soft/40 px-3 py-2 text-[12.5px]">
              <ShieldCheck className="size-3.5 text-brand" aria-hidden="true" />
              <span className="font-medium">Shortlist coverage:</span>
              <span className="text-muted-foreground">
                <span className="font-mono-tight text-foreground">≥90%</span> probability at{" "}
                <span className="font-mono-tight text-foreground">95%</span> confidence · calibrated
                on <span className="font-mono-tight text-foreground">412</span> labeled cases.
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Differentiators */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="flex items-end justify-between gap-6 mb-10">
          <div>
            <div className="text-[11px] font-mono-tight tracking-widest uppercase text-tertiary">
              What&apos;s different
            </div>
            <h2 className="mt-2 text-[28px] font-medium tracking-tight">
              Built for the clinician&apos;s next action.
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-px bg-border rounded-xl overflow-hidden border border-border">
          {DIFFERENTIATORS.map((f, i) => {
            const Icon = f.icon;
            return (
              <div
                key={f.title}
                className="group relative bg-card p-6 transition-colors hover:bg-secondary/40 min-h-[190px] flex flex-col"
                style={{ transitionDelay: mounted ? `${i * 40}ms` : "0ms" }}
              >
                <div className="flex items-center gap-3">
                  <div className="size-9 rounded-md border border-border bg-brand-soft grid place-items-center text-brand transition-transform group-hover:-translate-y-0.5">
                    <Icon className="size-4" aria-hidden="true" />
                  </div>
                  <h3 className="text-[15px] font-medium tracking-tight">{f.title}</h3>
                </div>
                <p className="mt-3 text-[13.5px] text-muted-foreground leading-relaxed">
                  {f.body}
                </p>
                <span
                  aria-hidden="true"
                  className="absolute bottom-0 left-0 h-px w-0 bg-brand transition-all duration-500 group-hover:w-full"
                />
              </div>
            );
          })}
        </div>
      </section>

      {/* Pipeline */}
      <section className="border-t border-border bg-secondary/30">
        <div className="mx-auto max-w-7xl px-6 py-20">
          <div className="text-[11px] font-mono-tight tracking-widest uppercase text-tertiary">
            The pipeline
          </div>
          <h2 className="mt-2 text-[28px] font-medium tracking-tight">
            Seven stages, every step visible.
          </h2>
          <p className="mt-2 text-[14px] text-muted-foreground max-w-2xl">
            Deterministic where the trial protocol is precise, narrative-LLM where the criterion is
            qualitative, and a hard cap on how much clarification we&apos;ll ask before flagging a
            criterion as unverifiable.
          </p>

          <ol className="mt-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7 gap-3">
            {STAGES.map((s) => (
              <li
                key={s.n}
                className="relative rounded-lg border border-border bg-card p-4 hover:border-brand/40 transition-colors"
              >
                <div className="text-[11px] font-mono-tight text-tertiary">{s.n}</div>
                <div className="mt-1.5 text-[14px] font-medium tracking-tight">{s.title}</div>
                <div className="mt-1 text-[12px] text-muted-foreground leading-relaxed">
                  {s.body}
                </div>
              </li>
            ))}
          </ol>

          <div className="mt-12 flex flex-wrap items-center gap-3">
            <Button
              size="lg"
              onClick={onStart}
              className="bg-brand hover:bg-brand/90 text-primary-foreground"
            >
              Run a sample match
              <ArrowRight className="size-4" />
            </Button>
            <span className="text-[12px] text-tertiary">
              Screening aid for research and education. Not clinical advice.
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}

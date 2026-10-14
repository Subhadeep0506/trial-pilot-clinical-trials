import { useEffect, useRef, useState } from "react";
import { MOCK_LOG } from "@/lib/trialmatch/mockData";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CheckCircle2, Loader2 } from "lucide-react";

type Variant = "full" | "rerun";

type Props = { onDone: () => void; variant?: Variant };

type Stage = {
  key: string;
  label: string;
  detail: string;
  ms: number;
  substeps: string[];
};

const FULL_STAGES: Stage[] = [
  {
    key: "intake",
    label: "Intake",
    detail: "Provenance-tagged snippets",
    ms: 2500,
    substeps: ["Normalising units", "Tagging source spans", "Building the fact table"],
  },
  {
    key: "extract",
    label: "Deterministic + NLP extraction",
    detail: "Structured fields + narrative claims",
    ms: 4000,
    substeps: ["Mapping RxNorm codes", "Mapping SNOMED codes", "Resolving temporal anchors"],
  },
  {
    key: "retrieve",
    label: "Retrieval",
    detail: "Recall-optimized hybrid search",
    ms: 3500,
    substeps: ["BM25 pass over registry", "Dense re-ranking", "Deduplicating 87 candidates"],
  },
  {
    key: "gap",
    label: "Gap analysis",
    detail: "Diffing 412 criteria against the profile",
    ms: 9000,
    substeps: [
      "Parsing eligibility blocks",
      "Aligning criteria to profile facts",
      "Scoring resolvability of each gap",
      "Estimating clinician minutes per gap",
    ],
  },
  {
    key: "classify",
    label: "Criterion classifier",
    detail: "Routing each criterion to a verifier",
    ms: 8000,
    substeps: [
      "Classifying semantic class",
      "Splitting deterministic vs narrative",
      "Flagging predicate-only criteria",
    ],
  },
  {
    key: "verify",
    label: "Neurosymbolic verifier",
    detail: "Predicates in code, entailment for prose",
    ms: 10000,
    substeps: [
      "Evaluating numeric predicates",
      "Running date-window logic",
      "LLM entailment with sentence citations",
      "Cross-checking contradictions",
    ],
  },
  {
    key: "calibrate",
    label: "Conformal calibration",
    detail: "≥90% coverage @ 95% confidence",
    ms: 4000,
    substeps: ["Loading calibration set (n=412)", "Computing nonconformity scores"],
  },
  {
    key: "rank",
    label: "Verification-cost ranking",
    detail: "Least clinician effort first",
    ms: 3500,
    substeps: ["Summing minutes-to-verify", "Assigning confidence tiers"],
  },
];

const RERUN_STAGES: Stage[] = [
  {
    key: "merge",
    label: "Merging clinician answers",
    detail: "Answers become first-class facts with provenance",
    ms: 3000,
    substeps: ["Attaching provenance", "Invalidating stale verdicts"],
  },
  {
    key: "reverify",
    label: "Neurosymbolic re-verification",
    detail: "Re-running only the affected criteria",
    ms: 9000,
    substeps: [
      "Re-evaluating gated predicates",
      "Re-running entailment on 8 criteria",
      "Propagating verdict changes",
    ],
  },
  {
    key: "recalibrate",
    label: "Re-calibration",
    detail: "Updated conformal coverage",
    ms: 4500,
    substeps: ["Recomputing nonconformity", "Refreshing coverage statement"],
  },
  {
    key: "rerank",
    label: "Re-ranking shortlist",
    detail: "Verification cost recalculated",
    ms: 3500,
    substeps: ["Recomputing minutes-to-verify", "Re-tiering trials"],
  },
];

export function LiveProcessing({ onDone, variant = "full" }: Props) {
  const stages = variant === "full" ? FULL_STAGES : RERUN_STAGES;
  const total = stages.reduce((a, s) => a + s.ms, 0);

  const [logEntries, setLogEntries] = useState<typeof MOCK_LOG>([]);
  const [stageIdx, setStageIdx] = useState(0);
  const [subIdx, setSubIdx] = useState(0);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const timers: number[] = [];

    if (reduce) {
      setStageIdx(stages.length);
      setLogEntries(MOCK_LOG);
      timers.push(window.setTimeout(() => doneRef.current(), 800));
      return () => timers.forEach((t) => window.clearTimeout(t));
    }

    let acc = 0;
    stages.forEach((s, i) => {
      acc += s.ms;
      timers.push(
        window.setTimeout(() => {
          setStageIdx(i + 1);
          setSubIdx(0);
        }, acc),
      );
    });

    MOCK_LOG.forEach((entry, i) => {
      timers.push(
        window.setTimeout(
          () => setLogEntries((prev) => [...prev, entry]),
          Math.round(((i + 1) / (MOCK_LOG.length + 1)) * total),
        ),
      );
    });

    timers.push(window.setTimeout(() => doneRef.current(), total + 600));
    return () => timers.forEach((t) => window.clearTimeout(t));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [variant]);

  // Cycle substeps of the active stage.
  useEffect(() => {
    const stage = stages[stageIdx];
    if (!stage) return;
    const every = Math.max(900, Math.round(stage.ms / stage.substeps.length));
    const id = window.setInterval(
      () => setSubIdx((i) => Math.min(i + 1, stage.substeps.length - 1)),
      every,
    );
    return () => window.clearInterval(id);
  }, [stageIdx, stages]);

  const elapsed = stages.slice(0, stageIdx).reduce((a, s) => a + s.ms, 0);
  const pct = Math.round((elapsed / total) * 100);
  const remainingSec = Math.max(0, Math.round((total - elapsed) / 1000));

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-6">
        <div className="text-[11px] font-mono-tight tracking-widest uppercase text-tertiary">
          {variant === "full" ? "Live pipeline" : "Live pipeline · re-run"}
        </div>
        <h2 className="mt-2 text-[22px] font-medium tracking-tight">
          {variant === "full"
            ? "Screening 87 recruiting trials"
            : "Re-screening with your answers"}
        </h2>
        <p className="mt-1 text-[13px] text-muted-foreground">
          {variant === "full"
            ? "Retrieval is fast; gap analysis, criterion classification and neurosymbolic verification are the heavy passes."
            : "Only the criteria your answers touched are re-verified, then coverage and ranking are recomputed."}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,42fr)_minmax(0,58fr)] gap-6">
        {/* Stage list */}
        <section>
          <ol className="rounded-lg border border-border bg-card overflow-hidden">
            {stages.map((s, i) => {
              const done = i < stageIdx;
              const active = i === stageIdx;
              return (
                <li
                  key={s.key}
                  className={cn(
                    "flex items-start gap-3 px-4 py-3 border-b border-border last:border-0 transition-colors",
                    active && "bg-brand-soft/40",
                  )}
                >
                  <span className="mt-0.5 shrink-0">
                    {done ? (
                      <CheckCircle2 className="size-4 text-met" aria-hidden="true" />
                    ) : active ? (
                      <Loader2 className="size-4 text-brand animate-spin" aria-hidden="true" />
                    ) : (
                      <span
                        className="block size-4 rounded-full border border-border"
                        aria-hidden="true"
                      />
                    )}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline justify-between gap-2">
                      <span
                        className={cn(
                          "text-[13px] font-medium",
                          done || active ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {s.label}
                      </span>
                      <span className="text-[10px] font-mono-tight text-tertiary shrink-0">
                        Stage {i + 1}
                      </span>
                    </div>
                    <p className="mt-0.5 text-[12px] text-muted-foreground leading-snug">
                      {s.detail}
                    </p>
                    {active && (
                      <p
                        aria-live="polite"
                        key={s.substeps[Math.min(subIdx, s.substeps.length - 1)]}
                        className="mt-1.5 text-[12px] font-mono-tight text-brand animate-in fade-in slide-in-from-bottom-1 duration-300"
                      >
                        {s.substeps[Math.min(subIdx, s.substeps.length - 1)]}…
                      </p>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
          <div className="mt-3 flex items-center gap-2">
            <div className="flex-1 h-1 rounded-full bg-secondary overflow-hidden">
              <div
                className="h-full bg-brand transition-all duration-500 ease-linear"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-[11px] font-mono-tight text-tertiary tabular-nums">
              {Math.min(stageIdx, stages.length)} / {stages.length}
            </span>
          </div>
          <p className="mt-2 text-[11px] text-tertiary">
            {remainingSec > 0
              ? `Roughly ${remainingSec}s of compute remaining.`
              : "Finalising shortlist…"}
          </p>
        </section>

        {/* Live log */}
        <aside className="rounded-lg border border-border bg-card shadow-sm h-[calc(100vh-260px)] flex flex-col">
          <header className="px-4 py-3 border-b border-border flex items-center justify-between">
            <h3 className="text-[13px] font-medium text-muted-foreground">Agent activity log</h3>
            <span className="text-[11px] text-tertiary font-mono-tight">
              {logEntries.length} events
            </span>
          </header>
          <ScrollArea className="flex-1">
            <ol aria-live="polite" className="py-1">
              {logEntries.map((e) => (
                <li
                  key={e.id}
                  className="px-4 py-2.5 border-b border-border/60 last:border-0 animate-in fade-in slide-in-from-top-1 duration-200"
                >
                  <div className="flex items-start gap-2.5">
                    <span
                      className={cn(
                        "mt-1.5 size-1.5 rounded-full shrink-0",
                        e.status === "in_progress"
                          ? "bg-brand animate-pulse"
                          : e.status === "reflection"
                            ? "bg-unknown"
                            : "bg-met",
                      )}
                      aria-hidden="true"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-[12px] font-medium text-foreground">{e.agent}</span>
                        <span className="text-[10px] text-tertiary font-mono-tight shrink-0">
                          {e.ts}
                        </span>
                      </div>
                      <p className="mt-0.5 text-[12px] leading-snug text-muted-foreground">
                        {e.action}
                      </p>
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          </ScrollArea>
        </aside>
      </div>
    </div>
  );
}

import { useMemo } from "react";
import type { TrialPilot, Criterion } from "@/lib/trialmatch/types";
import { VerdictIcon, VerdictPill, TierPill, PathBadge } from "./Verdict";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ExternalLink, X, MapPin, Bookmark, Search, Ban, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

type Props = { trial: TrialPilot; onClose: () => void };

export function TrialDetailPanel({ trial, onClose }: Props) {
  const met = trial.criteria.filter((c) => c.verdict === "met").length;
  const unmet = trial.criteria.filter((c) => c.verdict === "unmet").length;
  const unknown = trial.criteria.filter((c) => c.verdict === "unknown").length;
  const total = trial.criteria.length;

  const grouped = useMemo(() => {
    const inclusion = trial.criteria.filter((c) => c.type === "inclusion" && c.verdict !== "unknown");
    const exclusion = trial.criteria.filter((c) => c.type === "exclusion" && c.verdict !== "unknown");
    const unknowns = trial.criteria.filter((c) => c.verdict === "unknown");
    return { inclusion, exclusion, unknowns };
  }, [trial]);

  const detCount = trial.criteria.filter((c) => c.path === "deterministic").length;
  const narrCount = trial.criteria.filter((c) => c.path === "narrative").length;

  return (
    <div className="flex flex-col">
      {/* Header */}
      <header className="px-5 py-4 border-b border-border">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[11px] text-tertiary">
              <span className="font-mono-tight text-foreground">{trial.nct_id}</span>
              <span>·</span>
              <span>Phase {trial.phase}</span>
              <span>·</span>
              <span className="text-met">{trial.status}</span>
            </div>
            <h2 className="mt-1.5 text-[16px] font-medium leading-snug">{trial.title}</h2>
            <p className="mt-1 text-[12px] text-muted-foreground">{trial.sponsor}</p>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close detail panel" className="size-7 -mr-1">
            <X className="size-4" aria-hidden="true" />
          </Button>
        </div>
        <a
          href={`https://clinicaltrials.gov/study/${trial.nct_id}`}
          target="_blank"
          rel="noreferrer noopener"
          className="mt-3 inline-flex items-center gap-1 text-[12px] text-brand hover:underline underline-offset-4"
        >
          Open on ClinicalTrials.gov
          <ExternalLink className="size-3" aria-hidden="true" />
        </a>
      </header>

      {/* Tier + cost + confidence */}
      <section className="px-5 py-4 border-b border-border">
        <div className="flex flex-wrap items-center gap-2">
          <TierPill tier={trial.tier} />
          <VerdictPill verdict={trial.verdict} />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div className="rounded-md border border-border bg-background p-3">
            <div className="text-[10px] uppercase tracking-[0.08em] text-tertiary flex items-center gap-1.5">
              <Clock className="size-3" aria-hidden="true" />
              Verification cost
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-[22px] font-medium font-mono-tight tabular-nums">
                {trial.verification_cost_min}
              </span>
              <span className="text-[11px] text-muted-foreground">min</span>
            </div>
            <p className="mt-1 text-[11px] text-tertiary">
              Estimated clinician time to close all gaps.
            </p>
          </div>
          <div className="rounded-md border border-border bg-background p-3">
            <div className="text-[10px] uppercase tracking-[0.08em] text-tertiary">
              Calibrated confidence
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-[22px] font-medium font-mono-tight tabular-nums">
                {(trial.calibrated_confidence * 100).toFixed(0)}
              </span>
              <span className="text-[11px] text-muted-foreground">% shortlist inclusion</span>
            </div>
            <p className="mt-1 text-[11px] text-tertiary">
              From split conformal calibration (n=412).
            </p>
          </div>
        </div>

        <div
          className="mt-4 h-2 w-full rounded-full overflow-hidden flex bg-secondary"
          role="img"
          aria-label={`${met} met, ${unmet} unmet, ${unknown} unknown of ${total} criteria`}
        >
          <span className="bg-met" style={{ width: `${(met / total) * 100}%` }} />
          <span className="bg-unmet" style={{ width: `${(unmet / total) * 100}%` }} />
          <span className="bg-unknown" style={{ width: `${(unknown / total) * 100}%` }} />
        </div>
        <div className="mt-2 flex items-center justify-between text-[12px]">
          <div className="flex items-center gap-3 text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <span className="size-2 rounded-sm bg-met" aria-hidden="true" />
              <span className="font-mono-tight text-foreground">{met}</span> met
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="size-2 rounded-sm bg-unmet" aria-hidden="true" />
              <span className="font-mono-tight text-foreground">{unmet}</span> unmet
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="size-2 rounded-sm bg-unknown" aria-hidden="true" />
              <span className="font-mono-tight text-foreground">{unknown}</span> unknown
            </span>
          </div>
          <span className="text-[11px] text-tertiary">
            <span className="font-mono-tight text-foreground">{detCount}</span> deterministic ·{" "}
            <span className="font-mono-tight text-foreground">{narrCount}</span> narrative
          </span>
        </div>
      </section>

      {/* Why */}
      <section className="px-5 py-4 border-b border-border">
        <h3 className="text-[12px] font-medium text-muted-foreground tracking-wide mb-1.5">
          Why this verdict
        </h3>
        <p className="text-[13px] leading-relaxed text-foreground">{trial.explanation}</p>
      </section>

      {/* Criteria breakdown */}
      <section className="px-5 py-4 border-b border-border">
        <h3 className="text-[12px] font-medium text-muted-foreground tracking-wide mb-3">
          Criteria breakdown
        </h3>
        {grouped.unknowns.length > 0 && (
          <CriteriaTable label={`Unknown — ${grouped.unknowns.length} open`} criteria={grouped.unknowns} highlight />
        )}
        {grouped.inclusion.length > 0 && (
          <CriteriaTable label="Inclusion" criteria={grouped.inclusion} className={grouped.unknowns.length ? "mt-5" : ""} />
        )}
        {grouped.exclusion.length > 0 && (
          <CriteriaTable label="Exclusion" criteria={grouped.exclusion} className="mt-5" />
        )}
      </section>

      {/* Locations */}
      <section className="px-5 py-4 border-b border-border">
        <h3 className="text-[12px] font-medium text-muted-foreground tracking-wide mb-3">
          Locations
        </h3>
        <ul className="divide-y divide-border rounded-md border border-border overflow-hidden">
          {trial.locations.map((loc, i) => (
            <li key={i} className="flex items-center justify-between px-3 py-2.5 text-[13px] bg-background">
              <div className="flex items-center gap-2 min-w-0">
                <MapPin className="size-3.5 text-tertiary shrink-0" aria-hidden="true" />
                <span className="truncate">
                  {loc.city}
                  {loc.state ? `, ${loc.state}` : ""}, {loc.country}
                </span>
                <span
                  className={cn(
                    "ml-1 text-[11px] rounded-sm px-1.5 py-0.5",
                    loc.status === "Recruiting" ? "bg-met-soft text-met" : "bg-secondary text-muted-foreground",
                  )}
                >
                  {loc.status}
                </span>
              </div>
              {loc.distance_mi != null && (
                <span className="font-mono-tight text-[12px] text-muted-foreground shrink-0">
                  {loc.distance_mi} mi
                </span>
              )}
            </li>
          ))}
        </ul>
      </section>

      {/* Next steps */}
      <section className="px-5 py-4">
        <h3 className="text-[12px] font-medium text-muted-foreground tracking-wide mb-3">
          What to do next
        </h3>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" className="bg-background">
            <Bookmark className="size-4" aria-hidden="true" />
            Mark for follow-up
          </Button>
          <Button variant="outline" size="sm" className="bg-background">
            <Search className="size-4" aria-hidden="true" />
            Show similar trials
          </Button>
        </div>
      </section>
    </div>
  );
}

function CriteriaTable({
  label,
  criteria,
  className,
  highlight,
}: {
  label: string;
  criteria: Criterion[];
  className?: string;
  highlight?: boolean;
}) {
  return (
    <div className={className}>
      <p
        className={cn(
          "text-[10px] uppercase tracking-[0.08em] font-medium mb-2",
          highlight ? "text-unknown" : "text-tertiary",
        )}
      >
        {label}
      </p>
      <ul className="divide-y divide-border rounded-md border border-border overflow-hidden bg-background">
        {criteria.map((c) => (
          <li key={c.id} className="grid grid-cols-[1fr_auto] gap-3 px-3 py-3">
            <div className="min-w-0">
              <p className="text-[13px] leading-snug text-foreground">{c.text}</p>
              <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-tertiary">
                <span>{c.semantic_class}</span>
                <span>·</span>
                <PathBadge path={c.path} />
                {c.cost_min != null && c.cost_min > 0 && (
                  <>
                    <span>·</span>
                    <span className="inline-flex items-center gap-1 text-muted-foreground">
                      <Clock className="size-3" aria-hidden="true" />
                      <span className="font-mono-tight text-foreground">{c.cost_min}</span> min
                    </span>
                  </>
                )}
                {c.unverifiable && (
                  <>
                    <span>·</span>
                    <span className="inline-flex items-center gap-1 text-unmet">
                      <Ban className="size-3" aria-hidden="true" />
                      Unverifiable
                    </span>
                  </>
                )}
              </div>
              <div className="mt-2">
                {c.evidence ? (
                  <Popover>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className="text-left inline-flex flex-wrap items-baseline gap-1 text-[12px] font-mono-tight text-muted-foreground hover:text-foreground transition-colors underline decoration-dotted decoration-border underline-offset-4"
                      >
                        <span>{c.evidence.value}</span>
                      </button>
                    </PopoverTrigger>
                    <PopoverContent align="start" className="w-72 text-[12px]">
                      <p className="text-tertiary">Source</p>
                      <p className="mt-0.5 font-medium">{c.evidence.source}</p>
                      <p className="mt-2 text-tertiary">Value</p>
                      <p className="mt-0.5 font-mono-tight">{c.evidence.value}</p>
                      {c.evidence.date && (
                        <>
                          <p className="mt-2 text-tertiary">Recorded</p>
                          <p className="mt-0.5 font-mono-tight">{c.evidence.date}</p>
                        </>
                      )}
                      <a href="#" className="mt-3 inline-block text-brand hover:underline underline-offset-4">
                        View in full record →
                      </a>
                    </PopoverContent>
                  </Popover>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[12px] text-unknown">
                    No data in record
                    {c.missing_data && (
                      <Popover>
                        <PopoverTrigger asChild>
                          <button
                            type="button"
                            className="underline decoration-dotted decoration-unknown/40 underline-offset-4"
                          >
                            · why
                          </button>
                        </PopoverTrigger>
                        <PopoverContent align="start" className="w-72 text-[12px]">
                          <p className="text-tertiary">Missing data</p>
                          <p className="mt-1 text-foreground leading-relaxed">{c.missing_data}</p>
                          {c.clarification_options && (
                            <>
                              <p className="mt-2 text-tertiary">Bounded options</p>
                              <ul className="mt-1 space-y-1 list-disc pl-4">
                                {c.clarification_options.map((o) => (
                                  <li key={o} className="text-foreground">{o}</li>
                                ))}
                              </ul>
                            </>
                          )}
                        </PopoverContent>
                      </Popover>
                    )}
                  </span>
                )}
              </div>
            </div>
            <div className="pt-0.5">
              <VerdictIcon verdict={c.verdict} />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

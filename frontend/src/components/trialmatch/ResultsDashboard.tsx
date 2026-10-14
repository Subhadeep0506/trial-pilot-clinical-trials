import { useEffect, useMemo, useState } from "react";
import { MOCK_TRIALS, MOCK_LOG, COVERAGE } from "@/lib/trialmatch/mockData";
import type { TrialPilot, ConfidenceTier } from "@/lib/trialmatch/types";
import { VerdictPill, VerdictIcon, TierPill, tierLabel } from "./Verdict";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Download, MapPin, ChevronRight, ShieldCheck, Clock, Ban } from "lucide-react";
import { TrialDetailPanel } from "./TrialDetailPanel";
import { cn } from "@/lib/utils";

function useIsDesktop() {
  const [isDesktop, setIsDesktop] = useState(false);
  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1024px)");
    const onChange = () => setIsDesktop(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);
  return isDesktop;
}

const TIER_ORDER: ConfidenceTier[] = ["high", "quick_check", "manual_review", "ineligible"];

const TIER_FILTERS: { id: ConfidenceTier | "all"; label: string }[] = [
  { id: "all", label: "All tiers" },
  { id: "high", label: "High confidence" },
  { id: "quick_check", label: "Quick check" },
  { id: "manual_review", label: "Manual review" },
  { id: "ineligible", label: "Ineligible" },
];

type Props = { onStartOver: () => void };

export function ResultsDashboard({ onStartOver }: Props) {
  const isDesktop = useIsDesktop();
  const [sort, setSort] = useState("cost");
  const [tierFilter, setTierFilter] = useState<ConfidenceTier | "all">("all");
  const [selectedId, setSelectedId] = useState<string | null>(MOCK_TRIALS[0]?.nct_id ?? null);

  const trials = useMemo(() => {
    let arr = [...MOCK_TRIALS];
    if (tierFilter !== "all") arr = arr.filter((t) => t.tier === tierFilter);
    switch (sort) {
      case "distance":
        arr.sort((a, b) => (a.locations[0]?.distance_mi ?? 1e9) - (b.locations[0]?.distance_mi ?? 1e9));
        break;
      case "confidence":
        arr.sort((a, b) => b.calibrated_confidence - a.calibrated_confidence);
        break;
      case "phase":
        arr.sort((a, b) => a.phase.localeCompare(b.phase));
        break;
      default:
        // Verification-cost ascending, then tier, then distance
        arr.sort((a, b) => {
          const t = TIER_ORDER.indexOf(a.tier) - TIER_ORDER.indexOf(b.tier);
          if (t !== 0) return t;
          if (a.verification_cost_min !== b.verification_cost_min)
            return a.verification_cost_min - b.verification_cost_min;
          return (a.locations[0]?.distance_mi ?? 1e9) - (b.locations[0]?.distance_mi ?? 1e9);
        });
    }
    return arr;
  }, [sort, tierFilter]);

  const grouped = useMemo(() => {
    const g = new Map<ConfidenceTier, TrialPilot[]>();
    for (const t of trials) {
      if (!g.has(t.tier)) g.set(t.tier, []);
      g.get(t.tier)!.push(t);
    }
    return TIER_ORDER.filter((k) => g.has(k)).map((k) => [k, g.get(k)!] as const);
  }, [trials]);

  const selected = trials.find((t) => t.nct_id === selectedId) ?? null;

  const totalCost = MOCK_TRIALS.reduce((s, t) => s + t.verification_cost_min, 0);

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      {/* Coverage banner */}
      <section
        aria-label="Coverage statement"
        className="mb-5 rounded-lg border border-brand/25 bg-brand-soft/50 px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-2"
      >
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-4 text-brand shrink-0" aria-hidden="true" />
          <p className="text-[13px] leading-snug">
            <span className="font-medium">Shortlist coverage:</span>{" "}
            <span className="text-muted-foreground">
              Contains the true-eligible trial with{" "}
              <span className="font-mono-tight text-foreground">
                ≥{Math.round(COVERAGE.probability * 100)}%
              </span>{" "}
              probability at{" "}
              <span className="font-mono-tight text-foreground">
                {Math.round(COVERAGE.confidence * 100)}%
              </span>{" "}
              confidence. Split conformal, calibrated on{" "}
              <span className="font-mono-tight text-foreground">{COVERAGE.calibration_n}</span>{" "}
              labeled cases.
            </span>
          </p>
        </div>
        <div className="ml-auto flex items-center gap-3 text-[12px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <Clock className="size-3.5" aria-hidden="true" />
            Total verification budget:{" "}
            <span className="font-mono-tight text-foreground">{totalCost} min</span>
          </span>
        </div>
      </section>

      {/* Sub bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div className="flex items-center gap-3">
          <h2 className="text-[16px] font-medium">
            <span className="font-mono-tight">{MOCK_TRIALS.length}</span> of{" "}
            <span className="font-mono-tight">87</span> trials shortlisted
          </h2>
          <span className="text-tertiary text-xs">·</span>
          <span className="text-[12px] text-muted-foreground">
            ranked by clinician verification cost
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Select value={sort} onValueChange={setSort}>
            <SelectTrigger className="h-9 w-[220px] bg-card text-[13px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cost">Least verification effort</SelectItem>
              <SelectItem value="confidence">Highest calibrated confidence</SelectItem>
              <SelectItem value="distance">Nearest site</SelectItem>
              <SelectItem value="phase">Phase</SelectItem>
            </SelectContent>
          </Select>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="h-9 bg-card">
                <Download className="size-4" aria-hidden="true" />
                Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem>Export JSON with evidence trail</DropdownMenuItem>
              <DropdownMenuItem>Export PDF report</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="ghost" size="sm" onClick={onStartOver}>
            Start over
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)_380px] xl:grid-cols-[240px_minmax(0,1fr)_480px] gap-6">
        {/* Sidebar */}
        <aside className="hidden lg:block">
          <div className="sticky top-20 space-y-5">
            <div className="rounded-lg border border-border bg-card p-4">
              <h4 className="text-[12px] font-medium text-muted-foreground mb-3 tracking-wide">
                Confidence tier
              </h4>
              <div className="flex flex-col gap-1">
                {TIER_FILTERS.map((v) => (
                  <button
                    key={v.id}
                    type="button"
                    onClick={() => setTierFilter(v.id)}
                    className={cn(
                      "text-left text-[13px] px-2 py-1.5 rounded-md transition-colors",
                      tierFilter === v.id
                        ? "bg-brand-soft text-brand"
                        : "text-muted-foreground hover:bg-secondary",
                    )}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-lg border border-border bg-card p-4">
              <h4 className="text-[12px] font-medium text-muted-foreground mb-3 tracking-wide">
                Search filters
              </h4>
              <ul className="space-y-2 text-[13px]">
                <li className="flex justify-between text-muted-foreground">
                  <span>Location</span>
                  <span className="font-mono-tight text-foreground">Boston, MA</span>
                </li>
                <li className="flex justify-between text-muted-foreground">
                  <span>Radius</span>
                  <span className="font-mono-tight text-foreground">100 mi</span>
                </li>
                <li className="flex justify-between text-muted-foreground">
                  <span>Status</span>
                  <span className="text-foreground">Recruiting</span>
                </li>
                <li className="flex justify-between text-muted-foreground">
                  <span>Phase</span>
                  <span className="text-foreground">II, III</span>
                </li>
              </ul>
            </div>

            <details className="rounded-lg border border-border bg-card p-4 group" open>
              <summary className="cursor-pointer text-[12px] font-medium text-muted-foreground tracking-wide list-none flex items-center justify-between">
                Pipeline log
                <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" aria-hidden="true" />
              </summary>
              <ScrollArea className="mt-3 max-h-72 pr-1">
                <ol className="space-y-2">
                  {MOCK_LOG.slice(-6).map((e) => (
                    <li key={e.id} className="flex items-start gap-2">
                      <span className="mt-1.5 size-1.5 rounded-full shrink-0 bg-met" aria-hidden="true" />
                      <div className="min-w-0">
                        <p className="text-[11px] font-medium text-foreground">{e.agent}</p>
                        <p className="text-[11px] text-tertiary">{e.action}</p>
                      </div>
                    </li>
                  ))}
                </ol>
              </ScrollArea>
              <div className="mt-3 pt-3 border-t border-border text-[11px] text-tertiary">
                87 screened · 3 clarifications resolved · 1 unverifiable
              </div>
            </details>
          </div>
        </aside>

        {/* Cards grouped by tier */}
        <div className="flex flex-col gap-8">
          {grouped.map(([tier, list]) => (
            <section key={tier} aria-label={tierLabel(tier)}>
              <header className="flex items-baseline justify-between mb-3">
                <div className="flex items-center gap-2.5">
                  <TierPill tier={tier} />
                  <span className="text-[12px] text-tertiary font-mono-tight">
                    {list.length} trial{list.length > 1 ? "s" : ""}
                  </span>
                </div>
                <p className="text-[11.5px] text-tertiary">{groupHint(tier)}</p>
              </header>
              <div className="flex flex-col gap-3">
                {list.map((t) => (
                  <TrialCard
                    key={t.nct_id}
                    trial={t}
                    selected={t.nct_id === selectedId}
                    onSelect={() => setSelectedId((cur) => (cur === t.nct_id ? null : t.nct_id))}
                  />
                ))}
              </div>
            </section>
          ))}
          {trials.length === 0 && (
            <div className="rounded-lg border border-dashed border-border bg-card/50 p-12 text-center">
              <p className="text-[14px] font-medium">No trials match the current filters.</p>
              <p className="mt-1 text-[13px] text-muted-foreground">
                Try widening the radius or removing the tier filter.
              </p>
              <Button
                className="mt-4 bg-brand hover:bg-brand/90 text-primary-foreground"
                size="sm"
                onClick={() => setTierFilter("all")}
              >
                Show all tiers
              </Button>
            </div>
          )}
        </div>

        {/* Detail panel (desktop) */}
        <aside className="hidden lg:block">
          <div className="sticky top-20 max-h-[calc(100vh-100px)] rounded-lg border border-border bg-card shadow-sm overflow-hidden">
            <ScrollArea className="h-[calc(100vh-100px)]">
              {selected ? (
                <TrialDetailPanel trial={selected} onClose={() => setSelectedId(null)} />
              ) : (
                <div className="p-10 text-center text-[13px] text-tertiary">
                  Select a trial to see the per-criterion breakdown.
                </div>
              )}
            </ScrollArea>
          </div>
        </aside>
      </div>

      {!isDesktop && (
        <Sheet open={!!selected} onOpenChange={(o) => !o && setSelectedId(null)}>
          <SheetContent side="right" className="w-full sm:max-w-[520px] p-0">
            <ScrollArea className="h-full">
              {selected && <TrialDetailPanel trial={selected} onClose={() => setSelectedId(null)} />}
            </ScrollArea>
          </SheetContent>
        </Sheet>
      )}
    </div>
  );
}

function groupHint(tier: ConfidenceTier) {
  switch (tier) {
    case "high":
      return "No open items. Call the site.";
    case "quick_check":
      return "A few minutes of chart work resolves the gap.";
    case "manual_review":
      return "External data or specialist review required.";
    case "ineligible":
      return "Deterministically excluded — shown for transparency.";
  }
}

function TrialCard({
  trial,
  selected,
  onSelect,
}: {
  trial: TrialPilot;
  selected: boolean;
  onSelect: () => void;
}) {
  const met = trial.criteria.filter((c) => c.verdict === "met").length;
  const unknown = trial.criteria.filter((c) => c.verdict === "unknown").length;
  const unmet = trial.criteria.filter((c) => c.verdict === "unmet").length;
  const unverifiable = trial.criteria.some((c) => c.unverifiable);
  const top = trial.criteria.slice(0, 2);
  const nearest = trial.locations[0];

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "group relative text-left rounded-lg border bg-card shadow-sm p-5 transition-colors hover:bg-secondary/30",
        selected ? "border-brand/40" : "border-border",
      )}
    >
      {selected && (
        <span aria-hidden="true" className="absolute left-0 top-3 bottom-3 w-[2px] bg-brand rounded-r" />
      )}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
        <div className="min-w-0 flex-1 order-2 sm:order-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-tertiary">
            <span className="font-mono-tight text-foreground">{trial.nct_id}</span>
            <span>·</span>
            <span className="rounded-sm border border-border bg-background px-1.5 py-0.5 text-[10px] text-muted-foreground">
              Phase {trial.phase}
            </span>
            <span
              className={cn(
                "rounded-sm border px-1.5 py-0.5 text-[10px]",
                trial.status === "Recruiting"
                  ? "border-met/30 bg-met-soft text-met"
                  : "border-border bg-background text-muted-foreground",
              )}
            >
              {trial.status}
            </span>
            {nearest?.distance_mi != null && (
              <>
                <span>·</span>
                <span className="inline-flex items-center gap-1 font-mono-tight text-foreground">
                  <MapPin className="size-3 text-tertiary" aria-hidden="true" />
                  {nearest.distance_mi} mi
                </span>
              </>
            )}
            {unverifiable && (
              <span className="inline-flex items-center gap-1 rounded-sm border border-unmet/30 bg-unmet-soft px-1.5 py-0.5 text-[10px] text-unmet">
                <Ban className="size-2.5" aria-hidden="true" />
                Unverifiable item
              </span>
            )}
          </div>
          <h3 className="mt-1.5 text-[15px] font-medium leading-snug line-clamp-2 group-hover:text-foreground">
            {trial.title}
          </h3>
        </div>
        <div className="order-1 sm:order-2 shrink-0 max-w-full">
          <VerdictPill
            verdict={trial.verdict}
            fraction={`${met}/${trial.criteria.length} met · ${unknown} unknown${unmet ? ` · ${unmet} unmet` : ""}`}
          />
        </div>
      </div>

      {/* Verification cost row */}
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.08em] text-tertiary">
            Verification cost
          </div>
          <div className="mt-0.5 flex items-baseline gap-1.5">
            <span className="text-[20px] font-medium font-mono-tight tabular-nums">
              {trial.verification_cost_min}
            </span>
            <span className="text-[11px] text-muted-foreground">min of clinician work</span>
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.08em] text-tertiary">
            Calibrated confidence
          </div>
          <div className="mt-1.5 flex items-center gap-2">
            <div className="flex-1 h-1 rounded-full bg-secondary overflow-hidden">
              <div
                className="h-full rounded-full bg-brand"
                style={{ width: `${trial.calibrated_confidence * 100}%` }}
              />
            </div>
            <span className="text-[12px] font-mono-tight text-muted-foreground tabular-nums">
              {(trial.calibrated_confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Top criteria preview */}
      <ul className="mt-4 space-y-1.5">
        {top.map((c) => (
          <li key={c.id} className="flex items-start gap-2 text-[13px]">
            <VerdictIcon verdict={c.verdict} className="mt-0.5" />
            <span className="text-muted-foreground line-clamp-1">{c.text}</span>
          </li>
        ))}
      </ul>

      <div className="mt-4 flex items-center justify-between text-[12px] text-tertiary">
        <span className="inline-flex items-center gap-1">
          <MapPin className="size-3" aria-hidden="true" />
          {trial.locations.length} site{trial.locations.length > 1 ? "s" : ""}
          {nearest?.distance_mi != null && ` · nearest ${nearest.distance_mi} mi`}
        </span>
        <span className="inline-flex items-center gap-0.5 text-brand">
          {selected ? "Hide details" : "View details"}
          <ChevronRight className="size-3.5" aria-hidden="true" />
        </span>
      </div>
    </button>
  );
}

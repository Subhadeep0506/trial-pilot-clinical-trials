import { Check, X, HelpCircle, Zap, ClipboardList, Ban } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Verdict, TrialVerdict, ConfidenceTier, VerificationPath } from "@/lib/trialmatch/types";

export function VerdictIcon({ verdict, className }: { verdict: Verdict; className?: string }) {
  if (verdict === "met") {
    return (
      <span className={cn("inline-flex size-4 items-center justify-center rounded-full bg-met-soft text-met", className)} aria-label="Met">
        <Check className="size-3" strokeWidth={2.5} aria-hidden="true" />
      </span>
    );
  }
  if (verdict === "unmet") {
    return (
      <span className={cn("inline-flex size-4 items-center justify-center rounded-full bg-unmet-soft text-unmet", className)} aria-label="Unmet">
        <X className="size-3" strokeWidth={2.5} aria-hidden="true" />
      </span>
    );
  }
  return (
    <span className={cn("inline-flex size-4 items-center justify-center rounded-full bg-unknown-soft text-unknown", className)} aria-label="Unknown">
      <HelpCircle className="size-3" strokeWidth={2.5} aria-hidden="true" />
    </span>
  );
}

export function VerdictPill({ verdict, fraction }: { verdict: TrialVerdict; fraction?: string }) {
  const map = {
    eligible: { label: "Eligible", cls: "bg-met-soft text-met border-met/20", Icon: Check },
    eligible_pending_unknown: { label: "Eligible pending", cls: "bg-unknown-soft text-unknown border-unknown/20", Icon: HelpCircle },
    ineligible: { label: "Ineligible", cls: "bg-unmet-soft text-unmet border-unmet/20", Icon: X },
  } as const;
  const { label, cls, Icon } = map[verdict];
  return (
    <span className={cn("inline-flex flex-wrap items-center gap-x-1.5 gap-y-0.5 rounded-md border px-2 py-1 text-[13px] font-medium max-w-full", cls)}>
      <Icon className="size-3.5 shrink-0" strokeWidth={2.5} aria-hidden="true" />
      <span className="whitespace-nowrap">{label}</span>
      {fraction && (
        <span className="text-[11px] font-mono-tight opacity-80 pl-1.5 ml-0.5 border-l border-current/20 whitespace-nowrap">
          {fraction}
        </span>
      )}
    </span>
  );
}

const TIER_META: Record<ConfidenceTier, { label: string; short: string; cls: string; Icon: typeof Check }> = {
  high: {
    label: "High confidence",
    short: "High",
    cls: "bg-met-soft text-met border-met/30",
    Icon: Zap,
  },
  quick_check: {
    label: "Needs a quick check",
    short: "Quick check",
    cls: "bg-unknown-soft text-unknown border-unknown/30",
    Icon: HelpCircle,
  },
  manual_review: {
    label: "Needs manual review",
    short: "Manual review",
    cls: "bg-secondary text-foreground border-border",
    Icon: ClipboardList,
  },
  ineligible: {
    label: "Ineligible",
    short: "Ineligible",
    cls: "bg-unmet-soft text-unmet border-unmet/20",
    Icon: Ban,
  },
};

export function TierPill({ tier, size = "md" }: { tier: ConfidenceTier; size?: "sm" | "md" }) {
  const { label, cls, Icon } = TIER_META[tier];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border font-medium whitespace-nowrap",
        cls,
        size === "sm" ? "px-2 h-6 text-[11px]" : "px-2.5 h-7 text-[12px]",
      )}
    >
      <Icon className="size-3.5 shrink-0" strokeWidth={2.5} aria-hidden="true" />
      {label}
    </span>
  );
}

export function tierLabel(tier: ConfidenceTier) {
  return TIER_META[tier].label;
}

export function PathBadge({ path }: { path: VerificationPath }) {
  const map: Record<VerificationPath, { label: string; hint: string; cls: string }> = {
    deterministic: {
      label: "Deterministic",
      hint: "Exact numeric or date predicate evaluated in code",
      cls: "border-brand/30 bg-brand-soft text-brand",
    },
    narrative: {
      label: "Narrative",
      hint: "LLM entailment on qualitative text with sentence-level citation",
      cls: "border-border bg-background text-muted-foreground",
    },
    predicate: {
      label: "Predicate",
      hint: "Domain rule gated on a symbolic predicate",
      cls: "border-border bg-secondary text-muted-foreground",
    },
  };
  const m = map[path];
  return (
    <span
      title={m.hint}
      className={cn("inline-flex items-center rounded-sm border px-1.5 py-0.5 text-[10px] font-mono-tight tracking-wide", m.cls)}
    >
      {m.label}
    </span>
  );
}

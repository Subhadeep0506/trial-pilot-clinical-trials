import { Activity, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "./ThemeToggle";

type Props = {
  patientSummary?: string;
  onShowActivity?: () => void;
  onStartOver?: () => void;
  onLogoClick?: () => void;
  step: "landing" | "intake" | "confirm" | "processing" | "clarify" | "results";
};

const stepLabel: Record<Props["step"], string> = {
  landing: "Overview",
  intake: "Intake",
  confirm: "Profile confirmation",
  processing: "Live pipeline",
  clarify: "Clarification",
  results: "Ranked shortlist",
};

export function TopBar({ patientSummary, onShowActivity, onStartOver, onLogoClick, step }: Props) {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="mx-auto max-w-7xl px-6 h-14 flex items-center gap-4">
        <button
          type="button"
          onClick={onLogoClick}
          className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
        >
          <div
            aria-hidden="true"
            className="size-5 rounded-sm border border-brand/40 bg-brand-soft grid place-items-center"
          >
            <div className="size-1.5 rounded-full bg-brand" />
          </div>
          <span className="text-[15px] font-medium tracking-tight">TrialPilot</span>
        </button>

        <span className="text-tertiary text-xs">/</span>
        <span className="text-xs text-muted-foreground">{stepLabel[step]}</span>

        {patientSummary && (
          <button
            type="button"
            className="ml-2 inline-flex items-center gap-2 rounded-md border border-border bg-secondary/40 px-2.5 h-7 text-xs text-foreground hover:bg-secondary transition-colors"
          >
            <span className="size-1.5 rounded-full bg-brand" aria-hidden="true" />
            <span className="font-mono-tight">{patientSummary}</span>
          </button>
        )}

        <div className="ml-auto flex items-center gap-1">
          {onShowActivity && (
            <Button variant="ghost" size="sm" onClick={onShowActivity}>
              <Activity className="size-4" aria-hidden="true" />
              View activity
            </Button>
          )}
          {onStartOver && (
            <Button variant="ghost" size="sm" onClick={onStartOver}>
              <RotateCcw className="size-4" aria-hidden="true" />
              Start over
            </Button>
          )}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

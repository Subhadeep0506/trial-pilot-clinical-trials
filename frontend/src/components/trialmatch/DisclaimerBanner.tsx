import { ShieldAlert } from "lucide-react";

export function DisclaimerBanner({ variant = "footer" }: { variant?: "footer" | "inline" }) {
  return (
    <div
      role="note"
      className={
        variant === "footer"
          ? "border-t border-border bg-background"
          : "border border-border rounded-md bg-secondary/40"
      }
    >
      <div className="mx-auto max-w-7xl px-6 py-3 flex items-start gap-3">
        <ShieldAlert
          aria-hidden="true"
          className="size-4 mt-0.5 text-tertiary shrink-0"
        />
        <p className="text-xs text-muted-foreground leading-relaxed">
          TrialPilot is a research and educational screening aid. It is not clinical
          advice and must not be used as the sole basis for treatment decisions.
        </p>
      </div>
    </div>
  );
}

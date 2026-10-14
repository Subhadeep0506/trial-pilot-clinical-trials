import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type StreamStep = { label: string; ms: number };

type Props = {
  eyebrow: string;
  title: string;
  subtitle?: string;
  steps: StreamStep[];
  onDone: () => void;
};

/**
 * Vertically scrolling status stream used for the long-running extraction and
 * re-analysis phases. Purely presentational simulation.
 */
export function ProcessingStream({ eyebrow, title, subtitle, steps, onDone }: Props) {
  const [idx, setIdx] = useState(0);
  const total = steps.reduce((a, s) => a + s.ms, 0);
  const listRef = useRef<HTMLOListElement>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const timers: number[] = [];
    let acc = 0;
    steps.forEach((s, i) => {
      acc += reduce ? Math.min(s.ms, 120) : s.ms;
      timers.push(window.setTimeout(() => setIdx(i + 1), acc));
    });
    timers.push(window.setTimeout(() => doneRef.current(), acc + 500));
    return () => timers.forEach((t) => window.clearTimeout(t));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>("[data-active='true']");
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [idx]);

  const elapsedMs = steps.slice(0, idx).reduce((a, s) => a + s.ms, 0);
  const pct = Math.round((elapsedMs / total) * 100);
  const current = steps[Math.min(idx, steps.length - 1)];

  return (
    <div className="mx-auto max-w-[720px] px-6 py-16">
      <div className="text-[11px] font-mono-tight tracking-widest uppercase text-tertiary">
        {eyebrow}
      </div>
      <h2 className="mt-2 text-[22px] font-medium tracking-tight">{title}</h2>
      {subtitle && (
        <p className="mt-1 text-[13px] text-muted-foreground">{subtitle}</p>
      )}

      <div className="mt-8 rounded-lg border border-border bg-card shadow-sm overflow-hidden">
        <div className="flex items-center gap-3 px-5 py-4 border-b border-border">
          <Loader2 className="size-4 text-brand animate-spin shrink-0" aria-hidden="true" />
          <p
            aria-live="polite"
            className="text-[14px] font-medium text-foreground min-w-0 truncate"
          >
            {idx >= steps.length ? "Wrapping up…" : current.label}
          </p>
          <span className="ml-auto text-[12px] font-mono-tight text-tertiary tabular-nums shrink-0">
            {Math.min(idx + 1, steps.length)} / {steps.length}
          </span>
        </div>

        {/* Scrolling stream */}
        <div className="relative h-[280px]">
          <div className="pointer-events-none absolute inset-x-0 top-0 h-10 z-10 bg-gradient-to-b from-card to-transparent" />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 z-10 bg-gradient-to-t from-card to-transparent" />
          <ol ref={listRef} className="h-full overflow-hidden py-8">
            {steps.map((s, i) => {
              const done = i < idx;
              const active = i === idx;
              return (
                <li
                  key={s.label}
                  data-active={active}
                  className={cn(
                    "flex items-center gap-3 px-5 py-2.5 transition-all duration-500",
                    done && "opacity-45",
                    active && "opacity-100",
                    !done && !active && "opacity-20",
                  )}
                >
                  {done ? (
                    <CheckCircle2 className="size-3.5 text-met shrink-0" aria-hidden="true" />
                  ) : active ? (
                    <span
                      className="size-3.5 shrink-0 grid place-items-center"
                      aria-hidden="true"
                    >
                      <span className="size-1.5 rounded-full bg-brand animate-ping" />
                    </span>
                  ) : (
                    <span
                      className="size-3.5 shrink-0 grid place-items-center"
                      aria-hidden="true"
                    >
                      <span className="size-1.5 rounded-full border border-border" />
                    </span>
                  )}
                  <span
                    className={cn(
                      "text-[13px] font-mono-tight leading-snug",
                      active ? "text-foreground" : "text-muted-foreground",
                    )}
                  >
                    {s.label}
                  </span>
                </li>
              );
            })}
          </ol>
        </div>

        <div className="flex items-center gap-3 px-5 py-3 border-t border-border">
          <div className="flex-1 h-1 rounded-full bg-secondary overflow-hidden">
            <div
              className="h-full bg-brand transition-all duration-500 ease-linear"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-[11px] font-mono-tight text-tertiary tabular-nums">
            {pct}%
          </span>
        </div>
      </div>

      <p className="mt-4 text-[12px] text-tertiary">
        This runs a multi-agent pipeline — it takes a moment. Please keep this tab open.
      </p>
    </div>
  );
}

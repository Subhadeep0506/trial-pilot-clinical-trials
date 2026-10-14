import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { ArrowLeft, ArrowRight, AlertCircle, Ban, CheckCircle2 } from "lucide-react";
import { MOCK_CLARIFICATIONS } from "@/lib/trialmatch/mockData";
import type { ClarificationQuestion } from "@/lib/trialmatch/types";
import { cn } from "@/lib/utils";

type Props = { onBack: () => void; onContinue: () => void };

const MAX_ROUNDS = 3;

export function ClarificationLoop({ onBack, onContinue }: Props) {
  const [questions] = useState<ClarificationQuestion[]>(MOCK_CLARIFICATIONS);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [round, setRound] = useState(1);

  const answeredCount = useMemo(
    () => questions.filter((q) => answers[q.id]).length,
    [answers, questions],
  );
  const allAnswered = answeredCount === questions.length;

  const setAnswer = (qid: string, value: string) =>
    setAnswers((prev) => ({ ...prev, [qid]: value }));

  const markUnverifiable = (qid: string) =>
    setAnswers((prev) => ({ ...prev, [qid]: "unverifiable" }));

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <div className="flex flex-col gap-2">
        <div className="text-[11px] font-mono-tight tracking-widest uppercase text-tertiary">
          Stage 4 · Bounded clarification
        </div>
        <h2 className="text-[22px] font-medium tracking-tight">
          Resolve gaps before ranking
        </h2>
        <p className="text-[14px] text-muted-foreground max-w-2xl">
          The retriever flagged {questions.length} facts that we can&apos;t determine from the record.
          Each has 2–4 structured options. We&apos;ll stop after round{" "}
          <span className="font-mono-tight text-foreground">{MAX_ROUNDS}</span> and mark anything
          still unresolved as <em className="not-italic text-foreground">unverifiable</em> — no
          open-ended chat, no hallucinated facts.
        </p>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-3">
        <StatTile label="Gaps identified" value={questions.length.toString()} />
        <StatTile label="Answered this round" value={`${answeredCount} / ${questions.length}`} />
        <StatTile
          label="Round"
          value={`${round} of ${MAX_ROUNDS}`}
          hint={round === MAX_ROUNDS ? "Last round" : `${MAX_ROUNDS - round} remaining`}
        />
      </div>

      <ScrollArea className="mt-6 h-[calc(100vh-360px)]">
        <ol className="space-y-4 pr-2">
          {questions.map((q, i) => {
            const answer = answers[q.id];
            const unverifiable = answer === "unverifiable";
            return (
              <li
                key={q.id}
                className={cn(
                  "rounded-lg border bg-card shadow-sm p-5 transition-colors",
                  answer
                    ? unverifiable
                      ? "border-unmet/30"
                      : "border-met/30"
                    : "border-border",
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-[11px] text-tertiary">
                      <span className="font-mono-tight text-foreground">{q.trial_id}</span>
                      <span>·</span>
                      <span>Q{i + 1} of {questions.length}</span>
                    </div>
                    <h3 className="mt-1.5 text-[14px] font-medium leading-snug">
                      {q.criterion_text}
                    </h3>
                  </div>
                  {answer && !unverifiable && (
                    <CheckCircle2 className="size-4 text-met shrink-0" aria-hidden="true" />
                  )}
                  {unverifiable && (
                    <Ban className="size-4 text-unmet shrink-0" aria-hidden="true" />
                  )}
                </div>

                <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="rounded-md border border-border bg-background p-3">
                    <p className="text-[10px] uppercase tracking-[0.08em] text-tertiary">
                      Currently in the record
                    </p>
                    <p className="mt-1 text-[12.5px] text-foreground leading-relaxed">
                      {q.current_data}
                    </p>
                  </div>
                  <div className="rounded-md border border-unknown/30 bg-unknown-soft/60 p-3">
                    <p className="text-[10px] uppercase tracking-[0.08em] text-unknown flex items-center gap-1.5">
                      <AlertCircle className="size-3" aria-hidden="true" />
                      Gap
                    </p>
                    <p className="mt-1 text-[12.5px] text-foreground leading-relaxed">
                      {q.gap}
                    </p>
                  </div>
                </div>

                <div className="mt-4">
                  <p className="text-[11px] uppercase tracking-[0.08em] text-tertiary mb-2">
                    Pick one
                  </p>
                  <RadioGroup
                    value={answer && answer !== "unverifiable" ? answer : ""}
                    onValueChange={(v) => setAnswer(q.id, v)}
                    className="grid gap-1.5"
                  >
                    {q.options.map((opt) => (
                      <Label
                        key={opt}
                        className={cn(
                          "flex items-center gap-2.5 rounded-md border px-3 py-2 cursor-pointer transition-colors",
                          answer === opt
                            ? "border-brand/40 bg-brand-soft/60"
                            : "border-border bg-background hover:bg-secondary/40",
                        )}
                      >
                        <RadioGroupItem value={opt} className="shrink-0" />
                        <span className="text-[13px] leading-snug">{opt}</span>
                      </Label>
                    ))}
                  </RadioGroup>
                  <button
                    type="button"
                    onClick={() => markUnverifiable(q.id)}
                    className={cn(
                      "mt-2 text-[11.5px] underline-offset-4 hover:underline transition-colors",
                      unverifiable ? "text-unmet" : "text-tertiary hover:text-foreground",
                    )}
                  >
                    {unverifiable
                      ? "Marked unverifiable — trial will be tagged."
                      : "None of the above · mark unverifiable"}
                  </button>
                </div>
              </li>
            );
          })}
        </ol>
      </ScrollArea>

      <div className="mt-8 flex items-center justify-between gap-3">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft className="size-4" aria-hidden="true" />
          Back
        </Button>
        <div className="flex items-center gap-2">
          {round < MAX_ROUNDS && (
            <Button
              variant="outline"
              onClick={() => setRound((r) => Math.min(MAX_ROUNDS, r + 1))}
              className="bg-background"
              disabled={!answeredCount}
            >
              Ask another round
            </Button>
          )}
          <Button
            onClick={onContinue}
            disabled={!allAnswered}
            className="bg-brand hover:bg-brand/90 text-primary-foreground"
          >
            Rank remaining trials
            <ArrowRight className="size-4" aria-hidden="true" />
          </Button>
        </div>
      </div>
    </div>
  );
}

function StatTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-md border border-border bg-card px-4 py-3">
      <p className="text-[11px] uppercase tracking-[0.08em] text-tertiary">{label}</p>
      <p className="mt-1 text-[18px] font-medium font-mono-tight tabular-nums">{value}</p>
      {hint && <p className="mt-0.5 text-[11px] text-tertiary">{hint}</p>}
    </div>
  );
}

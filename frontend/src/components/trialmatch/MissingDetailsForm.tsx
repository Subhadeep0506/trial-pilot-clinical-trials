import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertCircle, ArrowLeft, ArrowRight, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type MissingDetailsAnswers = Record<string, string>;

type Field =
  | { id: string; label: string; hint: string; kind: "choice"; options: string[] }
  | { id: string; label: string; hint: string; kind: "text"; placeholder: string }
  | { id: string; label: string; hint: string; kind: "textarea"; placeholder: string };

export const MISSING_FIELDS: Field[] = [
  {
    id: "ecog",
    label: "Performance status (ECOG)",
    hint: "Referenced by nearly every oncology protocol. Not stated explicitly in the narrative.",
    kind: "choice",
    options: ["0 — fully active", "1 — restricted, ambulatory", "2 — ambulatory > 50% of day", "3 or 4", "Unknown"],
  },
  {
    id: "lvef_date",
    label: "Date of most recent LVEF assessment",
    hint: "Cardiac-gating criteria usually require an assessment within 28 days.",
    kind: "text",
    placeholder: "YYYY-MM-DD",
  },
  {
    id: "prior_lines",
    label: "Prior lines of systemic therapy in the metastatic setting",
    hint: "Line-count caps are a common hard exclusion.",
    kind: "choice",
    options: ["0", "1", "2", "3", "4 or more"],
  },
  {
    id: "smoking",
    label: "Smoking status",
    hint: "Used for pulmonary and ILD-related exclusions.",
    kind: "choice",
    options: ["Never smoker", "Former smoker", "Current smoker", "Unknown"],
  },
  {
    id: "repro",
    label: "Pregnancy / menopausal status",
    hint: "Required for contraception and pregnancy exclusions.",
    kind: "choice",
    options: ["Postmenopausal", "Premenopausal, not pregnant", "Pregnant or breastfeeding", "Not applicable", "Unknown"],
  },
  {
    id: "comorbid",
    label: "Active comorbidities and organ dysfunction",
    hint: "Free text. Include cardiac, hepatic, renal, autoimmune or infectious conditions.",
    kind: "textarea",
    placeholder: "e.g. controlled hypertension on amlodipine; no cardiac, hepatic or renal impairment",
  },
];

type Props = {
  onBack: () => void;
  onSubmit: (answers: MissingDetailsAnswers) => void;
};

export function MissingDetailsForm({ onBack, onSubmit }: Props) {
  const [answers, setAnswers] = useState<MissingDetailsAnswers>({});

  const answered = useMemo(
    () => MISSING_FIELDS.filter((f) => (answers[f.id] ?? "").trim().length > 0).length,
    [answers],
  );
  const complete = answered === MISSING_FIELDS.length;

  const set = (id: string, v: string) => setAnswers((p) => ({ ...p, [id]: v }));

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <div className="text-[11px] font-mono-tight tracking-widest uppercase text-tertiary">
        Stage 2b · Extraction gaps
      </div>
      <h2 className="mt-2 text-[22px] font-medium tracking-tight">
        A few details are missing from the record
      </h2>
      <p className="mt-1 text-[14px] text-muted-foreground max-w-2xl">
        Extraction finished, but {MISSING_FIELDS.length} facts that most protocols gate on
        weren&apos;t stated in the narrative. Fill these in so we don&apos;t guess — every field is
        required before we re-analyse.
      </p>

      <div className="mt-5 flex items-center gap-2.5 rounded-md border border-unknown/30 bg-unknown-soft px-3 py-2 text-[13px] text-unknown">
        <AlertCircle className="size-4 shrink-0" aria-hidden="true" />
        {answered} of {MISSING_FIELDS.length} answered — nothing here is inferred by the model.
      </div>

      <ScrollArea className="mt-6 h-[calc(100vh-400px)] min-h-[320px]">
        <ol className="space-y-4 pr-3">
          {MISSING_FIELDS.map((f, i) => {
            const value = answers[f.id] ?? "";
            const filled = value.trim().length > 0;
            return (
              <li
                key={f.id}
                className={cn(
                  "rounded-lg border bg-card shadow-sm p-5 transition-colors",
                  filled ? "border-met/30" : "border-border",
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-[11px] text-tertiary font-mono-tight">
                      Field {i + 1} of {MISSING_FIELDS.length}
                    </p>
                    <h3 className="mt-1 text-[14px] font-medium leading-snug">{f.label}</h3>
                    <p className="mt-1 text-[12.5px] text-muted-foreground leading-relaxed">
                      {f.hint}
                    </p>
                  </div>
                  {filled && (
                    <CheckCircle2 className="size-4 text-met shrink-0" aria-hidden="true" />
                  )}
                </div>

                <div className="mt-4">
                  {f.kind === "choice" && (
                    <RadioGroup
                      value={value}
                      onValueChange={(v) => set(f.id, v)}
                      className="grid gap-1.5"
                    >
                      {f.options.map((opt) => (
                        <Label
                          key={opt}
                          className={cn(
                            "flex items-center gap-2.5 rounded-md border px-3 py-2 cursor-pointer transition-colors",
                            value === opt
                              ? "border-brand/40 bg-brand-soft/60"
                              : "border-border bg-background hover:bg-secondary/40",
                          )}
                        >
                          <RadioGroupItem value={opt} className="shrink-0" />
                          <span className="text-[13px] leading-snug">{opt}</span>
                        </Label>
                      ))}
                    </RadioGroup>
                  )}
                  {f.kind === "text" && (
                    <Input
                      value={value}
                      onChange={(e) => set(f.id, e.target.value)}
                      placeholder={f.placeholder}
                      className="bg-background font-mono-tight"
                    />
                  )}
                  {f.kind === "textarea" && (
                    <Textarea
                      value={value}
                      onChange={(e) => set(f.id, e.target.value)}
                      placeholder={f.placeholder}
                      className="bg-background min-h-24 text-[13px] leading-relaxed resize-y"
                    />
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      </ScrollArea>

      <div className="mt-8 flex items-center justify-between gap-3">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft className="size-4" aria-hidden="true" />
          Back to intake
        </Button>
        <Button
          disabled={!complete}
          onClick={() => onSubmit(answers)}
          className="bg-brand hover:bg-brand/90 text-primary-foreground"
        >
          Submit and re-analyse
          <ArrowRight className="size-4" aria-hidden="true" />
        </Button>
      </div>
    </div>
  );
}

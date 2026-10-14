import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Pencil, AlertCircle, ArrowLeft, ArrowRight, Plus, Trash2 } from "lucide-react";
import type { PatientProfile } from "@/lib/trialmatch/types";

type Section = "demo" | "perf" | "cond" | "med" | "labs" | "cancer";

type Props = {
  profile: PatientProfile;
  onChange: (p: PatientProfile) => void;
  onBack: () => void;
  onContinue: () => void;
};

const SECTION_TITLES: Record<Section, string> = {
  demo: "Demographics",
  perf: "Performance status",
  cond: "Conditions",
  med: "Medications",
  labs: "Recent labs",
  cancer: "Cancer-specific",
};

function Card({
  title,
  children,
  onEdit,
}: {
  title: string;
  children: React.ReactNode;
  onEdit: () => void;
}) {
  return (
    <section className="rounded-lg border border-border bg-card shadow-sm">
      <header className="flex items-center justify-between px-5 py-3 border-b border-border">
        <h3 className="text-[13px] font-medium text-muted-foreground tracking-wide">{title}</h3>
        <button
          type="button"
          onClick={onEdit}
          className="inline-flex items-center gap-1 text-[12px] text-tertiary hover:text-foreground transition-colors cursor-pointer"
          aria-label={`Edit ${title}`}
        >
          <Pencil className="size-3" aria-hidden="true" />
          Edit
        </button>
      </header>
      <div className="p-5 text-[14px]">{children}</div>
    </section>
  );
}

function Chip({ label, code, lowConf }: { label: string; code?: string; lowConf?: boolean }) {
  const inner = (
    <span
      className={
        "inline-flex items-center gap-1.5 rounded-md border px-2 h-7 text-[13px] " +
        (lowConf
          ? "border-unknown/40 bg-unknown-soft text-unknown"
          : "border-border bg-secondary/50 text-foreground")
      }
    >
      {lowConf && <span className="size-1.5 rounded-full bg-unknown" aria-hidden="true" />}
      <span>{label}</span>
      {code && <span className="text-[11px] text-tertiary font-mono-tight">·</span>}
    </span>
  );
  if (!code) return inner;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{inner}</TooltipTrigger>
      <TooltipContent>
        <span className="font-mono-tight">{code}</span>
      </TooltipContent>
    </Tooltip>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-[12px] text-muted-foreground">{label}</Label>
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="bg-background"
      />
    </div>
  );
}

export function ProfileConfirmation({ profile, onChange, onBack, onContinue }: Props) {
  const [editing, setEditing] = useState<Section | null>(null);
  const [draft, setDraft] = useState<PatientProfile>(profile);

  useEffect(() => {
    if (editing) setDraft(structuredClone(profile));
  }, [editing, profile]);

  const lowConfCount = profile.conditions.filter((c) => c.confidence === "low").length;

  const save = () => {
    onChange(draft);
    setEditing(null);
  };

  const patch = (fn: (d: PatientProfile) => void) =>
    setDraft((prev) => {
      const next = structuredClone(prev);
      fn(next);
      return next;
    });

  return (
    <TooltipProvider delayDuration={150}>
      <div className="mx-auto max-w-5xl px-6 py-10">
        <div>
          <h2 className="text-[20px] font-medium tracking-tight">
            Confirm extracted patient profile
          </h2>
          <p className="mt-1 text-[14px] text-muted-foreground">
            Review what TrialPilot extracted. Edit anything that&apos;s wrong before we search.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-5">
          <Card title="Demographics" onEdit={() => setEditing("demo")}>
            <dl className="grid grid-cols-2 gap-y-2 text-[14px]">
              <dt className="text-muted-foreground">Age</dt>
              <dd className="font-mono-tight">{profile.demographics.age}</dd>
              <dt className="text-muted-foreground">Sex</dt>
              <dd className="font-mono-tight">{profile.demographics.sex}</dd>
              {profile.demographics.weight_kg != null && (
                <>
                  <dt className="text-muted-foreground">Weight</dt>
                  <dd className="font-mono-tight">{profile.demographics.weight_kg} kg</dd>
                </>
              )}
              {profile.demographics.height_cm != null && (
                <>
                  <dt className="text-muted-foreground">Height</dt>
                  <dd className="font-mono-tight">{profile.demographics.height_cm} cm</dd>
                </>
              )}
            </dl>
          </Card>

          <Card title="Performance status" onEdit={() => setEditing("perf")}>
            {profile.performance ? (
              <div className="flex items-baseline gap-2">
                <span className="text-muted-foreground">{profile.performance.name}</span>
                <span className="font-mono-tight text-[16px]">{profile.performance.value}</span>
              </div>
            ) : (
              <span className="text-tertiary">Not documented</span>
            )}
          </Card>

          <Card title="Conditions" onEdit={() => setEditing("cond")}>
            <div className="flex flex-wrap gap-1.5">
              {profile.conditions.map((c) => (
                <Chip
                  key={c.name}
                  label={c.name}
                  code={c.snomed && `SNOMED ${c.snomed}`}
                  lowConf={c.confidence === "low"}
                />
              ))}
            </div>
          </Card>

          <Card title="Medications" onEdit={() => setEditing("med")}>
            <div className="flex flex-wrap gap-1.5">
              {profile.medications.map((m) => (
                <Chip
                  key={m.name}
                  label={`${m.name}${m.start ? ` · ${m.start}${m.stop ? "—" + m.stop : "—present"}` : ""}`}
                  code={m.rxnorm && `RxNorm ${m.rxnorm}`}
                />
              ))}
            </div>
          </Card>

          <Card title="Recent labs" onEdit={() => setEditing("labs")}>
            <ul className="divide-y divide-border">
              {profile.labs.map((l) => (
                <li key={l.name} className="flex items-baseline justify-between py-1.5 text-[13px]">
                  <span className="text-muted-foreground">{l.name}</span>
                  <span className="font-mono-tight">
                    {l.value} {l.unit}
                    <span className="text-tertiary"> · {l.date}</span>
                  </span>
                </li>
              ))}
            </ul>
          </Card>

          {profile.cancer && (
            <Card title="Cancer-specific" onEdit={() => setEditing("cancer")}>
              <div className="space-y-3 text-[13px]">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Stage</span>
                  <span className="font-mono-tight">{profile.cancer.stage}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Prior lines of therapy</span>
                  <span className="font-mono-tight">{profile.cancer.prior_lines}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Biomarkers</span>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {profile.cancer.biomarkers?.map((b) => (
                      <Chip key={b.name} label={`${b.name}: ${b.value}`} />
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          )}
        </div>

        {lowConfCount > 0 && (
          <div className="mt-6 flex items-center gap-2.5 rounded-md border border-unknown/30 bg-unknown-soft px-3 py-2 text-[13px] text-unknown">
            <AlertCircle className="size-4" aria-hidden="true" />
            {lowConfCount} fact{lowConfCount > 1 ? "s have" : " has"} low confidence — review
            highlighted items.
          </div>
        )}

        <div className="mt-8 flex items-center justify-between">
          <Button variant="ghost" onClick={onBack}>
            <ArrowLeft className="size-4" aria-hidden="true" />
            Back
          </Button>
          <Button
            onClick={onContinue}
            className="bg-brand hover:bg-brand/90 text-primary-foreground"
          >
            Continue to trial search
            <ArrowRight className="size-4" aria-hidden="true" />
          </Button>
        </div>

        {/* Edit dialog */}
        <Dialog open={editing !== null} onOpenChange={(o) => !o && setEditing(null)}>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle className="text-[16px] font-medium">
                Edit {editing ? SECTION_TITLES[editing].toLowerCase() : ""}
              </DialogTitle>
              <DialogDescription className="text-[13px]">
                Changes here override what the extractor produced and are used for matching.
              </DialogDescription>
            </DialogHeader>

            <ScrollArea className="max-h-[55vh]">
              <div className="grid gap-4 pr-3 pb-1">
                {editing === "demo" && (
                  <div className="grid grid-cols-2 gap-4">
                    <Field
                      label="Age"
                      value={String(draft.demographics.age ?? "")}
                      onChange={(v) => patch((d) => { d.demographics.age = Number(v) || 0; })}
                    />
                    <Field
                      label="Sex (F / M)"
                      value={draft.demographics.sex}
                      onChange={(v) =>
                        patch((d) => {
                          d.demographics.sex = v.toUpperCase() === "M" ? "M" : "F";
                        })
                      }
                    />
                    <Field
                      label="Weight (kg)"
                      value={String(draft.demographics.weight_kg ?? "")}
                      onChange={(v) =>
                        patch((d) => {
                          d.demographics.weight_kg = v ? Number(v) : undefined;
                        })
                      }
                    />
                    <Field
                      label="Height (cm)"
                      value={String(draft.demographics.height_cm ?? "")}
                      onChange={(v) =>
                        patch((d) => {
                          d.demographics.height_cm = v ? Number(v) : undefined;
                        })
                      }
                    />
                  </div>
                )}

                {editing === "perf" && (
                  <div className="grid grid-cols-2 gap-4">
                    <Field
                      label="Scale"
                      value={draft.performance?.name ?? ""}
                      placeholder="ECOG"
                      onChange={(v) =>
                        patch((d) => {
                          d.performance = { name: v, value: d.performance?.value ?? "" };
                        })
                      }
                    />
                    <Field
                      label="Value"
                      value={draft.performance?.value ?? ""}
                      placeholder="1"
                      onChange={(v) =>
                        patch((d) => {
                          d.performance = { name: d.performance?.name ?? "ECOG", value: v };
                        })
                      }
                    />
                  </div>
                )}

                {editing === "cond" &&
                  draft.conditions.map((c, i) => (
                    <div key={i} className="grid grid-cols-[1fr_140px_auto] gap-3 items-end">
                      <Field
                        label="Condition"
                        value={c.name}
                        onChange={(v) => patch((d) => { d.conditions[i].name = v; })}
                      />
                      <Field
                        label="SNOMED"
                        value={c.snomed ?? ""}
                        onChange={(v) => patch((d) => { d.conditions[i].snomed = v || undefined; })}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label="Remove condition"
                        onClick={() => patch((d) => { d.conditions.splice(i, 1); })}
                      >
                        <Trash2 className="size-4 text-tertiary" aria-hidden="true" />
                      </Button>
                    </div>
                  ))}

                {editing === "med" &&
                  draft.medications.map((m, i) => (
                    <div key={i} className="grid grid-cols-[1fr_120px_120px_auto] gap-3 items-end">
                      <Field
                        label="Medication"
                        value={m.name}
                        onChange={(v) => patch((d) => { d.medications[i].name = v; })}
                      />
                      <Field
                        label="Start"
                        value={m.start ?? ""}
                        onChange={(v) => patch((d) => { d.medications[i].start = v || undefined; })}
                      />
                      <Field
                        label="Stop"
                        value={m.stop ?? ""}
                        onChange={(v) => patch((d) => { d.medications[i].stop = v || undefined; })}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label="Remove medication"
                        onClick={() => patch((d) => { d.medications.splice(i, 1); })}
                      >
                        <Trash2 className="size-4 text-tertiary" aria-hidden="true" />
                      </Button>
                    </div>
                  ))}

                {editing === "labs" &&
                  draft.labs.map((l, i) => (
                    <div key={i} className="grid grid-cols-[1fr_90px_90px_120px_auto] gap-3 items-end">
                      <Field
                        label="Lab"
                        value={l.name}
                        onChange={(v) => patch((d) => { d.labs[i].name = v; })}
                      />
                      <Field
                        label="Value"
                        value={l.value}
                        onChange={(v) => patch((d) => { d.labs[i].value = v; })}
                      />
                      <Field
                        label="Unit"
                        value={l.unit}
                        onChange={(v) => patch((d) => { d.labs[i].unit = v; })}
                      />
                      <Field
                        label="Date"
                        value={l.date}
                        onChange={(v) => patch((d) => { d.labs[i].date = v; })}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label="Remove lab"
                        onClick={() => patch((d) => { d.labs.splice(i, 1); })}
                      >
                        <Trash2 className="size-4 text-tertiary" aria-hidden="true" />
                      </Button>
                    </div>
                  ))}

                {editing === "cancer" && (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <Field
                        label="Stage"
                        value={draft.cancer?.stage ?? ""}
                        onChange={(v) =>
                          patch((d) => {
                            d.cancer = { ...(d.cancer ?? {}), stage: v };
                          })
                        }
                      />
                      <Field
                        label="Prior lines of therapy"
                        value={String(draft.cancer?.prior_lines ?? "")}
                        onChange={(v) =>
                          patch((d) => {
                            d.cancer = { ...(d.cancer ?? {}), prior_lines: Number(v) || 0 };
                          })
                        }
                      />
                    </div>
                    {(draft.cancer?.biomarkers ?? []).map((b, i) => (
                      <div key={i} className="grid grid-cols-[1fr_1fr_auto] gap-3 items-end">
                        <Field
                          label="Biomarker"
                          value={b.name}
                          onChange={(v) => patch((d) => { d.cancer!.biomarkers![i].name = v; })}
                        />
                        <Field
                          label="Value"
                          value={b.value}
                          onChange={(v) => patch((d) => { d.cancer!.biomarkers![i].value = v; })}
                        />
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label="Remove biomarker"
                          onClick={() => patch((d) => { d.cancer!.biomarkers!.splice(i, 1); })}
                        >
                          <Trash2 className="size-4 text-tertiary" aria-hidden="true" />
                        </Button>
                      </div>
                    ))}
                  </>
                )}

                {(editing === "cond" || editing === "med" || editing === "labs" || editing === "cancer") && (
                  <Button
                    variant="outline"
                    className="bg-background justify-self-start"
                    onClick={() =>
                      patch((d) => {
                        if (editing === "cond") d.conditions.push({ name: "", confidence: "low" });
                        if (editing === "med") d.medications.push({ name: "" });
                        if (editing === "labs") d.labs.push({ name: "", value: "", unit: "", date: "" });
                        if (editing === "cancer") {
                          d.cancer = d.cancer ?? {};
                          d.cancer.biomarkers = [...(d.cancer.biomarkers ?? []), { name: "", value: "" }];
                        }
                      })
                    }
                  >
                    <Plus className="size-4" aria-hidden="true" />
                    Add row
                  </Button>
                )}
              </div>
            </ScrollArea>

            <DialogFooter>
              <Button variant="ghost" onClick={() => setEditing(null)}>
                Cancel
              </Button>
              <Button onClick={save} className="bg-brand hover:bg-brand/90 text-primary-foreground">
                Save changes
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
}

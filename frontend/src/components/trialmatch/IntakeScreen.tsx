import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Upload, FileText } from "lucide-react";
import { SAMPLE_NARRATIVES } from "@/lib/trialmatch/mockData";
import { cn } from "@/lib/utils";

type Props = { onSubmit: (narrative: string) => void };

const STATUS_OPTIONS = ["Recruiting", "Active, not recruiting", "Enrolling by invitation"];
const PHASE_OPTIONS = ["I", "II", "III", "IV", "N/A"];
const INTERVENTION_OPTIONS = ["Drug", "Biological", "Device", "Behavioral", "Procedure", "Other"];

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "h-7 px-2.5 rounded-md border text-xs font-medium transition-colors cursor-pointer",
        active
          ? "border-brand bg-brand-soft text-brand"
          : "border-border bg-background text-muted-foreground hover:bg-secondary",
      )}
    >
      {children}
    </button>
  );
}

export function IntakeScreen({ onSubmit }: Props) {
  const [narrative, setNarrative] = useState("");
  const [location, setLocation] = useState("Boston, MA");
  const [radius, setRadius] = useState([100]);
  const [statuses, setStatuses] = useState<string[]>(["Recruiting"]);
  const [phases, setPhases] = useState<string[]>(["II", "III"]);
  const [interventions, setInterventions] = useState<string[]>(["Drug", "Biological"]);
  const [refinement, setRefinement] = useState("");

  const toggle = (arr: string[], set: (a: string[]) => void, v: string) =>
    set(arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);

  const loadSample = (id: string) => {
    const s = SAMPLE_NARRATIVES.find((x) => x.id === id);
    if (s) setNarrative(s.text);
  };

  return (
    <div className="mx-auto max-w-[720px] px-6 py-12">
      <h1 className="text-[24px] font-medium tracking-tight">
        Find clinical trials for a patient
      </h1>
      <p className="mt-2 text-[14px] text-muted-foreground">
        Paste a clinical narrative, upload a FHIR bundle, or load a sample patient.
      </p>

      <div className="mt-8 rounded-lg border border-border bg-card shadow-sm">
        <Tabs defaultValue="narrative" className="w-full">
          <div className="px-5 pt-5">
            <TabsList className="bg-secondary/60">
              <TabsTrigger value="narrative">Narrative</TabsTrigger>
              <TabsTrigger value="fhir">FHIR bundle</TabsTrigger>
              <TabsTrigger value="sample">Sample patient</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="narrative" className="px-5 pb-5 pt-4">
            <Textarea
              value={narrative}
              onChange={(e) => setNarrative(e.target.value)}
              placeholder="e.g. 58F with HER2+ metastatic breast cancer, progressed on trastuzumab + pertuzumab, brain mets treated with SRS 8 weeks ago. ECOG 1. LVEF 58%."
              className="min-h-44 font-mono-tight text-[13px] leading-relaxed resize-y bg-background"
            />
            <p className="mt-2 text-[12px] text-tertiary">
              Free text. We'll extract structured facts before searching trials.
            </p>
          </TabsContent>

          <TabsContent value="fhir" className="px-5 pb-5 pt-4">
            <label className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-background py-10 cursor-pointer hover:bg-secondary/40 transition-colors">
              <Upload className="size-5 text-tertiary" aria-hidden="true" />
              <span className="text-[13px] text-foreground">
                Drop a FHIR bundle (.json) or click to choose a file
              </span>
              <span className="text-[12px] text-tertiary">
                Patient, Condition, MedicationStatement, Observation resources are parsed
              </span>
              <input type="file" accept="application/json" className="hidden" />
            </label>
          </TabsContent>

          <TabsContent value="sample" className="px-5 pb-5 pt-4">
            <Select onValueChange={loadSample}>
              <SelectTrigger className="bg-background">
                <SelectValue placeholder="Choose a synthetic patient" />
              </SelectTrigger>
              <SelectContent>
                {SAMPLE_NARRATIVES.map((s) => (
                  <SelectItem key={s.id} value={s.id}>
                    <FileText className="size-3.5 text-tertiary inline mr-2" />
                    {s.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {narrative && (
              <div className="mt-4 rounded-md border border-border bg-background p-3 text-[13px] font-mono-tight text-muted-foreground leading-relaxed">
                {narrative}
              </div>
            )}
          </TabsContent>
        </Tabs>

        <div className="border-t border-border">
          <Accordion type="single" collapsible>
            <AccordionItem value="filters" className="border-0">
              <AccordionTrigger className="px-5 py-3 hover:no-underline text-[13px] font-medium text-muted-foreground">
                Optional filters
              </AccordionTrigger>
              <AccordionContent className="px-5 pb-5">
                <div className="grid gap-5">
                  <div>
                    <label className="text-[12px] text-muted-foreground">
                      Location
                    </label>
                    <Input
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      className="mt-1.5 bg-background"
                      placeholder="City, state"
                    />
                  </div>
                  <div>
                    <div className="flex items-center justify-between">
                      <label className="text-[12px] text-muted-foreground">
                        Search radius
                      </label>
                      <span className="text-[12px] font-mono-tight text-foreground">
                        {radius[0]} mi
                      </span>
                    </div>
                    <Slider
                      value={radius}
                      onValueChange={setRadius}
                      min={10}
                      max={500}
                      step={10}
                      className="mt-3"
                    />
                  </div>
                  <div>
                    <label className="text-[12px] text-muted-foreground">
                      Recruitment status
                    </label>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {STATUS_OPTIONS.map((s) => (
                        <Chip
                          key={s}
                          active={statuses.includes(s)}
                          onClick={() => toggle(statuses, setStatuses, s)}
                        >
                          {s}
                        </Chip>
                      ))}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-5">
                    <div>
                      <label className="text-[12px] text-muted-foreground">
                        Phase
                      </label>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {PHASE_OPTIONS.map((s) => (
                          <Chip
                            key={s}
                            active={phases.includes(s)}
                            onClick={() => toggle(phases, setPhases, s)}
                          >
                            {s}
                          </Chip>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="text-[12px] text-muted-foreground">
                        Intervention type
                      </label>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {INTERVENTION_OPTIONS.map((s) => (
                          <Chip
                            key={s}
                            active={interventions.includes(s)}
                            onClick={() => toggle(interventions, setInterventions, s)}
                          >
                            {s}
                          </Chip>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>

        <div className="border-t border-border px-5 py-4 flex flex-col gap-3">
          <Input
            value={refinement}
            onChange={(e) => setRefinement(e.target.value)}
            placeholder="Any specific focus? e.g. immunotherapy only, no chemo (optional)"
            className="bg-background"
          />
          <div className="flex items-center justify-between gap-3">
            {!narrative ? (
              <button
                type="button"
                onClick={() => loadSample("her2-mbc")}
                className="text-[13px] text-brand hover:underline underline-offset-4"
              >
                Load sample patient
              </button>
            ) : <span />}
            <Button
              disabled={!narrative.trim()}
              onClick={() => onSubmit(narrative)}
              className="bg-brand hover:bg-brand/90 text-primary-foreground"
            >
              Find matching trials
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

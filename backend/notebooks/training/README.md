# Fine-tuning the Laya decision model

Runbook for TRD stage 4b: fine-tune `laya-multilingual` on the synthetic intake
corpus (TRD §4.4), export an ONNX graph, and bring both back to this repo.

**Training does not run on the dev box.** That machine has a CPU-only torch build
(`2.14.1+cpu`), two performance cores and ~2.7 GB free; a full fine-tune of the
~322M-parameter checkpoint needs roughly 5.2 GB of optimiser state alone (TRD R10).
These scripts are written to be uploaded to a Colab or Kaggle GPU runtime and run
there. `train_laya.py` refuses to start without CUDA unless you ask for `--device cpu`
explicitly.

| | |
|---|---|
| `train_laya.py` | Fine-tune. Wraps `laya.train.finetune`. |
| `export_onnx.py` | Checkpoint → `model.onnx` + `model.onnx_data`. |
| `requirements.txt` | What to pip install. Deliberately does **not** name torch. |

Neither script imports anything from `src/`, and neither writes outside its own
output directory. The fitted abstention thresholds and the new model paths are
**printed** for you to paste into `backend/.env` when you choose to.

---

## 1. Upload

New Colab notebook → **Runtime → Change runtime type → T4 GPU** (or better).

Upload these five files into `/content` via the file pane:

```
train_laya.py
export_onnx.py
requirements.txt
intake_v1.train.jsonl      from backend/data/synthetic/
intake_v1.eval.jsonl       from backend/data/synthetic/
```

Both scripts look for the corpus beside themselves first, then in the working
directory, then at `backend/data/synthetic/` — so a flat `/content` needs no flags.

## 2. Cells

```python
!nvidia-smi
```

Confirm a GPU is attached before paying for anything else.

```python
!pip install -q -r requirements.txt
import torch; print(torch.__version__, torch.cuda.is_available())
```

Must print `True`. If it prints `False`, the runtime has no GPU and the next cell
will stop with a message saying so.

```python
!python train_laya.py --dry-run
```

The acceptance gate from TRD §4.4. It builds every training item **without loading
model weights** and must report `skipped: {}`. This matters more than it looks:
a wrong label literal does **not** raise — `laya/train.py` counts it under
`invalid_target` and drops the item silently — so this check is the only thing
between a format mistake and a wasted multi-hour run. The real run performs the
same check first and aborts if anything was dropped (`--allow-skipped` overrides).

```python
!python train_laya.py
```

The run. Downloads `convaiinnovations/laya` (multilingual subfolder) on first use —
`backend/models/laya-ml-checkpoint` in this repo holds only configs and the
tokenizer, with no `model.safetensors`, so it cannot be trained from.

```python
!python export_onnx.py
```

Exports `laya-intake-v1-onnx/model.onnx` and `model.onnx_data`, then verifies the
graph against the contract `DecisionLMService` enforces.

```python
import shutil
from google.colab import files

shutil.make_archive("laya-intake-v1", "zip", "laya-intake-v1")
shutil.make_archive("laya-intake-v1-onnx", "zip", "laya-intake-v1-onnx")
files.download("laya-intake-v1.zip")        # ~640 MB, fp16 weights
files.download("laya-intake-v1-onnx.zip")   # ~1.3 GB, fp32 weights
```

Two separate archives: ~1.9 GB in one download is a long way to fall.

## 3. Defaults, and why

`train_laya.py` ships laya's defaults with two deliberate changes:

| | Default here | laya's default | Why |
|---|---|---|---|
| `--epochs` | **6** | 4 | The 4-epoch budget is sized for a few thousand typed decisions. The corpus is ~956 rows × 11 questions ≈ 10.5k items, small enough that the head can finish with every option at the same logit — chance-level, and silent. `finetune` warns when it detects this; more epochs is the first remedy. |
| `--shuffle-options` | **on** | off | Without it, options are always presented in the order the question lists them and the model learns a position prior. The measured failure is `performance_status`, a 6-option head — the widest in the catalogue and the most exposed. |

Everything else is laya's: `rlcd` loss, micro-batch 8 × grad-accum 8 (effective
batch 64), encoder LR 2.5e-5, head LR 1e-4, `--target-error 0.10`. All are flags.

Mixed precision and gradient checkpointing turn themselves on when the device is
CUDA. `--amp off` is the first thing to try if the loss goes NaN;
`--gradient-checkpointing off` trades VRAM for speed.

## 4. Time, and sessions that drop

The dry run on the corpus as of 2026-10-14 reports:

```
956 rows read, 10135 valid items, skipped: {}
eval: 107 rows read, 1128 valid items, skipped: {}
optimizer budget: 9735 train item(s), effective batch up to 64
                  (micro_batch 8 x grad_accum 8), 918 optimizer updates over 6 epoch(s)
```

9,735 training items at micro-batch 8 is 1,217 forward/backward passes per epoch —
7,302 over six epochs — through a 22-layer encoder at `max_len` 1024 with gradient
checkpointing on. On a T4 that is hours, and free Colab disconnects on idle.

Two levers, in order of how much they cost you:

- **`--max-len 512`** roughly halves the compute. The narratives run ~160–400 state
  tokens, so little is actually truncated. The catch: whatever value is used is
  written into the saved config, so **inference must then use the same value** —
  it becomes a property of the checkpoint, not of the run.
- **`--epochs 4`** back to laya's budget, at a higher risk of prior collapse.

If the session dies mid-run, `finetune` has been saving a full checkpoint after
every epoch:

```python
!python train_laya.py --base ./laya-intake-v1/checkpoint_latest --epochs 2
```

Honest caveat: this restarts the LR schedule from the top and re-draws the
calibration split, so it is a continuation, not a resume.

## 5. Reading the output

In order of what will actually tell you something:

1. **The collapse warning.** `laya.train: predictions collapsed to the class prior`
   means the run finished and learned nothing. Everything below is then noise.
2. **The before/after table.** Printed by `finetune`: accuracy, loss, ECE, Brier on
   the eval split, base checkpoint versus fine-tuned. Check `eval_mode` is
   `held_out` — `overlapping_eval` or `in_sample_calibration` means the number is
   not a generalisation estimate, and the script warns when it is not.
3. **Accuracy by question type.** `noul` is the `*_stated` head, which was collapsed
   and unusable on the stock checkpoint — stated and unstated facts scored
   indistinguishably. Whether it learned is the single most interesting result here.
   Anything near 0.5 is still collapsed.
4. **The baseline block.** Printed at the end: the measured stock-checkpoint answers
   on the §6.1 catalogue, including `ecog_0` at 0.836 confidence for a narrative
   describing ECOG 1. That is the bar.
5. **The fitted abstention thresholds.** One cut per option-count bucket, fitted at
   `--target-error`. The values in `backend/.env` today are guesses.

Everything is also in `laya-intake-v1/train.log`, `train_report.json` and
`run_summary.json`, so closing the tab does not lose the run.

Treat eval-split accuracy as an **upper bound**, not an estimate of field
performance. The corpus is entirely synthetic and LLM-written, and the model
inherits its blind spots invisibly (TRD R9).

## 6. The ONNX contract

`export_onnx.py` targets the layout of `onnx-community/laya-multilingual-ONNX`,
the only prebuilt Laya graph that works in this project:

```
ir_version 10 | opset 18 | fp32 weights, external, in model.onnx_data
inputs : input_ids      INT64 [batch, seq]
         attention_mask INT64 [batch, seq]
         marker_pos     INT64 [batch, markers]
         marker_mask    BOOL  [batch, markers]
         qtype          INT64 [batch]
outputs: logits         FLOAT [batch, markers]
         act_logits     FLOAT [batch, 2]
```

Three things in there are load-bearing, and the script checks each:

- **`marker_pos` dim 1 must stay dynamic.** `ONNXAgent` feeds a variable-width
  `marker_pos`, and `DecisionLMService._check_graph` refuses a graph with a fixed
  option axis. `Mattepiu/laya-onnx` is pinned to `[batch, 2]` and is unusable for
  precisely this reason.
- **The outputs must be named** `logits` and `act_logits` — `ONNXAgent` runs
  `session.run(["logits", "act_logits"], ...)` by name.
- **The weights file must be named `model.onnx_data`** and sit beside the graph.

There is also a trap in the module being exported: `DecisionModel.forward` contains
`if p.size(-1) >= 2: top2 = p.topk(2, -1)`, a Python branch the tracer bakes in.
Exporting with a single-option example would freeze the one-option path and ONNX
Runtime would refuse every real call. The script always builds its example batch
from the widest question in the schema.

Verification runs torch and ONNX Runtime on **two batches of different sequence
length and different marker count** and compares. One batch proves the graph runs;
two prove the dynamic axes are real and not merely named. Expect a max absolute
difference around 1e-5.

The re-save step holds the full fp32 weights in memory briefly (~2.6 GB). Fine on
a Colab runtime, worth knowing if you run it somewhere smaller.

`--dynamo` switches to torch's newer exporter. The legacy tracer is the default
because laya's `_DynamicMultiheadAttention` was written so that tracing does not
bake in the sequence length, and because it reproduces the reference axis names.

## 7. Bringing it back

Unzip into the repo:

```
backend/models/laya-intake-v1/          the checkpoint: weights, tokenizer, config,
                                        questions.json, train_report.json
backend/models/laya-intake-v1-onnx/     model.onnx + model.onnx_data
```

Copy **both**. The graph holds the weights; the checkpoint directory supplies the
tokenizer, the config and the fitted temperatures, and `ONNXAgent` needs both.

Then paste the lines the scripts printed into `backend/.env`:

```
DECISION_CHECKPOINT_ID="./models/laya-intake-v1"
DECISION_ONNX_PATH="./models/laya-intake-v1-onnx/model.onnx"
DECISION_MIN_CONFIDENCE='{...}'
```

Nothing is rewired automatically — by decision.

Finally, re-measure the §6.1 catalogue against the new graph through the
`/decision/predict` harness route before wiring the field catalogue into intake
(TRD stage 4c). The comparison against the stock-checkpoint baseline is what
decides whether §6.1's contingency options are needed at all.

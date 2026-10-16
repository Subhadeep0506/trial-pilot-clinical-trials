#!/usr/bin/env python
"""Fine-tune the Laya decision model on the synthetic intake corpus (TRD stage 4b).
python train_laya.py --dry-run          # acceptance gate, no weights loaded
python train_laya.py                    # the real run, requires CUDA
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
# backend/notebooks/training -> backend
BACKEND_DIR = SCRIPT_DIR.parent.parent
SYNTHETIC_DIR = BACKEND_DIR / "data" / "synthetic"

TRAIN_FILE = "intake_v1.train.jsonl"
EVAL_FILE = "intake_v1.eval.jsonl"
DEFAULT_BASE = "multilingual"
DEFAULT_OUT = "laya-intake-v1"

HUB_DATASET = "DenCT/laya-decisions-intake-10k"
# Hub split -> the filename _find_input already looks for.
HUB_SPLITS = {"train": TRAIN_FILE, "validation": EVAL_FILE}
# Key order of backend/data/synthetic/intake_v1.*.jsonl, reproduced so a
# downloaded corpus diffs cleanly against the committed one.
CORPUS_KEYS = (
    "state",
    "questions",
    "expected",
    "gold",  # soft-label probabilities; present on ~29% of rows, omitted elsewhere
    "seed",
    "generator",
    "domain",
    "presentation",
)

BASELINE_NOTE = """Stock-checkpoint baseline on the section 6.1 catalogue (measured 2026-10-14):
  performance_status -> ecog_0      @ 0.836  WRONG (truth ECOG 1), clears any gate
  age_band           -> under_18    @ 0.297  wrong (truth 63), caught by abstention
  smoking_status     -> former      @ 0.634  right, discarded by the 0.65 gate
  prior_lines        -> 1           @ 0.50   wrong (truth 3), caught by abstention
  sex                -> female      @ 0.908  right
  menopausal         -> postmeno.   @ 0.864  right
  *_stated (noul)    -> collapsed; stated and unstated are indistinguishable"""


def _disable_broken_truststore() -> None:
    """Undo truststore's SSL injection on Python 3.14, where it recurses forever."""
    if sys.version_info < (3, 14):
        return
    import ssl

    if "truststore" not in getattr(ssl.SSLContext, "__module__", ""):
        return
    for name in ("pip._vendor.truststore", "truststore"):
        try:
            module = __import__(name, fromlist=["extract_from_ssl"])
            module.extract_from_ssl()
        except Exception:  # noqa: BLE001 - try the next candidate module
            continue
        if "truststore" not in getattr(ssl.SSLContext, "__module__", ""):
            print(
                "note: removed truststore's SSL injection (%s); certificates now come "
                "from certifi." % name,
                flush=True,
            )
            return


def _tristate(value: str) -> bool | None:
    """'auto' -> None, which is how TrainConfig spells 'follow the device'."""
    return {"auto": None, "on": True, "off": False}[value]


def _find_input(explicit: str | None, filename: str, what: str) -> str:
    """Resolve a corpus path: the flag, then beside this script, then the repo."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            sys.exit("train_laya: %s file not found: %s" % (what, path))
        return str(path)

    candidates = [
        SCRIPT_DIR / filename,
        Path.cwd() / filename,
        SYNTHETIC_DIR / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    sys.exit(
        "train_laya: could not find the %s corpus (%s). Looked in:\n  %s\n"
        "Pass an explicit path, or upload the file next to this script."
        % (what, filename, "\n  ".join(str(c) for c in candidates))
    )


def _corpus_row(row: dict[str, Any]) -> dict[str, Any]:
    """One Hub row in the synthetic corpus's key order.

    A parquet row carries every column, so an absent `gold` arrives as None;
    the synthetic corpus omits the key instead, and so does this.
    """
    extras = [key for key in row if key not in CORPUS_KEYS]
    return {
        key: row[key]
        for key in (*CORPUS_KEYS, *extras)
        if row.get(key) is not None
    }


def download_corpus(
    repo_id: str = HUB_DATASET,
    dest: str | Path | None = None,
    revision: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download the intake corpus from the Hub as intake_v1.{train,eval}.jsonl.

    The Hub copy is backend/data/synthetic re-serialised as parquet: same
    columns, same 3071/342 rows, so what lands here is the format laya's
    loader already consumes. It is not byte-identical -- parquet normalises a
    few soft-label floats (0.39999999999999997 -> 0.4) -- but it parses equal.
    Writes beside this script by default, the first place _find_input looks,
    so a download does not clobber the committed corpus.
    """
    _disable_broken_truststore()
    try:
        from datasets import load_dataset
    except ImportError as err:
        sys.exit(
            "train_laya: downloading the corpus needs the 'datasets' package (%s)."
            % err
        )

    dest_dir = Path(dest).expanduser() if dest else SCRIPT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split, filename in HUB_SPLITS.items():
        path = dest_dir / filename
        if path.exists() and not overwrite:
            sys.exit(
                "train_laya: %s already exists. Pass --overwrite to replace it, or "
                "--download-to another directory." % path
            )

        print("downloading %s [%s] ..." % (repo_id, split), flush=True)
        try:
            dataset = load_dataset(repo_id, split=split, revision=revision)
        except Exception as err:  # noqa: BLE001 - datasets raises many types here
            sys.exit("train_laya: could not load %s [%s]: %s" % (repo_id, split, err))

        tmp = path.parent / (path.name + ".tmp")
        rows = 0
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in dataset:
                handle.write(json.dumps(_corpus_row(row), ensure_ascii=False) + "\n")
                rows += 1
        os.replace(tmp, path)

        written[split] = path
        print(
            "  %-24s %6d rows, %8.1f KB"
            % (path.name, rows, path.stat().st_size / 1024),
            flush=True,
        )
    return written


class _Tee:
    """Mirror a stream to a file, so a closed Colab tab does not lose the run."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._handle.write(text)
        self._handle.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


# laya's two training-loop log lines, which the progress bar consumes.
_STEP_LINE = re.compile(r"^epoch (\d+)/(\d+) step (\d+) loss ([\d.]+)$")
_EPOCH_LINE = re.compile(r"^epoch (\d+)/(\d+) mean loss ([\d.]+)$")


class _ProgressBar:
    """Render laya's per-step log lines as a tqdm bar.

    laya hardcodes `print("epoch %d/%d step %d loss %.4f")` in its training
    loop and exposes no progress hook, so there is nothing to subscribe to.
    What there is: `print` inside laya/train.py resolves as a module global
    before it reaches builtins, so binding one there reroutes those lines
    without patching the installed package -- which matters, because on a GPU
    runtime laya is a fresh pip install that any local edit would not survive.
    """

    def __init__(self, tqdm_cls, epochs: int, steps_per_epoch: int | None, stream, log_handle=None):
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epoch
        self._log = log_handle
        self._epoch = 1
        self._step = 0  # last step number seen inside the current epoch
        self._ema: float | None = None
        self._bar = tqdm_cls(
            total=epochs * steps_per_epoch if steps_per_epoch else None,
            unit="step",
            dynamic_ncols=True,
            smoothing=0.05,
            file=stream,
            desc="epoch 1/%d" % epochs,
        )

    def handle(self, text: str) -> bool:
        """Absorb a step line into the bar. True when the line was consumed."""
        match = _STEP_LINE.match(text)
        if match:
            self._advance(int(match.group(1)), int(match.group(3)), float(match.group(4)))
            if self._log is not None:  # the per-step trace still reaches train.log
                self._log.write(text + "\n")
            return True

        match = _EPOCH_LINE.match(text)
        if match:
            self._end_epoch(int(match.group(1)))
        return False  # mean-loss lines are the ones worth reading; let them print

    def _advance(self, epoch: int, step: int, loss: float) -> None:
        if epoch != self._epoch:  # laya restarts `step` at 1 every epoch
            self._epoch, self._step = epoch, 0
            self._bar.set_description("epoch %d/%d" % (epoch, self._epochs))
        delta, self._step = max(0, step - self._step), step
        # Raw per-micro-batch loss swings hard on 8 items; show it smoothed too.
        self._ema = loss if self._ema is None else 0.9 * self._ema + 0.1 * loss
        self._bar.set_postfix(loss="%.4f" % loss, ema="%.4f" % self._ema, refresh=False)
        self._bar.update(delta)

    def _end_epoch(self, epoch: int) -> None:
        # Without a dry run the steps-per-epoch is unknown until one finishes;
        # adopt the count so the bar gets a real total from epoch 2 onwards.
        if not self._steps_per_epoch and self._step:
            self._steps_per_epoch = self._step
            self._bar.total = self._epochs * self._steps_per_epoch
        if self._steps_per_epoch:  # resync: log_every rarely divides the epoch
            self._bar.n = min(epoch * self._steps_per_epoch, self._bar.total)
            self._bar.refresh()

    def close(self) -> None:
        self._bar.close()


def _install_progress_bar(epochs: int, steps_per_epoch: int | None, stream, log_handle=None):
    """Point laya.train's `print` at a tqdm bar. Returns a restore callable."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(
            "note: tqdm is not installed, so laya's plain step lines stay as they are.",
            flush=True,
        )
        return None

    import laya.train as laya_train

    bar = _ProgressBar(tqdm, epochs, steps_per_epoch, stream, log_handle)
    had_print = "print" in vars(laya_train)
    original = vars(laya_train).get("print", print)

    def hook(*args, **kwargs):
        if "file" not in kwargs:  # never swallow anything aimed at stderr
            text = " ".join(str(a) for a in args)
            if bar.handle(text):
                return
        original(*args, **kwargs)

    laya_train.print = hook

    def restore() -> None:
        bar.close()
        if had_print:
            laya_train.print = original
        else:
            del laya_train.print

    return restore


def _describe_gpu() -> str:
    import torch

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return "cuda:%d %s, %.1f GB VRAM, capability %d.%d" % (
        index,
        props.name,
        props.total_memory / 1024**3,
        props.major,
        props.minor,
    )


def _require_device(device: str) -> None:
    """Refuse to start a multi-hour run on the wrong device, with a usable message."""
    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            sys.exit(
                "train_laya: --device %s was requested but no CUDA device is visible.\n"
                "  torch %s\n"
                "  This is the expected failure on the TrialMatch dev box, which has a\n"
                "  CPU-only torch build and no CUDA GPU (TRD R10). Run this on a GPU\n"
                "  runtime -- see README.md -- or pass --device cpu to train anyway,\n"
                "  which will take many hours and may not fit in memory."
                % (device, torch.__version__)
            )
        print("device: %s" % _describe_gpu(), flush=True)
        print(
            "        AMP (fp16 autocast + GradScaler) and gradient checkpointing resolve to\n"
            "        on for CUDA; override with --amp off / --gradient-checkpointing off.",
            flush=True,
        )
        return

    if device == "cpu":
        print(
            "device: cpu -- WARNING. A full fine-tune of the ~322M-parameter checkpoint\n"
            "        needs roughly 5.2 GB of optimiser state alone (TRD R10). Expect many\n"
            "        hours, and consider --freeze-encoder, which fits in less memory but\n"
            "        trains only the heads.",
            flush=True,
        )
        return

    print("device: %s (unverified)" % device, flush=True)


def _print_by_type(report: dict[str, Any]) -> None:
    """Per-question-type accuracy before and after."""
    before = (report.get("before") or {}).get("by_type") or {}
    after = (report.get("after") or {}).get("by_type") or {}
    names = [n for n in ("choice", "score", "noul") if n in before or n in after]
    if not names:
        return

    print("\n=== Accuracy by question type ===", flush=True)
    print(
        "%-10s %-8s %-10s %-10s %-10s" % ("Type", "Items", "Before", "After", "Delta"),
        flush=True,
    )
    print("-" * 52, flush=True)
    for name in names:
        b = before.get(name) or {}
        a = after.get(name) or {}
        b_acc, a_acc = b.get("accuracy"), a.get("accuracy")
        delta = (
            ("%+.4f" % (a_acc - b_acc))
            if (b_acc is not None and a_acc is not None)
            else "N/A"
        )
        print(
            "%-10s %-8s %-10s %-10s %-10s"
            % (
                name,
                a.get("items", b.get("items", "-")),
                "%.4f" % b_acc if b_acc is not None else "N/A",
                "%.4f" % a_acc if a_acc is not None else "N/A",
                delta,
            ),
            flush=True,
        )
    print("-" * 52, flush=True)
    if "noul" in after:
        print(
            "noul is the '*_stated' head, which was collapsed and unusable on the stock\n"
            "checkpoint. Treat any accuracy near 0.5 here as still collapsed.",
            flush=True,
        )


def _print_env_block(summary: dict[str, Any], out_dir: Path) -> None:
    """The lines to paste into backend/.env. Printed, never written (by decision)."""
    name = out_dir.name
    thresholds = summary.get("abstention_thresholds") or {}

    print(
        "\n=== Paste into backend/.env when you bring the artifacts back ===",
        flush=True,
    )
    print('DECISION_CHECKPOINT_ID="./models/%s"' % name, flush=True)
    print('DECISION_ONNX_PATH="./models/%s-onnx/model.onnx"' % name, flush=True)
    if thresholds:
        compact = json.dumps(
            {k: round(float(v), 4) for k, v in sorted(thresholds.items())},
            separators=(",", ":"),
        )
        print("DECISION_MIN_CONFIDENCE='%s'" % compact, flush=True)
        print(
            "\n  These thresholds are FITTED at --target-error, one cut per option-count\n"
            "  bucket. The values currently in .env are guesses.",
            flush=True,
        )
    else:
        print(
            "\n  No abstention thresholds were fitted: every bucket had fewer than\n"
            "  --min-abstain-n calibration items. Leave DECISION_MIN_CONFIDENCE as it is;\n"
            "  an empty map is refused by the validator.",
            flush=True,
        )

    temperature = summary.get("temperature")
    if temperature:
        print(
            "\nFitted temperature (written into the checkpoint config): %s"
            % [round(float(t), 4) for t in temperature],
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_laya.py",
        description="Fine-tune Laya on the synthetic intake corpus (TRD section 4.4 / stage 4b).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data and artifacts")
    data.add_argument(
        "--data",
        default=None,
        help="Training JSONL. Default: %s beside this script, the cwd, or "
        "backend/data/synthetic/." % TRAIN_FILE,
    )
    data.add_argument(
        "--eval",
        dest="eval_data",
        default=None,
        help="Eval JSONL, used for the before/after comparison. Default: %s, "
        "found the same way." % EVAL_FILE,
    )
    data.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the held-out eval split; metrics then come from the in-sample "
        "calibration slice only.",
    )
    data.add_argument(
        "--download",
        nargs="?",
        const=HUB_DATASET,
        default=None,
        metavar="REPO_ID",
        help="Download the corpus from the Hub as %s / %s and exit. Takes an "
        "optional dataset id; defaults to %s."
        % (TRAIN_FILE, EVAL_FILE, HUB_DATASET),
    )
    data.add_argument(
        "--download-to",
        default=None,
        metavar="DIR",
        help="Where --download writes. Default: beside this script, which is "
        "where --data and --eval look first.",
    )
    data.add_argument(
        "--revision",
        default=None,
        help="Hub revision (branch, tag or commit) for --download.",
    )
    data.add_argument(
        "--overwrite",
        action="store_true",
        help="Let --download replace corpus files that already exist.",
    )
    data.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help="Base checkpoint: a local directory, or a name laya resolves and "
        "downloads from the Hub. NOTE: backend/models/laya-ml-checkpoint has "
        "no model.safetensors and cannot be trained from.",
    )
    data.add_argument(
        "--out",
        default=None,
        help="Output directory for the fine-tuned checkpoint. "
        "Default: ./%s beside this script." % DEFAULT_OUT,
    )

    gate = parser.add_argument_group("acceptance gate")
    gate.add_argument(
        "--dry-run",
        action="store_true",
        help="Build every training item WITHOUT loading model weights, print the "
        "stats, and stop. This is TRD section 4.4's acceptance gate.",
    )
    gate.add_argument(
        "--skip-dry-run",
        action="store_true",
        help="Do not run the dry-run check before training.",
    )
    gate.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Train even though the dry run dropped items. A wrong label literal "
        "does not raise -- laya silently drops the item -- so by default any "
        "non-empty 'skipped' aborts before the GPU is touched.",
    )

    hp = parser.add_argument_group("hyperparameters")
    hp.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="laya's own default is 4; 6 is this project's default because a ~10k-item "
        "corpus is small enough for the head to collapse to the class prior.",
    )
    hp.add_argument(
        "--shuffle-options",
        choices=["on", "off"],
        default="on",
        help="Re-encode choice questions with a random option order each epoch, so "
        "the model cannot learn a position prior. laya's default is off; on "
        "here because the measured failure is a 6-option ECOG head.",
    )
    hp.add_argument("--micro-batch", type=int, default=8, help="Rows per forward pass.")
    hp.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Accumulation steps; micro-batch x this is the effective batch.",
    )
    hp.add_argument("--encoder-lr", type=float, default=2.5e-5)
    hp.add_argument("--head-lr", type=float, default=1e-4)
    hp.add_argument("--loss", choices=["soft-ce", "rlcd"], default="rlcd")
    hp.add_argument("--label-smoothing", type=float, default=0.0)
    hp.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train the heads only. Fits in far less memory, but leaves ECOG "
        "inference -- an encoder-level task -- untouched (TRD R10).",
    )
    hp.add_argument("--seed", type=int, default=0)

    budget = parser.add_argument_group("token budgets and precision")
    budget.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="State token budget. Defaults to the checkpoint's (1024). Narratives "
        "run ~160-400 state tokens, so --max-len 512 roughly halves compute "
        "-- but whatever is used is written into the saved config, so "
        "inference must then use the same value.",
    )
    budget.add_argument(
        "--head-max-len",
        type=int,
        default=None,
        help="Question-head token budget. Defaults to the checkpoint's (256).",
    )
    budget.add_argument(
        "--amp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Mixed precision. 'auto' follows the device: on for CUDA, off "
        "otherwise. Try 'off' first if the loss goes NaN.",
    )
    budget.add_argument(
        "--gradient-checkpointing",
        choices=["auto", "on", "off"],
        default="auto",
        help="Trade compute for VRAM in the encoder and head. 'auto' follows "
        "whatever AMP resolved to.",
    )

    calib = parser.add_argument_group("calibration and abstention")
    calib.add_argument(
        "--target-error",
        type=float,
        default=0.10,
        help="Tolerated error among accepted answers when fitting the per-bucket "
        "abstention thresholds.",
    )
    calib.add_argument(
        "--min-abstain-n",
        type=int,
        default=10,
        help="Minimum calibration items in a bucket before a threshold is fitted.",
    )

    progress = parser.add_argument_group("progress reporting")
    progress.add_argument(
        "--progress",
        choices=["auto", "off"],
        default="auto",
        help="Replace laya's per-step log lines with a tqdm bar (epoch, ETA, "
        "raw and smoothed loss). 'auto' falls back to the plain lines when "
        "tqdm is not installed. The lines still go to train.log either way.",
    )
    progress.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Micro-batches between laya's loss lines, which is also how often "
        "the bar moves. Default: 10 with the bar, laya's 100 without.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="'cuda', 'cuda:N', 'cpu', 'mps', or 'auto'. Defaults to cuda and "
        "fails fast when no GPU is visible.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    _disable_broken_truststore()

    if args.download:
        download_corpus(
            args.download,
            dest=args.download_to,
            revision=args.revision,
            overwrite=args.overwrite,
        )
        print("\nNext: python train_laya.py --dry-run", flush=True)
        return 0

    data_path = _find_input(args.data, TRAIN_FILE, "training")
    eval_path = None if args.no_eval else _find_input(args.eval_data, EVAL_FILE, "eval")
    out_dir = Path(args.out).expanduser() if args.out else SCRIPT_DIR / DEFAULT_OUT

    try:
        import torch
        from laya.train import TrainConfig, dry_run, finetune
    except ImportError as err:
        sys.exit(
            "train_laya: missing dependency (%s). Install requirements.txt first." % err
        )

    if not args.dry_run:
        _require_device(args.device)
        out_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        loss=args.loss,
        label_smoothing=args.label_smoothing,
        shuffle_options=("choice",) if args.shuffle_options == "on" else (),
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        amp=_tristate(args.amp),
        gradient_checkpointing=_tristate(args.gradient_checkpointing),
        eval_data=eval_path,
        target_error=args.target_error,
        calib_max=args.calib_max,
        min_abstain_n=args.min_abstain_n,
        # A bar that only moves every 100 micro-batches looks stuck.
        log_every=args.log_every
        if args.log_every is not None
        else (10 if args.progress != "off" else 100),
    )
    config.validate()

    if args.dry_run:
        return _run_dry(dry_run, data_path, args.base, config, gate=False)[0]

    log_path = out_dir / "train.log"
    handle = log_path.open("a", encoding="utf-8")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(real_stdout, handle), _Tee(real_stderr, handle)
    warnings.simplefilter("always", RuntimeWarning)

    try:
        print("=" * 72, flush=True)
        print("laya fine-tune  %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        print("  torch        %s" % torch.__version__, flush=True)
        print("  train data   %s" % data_path, flush=True)
        print("  eval data    %s" % (eval_path or "(none)"), flush=True)
        print("  base         %s" % args.base, flush=True)
        print("  out          %s" % out_dir, flush=True)
        print(
            "  epochs %d, micro-batch %d, grad-accum %d (effective %d), loss %s, "
            "shuffle-options %s"
            % (
                config.epochs,
                config.micro_batch,
                config.grad_accum,
                config.micro_batch * config.grad_accum,
                config.loss,
                config.shuffle_options or "off",
            ),
            flush=True,
        )
        print("=" * 72, flush=True)

        steps_per_epoch = None
        if not args.skip_dry_run:
            print("\n--- acceptance gate (no weights loaded) ---", flush=True)
            code, dry_summary = _run_dry(
                dry_run, data_path, args.base, config, gate=not args.allow_skipped
            )
            if code != 0:
                return code
            train_items = dry_summary.get("train_items")
            if train_items:  # ceil: laya's last micro-batch is a short one
                steps_per_epoch = -(-train_items // config.micro_batch)
            print("--- gate passed ---\n", flush=True)

        restore_print = None
        if args.progress != "off":
            # The bar goes to the real stderr, never through the _Tee: a
            # redrawn bar would otherwise fill train.log with \r frames.
            restore_print = _install_progress_bar(
                config.epochs, steps_per_epoch, real_stderr, handle
            )

        try:
            summary = finetune(
                data=data_path,
                model_dir=args.base,
                output_dir=str(out_dir),
                config=config,
                device=args.device,
            )
        except (ValueError, FileNotFoundError, TypeError) as err:
            print(
                "\ntrain_laya: training failed: %s" % err, file=sys.stderr, flush=True
            )
            if "model.safetensors" in str(err) or "Checkpoint directory" in str(err):
                print(
                    "  The base checkpoint has no PyTorch weights. backend/models/"
                    "laya-ml-checkpoint\n  holds only configs and the tokenizer -- the weights in "
                    "this project live only\n  in the ONNX graph. Use --base multilingual to "
                    "download the real checkpoint.",
                    file=sys.stderr,
                    flush=True,
                )
            return 1
        finally:
            if restore_print is not None:
                restore_print()

        elapsed = time.time() - started
        _report(summary, out_dir, elapsed)
        _write_run_summary(summary, out_dir, args, config, elapsed, torch)
        return 0
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        handle.close()
        print("log written to %s" % log_path, flush=True)


def _run_dry(
    dry_run, data_path: str, base: str, config, gate: bool
) -> tuple[int, dict[str, Any]]:
    """Run laya's dry run; when `gate`, refuse to continue if it dropped any item.

    Returns the exit code and the summary, whose train_items is what gives the
    progress bar an exact total.
    """
    try:
        summary = dry_run(data=data_path, model_dir=base, config=config)
    except (ValueError, FileNotFoundError, TypeError) as err:
        print("train_laya: dry run failed: %s" % err, file=sys.stderr, flush=True)
        return 1, {}

    skipped = summary.get("skipped") or {}
    eval_skipped = summary.get("eval_skipped") or {}
    if not (skipped or eval_skipped):
        print(
            "dry-run: skipped {} -- every labelled question became a training item.",
            flush=True,
        )
    if gate and (skipped or eval_skipped):
        print(
            "\ntrain_laya: refusing to train. The dry run dropped items:\n"
            "  train: %r\n  eval:  %r\n"
            "A wrong label literal does not raise -- laya counts it under 'invalid_target'\n"
            "and drops it silently -- so this is the only thing between a format mistake\n"
            "and a wasted multi-hour run. Fix the corpus, or pass --allow-skipped."
            % (skipped, eval_skipped),
            file=sys.stderr,
            flush=True,
        )
        return 1, summary
    return 0, summary


def _report(summary: dict[str, Any], out_dir: Path, elapsed: float) -> None:
    report = summary.get("train_report") or {}

    print("\n=== Run ===", flush=True)
    print(
        "wall clock        %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        flush=True,
    )
    print("train items       %d" % summary.get("train_items", 0), flush=True)
    print("calibration items %d" % summary.get("calibration_items", 0), flush=True)
    print(
        "eval items        %d (%s)"
        % (summary.get("eval_items", 0), report.get("eval_source") or "none"),
        flush=True,
    )
    print("skipped           %r" % (summary.get("skipped") or {},), flush=True)
    print(
        "epoch mean loss   %s"
        % [round(float(v), 4) for v in summary.get("epoch_loss") or []],
        flush=True,
    )

    mode = report.get("eval_mode")
    if mode and mode != "held_out":
        print(
            "\nWARNING: eval_mode is %r, not 'held_out'.\n  %s"
            % (mode, report.get("note")),
            flush=True,
        )

    comparison = report.get("comparison")
    if comparison:
        print("\n=== Overall deltas (after - before) ===", flush=True)
        for key in (
            "delta_accuracy",
            "delta_loss",
            "delta_ece",
            "delta_brier",
            "delta_brier_top1",
            "delta_mean_confidence",
        ):
            value = comparison.get(key)
            print(
                "%-24s %s" % (key, ("%+.4f" % value) if value is not None else "N/A"),
                flush=True,
            )
        if (comparison.get("delta_accuracy") or 0) <= 0:
            print(
                "\nWARNING: accuracy did not improve. Check the collapse warning above, and "
                "consider\n  more epochs, more corpus, or another --seed before trusting this "
                "checkpoint.",
                flush=True,
            )

    _print_by_type(report)
    _print_env_block(summary, out_dir)

    print("\n=== Baseline to beat ===", flush=True)
    print(BASELINE_NOTE, flush=True)
    print(
        "\nRe-measure these against the fine-tuned graph before wiring the catalogue into\n"
        "intake (TRD stage 4c). Eval-split accuracy is an upper bound: the corpus is\n"
        "entirely synthetic and the model inherits its blind spots (TRD R9).",
        flush=True,
    )

    print("\n=== Artifacts in %s ===" % out_dir, flush=True)
    for name in (
        "model.safetensors",
        "rl_agent_config.json",
        "questions.json",
        "train_report.json",
        "run_summary.json",
        "train.log",
    ):
        path = out_dir / name
        if path.exists():
            print(
                "  %-24s %9.1f MB" % (name, path.stat().st_size / 1024**2), flush=True
            )
    for name in ("encoder", "tokenizer", "checkpoint_latest"):
        if (out_dir / name).is_dir():
            print("  %-24s (directory)" % name, flush=True)
    print(
        "\ncheckpoint_latest/ is a full checkpoint saved after every epoch. If the\n"
        "session dropped mid-run you can continue with --base <out>/checkpoint_latest,\n"
        "though that restarts the LR schedule and re-draws the calibration split.",
        flush=True,
    )
    print("\nNext: python export_onnx.py --checkpoint %s" % out_dir, flush=True)


def _write_run_summary(
    summary: dict[str, Any], out_dir: Path, args, config, elapsed: float, torch
) -> None:
    """Everything needed to reconstruct the run, beside the checkpoint it produced."""
    from dataclasses import asdict

    try:
        import laya

        laya_version = getattr(laya, "__version__", "unknown")
    except Exception:
        laya_version = "unknown"

    payload = {
        "argv": sys.argv,
        "elapsed_seconds": round(elapsed, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "laya_version": laya_version,
        "device": args.device,
        "gpu": _describe_gpu() if torch.cuda.is_available() else None,
        "base": args.base,
        "config": asdict(config),
        "result": {k: v for k, v in summary.items() if k != "train_report"},
        "train_report": summary.get("train_report"),
    }
    path = out_dir / "run_summary.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())

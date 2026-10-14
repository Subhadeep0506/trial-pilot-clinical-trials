#!/usr/bin/env python
"""Export a fine-tuned Laya checkpoint to ONNX, in the layout this project loads.

The goal is a graph byte-compatible in shape with ``onnx-community/laya-multilingual-ONNX``,
which is the only prebuilt Laya graph that works here. Reference layout, read off
``backend/models/laya-multilingual-onnx/model.onnx``:

    ir_version 10 | opset [('', 18)] | producer pytorch
    inputs : input_ids      INT64 [batch, seq]
             attention_mask INT64 [batch, seq]
             marker_pos     INT64 [batch, markers]
             marker_mask    BOOL  [batch, markers]
             qtype          INT64 [batch]
    outputs: logits         FLOAT [batch, markers]
             act_logits     FLOAT [batch, 2]
    weights: fp32, external, in a sibling file named model.onnx_data

Three things about that layout are load-bearing, and each has already cost this
project a dead end:

- ``marker_pos`` dim 1 **must stay dynamic**. ``ONNXAgent`` feeds a variable-width
  ``marker_pos``, and ``DecisionLMService._check_graph`` refuses a graph with a fixed
  option axis outright. ``Mattepiu/laya-onnx`` is pinned to ``[batch, 2]`` and is
  unusable for exactly this reason.
- The outputs **must be named** ``logits`` and ``act_logits``: ``ONNXAgent`` calls
  ``session.run(["logits", "act_logits"], ...)`` by name.
- The weights file **must be named** ``model.onnx_data``.

A further trap lives in the module being exported. ``DecisionModel.forward`` contains
``if p.size(-1) >= 2: top2 = p.topk(2, -1)`` -- a Python branch the tracer bakes in.
Exporting with a single-option example would freeze the one-option path and ONNX
Runtime would then refuse every real call. The example batch built below therefore
always has at least two markers.

    python export_onnx.py                            # exports ./laya-intake-v1
    python export_onnx.py --checkpoint ./my-ckpt --out-dir ./my-onnx
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent.parent
SYNTHETIC_DIR = BACKEND_DIR / "data" / "synthetic"

DEFAULT_CHECKPOINT = "laya-intake-v1"
WEIGHTS_FILE = "model.onnx_data"
GRAPH_FILE = "model.onnx"

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]

DYNAMIC_AXES = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "marker_pos": {0: "batch", 1: "markers"},
    "marker_mask": {0: "batch", 1: "markers"},
    "qtype": {0: "batch"},
    "logits": {0: "batch", 1: "markers"},
    "act_logits": {0: "batch"},
}

# Two states of different lengths, so the example batches differ in sequence length
# as well as in marker count. Nothing is learned from them; they only shape the trace.
STATES = [
    "63-year-old woman, stage IV lung adenocarcinoma. ECOG 1 -- up and about, does "
    "light housework, but tires on the stairs. Three prior lines: carboplatin/pemetrexed, "
    "docetaxel, then osimertinib. Quit smoking about fifteen years ago. Postmenopausal.",
    "My dad has lymphoma and we are looking for options.",
]

# Fallback questions, used only when the checkpoint has no questions.json and no corpus
# is on disk. Deliberately mixed: a 6-option choice, a 2-option noul and a 5-level score,
# so the exported trace sees the widest head the catalogue produces.
FALLBACK_QUESTIONS = {
    "performance_status": {
        "type": "choice",
        "instructions": "What is the patient's ECOG performance status?",
        "criteria": {
            "ecog_0": "fully active",
            "ecog_1": "restricted in strenuous activity but ambulatory",
            "ecog_2": "ambulatory and self-caring but unable to work",
            "ecog_3": "capable of only limited self-care",
            "ecog_4": "completely disabled",
            "unknown": "performance status is not stated or implied",
        },
    },
    "sex_stated": {
        "type": "noul",
        "instructions": "Is the patient's sex explicitly stated in the text?",
    },
    "prior_lines": {
        "type": "score",
        "instructions": "How many prior lines of systemic therapy has the patient received?",
        "criteria": ["0", "1", "2", "3", "4+"],
    },
}


# ----------------------------------------------------------------------- bootstrap

def _disable_broken_truststore() -> None:
    """Undo truststore's SSL injection on Python 3.14, where it recurses forever.

    Irrelevant on Colab, and essential on the TrialMatch dev box, where
    `pip_system_certs.pth` injects truststore into every process and its `wrap_socket`
    then recurses until the C stack is exhausted. Any Hub call -- loading a tokenizer
    that is not fully cached, for instance -- otherwise dies with a `RecursionError`
    deep inside `ssl.py` that points nowhere near the cause. Mirrors the copy in
    train_laya.py; duplicated because these scripts are uploaded on their own.
    """
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
            print("note: removed truststore's SSL injection (%s); certificates now come "
                  "from certifi." % name, flush=True)
            return


# ------------------------------------------------------------------- example batches

def _load_questions(checkpoint: Path) -> dict[str, Any]:
    """The real question schema, so the traced head matches what inference will send.

    ``finetune`` writes ``questions.json`` beside the checkpoint; failing that, take the
    schema off the first row of the training corpus; failing that, use the catalogue
    subset above.
    """
    path = checkpoint / "questions.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            questions = json.load(f)
        if questions:
            print("questions: %d from %s" % (len(questions), path), flush=True)
            return questions

    corpus = SYNTHETIC_DIR / "intake_v1.train.jsonl"
    for candidate in (SCRIPT_DIR / "intake_v1.train.jsonl", Path.cwd() / "intake_v1.train.jsonl",
                      corpus):
        if candidate.exists():
            with candidate.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        questions = json.loads(line).get("questions") or {}
                        if questions:
                            print("questions: %d from %s (first row)"
                                  % (len(questions), candidate), flush=True)
                            return questions
                    break

    print("questions: using the built-in fallback schema", flush=True)
    return dict(FALLBACK_QUESTIONS)


def _option_count(question: dict[str, Any]) -> int:
    criteria = question.get("criteria")
    if isinstance(criteria, (dict, list)):
        return len(criteria)
    return 2  # noul is always two options


def _order_questions(questions: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every question, widest head first.

    Width matters twice over. The trace is built from the widest, because it must take
    the ``topk(2)`` branch and because a wide head exercises more of the marker gather
    than a binary one. The narrowest then gives the verification batch a *different*
    marker count, which is the only way to show the option axis is genuinely dynamic
    rather than merely named.
    """
    ordered = sorted(questions.items(), key=lambda item: _option_count(item[1]), reverse=True)
    if not ordered:
        ordered = sorted(FALLBACK_QUESTIONS.items(),
                         key=lambda item: _option_count(item[1]), reverse=True)
    if _option_count(ordered[0][1]) < 2:
        sys.exit("export_onnx: no question with 2 or more options; the traced graph would "
                 "bake in the single-option branch and ONNX Runtime would refuse every call.")
    return ordered


def _make_batch(tok, picks, states, max_len: int, head_max_len: int):
    """A collated batch, one row per (question, state) pair."""
    from laya.common import QTYPES, build_sequence, collate_items
    from laya.train import to_internal

    items = []
    for (qid, question), state in zip(picks, states):
        q = to_internal(qid, question)
        ids, markers = build_sequence(tok, state, q, max_len, head_max_len)
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], tok.pad_token_id)
    return (batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
            batch["marker_mask"], batch["qtype"])


# -------------------------------------------------------------------------- export

def _export(model, example, path: Path, opset: int, use_dynamo: bool) -> None:
    import torch

    if use_dynamo:
        # Best-effort alternative exporter. The legacy tracer is the default because
        # laya's _DynamicMultiheadAttention was written so that tracing does not bake
        # in the sequence length, and because it reproduces the reference axis names.
        from torch.export import Dim

        batch, seq, markers = Dim("batch", min=1), Dim("seq", min=2), Dim("markers", min=2)
        torch.onnx.export(
            model, example, str(path),
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            opset_version=opset, dynamo=True, external_data=False,
            dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq},
                            {0: batch, 1: markers}, {0: batch, 1: markers}, {0: batch}),
        )
        return

    torch.onnx.export(
        model, example, str(path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        dynamic_axes=DYNAMIC_AXES,
        opset_version=opset, dynamo=False, do_constant_folding=True,
    )


def _save_with_external_weights(tmp_graph: Path, out_dir: Path) -> Path:
    """Re-save as model.onnx + model.onnx_data.

    torch picks its own sidecar name, and the project requires this one, so the graph is
    round-tripped through onnx. Holds the weights in memory briefly (~2.6 GB for a fp32
    322M-parameter model); that is fine on a Colab runtime and noted in the README.
    """
    import onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    graph = out_dir / GRAPH_FILE
    weights = out_dir / WEIGHTS_FILE
    for stale in (graph, weights):
        if stale.exists():
            stale.unlink()

    model = onnx.load(str(tmp_graph), load_external_data=True)
    onnx.save_model(
        model, str(graph),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=WEIGHTS_FILE, size_threshold=1024, convert_attribute=False,
    )
    if not weights.exists():
        sys.exit("export_onnx: %s was not written. Every weight landed inside the graph, "
                 "which diverges from the layout this project loads." % weights)
    return graph


# -------------------------------------------------------------------------- verify

def _shape(value_info) -> list[Any]:
    dims = value_info.type.tensor_type.shape.dim
    return [d.dim_param if d.dim_param else d.dim_value for d in dims]


def _check_contract(graph_path: Path) -> None:
    """The three things ONNXAgent and DecisionLMService require, each failing loudly."""
    import onnx

    model = onnx.load(str(graph_path), load_external_data=False)
    graph = model.graph
    inputs = {i.name: i for i in graph.input}
    outputs = {o.name: o for o in graph.output}

    print("\n=== Graph contract ===", flush=True)
    print("ir_version %d | opset %s | producer %s %s"
          % (model.ir_version, [(o.domain, o.version) for o in model.opset_import],
             model.producer_name, model.producer_version), flush=True)

    print("inputs:", flush=True)
    for name in INPUT_NAMES:
        if name not in inputs:
            sys.exit("export_onnx: exported graph has no input %r. ONNXAgent feeds exactly "
                     "%r." % (name, INPUT_NAMES))
        info = inputs[name]
        dtype = onnx.TensorProto.DataType.Name(info.type.tensor_type.elem_type)
        print("  %-16s %-6s %s" % (name, dtype, _shape(info)), flush=True)

    print("outputs:", flush=True)
    for name in OUTPUT_NAMES:
        if name not in outputs:
            sys.exit("export_onnx: exported graph has no output %r. ONNXAgent runs "
                     "session.run(%r, ...) by name." % (name, OUTPUT_NAMES))
        info = outputs[name]
        dtype = onnx.TensorProto.DataType.Name(info.type.tensor_type.elem_type)
        print("  %-16s %-6s %s" % (name, dtype, _shape(info)), flush=True)

    # The check DecisionLMService._check_graph performs at load time, applied here so the
    # failure surfaces at export rather than on the first multi-option question.
    marker_dims = inputs["marker_pos"].type.tensor_type.shape.dim
    if len(marker_dims) < 2 or not marker_dims[1].dim_param:
        sys.exit(
            "export_onnx: marker_pos dim 1 is fixed at %r, not a dynamic axis.\n"
            "  DecisionLMService._check_graph refuses such a graph: it cannot answer a\n"
            "  question with a different number of options than the one it was traced on."
            % (_shape(inputs["marker_pos"])[1:2] or None,)
        )
    print("\nmarker_pos dim 1 is dynamic (%r) -- the option axis is not pinned."
          % marker_dims[1].dim_param, flush=True)


def _check_numerics(model, graph_path: Path, batches: list[tuple], atol: float) -> None:
    """Torch and ONNX Runtime must agree, on two batches of different shapes.

    One batch proves the graph runs. Two batches with different sequence lengths and
    different marker counts prove the dynamic axes are real rather than named.
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    print("\n=== Numeric agreement (torch vs ONNX Runtime) ===", flush=True)
    worst = 0.0
    for index, example in enumerate(batches):
        with torch.no_grad():
            torch_logits, torch_act = model(*example)

        feeds = {
            "input_ids": example[0].numpy().astype(np.int64),
            "attention_mask": example[1].numpy().astype(np.int64),
            "marker_pos": example[2].numpy().astype(np.int64),
            "marker_mask": example[3].numpy().astype(bool),
            "qtype": example[4].numpy().astype(np.int64),
        }
        onnx_logits, onnx_act = session.run(OUTPUT_NAMES, feeds)

        d_logits = float(np.max(np.abs(torch_logits.numpy() - onnx_logits)))
        d_act = float(np.max(np.abs(torch_act.numpy() - onnx_act)))
        worst = max(worst, d_logits, d_act)
        print("batch %d  shape %s markers %d  max|dlogits| %.3e  max|dact| %.3e"
              % (index, tuple(feeds["input_ids"].shape), feeds["marker_pos"].shape[1],
                 d_logits, d_act), flush=True)

        if d_logits > atol or d_act > atol:
            sys.exit("export_onnx: ONNX output disagrees with torch by more than %g on batch "
                     "%d. The exported graph is not the model." % (atol, index))

    print("agreement within %g (worst %.3e) on both shapes." % (atol, worst), flush=True)


# ---------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_onnx.py",
        description="Export a fine-tuned Laya checkpoint to the ONNX layout this project loads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Fine-tuned checkpoint directory. Default: ./%s beside this script."
                             % DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", default=None,
                        help="Where to write %s and %s. Default: <checkpoint>-onnx."
                             % (GRAPH_FILE, WEIGHTS_FILE))
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset. 18 is what the reference graph uses.")
    parser.add_argument("--dynamo", dest="dynamo", action="store_true", default=False,
                        help="Use torch's dynamo exporter instead of the legacy tracer. "
                             "Best-effort fallback only.")
    parser.add_argument("--verify", choices=["on", "off"], default="on",
                        help="Run the graph-contract and torch-vs-ONNX numeric checks after "
                             "exporting. Turning this off is not recommended: the checks are "
                             "what catch a pinned option axis or a mistraced branch.")
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Tolerance for the torch-vs-ONNX comparison.")
    parser.add_argument("--max-len", type=int, default=None,
                        help="Override the state token budget used to build the example batch. "
                             "Defaults to the checkpoint's own value.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _disable_broken_truststore()

    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint \
        else SCRIPT_DIR / DEFAULT_CHECKPOINT
    if not checkpoint.is_dir():
        sys.exit("export_onnx: checkpoint directory not found: %s\n"
                 "  Run train_laya.py first, or pass --checkpoint." % checkpoint)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir \
        else checkpoint.parent / (checkpoint.name + "-onnx")

    try:
        import torch
        from laya.common import uses_parallel_layout
        from laya.train import load_checkpoint
    except ImportError as err:
        sys.exit("export_onnx: missing dependency (%s). Install requirements.txt first." % err)

    print("=" * 72, flush=True)
    print("laya ONNX export", flush=True)
    print("  torch       %s" % torch.__version__, flush=True)
    print("  checkpoint  %s" % checkpoint, flush=True)
    print("  out dir     %s" % out_dir, flush=True)
    print("  opset       %d, exporter %s"
          % (args.opset, "dynamo" if args.dynamo else "legacy tracer"), flush=True)
    print("=" * 72, flush=True)

    try:
        model, tok, cfg = load_checkpoint(str(checkpoint))
    except (FileNotFoundError, ValueError) as err:
        sys.exit("export_onnx: could not load the checkpoint: %s" % err)

    if uses_parallel_layout(cfg):
        # The parallel layout needs position_ids and option_ids as well, and ONNXAgent
        # feeds neither -- such a graph could not be served by this project.
        sys.exit("export_onnx: this checkpoint uses the 'parallel' option layout, which needs "
                 "two extra inputs ONNXAgent does not feed. Re-train without --option-layout "
                 "parallel.")

    # Export on CPU in fp32. save_checkpoint stores fp16; load_checkpoint already called
    # .float(), which is what the reference graph's 1.29 GB of fp32 weights corresponds to.
    model = model.float().cpu().eval()
    model.head_checkpointing = False

    max_len = args.max_len or cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    print("budgets: max_len %d, head_max_len %d (from the checkpoint config)"
          % (max_len, head_max_len), flush=True)

    questions = _load_questions(checkpoint)
    ordered = _order_questions(questions)

    # Trace on the widest heads; verify on the narrowest, with a shorter state, so the
    # two batches differ in batch size, sequence length AND marker count.
    trace_picks = ordered[:2] if len(ordered) > 1 else ordered * 2
    verify_picks = ordered[-1:]
    print("trace rows:  %s" % ", ".join("%s (%d options)" % (qid, _option_count(q))
                                        for qid, q in trace_picks), flush=True)
    print("verify row:  %s" % ", ".join("%s (%d options)" % (qid, _option_count(q))
                                        for qid, q in verify_picks), flush=True)

    trace_batch = _make_batch(tok, trace_picks, STATES, max_len, head_max_len)
    verify_batch = _make_batch(tok, verify_picks, STATES[1:], max_len, head_max_len)

    print("trace batch:  input_ids %s, marker_pos %s"
          % (tuple(trace_batch[0].shape), tuple(trace_batch[2].shape)), flush=True)
    print("verify batch: input_ids %s, marker_pos %s"
          % (tuple(verify_batch[0].shape), tuple(verify_batch[2].shape)), flush=True)
    if trace_batch[2].shape[1] < 2:
        sys.exit("export_onnx: the trace batch has fewer than 2 markers; the exported graph "
                 "would bake in the single-option branch.")
    if verify_batch[2].shape[1] == trace_batch[2].shape[1]:
        print("NOTE: both batches have the same marker count, so the numeric check cannot "
              "prove\n      the option axis is dynamic. The graph contract check below still "
              "does.", flush=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="laya-onnx-"))
    try:
        tmp_graph = tmp_dir / GRAPH_FILE
        print("\nexporting...", flush=True)
        _export(model, trace_batch, tmp_graph, args.opset, args.dynamo)
        graph = _save_with_external_weights(tmp_graph, out_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    weights = out_dir / WEIGHTS_FILE
    print("\nwrote %s (%.1f MB) and %s (%.1f MB)"
          % (graph, graph.stat().st_size / 1024 ** 2,
             weights, weights.stat().st_size / 1024 ** 2), flush=True)

    if args.verify == "on":
        _check_contract(graph)
        _check_numerics(model, graph, [trace_batch, verify_batch], args.atol)

    print("\n=== Paste into backend/.env ===", flush=True)
    print('DECISION_CHECKPOINT_ID="./models/%s"' % checkpoint.name, flush=True)
    print('DECISION_ONNX_PATH="./models/%s/model.onnx"' % out_dir.name, flush=True)
    print(
        "\nCopy BOTH directories back: the checkpoint supplies the tokenizer, the config and\n"
        "the fitted temperatures, and %s must keep its name beside the graph."
        % WEIGHTS_FILE,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Export laya's DecisionModel (encoder + decision head) to a fixed-shape ONNX graph.

Why fixed shapes: torch.onnx.export bakes traced Reshape constants for
ModernBERT's attention views even with dynamic_axes declared — ORT then rejects
the graph for any other length (observed: Reshape {129,48,64} vs {512,3,1024}).
Fixed B/L/K removes all shape ambiguity; padding is inert because attention-mask
zeros + -1e4 marker fills hide padded positions. Dummy rows keep one real token
(cls) so a fully-masked row can't produce NaNs.

Inputs : input_ids i64 [6,512], attention_mask i64 [6,512], marker_pos i64 [6,64],
         marker_mask bool [6,64], qtype i64 [6]
Outputs: choice_logits f32 [6,64], act_logits f32 [6,n_act]

Also writes cli/model/meta.json (everything the Rust CLI needs) and validates
the graph against the torch forward on representative padded inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from laya import Agent
from laya.common import (
    TEMP_MAX,
    TEMP_MIN,
    QTYPES,
    build_sequence,
    render_options,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "cli" / "model" / "laya-en"
OUT = ROOT / "cli" / "model"
ONNX_PATH = OUT / "laya-en.onnx"

FIXED_L = 512
FIXED_K = 64
FIXED_B = 6

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Which team should handle the email in `body`?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "sales": "pricing, demos, new purchases",
            "security": "phishing, scams, account compromise",
            "hr": "hiring, leave, payroll",
            "other": "none of the above",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the request in `body`?",
        "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"],
    },
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
        "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"},
    },
}

SHORT_STATE = {
    "subject": "Invoice #4471 is attached",
    "body": "Hi team, please find attached the invoice for our March order. "
    "The total is $4,120.00 and it looks like the PO number is missing. "
    "Can you confirm the due date and send a corrected copy with our PO number PO-9931? Thanks, Priya",
    "from": "priya@example.com",
}

LONG_STATE = (
    "Subject: ongoing outage escalation\n\n"
    "Our production cluster has been failing intermittently since the deploy on Monday. "
    + ("Customers are seeing 500 errors on checkout, support tickets are piling up, and we lose "
       "roughly $2k per hour while the payments API rejects valid cards. ") * 12
    + "Please treat this as the highest priority."
)


def build_item(agent: Agent, state, qdef) -> dict:
    q = Agent._to_internal(qdef)
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
    if len(markers) != len(render_options(q)):
        raise ValueError("marker/option mismatch")
    if len(markers) > FIXED_K:
        raise ValueError(f"question has {len(markers)} options > fixed K={FIXED_K}")
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}


def pad_batch(items: list[dict], pad_id: int, cls_id: int) -> dict:
    """Pad items (+ inert dummy rows) to the fixed (B, L) / (B, K) shape."""
    n = FIXED_B
    ids = torch.full((n, FIXED_L), pad_id, dtype=torch.long)
    att = torch.zeros((n, FIXED_L), dtype=torch.long)
    mpos = torch.zeros((n, FIXED_K), dtype=torch.long)
    mmask = torch.zeros((n, FIXED_K), dtype=torch.bool)
    qtype = torch.zeros(n, dtype=torch.long)
    for i, it in enumerate(items):
        if i >= n:
            break
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        mpos[i, : len(it["markers"])] = torch.tensor(it["markers"])
        mmask[i, : len(it["markers"])] = True
        qtype[i] = it["qtype"]
    # dummy rows: one real token so attention rows stay finite
    for i in range(len(items), n):
        ids[i, 0] = cls_id
        att[i, 0] = 1
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": qtype,
    }


def do_export(model, example_args, input_names) -> str:
    last = None
    for name, kw in [
        ("legacy", {"dynamo": False}),
        ("dynamo", {"dynamo": True}),
    ]:
        try:
            print(f"exporting via {name} exporter ...")
            # inference_mode: without it the autograd graph retains per-layer
            # activations for the B=6 trace and the process gets OOM-killed.
            with torch.inference_mode():
                torch.onnx.export(
                    model,
                    example_args,
                    str(ONNX_PATH),
                    input_names=input_names,
                    output_names=["choice_logits", "act_logits"],
                    **kw,
                )
            return name
        except Exception as e:  # noqa: BLE001 - try next exporter
            print(f"  {name} exporter failed: {type(e).__name__}: {e}")
            last = e
    raise RuntimeError("all export paths failed") from last


def main() -> None:
    agent = Agent(str(MODEL_DIR), device="cpu")
    model = agent.model
    model.eval()
    tok = agent.tok

    items = [build_item(agent, SHORT_STATE, qdef) for qdef in QUESTIONS.values()]
    if len(items) > FIXED_B:
        raise ValueError(f"{len(items)} example questions exceed FIXED_B={FIXED_B}")

    meta = {
        "model_file": ONNX_PATH.name,
        "checkpoint": "convaiinnovations/laya (root, english)",
        "fixed_l": FIXED_L,
        "fixed_k": FIXED_K,
        "fixed_b": FIXED_B,
        "max_len": agent.cfg.get("max_len", 512),
        "head_max_len": agent.cfg.get("head_max_len", 192),
        "pad_id": tok.pad_token_id,
        "cls_id": tok.cls_token_id,
        "sep_id": tok.sep_token_id,
        "mask_id": tok.mask_token_id,
        "mask_token": tok.mask_token,
        "temperature_raw": agent.temperature_raw,
        "temperature_by_options_raw": agent.temperature_by_options_raw,
        "temperature": agent.temperature,
        "temperature_by_options": agent.temperature_by_options,
        "temp_min": TEMP_MIN,
        "temp_max": TEMP_MAX,
        "qtypes": QTYPES,
        "act_costs": agent.cfg.get("act_costs", {}),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT / 'meta.json'}")

    input_names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    example_args = tuple(pad_batch(items, tok.pad_token_id, tok.cls_token_id).values())
    exporter = do_export(model, example_args, input_names)
    print(f"exported {ONNX_PATH} via {exporter} exporter (fixed {FIXED_B}x{FIXED_L}x{FIXED_K})")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])

    def npy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    worst_l, worst_a = 0.0, 0.0
    with torch.no_grad():
        for label, state in [("short", SHORT_STATE), ("long", LONG_STATE)]:
            for qid, qdef in QUESTIONS.items():
                it = build_item(agent, state, qdef)
                p = pad_batch([it], tok.pad_token_id, tok.cls_token_id)
                tl, ta = model(**p)
                outs = sess.run(["choice_logits", "act_logits"], {k: npy(v) for k, v in p.items()})
                dl = float(np.abs(outs[0] - npy(tl)).max())
                da = float(np.abs(outs[1] - npy(ta)).max())
                worst_l, worst_a = max(worst_l, dl), max(worst_a, da)
                print(f"[{label}/{qid}] max|dlogits|={dl:.2e} max|dact|={da:.2e}")
                # act_logits pass through a 256-wide MLP, so f32 kernel-order
                # noise is amplified; post-softmax that stays well under the
                # 1e-3 probability gate used by the golden validation.
                if dl > 1e-4 or da > 1e-2:
                    raise SystemExit(
                        f"FAIL: export drifted from torch (logits {dl:.2e} > 1e-4 or act {da:.2e} > 1e-2)"
                    )
    print(f"export validation OK (worst logits {worst_l:.2e}, act {worst_a:.2e})")


if __name__ == "__main__":
    main()
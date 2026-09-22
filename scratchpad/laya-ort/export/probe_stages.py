"""Probe laya's Python model stage-by-stage for the burn bisection."""

import json
import pathlib

import torch

from laya import Agent
from laya.common import build_sequence, collate_items, render_options

ROOT = pathlib.Path(__file__).resolve().parent.parent
agent = Agent(str(ROOT / "cli" / "model" / "laya-en"), device="cpu")
model = agent.model
model.eval()

state = {
    "subject": "Invoice #4471 is attached",
    "body": "Hi team, please find attached the invoice for our March order. The total is "
    "$4,120.00 and it looks like the PO number is missing. Can you confirm the due date? "
    "Thanks, Priya",
    "from": "priya@example.com",
}
questions = json.loads((ROOT / "cli" / "presets" / "email.json").read_text())

max_len = agent.cfg.get("max_len", 512)
head_max_len = agent.cfg.get("head_max_len", 192)
items = []
for qid, qdef in questions.items():
    q = Agent._to_internal(qdef)
    seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
    items.append({"ids": seq, "markers": markers, "qtype": {"choice": 0, "score": 1, "noul": 2}[q["t"]]})

b = collate_items([items], agent.tok.pad_token_id)
ids = b["input_ids"]
att = b["attention_mask"]
print("shapes: ids", tuple(ids.shape), "markers", tuple(b["marker_pos"].shape))
print("first ids:", ids[0][:16].tolist())
print("marker_pos:", b["marker_pos"][0].tolist(), "mask:", b["marker_mask"][0].tolist())


def stats(name, t):
    t = t.detach().float()
    print(f"{name}: mean={t.mean().item():.6f} std={t.std().item():.6f} absmax={t.abs().max().item():.4f}")


with torch.no_grad():
    stats("tok_embeddings", model.encoder.embeddings.tok_embeddings(ids))
    stats("embeddings.norm", model.encoder.embeddings.norm(model.encoder.embeddings.tok_embeddings(ids)))
    h = model.encoder.embeddings.norm(model.encoder.embeddings.tok_embeddings(ids))
    hooks = {}
    def mk(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            outs[name] = t.detach()
        return hook
    for i in (0, 1, 2, 27):
        model.encoder.layers[i].register_forward_hook(mk(f"after_layer_{i}"))
    outs = {}
    out = model.encoder(ids, attention_mask=att)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    for name, t in outs.items():
        stats(name, t)
    stats("final_norm", model.encoder.final_norm(h))
    h2 = model.encoder.final_norm(h)
    logits, act = model(ids, att, b["marker_pos"], b["marker_mask"], torch.zeros(ids.shape[0], dtype=torch.long))
    stats("logits", logits)
    print("logits[0][:8]:", [round(v, 4) for v in logits[0][:8].tolist()])
    print("act[0]:", act[0].tolist())
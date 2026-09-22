# Plan 2 (do first): laya via `ort` + CLI playground

Goal: play with laya from Rust on CPU — an email-classification CLI — with the
lowest-risk port path: **one ONNX graph + ~150 lines of Rust arithmetic**.
No codegen, no wgpu, no model code to hand-write. This also produces the golden
harness that Plan 1 (`burn-plan-1.md`) reuses for validation.

## Why this shape

`Agent.system_one` (`laya/agent.py:265-368`) is:

1. per question: `build_sequence` → token ids + marker positions (pure text munging),
2. one batched forward of `DecisionModel` (5 inputs → 2 outputs),
3. per row: temperature scale, softmax, entropy confidence, answer assembly.

Everything except (2) is trivial arithmetic. So export the **whole `DecisionModel`**
(encoder + head) as a single ONNX graph and port (1)/(3) to Rust.

- Inputs:  `input_ids` i64 `[B,L]`, `attention_mask` i64 `[B,L]`,
  `marker_pos` i64 `[B,K]`, `marker_mask` bool `[B,K]`, `qtype` i64 `[B]`
- Outputs: `choice_logits` f32 `[B,K]`, `act_logits` f32 `[B,2]`

**Export constraint: K ≥ 2.** `DecisionModel.forward` branches on `p.size(-1) >= 2`
(`common.py:122-132`); tracing picks one branch. All laya presets have ≥2 options
(noul is exactly 2), so the CLI enforces ≥2. Python itself raises for single-option
choice questions in `system_one` (`agent.py:287-288`), so this matches laya.

## Layout

```
scratchpad/laya-ort/
  export/
    export_onnx.py       # checkpoint -> model/laya-en.onnx + model/meta.json
    export_golden.py     # golden cases: inputs + expected laya outputs
    requirements.txt     # laya, torch, transformers, onnxruntime, safetensors, huggingface_hub
  cli/                   # cargo project "laya-cli"
    Cargo.toml           # ort, tokenizers, serde/serde_json, clap, anyhow
    src/main.rs          # subcommands
    src/seq.rs           # port of common.py:15-86 + collate (padding)
    src/post.rs          # temperature, softmax, confidence, answer assembly
    presets/*.json       # dumped from laya.presets
    golden/*.json
    model/               # laya-en.onnx, meta.json, tokenizer/ (gitignored)
```

## Phases

### 0. Environment

- `pip install laya onnxruntime torch transformers` (laya needs Python ≥3.10).
- Download once, read-only, matching laya's own allow-list (`agent.py:133-135`):
  `hf download convaiinnovations/laya --include "rl_agent_config.json" "model.safetensors" "tokenizer/*" "encoder/*" --local-dir scratchpad/laya-ort/cli/model/laya-en`

### 1. Export (`export_onnx.py`)

- `laya.load("<local dir>")`; export `agent.model` with
  `torch.onnx.export(..., opset_version=17, dynamic_axes={"B","L","K"})`,
  example inputs `B=1, L=512, K=2`. Model is already eager
  (`reference_compile=False` is set by laya at load).
- Also write `model/meta.json`:
  `max_len` / `head_max_len` from `rl_agent_config.json` (root english: 512 / 192),
  raw + clamped `temperature` and `temperature_by_options`, special token ids
  (cls/sep/mask/pad — ModernBERT-large: 50281/50282/50284/50283), qtype map
  `choice=0, score=1, noul=2`.
- Copy `tokenizer/tokenizer.json` next to the model for the `tokenizers` crate.
- **Validate the export in Python before any Rust**: run the golden cases through
  both `agent.predict` and Python `onnxruntime`; require max |Δp| ≤ 1e-4.
  This is the cheapest place to catch export drift.
- Documented fallback if the head's `topk`/`cat` branches fight the exporter:
  split into two graphs (`encoder.onnx` — the path burn-onnx's `get_model.py`
  already proves works — plus `head.onnx`) and chain them in Rust.

### 2. Golden data (`export_golden.py`)

~30 representative email states: plain billing/technical/phishing/security/HR mails,
empty string, >512-token mail (exercises truncation), dict/JSON state
(`serialize_state` path), text containing `<mask>` (must be replaced with " "),
unicode. Questions = `laya.presets.email_questions()` + `triage_questions()`.
Store inputs (raw state + question JSON) and expected outputs (probabilities,
confidence, act_probability) as `golden/*.json`. Dump `presets/*.json` here too.

### 3. CLI crate

- Deps: `ort` (CPU EP, `download-binaries`), `tokenizers`, `clap`, `serde_json`,
  `anyhow`.
- `src/seq.rs` — port of `common.py:15-86` + `collate_items` (`common.py:247-280`):
  `render_options`, `render_criterion`, `serialize_state`, `build_sequence`,
  batch padding (L = max len in batch, K = max markers, pad id 50283,
  attention 1s for real tokens only).
- `src/post.rs` — port of `agent.py:325-362`:
  - bucket key `"{choice|score|noul}:{2|3-5|6-10|11+}"` (`temp_bucket`, `common.py:219-221`)
  - temperature = `temperature_by_options[bucket]` else `temperature[qtype]`, clamped [0.5, 5.0]
  - softmax over the first `k` logits, `confidence = 1 − H(p)/log k`
  - `score` → expected value `Σ i·pᵢ`; `noul` → `p[1]` and `max(p[1], 1−p[1])`
  - `act_probability = softmax(act_logits)[0]`
- Subcommands:
  - `classify --state <file|-> [--questions presets/email.json]` — one-shot, prints
    per-question answer, probability bars, confidence, act_probability, latency.
  - `repl` — paste an email (EOF or blank line to run), all preset questions per
    iteration; `:questions <file>` swaps the set; `:q` quits.
  - `validate --golden golden/` — max |Δp| vs expected, per-case table, nonzero exit
    if any case exceeds tolerance.
  - `bench` — single-question and all-5-preset batch latency, p50/p95.

### 4. Acceptance

- `validate`: fp32 CPU, max |Δp| ≤ 1e-3 across all golden cases (top choice identical).
- `repl` round-trip: paste a fake invoice email → `category=billing`,
  urgency/scores sane vs `laya` Python output for the same text.
- `bench`: one question ≤ ~500 ms and batch-of-5 ≤ ~1 s on this laptop's CPU
  (PyTorch eager measured 193–464 ms/question for -large CPU; ORT has no
  ModernBERT-specific fusion, so expect similar or a bit worse — measure, don't assume).

### 5. Optional, only if needed

- `onnxruntime.quantization.quantize_dynamic` → ~430 MB int8 file; re-run
  `validate` (expect small drift; check top-choice agreement) and re-bench.
- GPU EPs later (`cuda` / `directml` features) — out of scope for the playground.

## Exact-match gotchas (each one silently shifts probabilities if missed)

- **JSON separators**: `render_criterion` uses `separators=(", ", ": ")` and
  `serialize_state` uses Python's default `json.dumps` separators (`", "` / `": "`).
  `serde_json::to_string` is compact (`","`/`":"`) — **not** equivalent after
  tokenization. `serde_json` can't configure separators → hand-roll a ~50-line
  writer matching Python's output, or restrict playground states/criteria to
  strings (presets' criteria are strings; dict states are the risk).
- **Dict ordering**: Python preserves insertion order; enable `serde_json`
  `preserve_order` or use the hand-rolled writer above.
- **Tokenization**: `str.replace("<mask>", " ")` before encoding and
  `add_special_tokens=false` everywhere (`build_sequence` adds cls/sep itself).
- **Option cap**: option text tokens capped at 48; `opt_budget`/`per` shrink logic at
  `common.py:70-75` must be ported as-is, and `head_max_len` read from config.
- **State truncation**: right-truncate (`st[:room]`) by default — `truncate_left`
  stays False.
- **mask fill value**: the graph applies `masked_fill(~marker_mask, -1e4)`; Rust only
  ever reads the first `k` logits per row.

## Non-goals

No engine integration (Plan 1 after its gate), no multilingual checkpoint, no GPU EP,
no training/fine-tuning. The golden JSONs and `seq.rs`/`post.rs` carry over into
Plan 1's validation harness unchanged.

## Results (2026-09-22) — shipped and validated

### What shipped

- `export/export_onnx.py` — one-command export of laya's `DecisionModel` (encoder +
  head) to a **fixed-shape ONNX graph (B=6, L=512, K=64)** plus `cli/model/meta.json`.
- `cli/` — Rust `laya-cli` (`ort` crate, `load-dynamic`): `classify`, `email`,
  `repl`, `validate`, `bench`. The sequence/collate/post math is a faithful port of
  `common.py` + `agent.py:265-368`.
- Golden harness: 42 cases × 206 answers dumped from laya 0.3.5's own `predict`
  (dict/list states, `<mask>` text, 512-token truncation, unicode, empty bodies,
  all five presets) + a `validate` subcommand that diffs Rust vs Python.
- **Final gate: 42/42 PASS, worst |Δp| = 1.0e-4** vs Python (tolerance 1e-3), on both
  the sequential-B=1 and batched-B=6 graphs. Invoice smoke: category=billing 0.9945,
  spam 0.001, phishing 0.03, urgency 1.33.

### Measured performance (i7-10750H, 6c/12t, f32)

| Configuration | 5-question call | per question |
|---|---|---|
| Python `agent.predict` (torch eager, SDPA, pads to ~130 tok) | **1.93 s** p50 | ~386 ms |
| Python, 1 question | 0.63 s | 0.63 s |
| Rust, legacy graph, B=1, fixed L=512 (5 sequential forwards) | 10.5 s | 2.1 s |
| Rust, dynamo graph, B=6, fixed L=512 (1 batched forward) | **9.2 s** p50 | 1.84 s |

Reading: batching bought only ~13% in ORT; the dominant cost of our port vs Python
is the **fixed L=512 padding** (~3x on short inputs) plus ORT having no
ModernBERT-specific fusion (sliding-window attention decomposed to generic ops).
The README's 193–464 ms/question claim matches our torch measurement. ORT CPU is
~4x slower than torch CPU for this model — Plan 1 runs on GPU via burn, where this
is moot; if CPU is ever needed, int8 dynamic quantization (2–4x, must re-run the
golden gate) and 256/512 length buckets are the levers.

### Export-path war story (directly relevant to the burn port)

1. **Dynamic axes do not survive ModernBERT export.** With `dynamic_axes`, torch
   bakes traced Reshape constants from the example batch (observed ORT failure:
   `Reshape input {512,3,1024} requested {129,48,64}`). Only fixed shapes work.
   → burn-onnx import must consume **fixed-shape graphs** (length buckets), as
   Plan 1 already planned. Now proven, not assumed.
2. **`inference_mode` is required to export B>1** — without it the autograd graph
   retains per-layer activations and the process is OOM-killed (8 GB available).
3. **But `inference_mode` kills the legacy (TorchScript) exporter**: eval-mode
   TransformerEncoderLayer then traces PyTorch's fused
   `aten::_transformer_encoder_layer_fwd`, which has **no ONNX export at any
   opset** (`UnsupportedOperatorError`, torch 2.14). The B=1 legacy export only
   ever worked because grad-enabled tracing kept the slow math path.
4. **The dynamo exporter (onnxscript) is the reliable path**: exports the whole
   model at native opset 18, and ORT 1.30 loads it (the earlier "Invalid
   attribute: num_outputs for Split" failure only happened when a dynamo graph
   was forced down to opset 17 — never downgrade the opset).
5. Export memory: weights (1.7 GB fp32) + ONNX proto peak ~6-8 GB RAM; the
   OOM-kill (exit 137) looked like a crash, not a memory error.

### Environment lessons (sandbox + TLS-intercepting proxy)

- All HTTPS is MITM'd by the org CA (in the system trust store). Tools using
  bundled TLS roots fail: **`ort`'s `download-binaries` cannot work here** — use
  `load-dynamic` + the `libonnxruntime.so` from the pypi `onnxruntime` wheel.
- `~/.cache` is unwritable in the agent sandbox → `HF_HOME`, `XDG_CACHE_HOME`,
  `UV_CACHE_DIR` redirected into the repo; HF downloads work via `SSL_CERT_FILE`
  + `HF_HUB_DISABLE_XET=1`.
- torch pinned `==2.14.0+cpu` requires uv `index-strategy = "unsafe-best-match"`
  (its default `first-index` refuses to cross indexes for a `+cpu` local version).

### Lessons for the burn port (Plan 1)

1. **Reuse this harness.** `cli/golden/cases.json` + the `validate` command are
   exporter-agnostic: point any new backend (burn) at the same inputs/expected
   outputs and require the same 1e-3 gate (observed worst case is the 4-decimal
   rounding boundary, 1.0e-4). `seq.rs`/`post.rs` (sequence building, JSON
   writer, post-processing) are backend-independent and drop into the burn port.
2. **Plan 1 must use fixed length buckets** (256/512/1024) for burn-onnx import —
   dynamic dims are proven broken for this architecture. Pad like `pad_batch`:
   attention zeros + `-1e4` marker fills + dummy rows with ≥1 real token.
3. **Batched-question API is the natural fit** (`B=6` covers every shipped
   preset; the K≥2 constraint from `common.py:122-132` stands). Dummy rows must
   keep one real token or fully-masked attention rows produce NaNs.
4. **Expect the burn wgpu backend to be the perf story**: ORT CPU is ~4x slower
   than torch CPU on this model; a GPU (even the GTX 1650-class) should beat both
   comfortably — but that is exactly the Phase 0 Vulkan gate that Plan 1 must
   measure before committing.
5. **The `K=64` marker ceiling is a playground convenience**, not a laya limit —
   if the game needs more options per question, raise `FIXED_K` and re-export.

## Burn port outcome (2026-09-22) — completed after this plan

Plan 1 was executed **without burn-onnx/ONNX**: a hand-written burn port
(`scratchpad/laya-burn/`) loads the **native safetensors directly** (answer to
"can burn load native safetensors": **yes**). Result: **42/42 golden PASS,
worst |Δp| = 1.0e-4, 39/42 exactly 0.0** — full parity with Python, no ONNX
involved. CPU perf: 6.0 s per 5-question batched call (ndarray backend) vs
this playground's 9.2 s (ort CPU) and torch's 1.9 s. Full findings, the
GeGLU-convention-swap root cause, verified-equal list, and 0.22-pre porting
notes are in `burn-plan-1.md` § "Results: native-safetensors burn port".
This playground stays as the ONNX reference path and its golden harness
(`cli/golden/`) is what the burn port validated against.


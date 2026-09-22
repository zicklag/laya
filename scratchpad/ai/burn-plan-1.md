# Plan 1: laya in the game engine via Burn (wgpu)

Status: **deferred** — we are playing first with Plan 2 (`ort` + CLI, see `ort-plan-1.md`).
This plan starts after that has proven the model's behavior in Rust and we're ready to
commit to in-engine integration.

## Why Burn (given the engine)

- Engine is wgpu-based. Burn's wgpu backend runs on the same graphics stack
  (Vulkan / DX12 / Metal / WebGPU) and the same dependency world — no second runtime.
- English-only ⇒ **one architecture to port**: ModernBERT-large (421M). The multilingual
  checkpoint (mmBERT-base, `sans_pos` variant) is out of scope.

## Verified facts (do not re-derive)

- Both checkpoints are `model_type: modernbert` (raw `encoder/config.json` on HF):
  - root + `typed-decisions`: 28 layers, hidden 1024, vocab 50368, seq 512/1024,
    sliding window 128, global attention every 3 layers, **separate RoPE thetas**
    (160000 full / 10000 sliding).
  - multilingual (excluded): 22 layers, hidden 768, vocab 256000,
    `position_embedding_type: sans_pos`.
- Laya's head is trivial and fully specified in `laya/common.py:89-136`
  (2 norm-first transformer layers, type embedding, scorer MLP, act head, marker gather).
- Weights ship as safetensors F16 (~843 MB for -large); **no ONNX shipped**.
- Burn ecosystem state (checked 2026-09):
  - `burn-store` loads PyTorch/safetensors into hand-written modules (name remap only).
  - `burn-onnx` has a **proven ModernBERT-base model check** (tracel-ai/burn#3814,
    merged 2025-10-01, opset 16, weights via `.bpk`):
    - tch backend: 134 ms @ seq 512, max |Δ| 0.007 vs PyTorch ✓
    - ndarray: 1.10 s ✗ (pure-CPU fallback is a non-starter)
    - Metal: **crashed** (wgpu shader-codegen bug in `reduce_kernel_f32_f32_f32`)
    - **no wgpu/Vulkan number exists anywhere** — this is the open risk.

## Phase 0 — retire the perf risk (~1 day, before any port work)

1. Build burn-onnx's existing `modernbert-base` model-check crate with the
   wgpu/Vulkan backend on a target-class GPU (this machine's GTX 1650 Mobile 4 GB
   is a good low-end proxy).
2. Export laya's `encoder/` to ONNX (burn-onnx's `get_model.py` does exactly this),
   codegen, run at seq 512 and 1024 on Vulkan.
3. **Decision gate**: Vulkan latency for -large ≈ base × 3.8 (FLOPs) must fit the
   frame budget. Escape hatches in order: F16 weights → int8 PTQ → custom CubeCL
   flash-attention kernel for the sliding-window layers.

## Phases 1–3: integration shape

1. **Encoder via burn-onnx import** (not hand-written): the model check proves the op
   graph converts and validates. Export at fixed length buckets (256/512/1024, pad to
   next bucket — option count per question varies; padding is cheap and the marker
   gather only touches real positions). If 28-layer codegen compile time hurts,
   fall back to hand-porting candle's proven `modernbert.rs` (~400 lines) into burn.
2. **Head hand-written in burn** (~100 lines, port of `common.py:89-136` minus encoder):
   2-layer norm-first transformer, type embedding, scorer, act head, marker gather,
   masked softmax over options.
3. **Sequence builder + confidence**: port `build_sequence` (`common.py:49-86`) and
   entropy confidence + `clamp_temperature` ([0.5, 5.0] — the clamp exists because a
   fitted 0.1006 temperature publishes coin flips as certainties).
4. **Weights**: safetensors → `burn-store` with name remap
   (`encoder.*`, `head.*`, `type_emb`, `scorer`, `act_head`, `temperature` buffer).
   Skip the MLM decoder weights.
5. **Golden test**: laya's benchmark notebook is fixed-seed. Same questions through
   Python and Rust; top choice + probability within ~1e-2. Calibration is the
   sensitive part (ECE refit on temperatures) — don't accept "close enough".

## Hardware notes

- 4GB-class cards: 843 MB F16 weights + activations + renderer on the same adapter is
  tight. Burn creates its own wgpu device; coexistence is fine, VRAM is the shared
  budget. Int8 (~430 MB) is the fallback; burn PTQ 8-bit is "some backends",
  wgpu support unverified — part of Phase 0.
- Burn pure-Rust CPU (`flex`) is seconds-per-question for this class → design decisions
  as async, event-driven calls off the frame loop; integrated-GPU machines get slow
  decisions, not hard failures.

## Checkpoint choice

English-only still leaves two, same architecture either way:

- root `convaiinnovations/laya`: seq 512 — guardrails / small triage
- `typed-decisions` subfolder: seq 1024 — the four typed workflows (likely the right
  one for game NPC decisions)

Ship exactly one.

## CubeCL: raises the ceiling, not the decision

- GEMM-dominated workload (seq ≤ 1024); burn already auto-fuses with autotune.
  The one real custom-kernel candidate is a flash-attention kernel for the
  alternating global/sliding-window pattern — days-to-weeks, only if profiling shows
  attention (not matmuls) dominating.
- Real value: vendor-neutral GPUs (AMD/Intel on Linux/Windows, where ORT falls back to
  CPU/DirectML) and pure-Rust hot-spot fixes. On NVIDIA it buys nothing over ORT/TensorRT.
- Cannot fix backend maturity bugs (see the Metal failure in burn's own check).

## If we plan to fine-tune/distill later

Laya's README notes the base checkpoints are near chance on typed-decisions zero-shot.
If we distill a smaller backbone for the game (worth it — 421M is heavy to ship),
**freeze this port until the final architecture exists**, then port once.

## References

- burn ModernBERT model check: https://github.com/tracel-ai/burn/pull/3814
- burn-onnx: https://github.com/tracel-ai/burn-onnx
- candle ModernBERT (reference impl for a hand-port):
  https://github.com/huggingface/candle/tree/master/candle-examples/examples/modernbert
- laya encoder configs:
  https://huggingface.co/convaiinnovations/laya/raw/main/encoder/config.json
  https://huggingface.co/convaiinnovations/laya/raw/main/multilingual/encoder/config.json

## Results: native-safetensors burn port (2026-09-22)

**42/42 golden PASS, worst |Δp| = 1.0e-4 (39/42 exactly 0.0), pure Rust, no ONNX.**
`scratchpad/laya-burn/` — burn + burn-store `=0.22.0-pre.4`, `ndarray` CPU backend,
hand-written ModernBERT-large + laya head. Loads the F16 checkpoint converted to
f32 with burn-layout transposes baked in.

### What was measured

| path | 5-question batched call | per question |
|---|---|---|
| PyTorch CPU (torch 2.14) | 1.93 s | ~0.39 s |
| ort CLI (fixed 512 ONNX) | 9.2 s | 1.53 s |
| burn ndarray (dynamic shapes) | 6.0 s | 1.19 s |

All CPU. The wgpu backend is still the untested perf lever (Phase 0).

### How parity was found (bisection that worked)

Hook the **true HF model** (installed transformers, not laya's hook path) and
compare stage-by-stage with the port, on the **exact golden inputs**. Trap:
probes on hand-made states or laya's own hooks compare different pipelines and
lie. Sequence: ids → embeddings → qkv → rope → attention → MLP → layer out.

Two real bugs found:

1. **GeGLU convention swap (the killer).** transformers 5.x `ModernBertMLP`:
   `input, gate = Wi(hn).chunk(2); Wo(act(input) * gate)` — GELU on the **FIRST**
   half. laya's `common.py` reads `inp, gate = chunk(2); act(gate) * inp` — the
   halves are swapped. My port copied laya's convention → layer-0 output std 2×
   too big → probabilities collapse toward uniform. The ONNX export was immune
   (traced from the real model).
2. **Stale checkpoints/inputs during bisection.** Early "mismatch" numbers came
   from comparing against a hand-typed probe state and a pre-conversion f32
   checkpoint. Always re-dump both sides' ids/tensors from the same case before
   believing a mismatch.

### Verified equal (do not re-litigate)

- Input ids byte-identical (42 cases, all rows); marker positions identical.
- rope tables identical (cos/sin to f32 noise; `theta` 160000 full / 10000
  sliding, `cos = cos(pos·f)` with `f = theta^(-2j/d)`).
- Embeddings: `norm(W(ids))` only — transformers 5.x has **no token_types add**.
- QKV: `view(B,L,3,H,hd)` == contiguous chunk(3) (the "per-head interleave"
  theory was a no-op). Layer 0 has **no attn_norm** (checkpoint lacks it).
- Mask: HF passes a **bool [B,1,L,L]** mask; SDPA converts internally. Additive
  `±1e9`/`-1e4` with positive-after-subtract constants is equivalent.
- Encoder MLP activation: `config.hidden_activation = gelu` (exact erf).
  Head FFN: relu. Scorer: LN→Linear→GELU→Linear.
- Sliding band `|i-j| ≤ sliding_window(=64)` passes the gate (bool-mask
  semantics confirmed equal to SDPA within f32 noise).

### burn 0.22-pre porting notes

- `Tensor` is NOT generic over the backend in 0.22-pre; runtime `Device`
  (`Device::ndarray()` — deprecated in favor of `Device::flex()`).
- burn `Linear` weight is `[in,out]` (torch `[out,in]`) — transpose
  head.layers.N.linear1/2, self_attn.out_proj, scorer.1/3, act_head.0/2.
- LayerNorm params are `gamma`/`beta`; remap `.weight`/`.bias`.
- `Tensor` is Clone-not-Copy; `softmax`/`gelu`/`relu` are free functions in
  `burn::tensor::activation`; `module::linear` expects `[in,out]`.
- F16 safetensors are rejected by burn-ndarray → ship a converted f32 file
  (or int8 later); `SafetensorsStore::load_from` + key remaps +
  `allow_partial(true)` + applied-count verification worked cleanly.
- burn-store only exists on the 0.22 pre-release line — pin `=0.22.0-pre.4`.

### Phase 0 executed (2026-09-22): wgpu/Vulkan on the game GPU

GTX 1650 (4 GB, NVIDIA 580.173.02), burn `features = ["ndarray", "vulkan"]`,
`Device::vulkan(DeviceKind::DiscreteGpu(0))` selected by `LAYA_DEVICE=vulkan`.

**Golden gate: 42/42 PASS, worst |Δp| = 0.00e0 — every case exact on GPU.**

| runtime | 5-question batched call | per question |
|---|---|---|
| PyTorch CPU | 1.93 s | ~0.39 s |
| ort CPU (fixed-512) | 9.2 s | ~1.53 s |
| burn ndarray CPU | 6.0 s | ~1.19 s |
| **burn wgpu/Vulkan (f32)** | **0.74 s (p50)** | **~0.15 s** |

~82% GPU utilization, ~2.1 GB VRAM during forward (843 MB f32 weights +
activations + workspace). Steady-state wall for the whole 42-case validate:
~75 s including warm cache.

Notes from the run:

- **Dependency fix required**: burn 0.22.0-pre.4 pins `cubek-fft
  =0.3.0-pre.4`, never published to crates.io (only `0.3.0-pre.1` exists).
  Fixed with `[patch.crates-io] cubek-fft = { git =
  "https://github.com/tracel-ai/cubek", tag = "v0.3.0-pre.4" }` in the
  workspace `Cargo.toml`. Expect this class of breakage from pre-release
  lines; pin the patch alongside the version.
- **Dynamic shapes JIT-compile + autotune per shape, per process**: first
  execution of a distinct (B, L, K) in a process pays kernel compilation and
  autotune benchmarking (observed ~1-13 s depending on shape size; the biggest
  sequences worst). Compiled kernel binaries persist in `CUBECL_CACHE_DIR`
  across runs, but tuning still re-benchmarks unseen shapes per process.
  Steady state after warmup: ~0.73-0.88 s/case. For the game: bucket L
  (e.g. pad to 128/256/512) so the kernel set is small and fully warmed.
- **CubeCL fusion feature: no measurable effect** (bench p50 736.9 ms with
  fusion vs 737.1 ms without) — the workload is large-GEMM-dominated; fusion
  targets small-op chains. Fusion also passes the full gate (0.00e0).
- The port was written backend-agnostic (runtime `Device`), so the GPU switch
  was one line + features. No model changes needed.
- First wgpu build: ~10 min (burn-wgpu + cubecl + wgpu graph).

### Next: model-size / architecture decision (now unblocked)

Apples-to-apples is in. The remaining decision inputs: int8 quantization
(~420 MB weights, 2-4× CPU), f16 GPU weights (421 MB), and whether to distill
a smaller backbone before freezing the port. The f32 port above is the
reference implementation for any of those.

# TODO

## CubeCL memory-pool patch (deferred — re-apply when integrating or upstreaming)

**Status: removed from the workspace for now.** The change is a one-line edit;
recorded here so it can be re-applied in minutes when we pick it back up.

### What it is

CubeCL's wgpu runtime ships with a memory allocator that, by default, caches
freed GPU buffers by size class (`MemoryConfiguration::SubSlices` in
`cubecl-wgpu/src/runtime.rs`, `RuntimeOptions::default()`). Freed buffers are
parked in a pool and handed to the next same-size allocation instead of being
returned to the card — a performance win (no driver round-trip per alloc), but
the pool never shrinks, so peak VRAM includes every buffer size ever touched.

The patch flips that preset:

```rust
// vendor/cubecl-wgpu/src/runtime.rs, RuntimeOptions::default()
-            memory_config: MemoryConfiguration::default(),
+            memory_config: MemoryConfiguration::ExclusivePages,
```

`ExclusivePages` gives every allocation its own page that is freed as soon as
the tensor drops — no cache, slightly slower allocation, much lower VRAM peak.

### Why it was needed (measured, GTX 1650 / 4 GB, laya f32 421M)

| config | peak VRAM stock (SubSlices) | peak VRAM patched (ExclusivePages) |
|---|---|---|
| f32 weights, f32 compute | 2136 MiB (steady 1711) | 1885 MiB (steady 1711) |
| f16 storage + f32 compute (mixed) | 2316 MiB | 2161 MiB |

1. **Pure f32 hoards ~425 MiB of dead activation buffers** — every distinct
   tensor size in the forward ([B,L,1024], [B,L,3072], [B,L,5248], attention
   scores, ...) leaves a cached slice behind.
2. **It defeated the f16-storage experiment**: with f16 weights, each matmul
   upcasts its weight to a transient f32 copy (~150 casts/forward, ~1 GiB
   accumulated); the pool cached those copies, so f16 storage cost f32-scale
   VRAM. Note kernel dequantization (cubek-quant, int8/int4) removes the
   upcast copies entirely — that motivation is quantization-specific — but the
   activation-buffer hoard (~425 MiB) affects every config.

### Why it was removed

- It requires vendoring `cubecl-wgpu` (a modified dependency), which we don't
  want to carry in the workspace while the burn 0.22 pre-release line is
  already moving.
- It's orthogonal to correctness (both golden gates pass either way) and easy
  to re-apply; upstreaming is the better end state.

### How to re-apply

1. Copy the cached crate source into the workspace:
   `cp -r ~/.cargo/registry/src/index.crates.io-*/cubecl-wgpu-0.11.0-pre.4 scratchpad/vendor/cubecl-wgpu`
   (and `chmod -R u+w`).
2. Apply the one-line change above in `scratchpad/vendor/cubecl-wgpu/src/runtime.rs`.
3. Add to `scratchpad/Cargo.toml`:

   ```toml
   [patch.crates-io]
   cubecl-wgpu = { path = "vendor/cubecl-wgpu" }
   ```

4. `cargo build --release -p laya-burn` — verify with the golden gate
   (`LAYA_DEVICE=vulkan LAYA_DTYPE=f16 laya-burn validate`) and a VRAM sample
   (`nvidia-smi` during a classify run).

### Upstream angle

`burn-cubecl` hardcodes `RuntimeOptions::default()` with no public knob or env
var to select the pool strategy (only `CUBECL_WGPU_MAX_TASKS` exists). Worth
raising with the burn/CubeCL maintainers: either expose `memory_config` on the
burn `Device`/runtime options, or honor an env var like
`CUBECL_MEMORY_CONFIG=exclusive-pages`. If that lands, drop the vendored patch.

Also keep in mind the related (still active) workspace patch: `cubek-fft`
pinned to `v0.3.0-pre.4` via git — that one is *required to build* (the version
was never published to crates.io) and is unrelated to memory.

## Backlog / decision items

- **int8 quantization** (`Q8S` via burn's Quantizer + cubek-quant): ~420 MB
  artifact, dequant-in-kernel (weights stay packed, f32 accumulate), expected
  ~1.0-1.2 GiB VRAM with f32 activations. Sequence: int8 first (validates
  calibrate → store → quantized matmul → 1e-3 gate at low accuracy risk),
  then int4/fp4 selective (~330 MB artifact) with a behavioral gate (top-1
  agreement, act-head agreement) — relaxing the parity gate needs sign-off.
- **Model-size / architecture decision** (deferred): distill a smaller backbone
  vs int8/int4 vs f32 — apples-to-apples numbers are in `scratchpad/ai/burn-plan-1.md`.
- **CPU backend removed**: wgpu/Vulkan is the only backend (`Device::vulkan`).
  burn-ndarray couldn't hold f16 storage, and the only non-ndarray pure-Rust
  CPU alternative (CubeCL's `cpu` feature) needs a `tracel-llvm-bundler` build
  step that downloads a bundled-CA LLVM toolchain — fails behind the
  TLS-intercepting proxy (`InvalidCertificate(UnknownIssuer)`; pre-seeding
  `~/.cache/tracel/` is blocked by the read-only `~/.cache`). Revisit only if
  a CPU fallback is ever needed (the ort playground already covers CPU).
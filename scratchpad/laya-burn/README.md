# laya-burn

laya's DecisionModel as a native [burn](https://burn.dev) module, weights loaded directly
from the HF safetensors checkpoint. Library entry point: `laya_burn::{load, Session,
Question, Questions}` — see `src/lib.rs` for the full surface.

## Consuming from another workspace

`burn` is pinned to `=0.22.0-pre.4`, which pulls in `cubek-fft =0.3.0-pre.4` — a version
that was **never published to crates.io** (only `0.3.0-pre.1` exists there). The
`scratchpad/` workspace already carries the fix, but Cargo applies `[patch.crates-io]`
per-workspace only, so if you depend on `laya-burn` from a different workspace (e.g. a
game), copy the patch into **that workspace's root** `Cargo.toml`:

```toml
[patch.crates-io]
cubek-fft = { git = "https://github.com/tracel-ai/cubek", tag = "v0.3.0-pre.4" }
```

The `v0.3.0-pre.4` tag of the upstream repo carries the matching `cubek-fft` package.

Related constraints:

- burn's exact pre-release pin (`=0.22.0-pre.4`) means no other burn version can coexist
  in the same dependency graph.
- The crate is instantiated on burn's Vulkan backend (`vulkan` + `fusion` features);
  pick the device via `load(model_dir, device)`.
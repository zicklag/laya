//! Loads the HF safetensors checkpoint into the burn module.
//!
//! Key strategy: burn module field names mirror the HF paths verbatim
//! (`encoder.layers.0.attn.Wqkv.weight`, …), so the only remaps needed are the
//! scorer/act_head Sequential indices → named fields. `allow_partial` tolerates
//! layer 0's absent attn_norm; the applied count must be exactly 206 - 1.

use anyhow::{anyhow, Context, Result};
use burn_store::{ModuleSnapshot, SafetensorsStore};

use crate::model::DecisionModel;

pub fn load(mut model: DecisionModel, path: &std::path::Path) -> Result<DecisionModel> {
    // The f16 checkpoint loads verbatim; the model upcasts to f32 per op
    // (see model.rs). No on-disk conversion is involved.
    let mut store = SafetensorsStore::from_file(path)
        .allow_partial(true)
        .with_key_remapping("scorer.0", "scorer.norm")
        .with_key_remapping("scorer.1", "scorer.linear1")
        .with_key_remapping("scorer.3", "scorer.linear2")
        .with_key_remapping("act_head.0", "act_head.linear1")
        .with_key_remapping("act_head.2", "act_head.linear2");
    let result = model
        .load_from(&mut store)
        .map_err(|e| anyhow!("safetensors apply failed: {e}"))
        .with_context(|| format!("loading {}", path.display()))?;

    if !result.errors.is_empty() {
        return Err(anyhow!("apply errors: {:?}", result.errors));
    }
    // 206 store tensors applied; the only absent one is layer 0's attn_norm
    // (absent from the HF checkpoint by design — the dummy param stays at init).
    if result.applied.len() != 206 {
        return Err(anyhow!(
            "expected 206 applied tensors, got {}: missing={:?} unused={:?}",
            result.applied.len(),
            result.missing,
            result.unused
        ));
    }
    if !(result.unused.is_empty()
        && result.missing.len() == 1
        && result.missing[0].0 == "encoder.layers.0.attn_norm.weight")
    {
        return Err(anyhow!(
            "unexpected load state: missing={:?} unused={:?}",
            result.missing,
            result.unused
        ));
    }
    Ok(model)
}
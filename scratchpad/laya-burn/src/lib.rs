//! laya DecisionModel as a native burn module, weights loaded directly from
//! the HF safetensors checkpoint (`laya-en/model.safetensors`).
//!
//! Library surface:
//! - [`session`]: [`Session`] (model + tokenizer + meta + email cleaner),
//!   [`load`] and [`predict`] — the full `Agent.system_one` pipeline behind
//!   one batched forward. This is the entry point for a game or a custom CLI.
//! - [`model`]: the burn module tree; field names mirror the checkpoint keys
//!   verbatim so `SafetensorsStore` loads with remaps only.
//! - [`seq`]/[`post`]: Python-faithful (byte-exact) sequence construction and
//!   answer post-processing; [`loader`] applies the safetensors checkpoint.
//!
//! The crate is instantiated on burn's Vulkan backend (fused kernels). Pass
//! the `Device` you want to [`load`] — e.g.
//! `Device::vulkan(DeviceKind::DiscreteGpu(0))`.
//!
//! ```no_run
//! use laya_burn::{load, predict};
//! use serde_json::{json, Map, Value};
//!
//! # fn main() -> anyhow::Result<()> {
//! let device = laya_burn::burn::prelude::Device::vulkan(
//!     laya_burn::burn::prelude::DeviceKind::DiscreteGpu(0),
//! );
//! let mut sess = load("model", device)?;
//! let state = json!({"subject": "Invoice attached", "body": "Due Friday?", "from": "a@b.c"});
//! let questions: Map<String, Value> = serde_json::from_str(r#"{}"#)?;
//! let result = predict(&mut sess, &state, &questions)?;
//! # Ok(())
//! # }
//! ```
pub use burn;

pub mod email;
pub mod jsonw;
pub mod loader;
pub mod model;
pub mod post;
pub mod seq;
pub mod session;

pub use session::{load, predict, Rt, Session};
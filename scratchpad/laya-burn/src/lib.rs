//! laya DecisionModel as a native burn module, weights loaded directly from
//! the HF safetensors checkpoint (`laya-en/model.safetensors`).
//!
//! Library surface:
//! - [`session`]: [`Session`] (model + tokenizer + meta) with
//!   [`Session::predict`](session::Session::predict) — the full
//!   `Agent.system_one` pipeline behind one batched forward; [`load`]
//!   constructs it.
//! - [`question`]: typed builders for `choice` / `score` / `noul` questions
//!   and the ordered [`Questions`] set — or hand-roll raw laya defs.
//! - [`model`]: the burn module tree; field names mirror the checkpoint keys
//!   verbatim so `SafetensorsStore` loads with remaps only.
//! - [`seq`]/[`post`]: Python-faithful (byte-exact) sequence construction and
//!   answer post-processing — public for custom pipelines:
//!   `seq::build_sequence` → `seq::collate` → [`Rt::run`] →
//!   `post::answer_value`.
//! - [`loader`] applies the safetensors checkpoint; [`jsonw`] is the
//!   Python-`json.dumps`-compatible writer.
//!
//! The crate is instantiated on burn's Vulkan backend (fused kernels). Pass
//! the `Device` you want to [`load`] — e.g.
//! `Device::vulkan(DeviceKind::DiscreteGpu(0))`.
//!
//! ```no_run
//! use std::path::Path;
//! use laya_burn::{load, Question, Questions};
//! use serde_json::json;
//!
//! # fn main() -> anyhow::Result<()> {
//! let device = laya_burn::burn::prelude::Device::vulkan(
//!     laya_burn::burn::prelude::DeviceKind::DiscreteGpu(0),
//! );
//! let mut sess = load(Path::new("model"), device)?;
//!
//! let mut questions = Questions::new();
//! questions.add(
//!     Question::choice("category")
//!         .instructions("Which team should handle the request in `state`?")
//!         .option_with("billing", "invoices, payments, refunds")
//!         .option_with("technical", "bugs, outages, integrations")
//!         .option("other")
//!         .build()?,
//! )?;
//! questions.add(Question::noul("is_spam").instructions("Is this spam?").build()?)?;
//!
//! let state = json!({"subject": "Invoice attached", "body": "Due Friday?"});
//! let result = sess.predict(&state, &questions)?;
//! # Ok(())
//! # }
//! ```

pub mod jsonw;
pub mod loader;
pub mod model;
pub mod post;
pub mod question;
pub mod seq;
pub mod session;

pub use burn;
pub use question::{Question, Questions};
pub use session::{load, Rt, Session};

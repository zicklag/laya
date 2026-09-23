//! Session runtime: model + tokenizer + meta + email cleaner, and the
//! `Agent.system_one` predict pipeline. This is the library surface a game
//! or custom CLI imports.
//!
//! The original checkpoint is f16 and loads verbatim; the model computes in
//! f32, upcasting params at each op (see `model.rs`).

use anyhow::{anyhow, bail, Context, Result};
use burn::prelude::*;
use burn::tensor::FloatDType;
use serde_json::{json, Map, Value};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::email;
use crate::loader;
use crate::model::DecisionModel;
use crate::post;
use crate::seq::{self, Meta};

/// Loaded engine: weights, tokenizer, question meta, email cleaner.
pub struct Session {
    pub rt: Rt,
    pub tok: Tokenizer,
    pub meta: Meta,
    pub cleaner: email::EmailCleaner,
}

pub struct Rt {
    pub model: DecisionModel,
    pub device: Device,
}

impl Rt {
    pub fn new(model_dir: &Path, device: Device) -> Result<Self> {
        let model = DecisionModel::init(&device);
        let model = loader::load(model, &model_dir.join("laya-en/model.safetensors"))?;
        Ok(Self { model, device })
    }

    /// One batched forward. Returns (logits [b*k], act [b*2]).
    pub fn run(&self, batch: &seq::Batch) -> Result<(Vec<f32>, Vec<f32>)> {
        let ids = Tensor::<2, Int>::from_ints(
            TensorData::new(batch.input_ids.clone(), [batch.b, batch.l]),
            &self.device,
        );
        let att = Tensor::<2, Int>::from_ints(
            TensorData::new(batch.attention_mask.clone(), [batch.b, batch.l]),
            &self.device,
        );
        let mpos = Tensor::<2, Int>::from_ints(
            TensorData::new(batch.marker_pos.clone(), [batch.b, batch.k]),
            &self.device,
        );
        let mmask = Tensor::<2, Bool>::from_bool(
            TensorData::new(batch.marker_mask.clone(), [batch.b, batch.k]),
            &self.device,
        );
        let qt = Tensor::<1, Int>::from_ints(
            TensorData::new(batch.qtype.clone(), [batch.b]),
            &self.device,
        );
        let (logits, act) = self.model.forward(ids, att, mpos, mmask, qt);
        // The model's dtype follows the checkpoint (f16 on GPU); read the
        // results back as f32 for the post-processing layer.
        let l = logits
            .cast::<FloatDType>(FloatDType::F32)
            .into_data()
            .try_to_vec::<f32>()?;
        let a = act
            .cast::<FloatDType>(FloatDType::F32)
            .into_data()
            .try_to_vec::<f32>()?;
        Ok((l, a))
    }
}

/// Load a session from `model_dir` (layout: `meta.json`,
/// `laya-en/model.safetensors`, `laya-en/tokenizer/tokenizer.json`).
pub fn load(model_dir: &Path, device: Device) -> Result<Session> {
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("meta.json"))
            .with_context(|| format!("reading {}", model_dir.join("meta.json").display()))?,
    )?;
    let tok = Tokenizer::from_file(model_dir.join("laya-en/tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("loading tokenizer: {e}"))?;
    let rt = Rt::new(model_dir, device)?;
    let cleaner = email::EmailCleaner::new()?;
    Ok(Session {
        rt,
        tok,
        meta,
        cleaner,
    })
}

/// Agent.system_one on burn: state + question defs -> answers JSON.
pub fn predict(sess: &mut Session, state: &Value, questions: &Map<String, Value>) -> Result<Value> {
    let state_text = seq::serialize_state(state);
    let mut items: Vec<seq::Item> = Vec::new();
    for (qid, qdef) in questions {
        let q = seq::to_internal(qdef)?;
        let (ids, markers) = seq::build_sequence(&sess.tok, &sess.meta, &state_text, &q)?;
        let n_opts = seq::render_options(&q).len();
        if markers.len() != n_opts {
            bail!(
                "question {qid:?}: options exceed head_max_len={} ({} markers for {} options)",
                sess.meta.head_max_len,
                markers.len(),
                n_opts
            );
        }
        if markers.len() < 2 {
            bail!("question {qid:?}: single-option questions are unsupported (K >= 2 required)");
        }
        let qt = sess.meta.qtype(&q.t)? as i64;
        items.push(seq::Item {
            qid: qid.clone(),
            q,
            ids,
            markers,
            qtype: qt,
        });
    }

    let batch = seq::collate(&items, sess.meta.pad_id);
    let (logits, act) = sess.rt.run(&batch)?;
    let lstride = logits.len() / batch.b;
    let astride = act.len() / batch.b;

    let mut answers = Map::new();
    let mut n_tokens = 0u64;
    for (r, it) in items.iter().enumerate() {
        let row = &logits[r * lstride..][..lstride];
        let actr = &act[r * astride..][..astride];
        let v = post::answer_value(
            &it.q,
            &it.qid,
            it.qtype as usize,
            batch.ks[r],
            row,
            actr,
            &sess.meta,
        )?;
        answers.insert(it.qid.clone(), v);
        n_tokens += batch.attention_mask[r * batch.l..(r + 1) * batch.l]
            .iter()
            .filter(|&&x| x == 1)
            .count() as u64;
    }
    Ok(json!({
        "model": "laya-burn",
        "answers": answers,
        "usage": {"input_tokens": n_tokens, "output_tokens": 0},
    }))
}
//! Port of laya/common.py:15-86 (rendering + sequence construction) and
//! collate_items (common.py:247-280). Faithful to the byte: JSON separators,
//! token caps, budgets and truncation all mirror the Python exactly.

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

pub const QTYPE_NAMES: [&str; 3] = ["choice", "score", "noul"];

#[derive(Debug, Clone)]
pub struct Q {
    pub t: String,
    pub ins: String,
    pub crit: Option<Value>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Meta {
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    pub pad_id: i64,
    pub cls_id: i64,
    pub sep_id: i64,
    pub mask_id: i64,
    pub mask_token: String,
    #[serde(default = "default_temps")]
    pub temperature: Vec<f64>,
    #[serde(default)]
    pub temperature_by_options: std::collections::HashMap<String, f64>,
    #[serde(default = "default_qtypes")]
    pub qtypes: std::collections::HashMap<String, usize>,
}

fn default_max_len() -> usize {
    512
}
fn default_head_max_len() -> usize {
    192
}
fn default_temps() -> Vec<f64> {
    vec![1.0, 1.0, 1.0]
}
fn default_qtypes() -> std::collections::HashMap<String, usize> {
    [("choice".into(), 0), ("score".into(), 1), ("noul".into(), 2)].into()
}

impl Meta {
    pub fn qtype(&self, t: &str) -> Result<usize> {
        self.qtypes
            .get(t)
            .copied()
            .ok_or_else(|| anyhow!("unknown question type {t:?}"))
    }
}

/// Agent._to_internal (agent.py:254-263).
pub fn to_internal(qdef: &Value) -> Result<Q> {
    let t = qdef["type"]
        .as_str()
        .ok_or_else(|| anyhow!("question missing 'type'"))?
        .to_string();
    let mut crit = qdef.get("criteria").cloned();
    if t == "choice" {
        if let Some(Value::Array(a)) = crit {
            let mut m = Map::new();
            for c in a {
                let k = c
                    .as_str()
                    .ok_or_else(|| anyhow!("choice criteria entries must be strings"))?;
                m.insert(k.to_string(), Value::Null);
            }
            crit = Some(Value::Object(m));
        }
    }
    let ins = match qdef.get("instructions") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => crate::jsonw::dump(other),
        None => return Err(anyhow!("question missing 'instructions'")),
    };
    Ok(Q { t, ins, crit })
}

/// render_criterion (common.py:21-30): strings pass through, everything else
/// becomes JSON with Python's (", ", ": ") separators.
pub fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => crate::jsonw::dump(other),
    }
}

/// serialize_state (common.py:15-18).
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => crate::jsonw::dump(other),
    }
}

/// render_options (common.py:33-46).
pub fn render_options(q: &Q) -> Vec<String> {
    match q.t.as_str() {
        "choice" => {
            let crit = q
                .crit
                .as_ref()
                .and_then(|c| c.as_object())
                .cloned()
                .unwrap_or_default();
            crit.into_iter()
                .map(|(k, v)| {
                    let plain = v.is_null() || matches!(&v, Value::String(s) if s.is_empty());
                    if plain {
                        k
                    } else {
                        format!("{}: {}", k, render_criterion(&v))
                    }
                })
                .collect()
        }
        "score" => {
            let crit = q
                .crit
                .as_ref()
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            crit.iter()
                .enumerate()
                .map(|(i, c)| format!("level {}: {}", i, render_criterion(c)))
                .collect()
        }
        _ => {
            let crit = q
                .crit
                .as_ref()
                .and_then(|c| c.as_object())
                .cloned()
                .unwrap_or_default();
            let part = |v: Option<&Value>, fallback: &str| -> String {
                let use_val = match v {
                    None | Some(Value::Null) => false,
                    Some(Value::String(s)) => !s.is_empty(),
                    Some(_) => true,
                };
                if use_val {
                    render_criterion(v.unwrap())
                } else {
                    fallback.to_string()
                }
            };
            vec![
                format!(
                    "false: {}",
                    part(crit.get("false"), "no, the statement does not hold")
                ),
                format!(
                    "true: {}",
                    part(crit.get("true"), "yes, the statement holds")
                ),
            ]
        }
    }
}

pub fn encode(tok: &Tokenizer, text: &str) -> Vec<i64> {
    let enc = tok
        .encode(text, false)
        .unwrap_or_else(|e| panic!("tokenization failed for {text:?}: {e}"));
    enc.get_ids().iter().map(|&i| i as i64).collect()
}

/// build_sequence (common.py:49-86).
/// Format: [CLS] <type> instructions [SEP]  opt0  opt1 ... [SEP] state [SEP].
pub fn build_sequence(
    tok: &Tokenizer,
    meta: &Meta,
    state_text: &str,
    q: &Q,
) -> Result<(Vec<i64>, Vec<usize>)> {
    let mask = &meta.mask_token;
    let opts = render_options(q);
    let ins = q.ins.replace(mask.as_str(), " ");
    let head_full = encode(tok, &format!("{} question: {}", q.t, ins));

    let mut opt_ids: Vec<Vec<i64>> = opts
        .iter()
        .map(|o| {
            let mut ids = vec![meta.mask_id];
            let text = o.replace(mask.as_str(), " ");
            ids.extend(encode(tok, &format!(" {text}")).into_iter().take(48));
            ids
        })
        .collect();

    let hl = meta.head_max_len as i64;
    let mut opt_budget = hl - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
    if opt_budget < 16 {
        let per = ((hl - 16) / opt_ids.len() as i64).max(4);
        for o in &mut opt_ids {
            o.truncate(per.max(0) as usize);
        }
        opt_budget = hl - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
    }
    let keep = (opt_budget.max(8) as usize).min(head_full.len());

    let mut ids: Vec<i64> = Vec::with_capacity(meta.max_len);
    ids.push(meta.cls_id);
    ids.extend_from_slice(&head_full[..keep]);
    ids.push(meta.sep_id);

    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(meta.sep_id);

    let room = ((meta.max_len as i64) - ids.len() as i64 - 1).max(0) as usize;
    let st = encode(tok, &state_text.replace(mask.as_str(), " "));
    ids.extend(st.into_iter().take(room));
    ids.push(meta.sep_id);

    ids.truncate(meta.max_len);
    markers.retain(|&m| m < meta.max_len);
    Ok((ids, markers))
}

pub struct Item {
    pub qid: String,
    pub q: Q,
    pub ids: Vec<i64>,
    pub markers: Vec<usize>,
    pub qtype: i64,
}

pub struct Batch {
    pub b: usize,
    pub l: usize,
    pub k: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub marker_pos: Vec<i64>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<i64>,
    pub ks: Vec<usize>,
}

/// collate_items (common.py:247-280), inference subset. Burn is shape-dynamic:
/// pad to the batch max only (no fixed export shape).
pub fn collate(items: &[Item], pad_id: i64) -> Batch {
    let b = items.len();
    let l = items.iter().map(|i| i.ids.len()).max().unwrap_or(0);
    let k = items
        .iter()
        .map(|i| i.markers.len())
        .max()
        .unwrap_or(0);
    let mut ids = vec![pad_id; b * l];
    let mut att = vec![0i64; b * l];
    let mut mpos = vec![0i64; b * k];
    let mut mmask = vec![false; b * k];
    for (i, it) in items.iter().enumerate() {
        ids[i * l..i * l + it.ids.len()].copy_from_slice(&it.ids);
        att[i * l..i * l + it.ids.len()].fill(1);
        let ki = it.markers.len();
        mpos[i * k..i * k + ki]
            .copy_from_slice(&it.markers.iter().map(|&m| m as i64).collect::<Vec<_>>());
        mmask[i * k..i * k + ki].fill(true);
    }
    let mut qtype = vec![0i64; b];
    for (i, it) in items.iter().enumerate() {
        qtype[i] = it.qtype;
    }
    let ks: Vec<usize> = items.iter().map(|i| i.markers.len()).collect();
    Batch {
        b,
        l,
        k,
        input_ids: ids,
        attention_mask: att,
        marker_pos: mpos,
        marker_mask: mmask,
        qtype,
        ks,
    }
}
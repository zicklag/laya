//! Port of the post-processing half of Agent.system_one (agent.py:319-368):
//! temperature buckets, softmax, entropy confidence, typed answers.

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};

use crate::seq::{Meta, Q, QTYPE_NAMES};

/// temp_bucket (common.py:219-221).
pub fn temp_bucket(qt: usize, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", QTYPE_NAMES[qt], size)
}

pub fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

/// confidence_from_probs (common.py:210-216): 1 - H(p)/log(k).
pub fn confidence(p: &[f64], k: usize) -> f64 {
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = p[..k].iter().map(|&x| -x * x.max(1e-12).ln()).sum();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// Build one answer object, mirroring agent.py:325-362.
pub fn answer_value(
    q: &Q,
    qid: &str,
    qt: usize,
    k: usize,
    logits: &[f32],
    act: &[f32],
    meta: &Meta,
) -> Result<Value> {
    if k < 2 {
        return Err(anyhow!(
            "question {qid}: {} option(s); laya's shipped path requires >= 2",
            k
        ));
    }
    let bucket = temp_bucket(qt, k);
    let t_scale = meta
        .temperature_by_options
        .get(&bucket)
        .copied()
        .unwrap_or_else(|| meta.temperature.get(qt).copied().unwrap_or(1.0));
    let z: Vec<f64> = logits[..k].iter().map(|&x| x as f64 / t_scale).collect();
    let p = softmax(&z);
    let conf = round4(confidence(&p, k));

    let act_f: Vec<f64> = act.iter().map(|&x| x as f64).collect();
    let act_p = round4(softmax(&act_f)[0]);

    match q.t.as_str() {
        "choice" => {
            let crit = q
                .crit
                .as_ref()
                .and_then(|c| c.as_object())
                .ok_or_else(|| anyhow!("question {qid}: choice requires object criteria"))?;
            let keys: Vec<&String> = crit.keys().collect();
            if keys.len() < k {
                return Err(anyhow!(
                    "question {qid}: {k} markers but only {} criteria",
                    keys.len()
                ));
            }
            let am = argmax(&p[..k]);
            let mut probs = Map::new();
            for (kk, &pv) in keys.iter().zip(p[..k].iter()) {
                probs.insert((*kk).clone(), json!(round4(pv)));
            }
            Ok(json!({
                "type": "choice",
                "choice": keys[am],
                "probabilities": probs,
                "confidence": conf,
                "action": {"act_probability": act_p},
            }))
        }
        "score" => {
            let crit = q
                .crit
                .as_ref()
                .and_then(|c| c.as_array())
                .ok_or_else(|| anyhow!("question {qid}: score requires array criteria"))?;
            if crit.len() < k {
                return Err(anyhow!(
                    "question {qid}: {k} markers but only {} levels",
                    crit.len()
                ));
            }
            let exp = (0..k).map(|i| i as f64 * p[i]).sum::<f64>();
            let mut legend = Map::new();
            let mut probs = Map::new();
            for (i, c) in crit.iter().enumerate().take(k) {
                legend.insert(i.to_string(), c.clone());
                probs.insert(i.to_string(), json!(round4(p[i])));
            }
            Ok(json!({
                "type": "score",
                "score": round4(exp),
                "legend": legend,
                "probabilities": probs,
                "confidence": conf,
                "action": {"act_probability": act_p},
            }))
        }
        _ => {
            let p1 = p[1];
            Ok(json!({
                "type": "noul",
                "noul": round4(p1),
                "confidence": round4(p1.max(1.0 - p1)),
                "action": {"act_probability": act_p},
            }))
        }
    }
}

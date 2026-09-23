//! laya-burn CLI: native burn port of laya's DecisionModel, weights loaded
//! directly from the HF safetensors checkpoint. Validated against the same
//! golden cases as the ort playground.
//!
//! The engine lives in the `laya_burn` library target (`src/session.rs` and
//! friends); this binary is only the clap wrapper and output formatting.

use anyhow::{anyhow, bail, Context, Result};
use burn::prelude::*;
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod email;

use laya_burn::{session, Questions};

const TOL: f64 = 1e-3;

#[derive(Parser)]
#[command(name = "laya-burn", about = "laya as a native burn module")]
struct Cli {
    /// dir with meta.json and the laya-en checkpoint (model.safetensors + tokenizer/)
    #[arg(long, default_value = "../laya-ort/cli/model")]
    model_dir: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build an email state (laya's email_state: subject/body/from) and classify
    Email {
        #[arg(long)]
        subject: String,
        #[arg(long)]
        sender: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        #[arg(short, long, default_value = "../laya-ort/cli/presets/email.json")]
        questions: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Classify one state: text file, `-` for stdin, or inline JSON object/list
    Classify {
        #[arg(long)]
        state_file: Option<String>,
        #[arg(short, long, default_value = "../laya-ort/cli/presets/email.json")]
        questions: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Compare against laya's Python outputs (golden cases.json)
    Validate {
        #[arg(long, default_value = "../laya-ort/cli/golden/cases.json")]
        golden: PathBuf,
    },
    /// Latency of one batched predict call
    Bench {
        #[arg(short, long, default_value = "../laya-ort/cli/presets/email.json")]
        questions: PathBuf,
        #[arg(long, default_value_t = 5)]
        n: usize,
    },
}

fn read_state_source(src: &str) -> Result<Value> {
    let text = if src == "-" {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        std::fs::read_to_string(src).with_context(|| format!("reading state file {src}"))?
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(v @ Value::Object(_)) | Ok(v @ Value::Array(_)) => Ok(v),
        _ => Ok(Value::String(text)),
    }
}

fn load_questions(path: &Path) -> Result<Questions> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )?;
    let defs = v.as_object().ok_or_else(|| {
        anyhow!(
            "{} must be an object of question-id -> question-def",
            path.display()
        )
    })?;
    Questions::from_defs(defs)
}

fn print_human(result: &Value) {
    let answers = result["answers"].as_object().expect("answers object");
    for (qid, a) in answers {
        let t = a["type"].as_str().unwrap_or("?");
        match t {
            "choice" => {
                println!(
                    "== {qid} (choice) -> {} [conf {:.4}, act {:.4}]",
                    a["choice"].as_str().unwrap_or("?"),
                    a["confidence"].as_f64().unwrap_or(0.0),
                    a["action"]["act_probability"].as_f64().unwrap_or(0.0),
                );
                if let Some(probs) = a["probabilities"].as_object() {
                    for (k, v) in probs {
                        println!("   {k:<12} {:.4}", v.as_f64().unwrap_or(0.0));
                    }
                }
            }
            "score" => {
                println!(
                    "== {qid} (score) -> {} [conf {:.4}, act {:.4}]",
                    a["score"].as_f64().unwrap_or(0.0),
                    a["confidence"].as_f64().unwrap_or(0.0),
                    a["action"]["act_probability"].as_f64().unwrap_or(0.0),
                );
                if let Some(probs) = a["probabilities"].as_object() {
                    for (k, v) in probs {
                        let desc = a["legend"]
                            .as_object()
                            .and_then(|l| l.get(k))
                            .and_then(|d| d.as_str())
                            .unwrap_or("");
                        println!("   level {k:<5} {:.4}  {desc}", v.as_f64().unwrap_or(0.0));
                    }
                }
            }
            _ => {
                println!(
                    "== {qid} (noul) -> {:.4} [conf {:.4}, act {:.4}]",
                    a["noul"].as_f64().unwrap_or(0.0),
                    a["confidence"].as_f64().unwrap_or(0.0),
                    a["action"]["act_probability"].as_f64().unwrap_or(0.0),
                );
            }
        }
    }
}

fn compare_answers(expected: &Value, ours: &Value) -> Result<(f64, Vec<String>)> {
    let exp = expected
        .as_object()
        .ok_or_else(|| anyhow!("expected answers not an object"))?;
    let got = ours
        .as_object()
        .ok_or_else(|| anyhow!("our answers not an object"))?;
    let mut worst = 0f64;
    let mut bad = Vec::new();

    let check = |name: &str, e: f64, g: f64, worst: &mut f64, bad: &mut Vec<String>| {
        let d = (e - g).abs();
        if d > *worst {
            *worst = d;
        }
        if d > TOL {
            bad.push(format!("{name}: expected {e}, got {g} (|d|={d:.6})"));
        }
    };
    let num = |v: &Value| -> Result<f64> {
        v.as_f64()
            .ok_or_else(|| anyhow!("expected numeric, got {v:?}"))
    };

    for (qid, e) in exp {
        let g = got
            .get(qid)
            .ok_or_else(|| anyhow!("question {qid} missing from our answers"))?;
        let t = e["type"].as_str().unwrap_or("");
        match t {
            "choice" => {
                let ec = e["choice"].as_str().unwrap_or("");
                let gc = g["choice"].as_str().unwrap_or("");
                if ec != gc {
                    bad.push(format!("{qid}: choice {ec:?} vs {gc:?}"));
                }
                let ep = e["probabilities"].as_object().expect("probabilities");
                let gp = g["probabilities"].as_object().expect("probabilities");
                for (k, v) in ep {
                    check(
                        &format!("{qid}.probabilities.{k}"),
                        v.as_f64().unwrap_or(0.0),
                        gp.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0),
                        &mut worst,
                        &mut bad,
                    );
                }
                check(
                    &format!("{qid}.confidence"),
                    num(&e["confidence"])?,
                    num(&g["confidence"])?,
                    &mut worst,
                    &mut bad,
                );
                check(
                    &format!("{qid}.act"),
                    num(&e["action"]["act_probability"])?,
                    num(&g["action"]["act_probability"])?,
                    &mut worst,
                    &mut bad,
                );
            }
            "score" => {
                check(
                    &format!("{qid}.score"),
                    num(&e["score"])?,
                    num(&g["score"])?,
                    &mut worst,
                    &mut bad,
                );
                let ep = e["probabilities"].as_object().expect("probabilities");
                let gp = g["probabilities"].as_object().expect("probabilities");
                for (k, v) in ep {
                    check(
                        &format!("{qid}.probabilities.{k}"),
                        v.as_f64().unwrap_or(0.0),
                        gp.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0),
                        &mut worst,
                        &mut bad,
                    );
                }
                check(
                    &format!("{qid}.confidence"),
                    num(&e["confidence"])?,
                    num(&g["confidence"])?,
                    &mut worst,
                    &mut bad,
                );
                check(
                    &format!("{qid}.act"),
                    num(&e["action"]["act_probability"])?,
                    num(&g["action"]["act_probability"])?,
                    &mut worst,
                    &mut bad,
                );
            }
            _ => {
                check(
                    &format!("{qid}.noul"),
                    num(&e["noul"])?,
                    num(&g["noul"])?,
                    &mut worst,
                    &mut bad,
                );
                check(
                    &format!("{qid}.confidence"),
                    num(&e["confidence"])?,
                    num(&g["confidence"])?,
                    &mut worst,
                    &mut bad,
                );
                check(
                    &format!("{qid}.act"),
                    num(&e["action"]["act_probability"])?,
                    num(&g["action"]["act_probability"])?,
                    &mut worst,
                    &mut bad,
                );
            }
        }
    }
    Ok((worst, bad))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut sess = session::load(&cli.model_dir, Device::vulkan(DeviceKind::DiscreteGpu(0)))?;
    match cli.cmd {
        Cmd::Email {
            subject,
            sender,
            body_file,
            questions,
            json: as_json,
        } => {
            let body = match body_file.as_deref() {
                Some("-") | None => {
                    let mut b = String::new();
                    std::io::Read::read_to_string(&mut std::io::stdin(), &mut b)?;
                    b
                }
                Some(f) => std::fs::read_to_string(f)?,
            };
            let cleaner = email::EmailCleaner::new()?;
            let state = cleaner.email_state(&subject, &body, sender.as_deref());
            let qs = load_questions(&questions)?;
            let t0 = Instant::now();
            let result = sess.predict(&state, &qs)?;
            let dt = t0.elapsed();
            if as_json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_human(&result);
                println!("   latency: {:.1} ms", dt.as_secs_f64() * 1e3);
            }
        }
        Cmd::Classify {
            state_file,
            questions,
            json: as_json,
        } => {
            let state = read_state_source(
                state_file
                    .as_deref()
                    .ok_or_else(|| anyhow!("--state-file required ('-' for stdin)"))?,
            )?;
            let qs = load_questions(&questions)?;
            let result = sess.predict(&state, &qs)?;
            if as_json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_human(&result);
            }
        }
        Cmd::Validate { golden } => {
            let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&golden)?)
                .with_context(|| format!("reading {}", golden.display()))?;
            let mut worst_all = 0f64;
            let mut n_bad = 0usize;
            for case in &cases {
                let name = case["name"].as_str().unwrap_or("<unnamed>");
                let qs = Questions::from_defs(
                    case["questions"]
                        .as_object()
                        .ok_or_else(|| anyhow!("case {name}: missing questions"))?,
                )?;
                let t0 = Instant::now();
                let ours = sess.predict(&case["state"], &qs)?;
                let dt = t0.elapsed();
                match compare_answers(&case["expected"]["answers"], &ours["answers"]) {
                    Ok((worst, bad)) => {
                        worst_all = worst_all.max(worst);
                        if bad.is_empty() {
                            println!(
                                "PASS {name} (worst |d|={worst:.2e}, {:.0} ms)",
                                dt.as_secs_f64() * 1e3
                            );
                        } else {
                            n_bad += 1;
                            println!("FAIL {name}:");
                            for b in bad {
                                println!("   {b}");
                            }
                        }
                    }
                    Err(e) => {
                        n_bad += 1;
                        println!("ERROR {name}: {e:#}");
                    }
                }
            }
            println!(
                "worst |d| across {} cases: {worst_all:.2e} (tol {TOL})",
                cases.len()
            );
            if n_bad > 0 {
                bail!("{n_bad} case(s) failed");
            }
            println!("validation OK");
        }
        Cmd::Bench { questions, n } => {
            let qs = load_questions(&questions)?;
            let state = json!({
                "subject": "Invoice #4471 is attached",
                "body": "Hi team, please find attached the invoice for our March order. The total is $4,120.00 and it looks like the PO number is missing. Can you confirm the due date? Thanks, Priya",
                "from": "priya@example.com",
            });
            for _ in 0..2 {
                sess.predict(&state, &qs)?;
            }
            let mut times = Vec::with_capacity(n);
            for _ in 0..n {
                let t0 = Instant::now();
                sess.predict(&state, &qs)?;
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{} questions in one batch, {} runs: p50 {:.1} ms, p95 {:.1} ms",
                qs.len(),
                n,
                times[n / 2],
                times[(n as f64 * 0.95) as usize % n]
            );
        }
    }
    Ok(())
}

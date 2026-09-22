//! laya-cli: playground CLI running laya's DecisionModel through ONNX Runtime.

mod email;
mod jsonw;
mod post;
mod rt;
mod seq;

use anyhow::{anyhow, bail, Context, Result};
use std::io::Read;
use clap::{Parser, Subcommand};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

const TOL: f64 = 1e-3;

#[derive(Parser)]
#[command(name = "laya-cli", about = "laya via ONNX Runtime (playground)")]
struct Cli {
    /// dir with laya-en.onnx, meta.json and laya-en/tokenizer/
    #[arg(long, default_value = "cli/model")]
    model_dir: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Classify one state: text file, `-` for stdin, or inline JSON object/list
    Classify {
        #[arg(long)]
        state_file: Option<String>,
        #[arg(short, long, default_value = "cli/presets/email.json")]
        questions: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Build an email state (laya's email_state: subject/body/from, cleaned) and classify
    Email {
        #[arg(long)]
        subject: String,
        #[arg(long)]
        sender: Option<String>,
        #[arg(long)]
        body_file: Option<String>,
        #[arg(short, long, default_value = "cli/presets/email.json")]
        questions: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Interactive: paste state, blank line runs, `:questions <file>` / `:f <file>` / `:q`
    Repl {
        #[arg(short, long, default_value = "cli/presets/email.json")]
        questions: PathBuf,
    },
    /// Compare against laya's Python outputs (export/golden/cases.json)
    Validate {
        #[arg(long, default_value = "cli/golden/cases.json")]
        golden: PathBuf,
    },
    /// Latency of one batched predict call
    Bench {
        #[arg(short, long, default_value = "cli/presets/email.json")]
        questions: PathBuf,
        #[arg(long)]
        state_file: Option<PathBuf>,
        #[arg(long, default_value_t = 20)]
        n: usize,
        #[arg(long)]
        json: bool,
    },
}

struct Sess {
    rt: rt::Rt,
    tok: Tokenizer,
    meta: seq::Meta,
    cleaner: email::EmailCleaner,
}

fn load(model_dir: &Path) -> Result<Sess> {
    rt::ensure_dylib()?;
    let meta: seq::Meta = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("meta.json"))
            .with_context(|| format!("reading {}", model_dir.join("meta.json").display()))?,
    )?;
    let tok = Tokenizer::from_file(model_dir.join("laya-en/tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("loading tokenizer: {e}"))?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let rt = rt::Rt::load(&model_dir.join("laya-en.onnx"), threads)?;
    let cleaner = email::EmailCleaner::new()?;
    Ok(Sess {
        rt,
        tok,
        meta,
        cleaner,
    })
}

/// Agent.system_one (agent.py:265-368) on ORT.
fn predict(sess: &mut Sess, state: &Value, questions: &Map<String, Value>) -> Result<Value> {
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
        if markers.len() > sess.meta.fixed_k {
            bail!(
                "question {qid:?}: {} options exceed fixed export K={}",
                markers.len(),
                sess.meta.fixed_k
            );
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
    if items.len() > sess.meta.fixed_b {
        bail!(
            "{} questions exceed fixed export B={}",
            items.len(),
            sess.meta.fixed_b
        );
    }

    // Batched fixed-shape forward (like Agent.system_one): all questions in
    // one call, padded with inert dummy rows to the graph's (B, L, K).
    let batch = seq::collate(
        &items,
        sess.meta.pad_id,
        sess.meta.cls_id,
        sess.meta.fixed_b,
        sess.meta.fixed_l,
        sess.meta.fixed_k,
    );
    let out = sess.rt.run(&batch)?;

    let lstride = out.logits.len() / batch.b;
    let astride = out.act.len() / batch.b;
    let mut answers = Map::new();
    let mut n_tokens = 0u64;
    for (r, it) in items.iter().enumerate() {
        let row = &out.logits[r * lstride..][..lstride];
        let actr = &out.act[r * astride..][..astride];
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
        "model": "laya-ort",
        "answers": answers,
        "usage": {"input_tokens": n_tokens, "output_tokens": 0},
    }))
}

fn read_state_source(src: &str) -> Result<Value> {
    let text = if src == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        std::fs::read_to_string(src)
            .with_context(|| format!("reading state file {src}"))?
    };
    // JSON object/list -> dict/list state; otherwise raw text state
    match serde_json::from_str::<Value>(&text) {
        Ok(v @ Value::Object(_)) | Ok(v @ Value::Array(_)) => Ok(v),
        _ => Ok(Value::String(text)),
    }
}

fn load_questions(path: &Path) -> Result<Map<String, Value>> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )?;
    v.as_object().cloned().ok_or_else(|| {
        anyhow!("{} must be an object of question-id -> question-def", path.display())
    })
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
    let usage = result["usage"]["input_tokens"].as_u64().unwrap_or(0);
    println!("   ({usage} input tokens)");
}

fn compare_answers(expected: &Value, ours: &Value) -> Result<(f64, Vec<String>)> {
    let exp = expected.as_object().ok_or_else(|| anyhow!("expected answers not an object"))?;
    let got = ours.as_object().ok_or_else(|| anyhow!("our answers not an object"))?;
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
    let mut sess = load(&cli.model_dir)?;
    match cli.cmd {
        Cmd::Classify {
            state_file,
            questions,
            json: as_json,
        } => {
            let state = read_state_source(
                state_file.as_deref().ok_or_else(|| anyhow!("--state-file required ('-' for stdin)"))?,
            )?;
            let qs = load_questions(&questions)?;
            let t0 = Instant::now();
            let result = predict(&mut sess, &state, &qs)?;
            let dt = t0.elapsed();
            if as_json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_human(&result);
                println!("   latency: {:.1} ms", dt.as_secs_f64() * 1e3);
            }
        }
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
                    std::io::stdin().read_to_string(&mut b)?;
                    b
                }
                Some(f) => std::fs::read_to_string(f)?,
            };
            let state = sess.cleaner.email_state(&subject, &body, sender.as_deref());
            let qs = load_questions(&questions)?;
            let t0 = Instant::now();
            let result = predict(&mut sess, &state, &qs)?;
            let dt = t0.elapsed();
            if as_json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_human(&result);
                println!("   latency: {:.1} ms", dt.as_secs_f64() * 1e3);
            }
        }
        Cmd::Repl { questions } => {
            let mut qs = load_questions(&questions)?;
            let mut buffer = String::new();
            println!("laya repl — paste state, blank line to run, :questions <file>, :f <file>, :q to quit");
            loop {
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line)? == 0 {
                    break; // EOF
                }
                let line = line.trim_end_matches(['\n', '\r']);
                match line {
                    ":q" => break,
                    _ if line.starts_with(":questions ") => {
                        let p = line.trim_start_matches(":questions ").trim();
                        qs = load_questions(Path::new(p))?;
                        println!("questions: {} entries from {p}", qs.len());
                        continue;
                    }
                    _ if line.starts_with(":f ") => {
                        let p = line.trim_start_matches(":f ").trim();
                        buffer = std::fs::read_to_string(p)
                            .with_context(|| format!("reading {p}"))?;
                        continue;
                    }
                    "" => {
                        if buffer.trim().is_empty() {
                            continue;
                        }
                        let state = Value::String(std::mem::take(&mut buffer));
                        let qs_now = qs.clone();
                        let t0 = Instant::now();
                        match predict(&mut sess, &state, &qs_now) {
                            Ok(r) => {
                                print_human(&r);
                                println!("   latency: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
                            }
                            Err(e) => eprintln!("error: {e:#}"),
                        }
                    }
                    _ => {
                        if !buffer.is_empty() {
                            buffer.push('\n');
                        }
                        buffer.push_str(line);
                    }
                }
            }
        }
        Cmd::Validate { golden } => {
            let cases: Vec<Value> =
                serde_json::from_str(&std::fs::read_to_string(&golden)?)
                    .with_context(|| format!("reading {}", golden.display()))?;
            let mut worst_all = 0f64;
            let mut n_bad = 0usize;
            for case in &cases {
                let name = case["name"].as_str().unwrap_or("<unnamed>");
                let qs = case["questions"]
                    .as_object()
                    .ok_or_else(|| anyhow!("case {name}: missing questions"))?;
                let t0 = Instant::now();
                let ours = predict(&mut sess, &case["state"], qs)?;
                let dt = t0.elapsed();
                match compare_answers(&case["expected"]["answers"], &ours["answers"]) {
                    Ok((worst, bad)) => {
                        worst_all = worst_all.max(worst);
                        if bad.is_empty() {
                            println!("PASS {name} (worst |d|={worst:.2e}, {:.0} ms)", dt.as_secs_f64() * 1e3);
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
            println!("worst |d| across {cases_len} cases: {worst_all:.2e} (tol {TOL})", cases_len = cases.len());
            if n_bad > 0 {
                bail!("{n_bad} case(s) failed");
            }
            println!("validation OK");
        }
        Cmd::Bench {
            questions,
            state_file,
            n,
            json: as_json,
        } => {
            let qs = load_questions(&questions)?;
            let state = match state_file.as_deref() {
                Some(p) => read_state_source(p.to_str().unwrap_or("-"))?,
                None => json!({
                    "subject": "Invoice #4471 is attached",
                    "body": "Hi team, please find attached the invoice for our March order. The total is $4,120.00 and it looks like the PO number is missing. Can you confirm the due date? Thanks, Priya",
                    "from": "priya@example.com",
                }),
            };
            for _ in 0..2 {
                predict(&mut sess, &state, &qs)?;
            }
            let mut times = Vec::with_capacity(n);
            for _ in 0..n {
                let t0 = Instant::now();
                predict(&mut sess, &state, &qs)?;
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = times[n / 2];
            let p95 = times[(n as f64 * 0.95) as usize % n];
            if as_json {
                println!("{}", serde_json::to_string_pretty(&json!({
                    "n": n, "questions": qs.len(), "p50_ms": p50, "p95_ms": p95,
                }))?);
            } else {
                println!(
                    "{} questions in one batch, {} runs: p50 {:.1} ms, p95 {:.1} ms",
                    qs.len(),
                    n,
                    p50,
                    p95
                );
            }
        }
    }
    Ok(())
}
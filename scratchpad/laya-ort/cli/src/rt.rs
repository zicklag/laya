//! Isolated ONNX Runtime surface. All ort-crate API usage lives here so
//! rc-version API drift is fixed in one file.

use anyhow::{anyhow, Context, Result};
use ort::{inputs, session::Session, value::Tensor};
use std::path::Path;

use crate::seq::Batch;

pub struct Rt {
    session: Session,
}

pub struct RunOut {
    pub logits: Vec<f32>,
    pub act: Vec<f32>,
}

/// ort's load-dynamic mode reads ORT_DYLIB_PATH; if unset, probe the pixi env
/// (CONDA_PREFIX) for the onnxruntime wheel's bundled shared library.
pub fn ensure_dylib() -> Result<()> {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return Ok(());
    }
    let prefix = std::env::var("CONDA_PREFIX").context(
        "ORT_DYLIB_PATH unset and CONDA_PREFIX not set: run via `pixi run`, \
         or set ORT_DYLIB_PATH to libonnxruntime.so",
    )?;
    let capi = Path::new(&prefix).join("lib/python3.12/site-packages/onnxruntime/capi");
    let mut candidates = vec![capi.join("libonnxruntime.so")];
    if let Ok(rd) = std::fs::read_dir(&capi) {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("libonnxruntime.so.") {
                candidates.push(capi.join(e.file_name()));
            }
        }
    }
    for c in candidates {
        if c.exists() {
            std::env::set_var("ORT_DYLIB_PATH", &c);
            return Ok(());
        }
    }
    Err(anyhow!(
        "libonnxruntime.so not found under {} — set ORT_DYLIB_PATH explicitly",
        capi.display()
    ))
}

impl Rt {
    pub fn load(model_path: &Path, threads: usize) -> Result<Self> {
        let builder = Session::builder().map_err(|e| anyhow!("ort session builder: {e}"))?;
        let mut builder = builder
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("ort thread config: {e}"))?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("ort session load: {e}"))
            .with_context(|| format!("loading {}", model_path.display()))?;
        Ok(Self { session })
    }

    pub fn run(&mut self, b: &Batch) -> Result<RunOut> {
        let outputs = self
            .session
            .run(inputs![
                "input_ids" => Tensor::from_array((vec![b.b as i64, b.l as i64], b.input_ids.clone()))?,
                "attention_mask" => Tensor::from_array((vec![b.b as i64, b.l as i64], b.attention_mask.clone()))?,
                "marker_pos" => Tensor::from_array((vec![b.b as i64, b.k as i64], b.marker_pos.clone()))?,
                "marker_mask" => Tensor::from_array((vec![b.b as i64, b.k as i64], b.marker_mask.clone()))?,
                "qtype" => Tensor::from_array((vec![b.b as i64], b.qtype.clone()))?,
            ])
            .map_err(|e| anyhow!("ort inference failed: {e}"))?;

        let logits = extract(&outputs, "choice_logits")?;
        let act = extract(&outputs, "act_logits")?;
        Ok(RunOut { logits, act })
    }
}

fn extract(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<Vec<f32>> {
    let v = outputs
        .get(name)
        .ok_or_else(|| anyhow!("missing output {name}"))?;
    let (_, data) = v
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("extracting {name}: {e}"))?;
    Ok(data.to_vec())
}
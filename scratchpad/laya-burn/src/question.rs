//! Typed builders for laya questions (`choice` / `score` / `noul`) and the
//! ordered [`Questions`] set consumed by
//! [`Session::predict`](crate::session::Session::predict).
//!
//! Builders produce the same internal `Q` as laya's JSON question defs; the
//! raw-def path ([`Question::from_def`] / [`Questions::from_defs`]) shares the
//! `seq::to_internal` parser, so both routes render byte-identically to
//! laya's Python.

use anyhow::{anyhow, Context, Result};
use serde_json::{Map, Value};

use crate::jsonw;
use crate::seq::{self, Q};

/// One question: an id plus the internal def the sequence builder consumes.
#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub q: Q,
}

impl Question {
    /// `choice`: pick one of >= 2 labeled options.
    pub fn choice(id: impl Into<String>) -> ChoiceBuilder {
        ChoiceBuilder {
            id: id.into(),
            instructions: None,
            options: Vec::new(),
        }
    }

    /// `score`: expected value over >= 2 ordered levels.
    pub fn score(id: impl Into<String>) -> ScoreBuilder {
        ScoreBuilder {
            id: id.into(),
            instructions: None,
            levels: Vec::new(),
        }
    }

    /// `noul`: binary true/false over one statement.
    pub fn noul(id: impl Into<String>) -> NoulBuilder {
        NoulBuilder {
            id: id.into(),
            instructions: None,
            true_desc: None,
            false_desc: None,
        }
    }

    /// Parse one laya question def (`{"type", "instructions", "criteria"}`).
    pub fn from_def(id: impl Into<String>, def: &Value) -> Result<Self> {
        Ok(Self {
            id: id.into(),
            q: seq::to_internal(def)?,
        })
    }
}

/// Builder for a `choice` question. Options render in insertion order; a
/// `null` or empty-string criterion renders as the bare label.
#[derive(Debug, Clone)]
pub struct ChoiceBuilder {
    id: String,
    instructions: Option<String>,
    options: Vec<(String, Value)>,
}

impl ChoiceBuilder {
    /// Instructions shown as `choice question: <ins>` in the prompt head.
    pub fn instructions(mut self, ins: impl Into<String>) -> Self {
        self.instructions = Some(ins.into());
        self
    }

    /// Instructions as an arbitrary JSON value, rendered like a non-string
    /// `instructions` def via laya's Python-JSON dump.
    pub fn instructions_json(mut self, ins: &Value) -> Self {
        self.instructions = Some(ins_to_string(ins));
        self
    }

    /// Add a plain option (no criterion text).
    pub fn option(mut self, label: impl Into<String>) -> Self {
        self.options.push((label.into(), Value::Null));
        self
    }

    /// Add an option with criterion text. Strings render verbatim; every
    /// other JSON value via laya's Python-JSON dump.
    pub fn option_with(mut self, label: impl Into<String>, criterion: impl Into<Value>) -> Self {
        self.options.push((label.into(), criterion.into()));
        self
    }

    pub fn build(self) -> Result<Question> {
        if self.options.len() < 2 {
            return Err(anyhow!(
                "choice question {:?}: {} option(s); the shipped path requires >= 2",
                self.id,
                self.options.len()
            ));
        }
        let ins = require_instructions(&self.id, &self.instructions)?;
        let mut crit = Map::new();
        for (label, v) in self.options {
            crit.insert(label, v);
        }
        Ok(Question {
            id: self.id,
            q: Q {
                t: "choice".into(),
                ins,
                crit: Some(Value::Object(crit)),
            },
        })
    }
}

/// Builder for a `score` question: the answer is the expected value over the
/// ordered levels (call order = level index).
#[derive(Debug, Clone)]
pub struct ScoreBuilder {
    id: String,
    instructions: Option<String>,
    levels: Vec<Value>,
}

impl ScoreBuilder {
    /// Instructions shown as `score question: <ins>` in the prompt head.
    pub fn instructions(mut self, ins: impl Into<String>) -> Self {
        self.instructions = Some(ins.into());
        self
    }

    /// Instructions as an arbitrary JSON value, rendered like a non-string
    /// `instructions` def via laya's Python-JSON dump.
    pub fn instructions_json(mut self, ins: &Value) -> Self {
        self.instructions = Some(ins_to_string(ins));
        self
    }

    /// Add one level description (level 0, 1, ... in call order).
    pub fn level(mut self, desc: impl Into<Value>) -> Self {
        self.levels.push(desc.into());
        self
    }

    pub fn build(self) -> Result<Question> {
        if self.levels.len() < 2 {
            return Err(anyhow!(
                "score question {:?}: {} level(s); the shipped path requires >= 2",
                self.id,
                self.levels.len()
            ));
        }
        let ins = require_instructions(&self.id, &self.instructions)?;
        Ok(Question {
            id: self.id,
            q: Q {
                t: "score".into(),
                ins,
                crit: Some(Value::Array(self.levels)),
            },
        })
    }
}

/// Builder for a `noul` question: two options, "false" then "true", with
/// laya's default descriptions unless overridden.
#[derive(Debug, Clone)]
pub struct NoulBuilder {
    id: String,
    instructions: Option<String>,
    true_desc: Option<Value>,
    false_desc: Option<Value>,
}

impl NoulBuilder {
    /// Instructions shown as `noul question: <ins>` in the prompt head.
    pub fn instructions(mut self, ins: impl Into<String>) -> Self {
        self.instructions = Some(ins.into());
        self
    }

    /// Instructions as an arbitrary JSON value, rendered like a non-string
    /// `instructions` def via laya's Python-JSON dump.
    pub fn instructions_json(mut self, ins: &Value) -> Self {
        self.instructions = Some(ins_to_string(ins));
        self
    }

    /// Description for the "true" option
    /// (default: "yes, the statement holds").
    pub fn true_desc(mut self, desc: impl Into<Value>) -> Self {
        self.true_desc = Some(desc.into());
        self
    }

    /// Description for the "false" option
    /// (default: "no, the statement does not hold").
    pub fn false_desc(mut self, desc: impl Into<Value>) -> Self {
        self.false_desc = Some(desc.into());
        self
    }

    pub fn build(self) -> Result<Question> {
        let ins = require_instructions(&self.id, &self.instructions)?;
        let crit = if self.true_desc.is_none() && self.false_desc.is_none() {
            None
        } else {
            let mut m = Map::new();
            if let Some(t) = self.true_desc {
                m.insert("true".into(), t);
            }
            if let Some(f) = self.false_desc {
                m.insert("false".into(), f);
            }
            Some(Value::Object(m))
        };
        Ok(Question {
            id: self.id,
            q: Q {
                t: "noul".into(),
                ins,
                crit,
            },
        })
    }
}

/// Ordered set of questions — insertion order defines batch order and the
/// answer key order.
#[derive(Debug, Clone, Default)]
pub struct Questions {
    items: Vec<Question>,
}

impl Questions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a question; ids must be unique (answers are keyed by id).
    pub fn add(&mut self, question: Question) -> Result<()> {
        if self.items.iter().any(|q| q.id == question.id) {
            return Err(anyhow!("duplicate question id {:?}", question.id));
        }
        self.items.push(question);
        Ok(())
    }

    /// Parse a laya question-def map (`qid -> def`), preserving key order.
    pub fn from_defs(defs: &Map<String, Value>) -> Result<Self> {
        let mut qs = Self::default();
        for (id, def) in defs {
            let question =
                Question::from_def(id.clone(), def).with_context(|| format!("question {id:?}"))?;
            qs.items.push(question);
        }
        Ok(qs)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Question> {
        self.items.iter()
    }
}

fn require_instructions(id: &str, ins: &Option<String>) -> Result<String> {
    ins.clone()
        .ok_or_else(|| anyhow!("question {id:?}: instructions required"))
}

/// Mirror `seq::to_internal`: strings pass through, everything else is dumped
/// with laya's Python-JSON formatting.
fn ins_to_string(ins: &Value) -> String {
    match ins {
        Value::String(s) => s.clone(),
        other => jsonw::dump(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn choice_builder_matches_json_def() {
        let def = json!({
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "invoices", "other": null, "sales": ""}
        });
        let from_json = Question::from_def("category", &def).unwrap();
        let built = Question::choice("category")
            .instructions("Which team?")
            .option_with("billing", "invoices")
            .option("other")
            .option_with("sales", "")
            .build()
            .unwrap();
        assert_eq!(from_json.q, built.q);
    }

    #[test]
    fn score_builder_matches_json_def() {
        let def = json!({
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["no pressure", "soon", "deadline"]
        });
        let from_json = Question::from_def("urgency", &def).unwrap();
        let built = Question::score("urgency")
            .instructions("How urgent?")
            .level("no pressure")
            .level("soon")
            .level("deadline")
            .build()
            .unwrap();
        assert_eq!(from_json.q, built.q);
    }

    #[test]
    fn noul_builder_matches_json_def() {
        let def = json!({"type": "noul", "instructions": "Is this spam?"});
        let from_json = Question::from_def("is_spam", &def).unwrap();
        let built = Question::noul("is_spam")
            .instructions("Is this spam?")
            .build()
            .unwrap();
        assert_eq!(from_json.q, built.q);

        let def = json!({
            "type": "noul",
            "instructions": "Phishing?",
            "criteria": {"true": "phishing", "false": "legit"}
        });
        let from_json = Question::from_def("is_phishing", &def).unwrap();
        let built = Question::noul("is_phishing")
            .instructions("Phishing?")
            .true_desc("phishing")
            .false_desc("legit")
            .build()
            .unwrap();
        assert_eq!(from_json.q, built.q);
    }

    #[test]
    fn non_string_instructions_dump_like_the_def_parser() {
        let def = json!({"type": "noul", "instructions": ["a", "b"]});
        let from_json = Question::from_def("x", &def).unwrap();
        let built = Question::noul("x")
            .instructions_json(&json!(["a", "b"]))
            .build()
            .unwrap();
        assert_eq!(from_json.q, built.q);
    }

    #[test]
    fn builders_enforce_minimums() {
        assert!(Question::choice("x")
            .instructions("i")
            .option("a")
            .build()
            .is_err());
        assert!(Question::score("x")
            .instructions("i")
            .level("only")
            .build()
            .is_err());
        assert!(Question::choice("x")
            .option("a")
            .option("b")
            .build()
            .is_err());
        assert!(Question::noul("x").build().is_err());
    }

    #[test]
    fn questions_reject_duplicate_ids() {
        let mut qs = Questions::new();
        qs.add(Question::noul("a").instructions("i").build().unwrap())
            .unwrap();
        assert!(qs
            .add(Question::noul("a").instructions("i").build().unwrap())
            .is_err());
    }
}

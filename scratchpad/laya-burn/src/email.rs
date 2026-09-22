//! Port of laya/email.py: email body cleaning + email_state construction.

use anyhow::{anyhow, Result};
use fancy_regex::Regex;
use serde_json::{json, Map, Value};

pub struct EmailCleaner {
    quote_headers: Vec<Regex>,
    signature_markers: Vec<Regex>,
    disclaimer: Regex,
    sentence: Regex,
    blank_split: Regex,
    spaces: Regex,
}

fn rx(pat: &str) -> Result<Regex> {
    Regex::new(pat).map_err(|e| anyhow!("bad regex {pat:?}: {e}"))
}

fn is_match(re: &Regex, s: &str) -> bool {
    re.is_match(s).unwrap_or(false)
}

impl EmailCleaner {
    pub fn new() -> Result<Self> {
        Ok(Self {
            // _QUOTE_HEADERS: all re.I
            quote_headers: vec![
                rx(r"(?i)^\s*On .{0,300}wrote:\s*$")?,
                rx(r"(?i)^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}")?,
                rx(r"(?i)^\s*_{8,}\s*$")?,
                rx(r"(?i)^\s*From:\s.+$")?,
            ],
            // _SIGNATURE_MARKERS: [0] has no re.I in Python; keep it exact
            signature_markers: vec![
                rx(r"^\s*--\s*$")?,
                rx(r"(?i)^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$")?,
                rx(r"(?i)^\s*sent from my (iphone|android|mobile|ipad)")?,
            ],
            disclaimer: rx(
                r"(?i)(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|if you (have )?received this (e-?mail|message) in error)",
            )?,
            sentence: rx(r"(?<=[.!?])\s+")?,
            blank_split: rx(r"\n\s*\n")?,
            spaces: rx(r"[ \t]+")?,
        })
    }

    /// _strip_disclaimer (email.py:24-34).
    fn strip_disclaimer(&self, paragraph: &str) -> String {
        if !is_match(&self.disclaimer, paragraph) {
            return paragraph.to_string();
        }
        let parts: Vec<String> = self
            .sentence
            .split(paragraph)
            .filter_map(|r| r.ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        parts
            .into_iter()
            .filter(|s| !is_match(&self.disclaimer, s))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// clean_email_body (email.py:37-55).
    pub fn clean_body(&self, body: &str, max_chars: usize) -> String {
        let text = body
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace("\\n", "\n");
        let mut lines: Vec<&str> = Vec::new();
        for line in text.split('\n') {
            if !lines.is_empty() && self.quote_headers.iter().any(|re| is_match(re, line)) {
                break;
            }
            if line.trim_start().starts_with('>') {
                continue;
            }
            lines.push(line.trim_end());
        }
        let mut cut = lines.len();
        if !lines.is_empty() {
            // range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines))
            let a = ((lines.len() as f64) * 0.6) as i64;
            let b = lines.len() as i64 - 8;
            let start = 1i64.max(a.min(b)) as usize;
            for i in start..lines.len() {
                if lines[i].trim().chars().count() <= 40
                    && self.signature_markers.iter().any(|re| is_match(re, lines[i]))
                {
                    cut = i;
                    break;
                }
            }
        }
        let joined = lines[..cut].join("\n");
        let paragraphs: Vec<String> = self
            .blank_split
            .split(&joined)
            .filter_map(|r| r.ok())
            .map(|p| self.strip_disclaimer(p))
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        let re_joined = paragraphs.join("\n\n");
        let text = self.spaces.replace_all(&re_joined, " ").to_string();
        text.chars().take(max_chars).collect()
    }

    /// email_state (email.py:58-67) — insertion order: subject, body, from.
    pub fn email_state(&self, subject: &str, body: &str, sender: Option<&str>) -> Value {
        let mut m = Map::new();
        m.insert("subject".into(), json!(subject.trim()));
        m.insert("body".into(), json!(self.clean_body(body, 3000)));
        if let Some(s) = sender {
            if !s.is_empty() {
                m.insert("from".into(), json!(s));
            }
        }
        Value::Object(m)
    }
}
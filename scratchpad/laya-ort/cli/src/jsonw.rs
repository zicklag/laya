//! Python `json.dumps(v, ensure_ascii=False)` byte-compatible writer.
//!
//! laya's prompts embed state/criterion text through Python's json module with
//! default separators (", " and ": "). serde_json's compact output differs
//! ("a,b" vs "a, b"), which changes tokenization and therefore probabilities —
//! so we emit Python's exact format here.

use serde_json::Value;

pub fn dump(v: &Value) -> String {
    let mut s = String::new();
    write_val(v, &mut s);
    s
}

fn write_val(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_num(n, out),
        Value::String(s) => out.push_str(&py_str(s)),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_val(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&py_str(k));
                out.push_str(": ");
                write_val(x, out);
            }
            out.push('}');
        }
    }
}

fn write_num(n: &serde_json::Number, out: &mut String) {
    if let Some(i) = n.as_i64() {
        out.push_str(&i.to_string());
    } else if let Some(u) = n.as_u64() {
        out.push_str(&u.to_string());
    } else {
        out.push_str(&py_float(n.as_f64().unwrap_or(f64::NAN)));
    }
}

fn py_str(s: &str) -> String {
    serde_json::to_string(s).expect("string escaping cannot fail")
}

/// Python `repr`-style float: integral values print with ".0"; magnitudes
/// outside [1e-4, 1e16) switch to e-notation with a signed 2+ digit exponent.
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let abs = f.abs();
    if (1e-4..1e16).contains(&abs) {
        if f == f.trunc() {
            return format!("{:.1}", f);
        }
        return format!("{f}");
    }
    let s = format!("{f:e}");
    match s.split_once('e') {
        Some((m, e)) => {
            let exp: i32 = e.parse().unwrap_or(0);
            if exp < 0 {
                format!("{m}e-{:02}", -exp)
            } else {
                format!("{m}e+{:02}", exp)
            }
        }
        None => s,
    }
}
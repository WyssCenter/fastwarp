#![allow(clippy::unwrap_used, clippy::expect_used)]
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

const F16_MAX: f64 = 65504.0;

fn shader_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("shaders")
}

fn wgsl_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(wgsl_files(&path));
        } else if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
            files.push(path);
        }
    }
    files
}

fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut depth = 0usize;
    while i < bytes.len() {
        if depth == 0 && bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if bytes[i..].starts_with(b"/*") {
            if depth == 0 {
                out.push(' ');
            }
            depth += 1;
            i += 2;
        } else if depth > 0 && bytes[i..].starts_with(b"*/") {
            depth -= 1;
            i += 2;
        } else {
            if depth == 0 || bytes[i] == b'\n' {
                out.push(bytes[i] as char);
            }
            i += 1;
        }
    }
    out
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn parse_hex_float(body: &str) -> Option<f64> {
    let (mantissa, exponent) = match body.find(['p', 'P']) {
        Some(p) => (&body[..p], body[p + 1..].parse::<i32>().ok()?),
        None => (body, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut value = 0f64;
    for c in int_part.chars().chain(frac_part.chars()) {
        value = value * 16.0 + f64::from(c.to_digit(16)?);
    }
    let shift = exponent - 4 * i32::try_from(frac_part.len()).ok()?;
    Some(value * 2f64.powi(shift))
}

struct Literal {
    line: usize,
    text: String,
    value: f64,
    half: bool,
}

fn float_literals(src: &str) -> Vec<Literal> {
    let code = strip_comments(src);
    let b = code.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        let starts =
            c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit());
        if !starts || (i > 0 && (is_ident(b[i - 1]) || b[i - 1] == b'.')) {
            i += 1;
            continue;
        }
        let start = i;
        let hex = b[i..].starts_with(b"0x") || b[i..].starts_with(b"0X");
        if hex {
            i += 2;
            while i < b.len() && (b[i].is_ascii_hexdigit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'p' || b[i] == b'P') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
        } else {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
        }
        let body = &code[start..i];
        let suffix = if i < b.len() && matches!(b[i], b'f' | b'h' | b'i' | b'u') {
            i += 1;
            b[i - 1]
        } else {
            0
        };
        let is_float = if hex {
            body.contains('.') || body.contains(['p', 'P']) || suffix == b'f' || suffix == b'h'
        } else {
            body.contains('.') || body.contains(['e', 'E']) || suffix == b'f' || suffix == b'h'
        };
        if is_float {
            let value = if hex {
                parse_hex_float(&body[2..])
            } else {
                body.parse::<f64>().ok()
            }
            .unwrap_or(f64::INFINITY);
            found.push(Literal {
                line,
                text: code[start..i].to_string(),
                value,
                half: suffix == b'h',
            });
        }
    }
    found
}

fn out_of_range(src: &str) -> Vec<String> {
    float_literals(src)
        .into_iter()
        .filter(|l| {
            let max = if l.half { F16_MAX } else { f64::from(f32::MAX) };
            !l.value.is_finite() || l.value.abs() > max
        })
        .map(|l| format!("line {}: {}", l.line, l.text))
        .collect()
}

#[test]
fn scanner_flags_literals_browsers_reject() {
    assert_eq!(
        out_of_range("const BIG: f32 = 3.40282347e38;"),
        vec!["line 1: 3.40282347e38".to_string()]
    );
    assert_eq!(out_of_range("let x = 1e39f;").len(), 1);
    assert_eq!(out_of_range("let h = 65520.0h;").len(), 1);
    assert!(out_of_range("const BIG: f32 = 0x1.fffffep+127f;").is_empty());
    assert!(out_of_range("let a = 3.4028234663852886e38; // 1e99").is_empty());
    assert!(out_of_range("let v = vec3<f32>(0.5, 1e-3, .25); let i = 4294967295u;").is_empty());
    assert_eq!(out_of_range("let x = a/**/1e39;").len(), 1);
    let hex = float_literals("const M: f32 = 0x1.fffffep+127f;");
    assert_eq!(hex.len(), 1);
    assert_eq!(hex[0].value, f64::from(f32::MAX));
}

#[test]
fn shader_float_literals_fit_their_type() {
    let mut files = 0;
    let mut literals = 0;
    let mut bad = Vec::new();
    for path in wgsl_files(&shader_dir()) {
        files += 1;
        let src = fs::read_to_string(&path).unwrap();
        literals += float_literals(&src).len();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        bad.extend(
            out_of_range(&src)
                .into_iter()
                .map(|m| format!("{name} {m}")),
        );
    }
    assert!(
        files >= 40,
        "only {files} shader files found in {:?}",
        shader_dir()
    );
    assert!(
        literals >= 50,
        "only {literals} float literals found; the scanner may be broken"
    );
    assert!(
        bad.is_empty(),
        "float literals outside the f32 / f16 range (browsers reject them when the shader is compiled; write the f32 maximum as 0x1.fffffep+127f): {bad:#?}"
    );
}

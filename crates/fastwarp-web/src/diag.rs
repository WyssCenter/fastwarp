// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! The engine's part of the browser demo's diagnostic report: where a panic happened, what the
//! last failure was, and how far a failed run got. Nothing here holds image data, file names or
//! paths of the user's computer.

use fastwarp_gpu::SelfTestRecord;
use serde_json::{json, Value};

/// The source file of a panic without the directories of the machine that built the module:
/// `wgpu-30.0.1/src/backend/webgpu.rs` for a crate from the registry, `crates/fastwarp-web/src/web.rs`
/// for a workspace file, `library/core/src/option.rs` for the standard library.
pub fn short_path(file: &str) -> String {
    let unix = file.replace('\\', "/");
    if let Some(i) = unix.find("/registry/src/") {
        let rest = &unix[i + "/registry/src/".len()..];
        return rest.split_once('/').map_or(rest, |(_, r)| r).to_string();
    }
    for marker in ["crates/", "library/"] {
        if let Some(i) = unix.find(marker) {
            return unix[i..].to_string();
        }
    }
    unix.rsplit('/').next().unwrap_or_default().to_string()
}

/// A panic as the report shows it: the message and `file:line:column`.
pub fn panic_json(message: &str, location: Option<(&str, u32, u32)>) -> Value {
    json!({
        "message": message,
        "location": location.map(|(file, line, column)| format!("{}:{line}:{column}", short_path(file))),
    })
}

/// What `last_failure` returns: the last panic and the size of the last allocation that failed.
pub fn failure_json(panic: Option<Value>, failed_alloc_bytes: usize) -> Value {
    json!({
        "panic": panic,
        "failed_alloc_bytes": (failed_alloc_bytes > 0).then_some(failed_alloc_bytes),
    })
}

/// The structured part of the error of a failed run: its kind, the last progress step, the plan
/// (grids and planned GPU memory) and the state of the GPU device when it failed.
pub fn run_failure_json(
    kind: &str,
    stage: Option<Value>,
    plan: Option<Value>,
    gpu: Option<Value>,
) -> Value {
    json!({ "kind": kind, "stage": stage, "plan": plan, "gpu": gpu })
}

/// The known-answer check of the device, as the report shows it.
pub fn selftest_json(record: &SelfTestRecord) -> Value {
    json!({
        "attempts": record.attempts,
        "syn_attempts": record.syn_attempts,
        "mi_attempts": record.mi_attempts,
        "lines": record.lines,
        "first_attempt_failures": record.first_attempt_failures,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn short_paths_drop_the_build_machine() {
        assert_eq!(
            short_path("/home/docs/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/wgpu-30.0.1/src/backend/webgpu.rs"),
            "wgpu-30.0.1/src/backend/webgpu.rs"
        );
        assert_eq!(
            short_path("C:\\Users\\someone\\.cargo\\registry\\src\\index.crates.io-1\\serde-1.0.0\\src\\de.rs"),
            "serde-1.0.0/src/de.rs"
        );
        assert_eq!(
            short_path("crates/fastwarp-web/src/web.rs"),
            "crates/fastwarp-web/src/web.rs"
        );
        assert_eq!(
            short_path("/home/runner/work/fastwarp/fastwarp/crates/fastwarp-reg/src/syn.rs"),
            "crates/fastwarp-reg/src/syn.rs"
        );
        assert_eq!(
            short_path("/rustc/abc123/library/core/src/option.rs"),
            "library/core/src/option.rs"
        );
        assert_eq!(short_path("/somewhere/else/main.rs"), "main.rs");
    }

    #[test]
    fn failures_and_runs_have_their_fields() {
        let panic = panic_json("boom", Some(("crates/fastwarp-web/src/web.rs", 7, 9)));
        assert_eq!(panic["location"], "crates/fastwarp-web/src/web.rs:7:9");
        let none = failure_json(None, 0);
        assert!(none["panic"].is_null());
        assert!(none["failed_alloc_bytes"].is_null());
        let some = failure_json(Some(panic), 1 << 30);
        assert_eq!(some["panic"]["message"], "boom");
        assert_eq!(some["failed_alloc_bytes"], 1u64 << 30);
        let run = run_failure_json("GpuMemory", Some(json!({"phase": "syn"})), None, None);
        assert_eq!(run["kind"], "GpuMemory");
        assert_eq!(run["stage"]["phase"], "syn");
        assert!(run["plan"].is_null());
        let record = SelfTestRecord {
            attempts: 1,
            syn_attempts: 1,
            mi_attempts: 1,
            lines: vec!["ok".into()],
            first_attempt_failures: Vec::new(),
        };
        let v = selftest_json(&record);
        assert_eq!(v["attempts"], 1);
        assert_eq!(v["lines"][0], "ok");
    }
}

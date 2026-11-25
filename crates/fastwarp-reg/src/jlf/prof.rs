//! Phase timings of a fusion, printed to stderr when `FASTWARP_JLF_PROFILE` is set. Not a stable API.

use std::sync::{Mutex, OnceLock};

static TIMES: Mutex<Vec<(&'static str, f64, u64)>> = Mutex::new(Vec::new());
static HASH: Mutex<u64> = Mutex::new(0xcbf2_9ce4_8422_2325);

/// Mixes `values` into the hash of the search offsets.
pub fn hash(values: &[u32]) {
    if !enabled() {
        return;
    }
    if let Ok(mut h) = HASH.lock() {
        for v in values {
            for b in v.to_le_bytes() {
                *h ^= u64::from(b);
                *h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
}

/// Whether `FASTWARP_JLF_PROFILE` is set.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FASTWARP_JLF_PROFILE").is_some())
}

/// Adds `value` to the phase `name`.
pub fn add(name: &'static str, value: f64) {
    if !enabled() {
        return;
    }
    if let Ok(mut t) = TIMES.lock() {
        if let Some(e) = t.iter_mut().find(|e| e.0 == name) {
            e.1 += value;
            e.2 += 1;
        } else {
            t.push((name, value, 1));
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// A running phase, added when dropped.
pub struct Span(&'static str, std::time::Instant);

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl Drop for Span {
    fn drop(&mut self) {
        add(self.0, self.1.elapsed().as_secs_f64());
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// Starts the phase `name`.
pub fn span(name: &'static str) -> Span {
    Span(name, std::time::Instant::now())
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// A running phase (not timed on wasm32).
pub struct Span;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// Starts the phase `name`.
pub fn span(_name: &'static str) -> Span {
    Span
}

/// Prints and clears the phases.
pub fn report(header: &str) {
    let taken = TIMES.lock().map(|mut t| std::mem::take(&mut *t)).unwrap_or_default();
    let h = HASH
        .lock()
        .map(|mut h| std::mem::replace(&mut *h, 0xcbf2_9ce4_8422_2325))
        .unwrap_or(0);
    if !enabled() {
        return;
    }
    let mut s = format!("JLFPROF {header} offsets_hash={h:016x}");
    for (name, value, count) in taken {
        s.push_str(&format!(" {name}={value:.4}/{count}"));
    }
    eprintln!("{s}");
}

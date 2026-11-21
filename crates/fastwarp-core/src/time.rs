// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch(std::time::Instant);

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl Stopwatch {
    pub fn start() -> Self {
        Stopwatch(std::time::Instant::now())
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.0.elapsed()
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch(f64);

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn now_ms() -> f64 {
    let global = js_sys::global();
    let perf = js_sys::Reflect::get(&global, &"performance".into()).ok();
    let now = perf
        .as_ref()
        .and_then(|p| js_sys::Reflect::get(p, &"now".into()).ok())
        .filter(|f| f.is_function());
    match (perf, now) {
        (Some(p), Some(f)) => js_sys::Function::from(f)
            .call0(&p)
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or_else(js_sys::Date::now),
        _ => js_sys::Date::now(),
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl Stopwatch {
    pub fn start() -> Self {
        Stopwatch(now_ms())
    }

    pub fn elapsed(&self) -> std::time::Duration {
        let ms = (now_ms() - self.0).max(0.0);
        std::time::Duration::from_secs_f64(ms / 1000.0)
    }
}

#[cfg(all(test, not(all(target_arch = "wasm32", target_os = "unknown"))))]
mod tests {
    use super::Stopwatch;

    #[test]
    fn elapsed_is_monotonic() {
        let t = Stopwatch::start();
        let a = t.elapsed();
        let b = t.elapsed();
        assert!(b >= a);
    }
}

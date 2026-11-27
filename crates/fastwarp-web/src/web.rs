// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fastwarp_gpu::device::required_limits;
use fastwarp_gpu::{BackendSelect, WgpuBackend};
use fastwarp_reg::Backend;
use js_sys::{Function, Uint8Array};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use crate::diag::{failure_json, panic_json, run_failure_json, selftest_json};
use crate::options::parse_options;
use crate::progress::Progress;
use crate::run::{
    classify_text, default_plan, run_job, short_plan, weigh_lost, DeviceLimits, Job, LostEvidence,
    RunError,
};
use crate::viewer::{Layer, Viewer};

static CANCEL: AtomicBool = AtomicBool::new(false);
static RUNNING: AtomicBool = AtomicBool::new(false);
static FAILED_ALLOC: AtomicUsize = AtomicUsize::new(0);

struct RecordingAllocator;

fn note_failure(block: *mut u8, size: usize) -> *mut u8 {
    if block.is_null() {
        FAILED_ALLOC.store(size, Ordering::Relaxed);
    }
    block
}

// SAFETY: every method hands its arguments unchanged to `System`, which upholds the `GlobalAlloc` contract; the wrapper only records the size of a request that `System` could not satisfy.
unsafe impl GlobalAlloc for RecordingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc` for `layout`, which `System::alloc` shares.
        note_failure(unsafe { System.alloc(layout) }, layout.size())
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc_zeroed` for `layout`, which `System::alloc_zeroed` shares.
        note_failure(unsafe { System.alloc_zeroed(layout) }, layout.size())
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        // SAFETY: `block` was returned by this allocator, that is by `System`, for `layout`, as the caller guarantees.
        unsafe { System.dealloc(block, layout) }
    }

    unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `block` was returned by `System` for `layout` and the caller upholds the contract of `GlobalAlloc::realloc` for `new_size`.
        note_failure(unsafe { System.realloc(block, layout, new_size) }, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: RecordingAllocator = RecordingAllocator;

thread_local! {
    static BACKEND: RefCell<Option<Rc<WgpuBackend>>> = const { RefCell::new(None) };
    static PROGRESS: RefCell<Option<Function>> = const { RefCell::new(None) };
    static VIEWER: RefCell<Option<Viewer>> = const { RefCell::new(None) };
    static PANIC: RefCell<Option<Value>> = const { RefCell::new(None) };
    static STAGE: RefCell<Option<Value>> = const { RefCell::new(None) };
    static PLAN: RefCell<Option<Value>> = const { RefCell::new(None) };
    static GPU_STATE: RefCell<Option<Value>> = const { RefCell::new(None) };
    static FAULT: RefCell<Option<(String, String)>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

fn keep(cell: &'static std::thread::LocalKey<RefCell<Option<Value>>>, value: Option<Value>) {
    cell.with(|c| {
        if let Ok(mut slot) = c.try_borrow_mut() {
            *slot = value;
        }
    });
}

fn kept(cell: &'static std::thread::LocalKey<RefCell<Option<Value>>>) -> Option<Value> {
    cell.with(|c| c.try_borrow().ok().and_then(|v| v.clone()))
}

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        let payload = info.payload();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic without a message".into());
        let location = info.location().map(|l| (l.file(), l.line(), l.column()));
        let panic = panic_json(&message, location);
        let place = panic["location"]
            .as_str()
            .unwrap_or("an unknown place")
            .to_string();
        keep(&PANIC, Some(panic));
        console_error(&format!("FastWarp: internal error: {message} (at {place})"));
    }));
}

fn js_error(e: &RunError) -> JsValue {
    let error = js_sys::Error::new(&e.to_string());
    error.set_name(e.kind());
    error.into()
}

fn js_run_error(e: &RunError) -> JsValue {
    let error = js_error(e);
    let diagnostics = run_failure_json(e.kind(), kept(&STAGE), kept(&PLAN), kept(&GPU_STATE));
    let _ = js_sys::Reflect::set(
        &error,
        &JsValue::from_str("diagnostics"),
        &JsValue::from_str(&diagnostics.to_string()),
    );
    error
}

fn exhaust_memory() -> ! {
    let mut held: Vec<Vec<u8>> = Vec::new();
    loop {
        let mut chunk: Vec<u8> = Vec::with_capacity(1 << 30);
        chunk.push(1);
        held.push(std::hint::black_box(chunk));
    }
}

fn inject_fault(phase: &str) {
    let fault = FAULT.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(_, at)| at == phase)
            .map(|(kind, _)| kind.clone())
    });
    match fault.as_deref() {
        Some("panic") => panic!("FastWarp fault injection: a panic during {phase}"),
        Some("alloc") => exhaust_memory(),
        _ => {}
    }
}

fn report(p: &Progress) {
    let step = p.to_json();
    keep(&STAGE, Some(step.clone()));
    PROGRESS.with(|cell| {
        if let Some(f) = cell.borrow().as_ref() {
            let _ = f.call1(&JsValue::NULL, &JsValue::from_str(&step.to_string()));
        }
    });
    inject_fault(p.phase);
}

fn forget_backend() {
    BACKEND.with(|cell| cell.borrow_mut().take());
}

async fn backend() -> Result<Rc<WgpuBackend>, RunError> {
    if let Some(b) = BACKEND.with(|cell| cell.borrow().clone()) {
        if !b.context().is_lost() {
            return Ok(b);
        }
        forget_backend();
    }
    FAILED_ALLOC.store(0, Ordering::Relaxed);
    let select = BackendSelect {
        allow_software: true,
        ..BackendSelect::default()
    };
    let b = WgpuBackend::new_async(&select)
        .await
        .map_err(|e| RunError::Registration(format!("WebGPU could not be started: {e}")))?;
    let b = Rc::new(b);
    BACKEND.with(|cell| *cell.borrow_mut() = Some(Rc::clone(&b)));
    Ok(b)
}

macro_rules! limits_json {
    ($limits:expr) => {{
        let l = $limits;
        json!({
            "max_bind_groups": l.max_bind_groups,
            "max_bindings_per_bind_group": l.max_bindings_per_bind_group,
            "max_buffer_size": l.max_buffer_size,
            "max_storage_buffer_binding_size": l.max_storage_buffer_binding_size,
            "max_uniform_buffer_binding_size": l.max_uniform_buffer_binding_size,
            "max_storage_buffers_per_shader_stage": l.max_storage_buffers_per_shader_stage,
            "max_uniform_buffers_per_shader_stage": l.max_uniform_buffers_per_shader_stage,
            "max_dynamic_storage_buffers_per_pipeline_layout": l.max_dynamic_storage_buffers_per_pipeline_layout,
            "min_storage_buffer_offset_alignment": l.min_storage_buffer_offset_alignment,
            "min_uniform_buffer_offset_alignment": l.min_uniform_buffer_offset_alignment,
            "max_compute_workgroup_storage_size": l.max_compute_workgroup_storage_size,
            "max_compute_invocations_per_workgroup": l.max_compute_invocations_per_workgroup,
            "max_compute_workgroup_size_x": l.max_compute_workgroup_size_x,
            "max_compute_workgroup_size_y": l.max_compute_workgroup_size_y,
            "max_compute_workgroup_size_z": l.max_compute_workgroup_size_z,
            "max_compute_workgroups_per_dimension": l.max_compute_workgroups_per_dimension,
        })
    }};
}

fn adapter_json(b: &WgpuBackend) -> Value {
    let ctx = b.context();
    let (i, c) = (&ctx.info, &ctx.caps);
    json!({
        "name": i.name,
        "vendor": i.vendor,
        "device": i.device,
        "device_type": i.device_type,
        "driver": i.driver,
        "driver_info": i.driver_info,
        "backend": i.backend,
        "software": i.is_software(),
        "limits": {
            "max_buffer_size": c.max_buffer_size,
            "max_storage_buffer_binding_size": c.max_storage_buffer_binding_size,
            "max_compute_workgroup_storage_size": c.max_compute_workgroup_storage_size,
            "max_compute_invocations_per_workgroup": c.max_compute_invocations_per_workgroup,
            "max_compute_workgroups_per_dimension": c.max_compute_workgroups_per_dimension,
            "max_storage_buffers_per_shader_stage": c.max_storage_buffers_per_shader_stage,
        },
        "limits_adapter": limits_json!(ctx.adapter.limits()),
        "limits_requested": limits_json!(required_limits(&ctx.adapter)),
        "limits_granted": limits_json!(ctx.device.limits()),
        "features": c.feature_names,
        "selftest": selftest_json(&b.selftest_record()),
    })
}

fn gpu_state(b: &WgpuBackend, out_of_memory_errors: u32) -> Value {
    let ctx = b.context();
    json!({
        "peak_allocated_bytes": b.peak_allocated_bytes(),
        "allocated_bytes": b.allocated_bytes(),
        "lost": ctx.is_lost(),
        "out_of_memory_errors": out_of_memory_errors,
        "pending_errors": ctx.check_errors().err().map(|e| e.to_string()),
    })
}

fn local_time() -> [u16; 6] {
    let d = js_sys::Date::new_0();
    [
        d.get_full_year() as u16,
        d.get_month() as u16 + 1,
        d.get_date() as u16,
        d.get_hours() as u16,
        d.get_minutes() as u16,
        d.get_seconds() as u16,
    ]
}

/// Opens the GPU (adapter, device, kernel self-test) and describes it as JSON: the adapter, the
/// limits the adapter offers, those FastWarp requested and those the device was granted, and the
/// self-test record.
#[wasm_bindgen]
pub async fn gpu_info() -> Result<String, JsValue> {
    let b = backend().await.map_err(|e| js_error(&e))?;
    let mut v = adapter_json(&b);
    v["fastwarp_version"] = json!(env!("CARGO_PKG_VERSION"));
    Ok(v.to_string())
}

/// The last failure as JSON: the last panic (`message`, `location`) and the size of the last
/// allocation that failed (`failed_alloc_bytes`). It can be called after the module stopped
/// on an internal error.
#[wasm_bindgen]
pub fn last_failure() -> String {
    failure_json(kept(&PANIC), FAILED_ALLOC.load(Ordering::Relaxed)).to_string()
}

/// Fault injection for tests of the page: `kind` `"panic"` panics, `"alloc"` allocates memory
/// until an allocation fails, at the first progress step of `phase`. Nothing happens unless the
/// page calls this, which it does only when its address asks for a fault.
#[wasm_bindgen]
pub fn debug_fault(kind: String, phase: String) -> Result<(), JsValue> {
    if !matches!(kind.as_str(), "panic" | "alloc") {
        return Err(js_error(&RunError::Invalid(format!(
            "unknown fault {kind:?}"
        ))));
    }
    FAULT.with(|c| *c.borrow_mut() = Some((kind, phase)));
    Ok(())
}

/// Stops a running registration at its next iteration.
#[wasm_bindgen]
pub fn cancel() {
    CANCEL.store(true, Ordering::SeqCst);
}

/// Describes the images kept for the result viewer of the last run as JSON (`dims`, `spacing`,
/// `before`, `windows`), or nothing when there are none. The copies are freed when the next run
/// starts.
#[wasm_bindgen]
pub fn view_info() -> Option<String> {
    VIEWER.with(|cell| cell.borrow().as_ref().map(|v| v.info().to_string()))
}

/// One slice of the result viewer: `axis` 0 sagittal, 1 coronal, 2 axial; `layer` 0 fixed, 1 warped,
/// 2 before (centre of mass only), 3 edges of the fixed image. The bytes are `width x height`
/// grey values from the top row down; the sizes follow from `view_info`.
#[wasm_bindgen]
pub fn view_slice(axis: u32, index: u32, layer: u32) -> Result<Uint8Array, JsValue> {
    let slice = VIEWER.with(|cell| {
        let cell = cell.borrow();
        let viewer = cell.as_ref()?;
        viewer.slice(axis as usize, index as usize, Layer::from_code(layer)?)
    });
    match slice {
        Some(s) => Ok(Uint8Array::from(s.data.as_slice())),
        None => Err(js_error(&RunError::Invalid(
            "there is no such slice".into(),
        ))),
    }
}

/// Registers `moving` onto `fixed` (NIfTI bytes, `.nii` or `.nii.gz`) and returns the zip
/// archive of the outputs. `options_json`: `{"grid": "1mm" | "2mm", "plan": "default" | "short"}`. `on_progress` is called
/// with a JSON text per progress step. A failed run's error carries `diagnostics`, a JSON text with
/// its kind, the last progress step, the plan and the state of the GPU device.
#[wasm_bindgen]
pub async fn register(
    fixed: Uint8Array,
    moving: Uint8Array,
    options_json: String,
    on_progress: Function,
) -> Result<Uint8Array, JsValue> {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return Err(js_error(&RunError::Invalid(
            "a registration is already running".into(),
        )));
    }
    CANCEL.store(false, Ordering::SeqCst);
    VIEWER.with(|cell| cell.borrow_mut().take());
    PROGRESS.with(|cell| *cell.borrow_mut() = Some(on_progress));
    keep(&STAGE, None);
    keep(&PLAN, None);
    keep(&GPU_STATE, None);
    FAILED_ALLOC.store(0, Ordering::Relaxed);
    let fixed = fixed.to_vec();
    let moving = moving.to_vec();
    let result = run(fixed, moving, &options_json).await;
    PROGRESS.with(|cell| cell.borrow_mut().take());
    RUNNING.store(false, Ordering::SeqCst);
    result.map_err(|e| js_run_error(&e))
}

async fn run(fixed: Vec<u8>, moving: Vec<u8>, options_json: &str) -> Result<Uint8Array, RunError> {
    let options = parse_options(options_json).map_err(RunError::Invalid)?;
    report(&Progress::phase("gpu"));
    let backend = backend().await?;
    let oom_before = backend.context().out_of_memory_errors();
    backend.reset_peak_allocated_bytes();
    let caps = &backend.context().caps;
    let job = Job {
        fixed,
        moving,
        options,
        plan: if options.short_plan {
            short_plan()
        } else {
            default_plan()
        },
        limits: DeviceLimits {
            max_buffer_size: caps.max_buffer_size,
            max_storage_buffer_binding_size: caps.max_storage_buffer_binding_size,
        },
        mi_unavailable: backend.selftest_record().mi_dense_failure,
        adapter: adapter_json(&backend),
        zip_time: Some(local_time()),
        on_plan: Some(Box::new(|plan: &Value| keep(&PLAN, Some(plan.clone())))),
    };
    let peak = || Some(backend.peak_allocated_bytes());
    let reporter: fn(&Progress) = report;
    let out = run_job(&*backend, &*backend, job, &CANCEL, &reporter, &peak).await;
    let lost = backend.context().is_lost();
    match out {
        Ok(finished) => {
            backend.release_idle_buffers();
            VIEWER.with(|cell| *cell.borrow_mut() = Some(finished.viewer));
            Ok(Uint8Array::from(finished.archive.as_slice()))
        }
        Err(e) => {
            let oom_errors = backend
                .context()
                .out_of_memory_errors()
                .saturating_sub(oom_before);
            keep(&GPU_STATE, Some(gpu_state(&backend, oom_errors)));
            let evidence = LostEvidence {
                out_of_memory_errors: oom_errors,
                failed_alloc_bytes: FAILED_ALLOC.load(Ordering::Relaxed),
                planned_gpu_bytes: kept(&PLAN)
                    .and_then(|p| p["planned_gpu_bytes"].as_u64())
                    .unwrap_or(0),
                peak_gpu_bytes: backend.peak_allocated_bytes(),
                max_buffer_size: backend.context().caps.max_buffer_size,
            };
            drop(backend);
            if lost || matches!(e, RunError::GpuMemory(_) | RunError::DeviceLost { .. }) {
                forget_backend();
            } else if let Some(b) = BACKEND.with(|cell| cell.borrow().clone()) {
                b.release_idle_buffers();
            }
            let e = match e {
                RunError::Registration(text) if lost => {
                    classify_text(&format!("device lost: {text}"), options.grid)
                }
                other => other,
            };
            Err(weigh_lost(e, options.grid, &evidence))
        }
    }
}

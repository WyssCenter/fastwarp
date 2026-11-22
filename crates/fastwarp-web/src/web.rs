// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

use fastwarp_gpu::{BackendSelect, WgpuBackend};
use fastwarp_reg::Backend;
use js_sys::{Function, Uint8Array};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use crate::options::parse_options;
use crate::progress::Progress;
use crate::run::{classify_text, default_plan, run_job, DeviceLimits, Job, RunError};
use crate::viewer::{Layer, Viewer};

static CANCEL: AtomicBool = AtomicBool::new(false);
static RUNNING: AtomicBool = AtomicBool::new(false);

thread_local! {
    static BACKEND: RefCell<Option<Rc<WgpuBackend>>> = const { RefCell::new(None) };
    static PROGRESS: RefCell<Option<Function>> = const { RefCell::new(None) };
    static VIEWER: RefCell<Option<Viewer>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        console_error(&format!("FastWarp: internal error: {info}"));
    }));
}

fn js_error(e: &RunError) -> JsValue {
    let error = js_sys::Error::new(&e.to_string());
    error.set_name(e.kind());
    error.into()
}

fn report(p: &Progress) {
    PROGRESS.with(|cell| {
        if let Some(f) = cell.borrow().as_ref() {
            let _ = f.call1(&JsValue::NULL, &JsValue::from_str(&p.to_json().to_string()));
        }
    });
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
        "features": c.feature_names,
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

/// Opens the GPU (adapter, device, kernel self-test) and describes it as JSON.
#[wasm_bindgen]
pub async fn gpu_info() -> Result<String, JsValue> {
    let b = backend().await.map_err(|e| js_error(&e))?;
    let mut v = adapter_json(&b);
    v["fastwarp_version"] = json!(env!("CARGO_PKG_VERSION"));
    Ok(v.to_string())
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
/// archive of the outputs. `options_json`: `{"grid": "1mm" | "2mm"}`. `on_progress` is called
/// with a JSON text per progress step.
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
    let fixed = fixed.to_vec();
    let moving = moving.to_vec();
    let result = run(fixed, moving, &options_json).await;
    PROGRESS.with(|cell| cell.borrow_mut().take());
    RUNNING.store(false, Ordering::SeqCst);
    result.map_err(|e| js_error(&e))
}

async fn run(fixed: Vec<u8>, moving: Vec<u8>, options_json: &str) -> Result<Uint8Array, RunError> {
    let options = parse_options(options_json).map_err(RunError::Invalid)?;
    report(&Progress::phase("gpu"));
    let backend = backend().await?;
    let caps = &backend.context().caps;
    let job = Job {
        fixed,
        moving,
        options,
        plan: default_plan(),
        limits: DeviceLimits {
            max_buffer_size: caps.max_buffer_size,
            max_storage_buffer_binding_size: caps.max_storage_buffer_binding_size,
        },
        adapter: adapter_json(&backend),
        zip_time: Some(local_time()),
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
            drop(backend);
            if lost || matches!(e, RunError::GpuMemory(_)) {
                forget_backend();
            } else if let Some(b) = BACKEND.with(|cell| cell.borrow().clone()) {
                b.release_idle_buffers();
            }
            match e {
                RunError::Registration(text) if lost => {
                    Err(classify_text(&format!("device lost: {text}"), options.grid))
                }
                other => Err(other),
            }
        }
    }
}

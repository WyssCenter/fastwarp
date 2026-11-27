// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! One registration from NIfTI bytes to a zip archive of the ANTs-compatible outputs.

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use fastwarp_core::Stopwatch;
use fastwarp_gpu::{
    plan_capacity_linear, plan_capacity_syn_stages, CapacityCaps, CapacityError, SynStageCapacity,
};
use fastwarp_reg::apply::{apply_chain, Interpolation};
use fastwarp_reg::linalg::IDENTITY3;
use fastwarp_reg::linear::LinearEngine;
use fastwarp_reg::recipe::SynRecipe;
use fastwarp_reg::register::LinearPlan;
use fastwarp_reg::stages::{
    register_stages_async, visit_stage_outputs, OutputLayout, StageContent, StagePlan, StagesOutput,
};
use fastwarp_reg::syn::SynOptions;
use fastwarp_reg::{Backend, Grid, RegError, TransformChain, TransformElem, Volume};
use serde_json::{json, Value};

use crate::options::{GridChoice, Options};
use crate::progress::{Observed, Progress, Tracker};
use crate::viewer::{quantize, Viewer};
use crate::zip::ZipWriter;

const BEFORE_SLAB: usize = 16;

pub const README: &str = include_str!("README.txt");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RunError {
    Invalid(String),
    Limit(String),
    GpuMemory(String),
    /// The browser lost the GPU device without a sign that memory ran short: the page's message
    /// and the browser's text.
    DeviceLost {
        message: String,
        raw: String,
    },
    Cancelled,
    Registration(String),
}

impl RunError {
    pub fn kind(&self) -> &'static str {
        match self {
            RunError::Invalid(_) => "InvalidInput",
            RunError::Limit(_) => "GpuLimit",
            RunError::GpuMemory(_) => "GpuMemory",
            RunError::DeviceLost { .. } => "GpuDeviceLost",
            RunError::Cancelled => "Cancelled",
            RunError::Registration(_) => "RegistrationError",
        }
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Cancelled => f.write_str("the registration was cancelled"),
            RunError::DeviceLost { message, .. } => f.write_str(message),
            RunError::Invalid(m)
            | RunError::Limit(m)
            | RunError::GpuMemory(m)
            | RunError::Registration(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RunError {}

/// The default recipe of `fastwarp register` (centre of mass, Rigid, Affine, SyN) for one
/// channel, with the smoothed images of the linear stages not cached between stages, which only
/// saves time and holds several image pairs in memory.
pub fn default_plan() -> StagePlan {
    let mut plan = StagePlan::cohort(&LinearPlan::cohort(), &SynRecipe::cohort_channels(1));
    for stage in &mut plan.linear {
        stage.cache_smoothed_images = false;
    }
    plan
}

/// The default recipe with two levels per stage and few iterations, so that a registration runs in
/// seconds: for tests, and for the page's test parameter `?plan=short`. Its results are not meant for
/// use.
pub fn short_plan() -> StagePlan {
    let mut plan = default_plan();
    for s in &mut plan.linear {
        s.iterations = vec![12, 8];
        s.shrink_factors = vec![2, 1];
        s.smoothing_sigmas = vec![1.0, 0.0];
    }
    for s in &mut plan.syn {
        s.iterations = vec![6, 4];
        s.shrink_factors = vec![2, 1];
        s.smoothing_sigmas = vec![1.0, 0.0];
    }
    plan
}

/// Voxels per axis of the isotropic registration grid at `mm`, by the rule of
/// `resample_isotropic`.
pub fn iso_dims(grid: &Grid, mm: f64) -> [usize; 3] {
    let (size, spacing) = (grid.size(), grid.spacing());
    let mut out = [1usize; 3];
    for a in 0..3 {
        if (spacing[a] - mm).abs() < 1e-6 {
            out[a] = size[a];
        } else {
            let last = size[a].saturating_sub(1) as f64;
            out[a] = (((last * spacing[a] / mm) + 1e-9).floor() as usize + 1).max(1);
        }
    }
    out
}

fn bytes_text(b: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    if b as f64 >= 1024.0 * MIB {
        format!("{:.1} GiB", b as f64 / (1024.0 * MIB))
    } else {
        format!("{:.0} MiB", (b as f64 / MIB).max(1.0))
    }
}

fn smaller_grid_hint(grid: GridChoice) -> &'static str {
    match grid {
        GridChoice::OneMm => "Choose the 2 mm grid.",
        GridChoice::TwoMm => "Use smaller images, or run FastWarp on a computer with a larger GPU.",
    }
}

/// Checks the buffers of every stage against the device limits before any work, and returns the
/// GPU memory the run is planned to need (the larger of the linear and the SyN stages).
pub fn check_limits(
    plan: &StagePlan,
    fixed: &Grid,
    moving: &Grid,
    grid: GridChoice,
    limits: &DeviceLimits,
) -> Result<u64, RunError> {
    let mm = grid.mm();
    let dims = iso_dims(fixed, mm);
    let moving_dims = iso_dims(moving, mm);
    let caps = CapacityCaps {
        max_storage_buffer_binding_size: limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size),
        memory_budget: u64::MAX,
    };
    let stages: Vec<SynStageCapacity<'_>> = plan
        .syn
        .iter()
        .map(|s| SynStageCapacity {
            shrink_factors: &s.shrink_factors,
            mask_dims: &[],
            mattes: s.uses(fastwarp_reg::recipe::SynMetric::Mattes),
        })
        .collect();
    let refuse = |need: u64| {
        let (name, limit) = if need > limits.max_storage_buffer_binding_size {
            (
                "maxStorageBufferBindingSize",
                limits.max_storage_buffer_binding_size,
            )
        } else {
            ("maxBufferSize", limits.max_buffer_size)
        };
        RunError::Limit(format!(
            "The {} mm grid (fixed image {}x{}x{}, moving image {}x{}x{} voxels) needs a GPU \
             buffer of {}, but this browser's WebGPU limit {name} is {}. {}",
            mm,
            dims[0],
            dims[1],
            dims[2],
            moving_dims[0],
            moving_dims[1],
            moving_dims[2],
            bytes_text(need),
            bytes_text(limit),
            smaller_grid_hint(grid)
        ))
    };
    let map = |e: CapacityError| match e {
        CapacityError::Binding { need, .. } | CapacityError::LinearBinding { need, .. } => {
            refuse(need)
        }
        other => RunError::Limit(other.to_string()),
    };
    let mut planned = 0;
    if !plan.linear.is_empty() {
        planned = plan_capacity_linear(dims, moving_dims, &plan.linear, &caps)
            .map_err(map)?
            .level_total_bytes;
    }
    if !stages.is_empty() {
        let syn = plan_capacity_syn_stages(dims, moving_dims, 1, &stages, &caps).map_err(map)?;
        planned = planned.max(syn.level_total_bytes);
    }
    Ok(planned)
}

/// Refuses before any work a plan whose SyN stages use Mattes MI on a device whose dense MI
/// kernels failed their known-answer check (`failure`, the reason).
pub fn check_mi_available(plan: &StagePlan, failure: Option<&str>) -> Result<(), RunError> {
    let mattes = plan
        .syn
        .iter()
        .any(|s| s.uses(fastwarp_reg::recipe::SynMetric::Mattes));
    match failure {
        Some(why) if mattes => Err(RunError::Limit(format!(
            "This GPU cannot run SyN stages with Mattes mutual information: its dense MI kernels \
             failed the known-answer check ({why})."
        ))),
        _ => Ok(()),
    }
}

/// Turns an engine error into what the page shows; GPU memory and a lost device suggest the
/// 2 mm grid.
pub fn classify(e: RegError, grid: GridChoice) -> RunError {
    if matches!(e, RegError::Cancelled) {
        return RunError::Cancelled;
    }
    classify_text(&e.to_string(), grid)
}

pub fn classify_text(text: &str, grid: GridChoice) -> RunError {
    let lower = text.to_ascii_lowercase();
    let hint = match grid {
        GridChoice::OneMm => {
            "Try the 2 mm grid, and close other tabs and programs that use the GPU."
        }
        GridChoice::TwoMm => {
            "Close other tabs and programs that use the GPU, or use smaller images."
        }
    };
    if lower.contains("cancelled") {
        RunError::Cancelled
    } else if lower.contains("out of memory") || lower.contains("outofmemory") {
        RunError::GpuMemory(format!("Not enough GPU memory. {hint} ({text})"))
    } else if lower.contains("device lost") || lower.contains("devicelost") {
        let again = match grid {
            GridChoice::OneMm => {
                "Run again usually works; if it happens again on the 1 mm grid, try the 2 mm grid, \
                 whose steps are shorter."
            }
            GridChoice::TwoMm => "Run again usually works.",
        };
        RunError::DeviceLost {
            message: format!(
                "The browser or the GPU driver reset the GPU during the registration. This \
                 happens now and then on some phones and computers and does not mean that \
                 something is wrong with the images. {again} ({text})"
            ),
            raw: text.to_string(),
        }
    } else if lower.contains("max buffer size") || lower.contains("exceeds the max") {
        RunError::Limit(format!(
            "A GPU buffer is larger than this browser allows. {} ({text})",
            smaller_grid_hint(grid)
        ))
    } else {
        RunError::Registration(text.to_string())
    }
}

/// What a run knew about memory when its GPU device was lost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LostEvidence {
    /// Out-of-memory errors the device's error scopes caught during the run.
    pub out_of_memory_errors: u32,
    /// Size of the last allocation of the WebAssembly module that failed (0: none).
    pub failed_alloc_bytes: usize,
    /// GPU memory the run planned to use.
    pub planned_gpu_bytes: u64,
    /// GPU memory the run used at most.
    pub peak_gpu_bytes: u64,
    /// The device's `maxBufferSize`.
    pub max_buffer_size: u64,
}

impl LostEvidence {
    /// Whether a lost device points to memory: an out-of-memory error was caught, an allocation
    /// of the WebAssembly module failed, the run used at least half of the device's
    /// `maxBufferSize` or planned at least all of it. A browser tells a page no other figure of
    /// the GPU's memory, so the last two tests are a rough proxy.
    pub fn points_to_memory(&self) -> bool {
        let limit = self.max_buffer_size;
        self.out_of_memory_errors > 0
            || self.failed_alloc_bytes > 0
            || (limit > 0
                && (self.peak_gpu_bytes.saturating_mul(2) >= limit
                    || self.planned_gpu_bytes >= limit))
    }
}

/// Turns a lost device into a GPU memory error when `evidence` points to memory; every other error
/// is returned unchanged.
pub fn weigh_lost(e: RunError, grid: GridChoice, evidence: &LostEvidence) -> RunError {
    match e {
        RunError::DeviceLost { raw, .. } if evidence.points_to_memory() => {
            let hint = match grid {
                GridChoice::OneMm => {
                    "Try the 2 mm grid, and close other tabs and programs that use the GPU."
                }
                GridChoice::TwoMm => {
                    "Close other tabs and programs that use the GPU, or use smaller images."
                }
            };
            RunError::GpuMemory(format!(
                "The browser stopped the GPU device after the GPU ran short of memory. {hint} \
                 ({raw})"
            ))
        }
        other => other,
    }
}

/// Receives the plan of a run (see [`Job::on_plan`]).
pub type PlanNote = Box<dyn Fn(&Value)>;

/// What one run needs besides the device.
pub struct Job {
    pub fixed: Vec<u8>,
    pub moving: Vec<u8>,
    pub options: Options,
    pub plan: StagePlan,
    pub limits: DeviceLimits,
    /// Why the device's dense Mattes MI kernels failed their known-answer check, if they did.
    pub mi_unavailable: Option<String>,
    pub adapter: Value,
    pub zip_time: Option<[u16; 6]>,
    /// Called once the images are read and the limits are checked, with the grids and the planned
    /// GPU memory (`registration_grid`, `fixed`, `moving`, `planned_gpu_bytes`), so that a
    /// failure later in the run can still report them.
    pub on_plan: Option<PlanNote>,
}

fn grid_json(g: &Grid) -> Value {
    json!({ "size": g.size(), "spacing": g.spacing(), "origin": g.origin() })
}

fn plan_json(plan: &StagePlan) -> Value {
    json!({
        "initial": "centre of mass",
        "winsorize": plan.winsorize.map(|(lo, hi)| [lo, hi]),
        "linear": plan.linear.iter().map(|s| json!({
            "transform": format!("{:?}", s.kind).to_ascii_lowercase(),
            "gradient_step": s.gradient_step,
            "iterations": s.iterations,
            "shrink_factors": s.shrink_factors,
            "smoothing_sigmas": s.smoothing_sigmas,
            "metric": "MI",
            "bins": s.bins,
            "sampling_percentage": s.sampling_percentage,
        })).collect::<Vec<_>>(),
        "syn": plan.syn.iter().map(|s| json!({
            "gradient_step": s.gradient_step,
            "update_field_variance": s.update_field_variance,
            "total_field_variance": s.total_field_variance,
            "metric": "CC",
            "cc_radius": s.cc_radius,
            "iterations": s.iterations,
            "shrink_factors": s.shrink_factors,
            "smoothing_sigmas": s.smoothing_sigmas,
        })).collect::<Vec<_>>(),
    })
}

fn results_json(plan: &StagePlan, out: &StagesOutput) -> Value {
    let linear: Vec<Value> = plan
        .linear
        .iter()
        .zip(&out.linear)
        .map(|(r, s)| {
            json!({
                "transform": format!("{:?}", r.kind).to_ascii_lowercase(),
                "iterations": s.levels.iter().map(|l| l.cg.iterations).collect::<Vec<_>>(),
                "metric": s.final_metric(),
            })
        })
        .collect();
    let warnings: Vec<String> = plan
        .linear
        .iter()
        .zip(&out.linear)
        .enumerate()
        .flat_map(|(i, (r, s))| {
            s.warnings(&format!(
                "linear stage {} ({})",
                i + 1,
                format!("{:?}", r.kind).to_ascii_lowercase()
            ))
        })
        .collect();
    let syn: Vec<Value> = out
        .syn_levels
        .iter()
        .flatten()
        .map(|l| {
            json!({
                "level": l.level,
                "grid": l.grid_size,
                "iterations": l.iterations.len(),
                "converged": l.converged,
                "energy": l.iterations.last().map(|i| i.energy),
            })
        })
        .collect();
    json!({
        "linear_stages": linear,
        "syn_levels": syn,
        "warnings": warnings,
        "affine_parameters": out.total.parameters(),
        "affine_center": out.total.center,
    })
}

/// What a finished run hands back: the zip archive, and the 8-bit copies for the result viewer.
#[derive(Debug)]
pub struct Finished {
    pub archive: Vec<u8>,
    pub viewer: Viewer,
}

fn zip_err(e: std::io::Error) -> RunError {
    RunError::Registration(format!("writing the zip archive: {e}"))
}

/// Registers the moving image of `job` onto its fixed image with `job.plan` and returns the zip
/// archive: `0GenericAffine.mat`, `1Warp.nii.gz`, `1InverseWarp.nii.gz`, `Warped.nii.gz` (the
/// moving image on the fixed image's grid), `run.json` and `README.txt`, with the 8-bit copies of
/// the images for the result viewer. `peak_gpu_bytes` is asked after the registration, for
/// `run.json`.
pub async fn run_job<B: Backend, E: LinearEngine>(
    backend: &B,
    engine: &E,
    job: Job,
    cancel: &AtomicBool,
    report: &(dyn Fn(&Progress) + Sync),
    peak_gpu_bytes: &dyn Fn() -> Option<u64>,
) -> Result<Finished, RunError> {
    let total = Stopwatch::start();
    let Job {
        fixed,
        moving,
        options,
        plan,
        limits,
        mi_unavailable,
        adapter,
        zip_time,
        on_plan,
    } = job;
    let grid_choice = options.grid;
    let tracker = Tracker::new(&plan, report);
    tracker.phase("read");
    let t = Stopwatch::start();
    let fixed = Volume::from_nifti_bytes_f64(fixed).map_err(|e| {
        RunError::Invalid(format!("The fixed image is not a usable NIfTI image: {e}"))
    })?;
    let moving = Volume::from_nifti_bytes_f64(moving).map_err(|e| {
        RunError::Invalid(format!("The moving image is not a usable NIfTI image: {e}"))
    })?;
    let read_s = t.elapsed().as_secs_f64();
    let fixed_grid = *fixed.grid();
    let moving_grid = *moving.grid();
    let planned = check_limits(&plan, &fixed_grid, &moving_grid, grid_choice, &limits)?;
    if let Some(note) = &on_plan {
        note(&json!({
            "registration_grid": iso_dims(&fixed_grid, grid_choice.mm()),
            "fixed": fixed_grid.size(),
            "moving": moving_grid.size(),
            "planned_gpu_bytes": planned,
        }));
    }
    check_mi_available(&plan, mi_unavailable.as_deref())?;
    if cancel.load(Ordering::SeqCst) {
        return Err(RunError::Cancelled);
    }
    tracker.phase("prepare");
    let observed = Observed::new(backend, engine, &tracker);
    let fixed = vec![fixed];
    let moving = vec![moving];
    let options_syn = SynOptions {
        cancel: Some(cancel),
    };
    let mut out = register_stages_async(
        &observed,
        &observed,
        &plan,
        &fixed,
        &moving,
        Some(grid_choice.mm()),
        &options_syn,
    )
    .await
    .map_err(|e| classify(e, grid_choice))?;
    let viewer_t = Stopwatch::start();
    let mut viewer = Viewer::new(&fixed[0]);
    let mut viewer_s = viewer_t.elapsed().as_secs_f64();
    drop(fixed);
    let peak = peak_gpu_bytes();
    backend.release_idle_buffers();
    tracker.phase("warp");
    let t = Stopwatch::start();
    let placeholder = Grid::new([1, 1, 1], [1.0; 3], [0.0; 3], IDENTITY3)
        .map_err(|e| classify(e, grid_choice))?;
    out.warped = Volume::zeros(placeholder);
    let warped = apply_chain(
        &moving[0],
        &fixed_grid,
        &out.pair.forward,
        Interpolation::Linear,
        0.0,
    )
    .map_err(|e| classify(e, grid_choice))?;
    let warp_native_s = t.elapsed().as_secs_f64();
    let viewer_t = Stopwatch::start();
    viewer.set_warped(&warped);
    let start = TransformChain::from_queue(vec![TransformElem::Affine(out.initial)]);
    let window = viewer.warped_window();
    let size = fixed_grid.size();
    let mut before_bytes = Vec::with_capacity(fixed_grid.num_voxels());
    let mut z0 = 0;
    while z0 < size[2] {
        let slab = BEFORE_SLAB.min(size[2] - z0);
        let grid = Grid::new(
            [size[0], size[1], slab],
            fixed_grid.spacing(),
            fixed_grid.index_to_point([0.0, 0.0, z0 as f64]),
            fixed_grid.direction(),
        )
        .map_err(|e| classify(e, grid_choice))?;
        let part = apply_chain(&moving[0], &grid, &start, Interpolation::Linear, 0.0)
            .map_err(|e| classify(e, grid_choice))?;
        before_bytes.extend(quantize(part.data(), window));
        z0 += slab;
    }
    viewer.set_before_bytes(&before_bytes);
    drop(before_bytes);
    drop(moving);
    viewer_s += viewer_t.elapsed().as_secs_f64();
    tracker.phase("write");
    let t = Stopwatch::start();
    let mut zip = ZipWriter::new();
    if let Some([y, mo, d, h, mi, s]) = zip_time {
        zip.set_time(y, mo, d, h, mi, s);
    }
    let mut files: Vec<(String, u64)> = Vec::new();
    visit_stage_outputs(
        Path::new(""),
        &out,
        OutputLayout::Split,
        false,
        |path, content| {
            let name = path.to_string_lossy().into_owned();
            let io = |source| RegError::Io {
                path: path.clone(),
                source,
            };
            let mut entry = zip.begin(&name).map_err(io)?;
            let mut counted = Counted::new(&mut entry);
            content.write_to(&path, &mut counted)?;
            let size = counted.bytes;
            entry.finish().map_err(io)?;
            files.push((name, size));
            Ok(())
        },
    )
    .map_err(|e| classify(e, grid_choice))?;
    let warped_name = "Warped.nii.gz";
    {
        let mut entry = zip.begin(warped_name).map_err(zip_err)?;
        let mut counted = Counted::new(&mut entry);
        StageContent::Warped(&warped)
            .write_to(Path::new(warped_name), &mut counted)
            .map_err(|e| classify(e, grid_choice))?;
        let size = counted.bytes;
        entry.finish().map_err(zip_err)?;
        files.push((warped_name.to_string(), size));
    }
    drop(warped);
    let results = results_json(&plan, &out);
    let registration_grid = out.grid;
    let timings = out.timings;
    drop(out);
    let write_s = t.elapsed().as_secs_f64();
    let run = json!({
        "program": "FastWarp in the browser (fastwarp-web)",
        "fastwarp_version": env!("CARGO_PKG_VERSION"),
        "options": options.to_json(),
        "recipe": plan_json(&plan),
        "adapter": adapter,
        "device_limits": {
            "max_buffer_size": limits.max_buffer_size,
            "max_storage_buffer_binding_size": limits.max_storage_buffer_binding_size,
        },
        "fixed": grid_json(&fixed_grid),
        "moving": grid_json(&moving_grid),
        "registration_grid": grid_json(&registration_grid),
        "planned_gpu_bytes": planned,
        "peak_gpu_bytes": peak,
        "timings_s": {
            "read": read_s,
            "resample": timings.resample_s,
            "preprocess": timings.preprocess_s,
            "linear": timings.linear_s,
            "syn": timings.syn_s,
            "warp": timings.warp_s,
            "warp_to_fixed_grid": warp_native_s,
            "viewer": viewer_s,
            "write": write_s,
            "total": total.elapsed().as_secs_f64(),
        },
        "results": results,
        "files": files.iter().map(|(n, s)| json!({"name": n, "bytes": s})).collect::<Vec<_>>(),
    });
    let text =
        serde_json::to_string_pretty(&run).map_err(|e| RunError::Registration(e.to_string()))?;
    zip.add("run.json", text.as_bytes()).map_err(zip_err)?;
    zip.add("README.txt", README.as_bytes()).map_err(zip_err)?;
    let archive = zip.finish().map_err(zip_err)?;
    tracker.phase("done");
    Ok(Finished { archive, viewer })
}

struct Counted<W> {
    inner: W,
    bytes: u64,
}

impl<W> Counted<W> {
    fn new(inner: W) -> Self {
        Self { inner, bytes: 0 }
    }
}

impl<W: std::io::Write> std::io::Write for Counted<W> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(data)?;
        self.bytes += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::future::Future;
    use std::rc::Rc;
    use std::sync::Mutex;
    use std::task::{Context, Poll, Waker};

    use fastwarp_reg::linear::CpuLinearEngine;
    use fastwarp_reg::stages::{register_stages, stage_output_files};
    use fastwarp_reg::CpuBackend;

    use crate::viewer::Layer;
    use crate::zip::tests::read_zip;

    fn ready<T>(f: impl Future<Output = T>) -> T {
        let mut f = std::pin::pin!(f);
        match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("a CPU future was pending"),
        }
    }

    fn blob(n: usize, spacing: f64, shift: [f64; 3]) -> Volume<f32> {
        let g = Grid::new([n, n, n], [spacing; 3], [0.0; 3], IDENTITY3).unwrap();
        let c = (n as f64 - 1.0) * spacing / 2.0;
        let data = (0..g.num_voxels())
            .map(|lin| {
                let [i, j, k] = g.ijk(lin);
                let p = g.index_to_point([i as f64, j as f64, k as f64]);
                let d = |o: [f64; 3]| {
                    (p[0] - c - shift[0] - o[0]).powi(2)
                        + (p[1] - c - shift[1] - o[1]).powi(2)
                        + (p[2] - c - shift[2] - o[2]).powi(2)
                };
                let noise = ((lin * 2654435761) % 1000) as f64 / 20000.0;
                ((-d([0.0; 3]) / (2.0 * (0.22 * n as f64 * spacing).powi(2))).exp()
                    + 0.6 * (-d([6.0, -3.0, 0.0]) / 18.0).exp()
                    + noise) as f32
            })
            .collect();
        Volume::from_vec(g, data).unwrap()
    }

    fn small_plan() -> StagePlan {
        short_plan()
    }

    const LIMITS: DeviceLimits = DeviceLimits {
        max_buffer_size: 1 << 30,
        max_storage_buffer_binding_size: 1 << 27,
    };

    fn job(plan: StagePlan, grid: GridChoice) -> Job {
        Job {
            fixed: blob(32, 1.0, [0.0; 3]).to_nifti_bytes(true).unwrap(),
            moving: blob(32, 1.0, [3.0, -2.0, 1.0])
                .to_nifti_bytes(false)
                .unwrap(),
            options: Options {
                grid,
                ..Options::default()
            },
            plan,
            limits: LIMITS,
            mi_unavailable: None,
            adapter: json!({"name": "CPU reference"}),
            zip_time: Some([2026, 11, 22, 9, 30, 0]),
            on_plan: None,
        }
    }

    #[test]
    fn the_default_plan_is_the_command_line_default_without_the_cache() {
        let plan = default_plan();
        let cli = StagePlan::cohort(&LinearPlan::cohort(), &SynRecipe::cohort_channels(1));
        assert_eq!(plan.linear.len(), 2);
        assert_eq!(plan.syn, cli.syn);
        assert_eq!(plan.initial, cli.initial);
        assert_eq!(plan.winsorize, cli.winsorize);
        for (a, b) in plan.linear.iter().zip(&cli.linear) {
            assert!(!a.cache_smoothed_images);
            let mut a = a.clone();
            a.cache_smoothed_images = b.cache_smoothed_images;
            assert_eq!(&a, b);
        }
    }

    #[test]
    fn the_plan_is_noted_before_the_registration_starts() {
        let noted: Rc<RefCell<Option<Value>>> = Rc::default();
        let sink = Rc::clone(&noted);
        let mut j = job(small_plan(), GridChoice::TwoMm);
        j.on_plan = Some(Box::new(move |v: &Value| {
            *sink.borrow_mut() = Some(v.clone())
        }));
        let e = ready(run_job(
            &CpuBackend,
            &CpuLinearEngine,
            j,
            &AtomicBool::new(true),
            &|_: &Progress| {},
            &|| None,
        ))
        .unwrap_err();
        assert_eq!(e, RunError::Cancelled);
        let v = noted.borrow().clone().unwrap();
        assert_eq!(v["registration_grid"], json!([16, 16, 16]));
        assert_eq!(v["fixed"], json!([32, 32, 32]));
        assert_eq!(v["moving"], json!([32, 32, 32]));
        assert!(v["planned_gpu_bytes"].as_u64().unwrap() > 0);
    }

    #[test]
    fn a_run_writes_the_ants_files_of_the_engine_and_reports_progress() {
        let plan = small_plan();
        let seen: Mutex<Vec<Progress>> = Mutex::new(Vec::new());
        let report = |p: &Progress| seen.lock().unwrap().push(p.clone());
        let cancel = AtomicBool::new(false);
        let finished = ready(run_job(
            &CpuBackend,
            &CpuLinearEngine,
            job(plan.clone(), GridChoice::TwoMm),
            &cancel,
            &report,
            &|| Some(123),
        ))
        .unwrap();
        let viewer = finished.viewer;
        let zip = finished.archive;
        let entries = read_zip(&zip);
        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "0GenericAffine.mat",
                "1Warp.nii.gz",
                "1InverseWarp.nii.gz",
                "Warped.nii.gz",
                "run.json",
                "README.txt"
            ]
        );

        let fixed = blob(32, 1.0, [0.0; 3]).to_f64();
        let moving = blob(32, 1.0, [3.0, -2.0, 1.0]).to_f64();
        let direct = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &[fixed],
            &[moving],
            Some(2.0),
            &SynOptions::default(),
        )
        .unwrap();
        let files = stage_output_files(Path::new(""), &direct, OutputLayout::Split, false).unwrap();
        assert_eq!(files.len(), 3);
        for ((name, bytes), entry) in files.iter().zip(&entries) {
            assert_eq!(name, &entry.name);
            assert!(
                bytes == &entry.data,
                "{name} differs from the engine's file"
            );
        }

        let warped = Volume::from_nifti_bytes_f64(entries[3].data.clone()).unwrap();
        assert_eq!(warped.grid().size(), [32, 32, 32]);
        assert!(warped.data().iter().any(|v| *v > 0.5));

        assert_eq!(viewer.dims(), [32, 32, 32]);
        assert!(viewer.has_before());
        let slice = |layer| viewer.slice(2, 16, layer).unwrap().data;
        let (fixed, after, before) = (
            slice(Layer::Fixed),
            slice(Layer::Warped),
            slice(Layer::Before),
        );
        assert!(fixed.iter().any(|v| *v > 100));
        assert!(after.iter().any(|v| *v > 100));
        assert_ne!(after, before);

        let run: Value = serde_json::from_slice(&entries[4].data).unwrap();
        assert_eq!(run["options"]["grid"], "2mm");
        assert_eq!(run["registration_grid"]["size"], json!([16, 16, 16]));
        assert_eq!(run["fixed"]["size"], json!([32, 32, 32]));
        assert_eq!(run["peak_gpu_bytes"], 123);
        assert_eq!(run["adapter"]["name"], "CPU reference");
        assert_eq!(run["fastwarp_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(run["files"].as_array().unwrap().len(), 4);
        assert_eq!(run["results"]["syn_levels"].as_array().unwrap().len(), 2);
        assert!(run["timings_s"]["syn"].as_f64().unwrap() > 0.0);
        assert!(run["planned_gpu_bytes"].as_u64().unwrap() > 0);
        assert_eq!(entries[5].data, README.as_bytes());
        assert_eq!(entries[0].date, (46 << 9) | (11 << 5) | 22);

        let seen = seen.into_inner().unwrap();
        let mut phases: Vec<&str> = Vec::new();
        for p in &seen {
            if phases.last() != Some(&p.phase) {
                phases.push(p.phase);
            }
        }
        assert_eq!(
            phases,
            ["read", "prepare", "rigid", "affine", "syn", "warp", "write", "done"]
        );
        let last = |phase: &str| {
            seen.iter()
                .rev()
                .find(|p| p.phase == phase)
                .unwrap()
                .clone()
        };
        let rigid = last("rigid");
        assert_eq!((rigid.level, rigid.levels, rigid.iterations), (2, 2, 8));
        let syn = last("syn");
        assert_eq!((syn.level, syn.levels, syn.iterations), (2, 2, 4));
        assert_eq!(syn.grid, Some([16, 16, 16]));
        assert!(seen
            .iter()
            .any(|p| p.phase == "syn" && p.level == 1 && p.grid == Some([8, 8, 8])));
    }

    #[test]
    fn the_warped_image_is_the_moving_image_through_the_written_transforms() {
        let plan = small_plan();
        let moving = blob(26, 1.25, [3.0, -2.0, 1.0]);
        let mut j = job(plan.clone(), GridChoice::OneMm);
        j.moving = moving.to_nifti_bytes(true).unwrap();
        let zip = ready(run_job(
            &CpuBackend,
            &CpuLinearEngine,
            j,
            &AtomicBool::new(false),
            &|_: &Progress| {},
            &|| None,
        ))
        .unwrap()
        .archive;
        let entries = read_zip(&zip);
        let warped = Volume::from_nifti_bytes_f64(entries[3].data.clone()).unwrap();
        let run: Value = serde_json::from_slice(&entries[4].data).unwrap();
        assert_eq!(run["registration_grid"]["size"], json!([32, 32, 32]));
        assert_eq!(run["peak_gpu_bytes"], Value::Null);

        let fixed = blob(32, 1.0, [0.0; 3]).to_f64();
        let moving = moving.to_f64();
        let direct = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            std::slice::from_ref(&fixed),
            std::slice::from_ref(&moving),
            Some(1.0),
            &SynOptions::default(),
        )
        .unwrap();
        let expected = apply_chain(
            &moving,
            fixed.grid(),
            &direct.pair.forward,
            Interpolation::Linear,
            0.0,
        )
        .unwrap();
        assert_eq!(warped.grid().size(), [32, 32, 32]);
        let worst = warped
            .data()
            .iter()
            .zip(expected.data())
            .map(|(a, b)| (a - (*b as f32) as f64).abs())
            .fold(0.0, f64::max);
        assert_eq!(worst, 0.0);
    }

    #[test]
    fn cancel_bad_input_and_limits_fail_with_their_kind() {
        let quiet = |_: &Progress| {};
        let cancelled = ready(run_job(
            &CpuBackend,
            &CpuLinearEngine,
            job(small_plan(), GridChoice::TwoMm),
            &AtomicBool::new(true),
            &quiet,
            &|| None,
        ));
        assert!(matches!(cancelled, Err(RunError::Cancelled)));
        let mut bad = job(small_plan(), GridChoice::TwoMm);
        bad.moving = b"not a nifti file".to_vec();
        let e = ready(run_job(
            &CpuBackend,
            &CpuLinearEngine,
            bad,
            &AtomicBool::new(false),
            &quiet,
            &|| None,
        ))
        .unwrap_err();
        assert_eq!(e.kind(), "InvalidInput");
        assert!(e.to_string().contains("moving image"), "{e}");

        let mni = Grid::new([193, 229, 193], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
        let small = DeviceLimits {
            max_buffer_size: 1 << 28,
            max_storage_buffer_binding_size: 1 << 26,
        };
        let e = check_limits(&default_plan(), &mni, &mni, GridChoice::OneMm, &small).unwrap_err();
        assert_eq!(e.kind(), "GpuLimit");
        let text = e.to_string();
        assert!(text.contains("maxStorageBufferBindingSize"), "{text}");
        assert!(text.contains("2 mm grid"), "{text}");
        assert!(text.contains("fixed image 193x229x193"), "{text}");
        let planned = check_limits(&default_plan(), &mni, &mni, GridChoice::TwoMm, &small).unwrap();
        assert!(planned > 0);
        let roomy = DeviceLimits {
            max_buffer_size: 1 << 26,
            max_storage_buffer_binding_size: 1 << 30,
        };
        let e = check_limits(&default_plan(), &mni, &mni, GridChoice::OneMm, &roomy).unwrap_err();
        assert!(e.to_string().contains("maxBufferSize"), "{e}");
    }

    #[test]
    fn mattes_on_a_device_that_failed_its_check_is_refused_before_any_work() {
        let mut plan = default_plan();
        assert!(check_mi_available(&plan, Some("mismatch")).is_ok());
        plan.syn[0].channel_metrics = vec![fastwarp_reg::recipe::SynMetric::Mattes];
        assert!(check_mi_available(&plan, None).is_ok());
        let e = check_mi_available(&plan, Some("mismatch")).unwrap_err();
        assert_eq!(e.kind(), "GpuLimit");
        assert!(e.to_string().contains("mismatch"), "{e}");
    }

    #[test]
    fn a_mattes_job_on_a_device_that_failed_its_check_stops_before_any_stage() {
        let mut plan = small_plan();
        plan.syn[0].channel_metrics = vec![fastwarp_reg::recipe::SynMetric::Mattes];
        let mut j = job(plan, GridChoice::TwoMm);
        j.mi_unavailable = Some("forced for the test".into());
        let seen: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
        let report = |p: &Progress| seen.lock().unwrap().push(p.phase);
        let e = ready(run_job(
            &CpuBackend,
            &CpuLinearEngine,
            j,
            &AtomicBool::new(false),
            &report,
            &|| None,
        ))
        .unwrap_err();
        assert_eq!(e.kind(), "GpuLimit");
        assert!(e.to_string().contains("forced for the test"), "{e}");
        let seen = seen.into_inner().unwrap();
        assert!(seen.contains(&"read"), "{seen:?}");
        assert!(seen.iter().all(|p| *p == "read"), "{seen:?}");
    }

    #[test]
    fn gpu_failures_suggest_the_coarser_grid() {
        let e = classify_text(
            "out of memory allocating 5 bytes for field",
            GridChoice::OneMm,
        );
        assert_eq!(e.kind(), "GpuMemory");
        assert!(e.to_string().contains("2 mm grid"), "{e}");
        let e = classify_text("device lost: Destroyed", GridChoice::OneMm);
        assert_eq!(e.kind(), "GpuDeviceLost");
        let e = classify_text("registration cancelled", GridChoice::OneMm);
        assert_eq!(e, RunError::Cancelled);
        let e = classify_text("something else", GridChoice::TwoMm);
        assert_eq!(e, RunError::Registration("something else".into()));
        assert_eq!(
            classify(RegError::Cancelled, GridChoice::OneMm),
            RunError::Cancelled
        );
    }

    #[test]
    fn a_lost_device_is_a_memory_error_only_with_evidence() {
        const MIB: u64 = 1 << 20;
        let raw = "device lost: GetFenceStatus failed with VK_ERROR_DEVICE_LOST";
        let lost = || classify_text(raw, GridChoice::TwoMm);
        let pixel = LostEvidence {
            planned_gpu_bytes: 194 * MIB,
            peak_gpu_bytes: 33 * MIB,
            max_buffer_size: 1024 * MIB,
            ..LostEvidence::default()
        };
        assert!(!pixel.points_to_memory());
        let e = weigh_lost(lost(), GridChoice::TwoMm, &pixel);
        assert_eq!(e.kind(), "GpuDeviceLost");
        let text = e.to_string();
        assert!(text.contains("reset the GPU"), "{text}");
        assert!(text.contains("Run again"), "{text}");
        assert!(text.contains("VK_ERROR_DEVICE_LOST"), "{text}");
        assert!(!text.contains("memory"), "{text}");
        assert!(!text.contains("2 mm grid"), "{text}");
        let one_mm = classify_text(raw, GridChoice::OneMm).to_string();
        assert!(one_mm.contains("try the 2 mm grid"), "{one_mm}");
        let low_limit = LostEvidence {
            max_buffer_size: 256 * MIB,
            ..pixel
        };
        assert!(!low_limit.points_to_memory());
        assert!(LostEvidence {
            planned_gpu_bytes: 512 * MIB,
            ..low_limit
        }
        .points_to_memory());

        let with = |change: &dyn Fn(&mut LostEvidence)| {
            let mut evidence = pixel;
            change(&mut evidence);
            weigh_lost(lost(), GridChoice::OneMm, &evidence)
        };
        for e in [
            with(&|v| v.out_of_memory_errors = 1),
            with(&|v| v.failed_alloc_bytes = 1 << 30),
            with(&|v| v.planned_gpu_bytes = 1024 * MIB),
            with(&|v| v.peak_gpu_bytes = 600 * MIB),
        ] {
            assert_eq!(e.kind(), "GpuMemory");
            let text = e.to_string();
            assert!(text.contains("short of memory"), "{text}");
            assert!(text.contains("2 mm grid"), "{text}");
            assert!(text.contains("VK_ERROR_DEVICE_LOST"), "{text}");
        }
        assert_eq!(with(&|v| v.max_buffer_size = 0).kind(), "GpuDeviceLost");

        let memory = classify_text("out of memory allocating 5 bytes", GridChoice::TwoMm);
        assert_eq!(
            weigh_lost(memory.clone(), GridChoice::TwoMm, &pixel),
            memory
        );
    }

    #[test]
    fn iso_dims_follow_the_resampling_rule() {
        let g = Grid::new([193, 229, 193], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
        assert_eq!(iso_dims(&g, 1.0), [193, 229, 193]);
        assert_eq!(iso_dims(&g, 2.0), [97, 115, 97]);
    }
}

//! Python extension `dbsloc_gpureg._core`: the dbs-localizer cohort registration (centre of mass,
//! Rigid, Affine, SyN) on the GPU, with ANTs-compatible `Composite.h5` / `InverseComposite.h5`.
//!
//! Images arrive as flat `float64` arrays in ITK voxel order (x fastest) with ITK (LPS) geometry,
//! i.e. exactly what an `ANTsImage` holds; the Python wrapper does the flattening.

use std::path::PathBuf;
use std::time::Instant;

use dbsloc_gpu::{BackendSelect, WgpuBackend};
use dbsloc_reg::recipe::SynRecipe;
use dbsloc_reg::register::{register_full, write_linear_output, write_outputs, LinearPlan};
use dbsloc_reg::syn::SynOptions;
use dbsloc_reg::transform::write_composite_pair;
use dbsloc_reg::{Grid, Mat3, Vec3, Volume};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// One image as handed over from Python: flat data (x fastest), size, spacing, origin, direction.
type ImageArgs<'py> = (PyReadonlyArray1<'py, f64>, [usize; 3], Vec3, Vec3, Mat3);

fn to_volume(img: &ImageArgs<'_>) -> PyResult<Volume<f64>> {
    let (data, size, spacing, origin, direction) = img;
    let grid = Grid::new(*size, *spacing, *origin, *direction)
        .map_err(|e| PyValueError::new_err(format!("bad image geometry: {e}")))?;
    let flat = data.as_slice().map_err(|_| PyValueError::new_err("image data must be contiguous"))?;
    Volume::from_vec(grid, flat.to_vec()).map_err(|e| PyValueError::new_err(format!("bad image: {e}")))
}

struct Run {
    adapter: String,
    grid: [usize; 3],
    total_s: f64,
    write_s: f64,
    timings: dbsloc_reg::register::RegisterTimings,
    levels: Vec<(usize, [usize; 3], usize, bool, f64)>,
    affine_parameters: Option<[f64; 12]>,
}

fn gpu(adapter_substring: Option<String>) -> PyResult<WgpuBackend> {
    let mut sel = BackendSelect::from_env().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if adapter_substring.is_some() {
        sel.adapter_substring = adapter_substring;
    }
    WgpuBackend::new(&sel).map_err(|e| PyRuntimeError::new_err(format!("no usable GPU: {e}")))
}

fn adapter_name(g: &WgpuBackend) -> String {
    let i = &g.context().info;
    format!("{} ({}, {})", i.name, i.backend, i.device_type)
}

/// Registers `moving` to `fixed` (1 or 2 channels each) with the dbs-localizer cohort recipe on
/// the GPU and writes `<out_prefix>Composite.h5`, `<out_prefix>InverseComposite.h5` (and, with
/// `write_warped`, `<out_prefix>_Warped.nii.gz`). Returns a dict of timings and diagnostics.
#[pyfunction]
#[pyo3(signature = (fixed, moving, out_prefix, iso_mm=1.0, write_warped=false, adapter=None))]
fn register_syn<'py>(
    py: Python<'py>,
    fixed: Vec<ImageArgs<'py>>,
    moving: Vec<ImageArgs<'py>>,
    out_prefix: PathBuf,
    iso_mm: f64,
    write_warped: bool,
    adapter: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    if fixed.is_empty() || fixed.len() > 2 || fixed.len() != moving.len() {
        return Err(PyValueError::new_err("give 1 or 2 fixed images and as many moving images"));
    }
    let fixed: Vec<Volume<f64>> = fixed.iter().map(to_volume).collect::<PyResult<_>>()?;
    let moving: Vec<Volume<f64>> = moving.iter().map(to_volume).collect::<PyResult<_>>()?;

    let run = py.detach(move || -> PyResult<Run> {
        let t_all = Instant::now();
        let g = gpu(adapter)?;
        let recipe = if fixed.len() == 2 {
            SynRecipe::cohort_multispectral()
        } else {
            SynRecipe::cohort()
        };
        let out = register_full(
            &g,
            &g,
            &LinearPlan::cohort(),
            &recipe,
            &fixed,
            &moving,
            Some(iso_mm),
            &SynOptions::default(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("registration failed: {e}")))?;
        drop((fixed, moving));
        let t = Instant::now();
        let io = |e: dbsloc_reg::RegError| PyRuntimeError::new_err(format!("writing outputs: {e}"));
        if write_warped {
            write_outputs(&out_prefix, &out).map_err(io)?;
        } else {
            write_composite_pair(&out_prefix, &out.pair).map_err(io)?;
        }
        if let Some(chain) = &out.linear {
            write_linear_output(&out_prefix, &chain.total).map_err(io)?;
        }
        let write_s = t.elapsed().as_secs_f64();
        Ok(Run {
            adapter: adapter_name(&g),
            grid: out.grid.size(),
            total_s: t_all.elapsed().as_secs_f64(),
            write_s,
            timings: out.timings,
            levels: out
                .levels
                .iter()
                .map(|l| {
                    let e = l.iterations.last().map(|i| i.energy).unwrap_or(0.0);
                    (l.level, l.grid_size, l.iterations.len(), l.converged, e)
                })
                .collect(),
            affine_parameters: out.linear.as_ref().map(|c| c.total.parameters()),
        })
    })?;

    let d = PyDict::new(py);
    d.set_item("adapter", run.adapter)?;
    d.set_item("grid", run.grid)?;
    d.set_item("total_s", run.total_s)?;
    d.set_item("resample_s", run.timings.resample_s)?;
    d.set_item("preprocess_s", run.timings.preprocess_s)?;
    d.set_item("linear_s", run.timings.linear_s)?;
    d.set_item("syn_s", run.timings.syn_s)?;
    d.set_item("warp_s", run.timings.warp_s)?;
    d.set_item("write_s", run.write_s)?;
    d.set_item("levels", run.levels)?;
    d.set_item("affine_parameters", run.affine_parameters)?;
    Ok(d)
}

/// Opens the GPU the engine would use and returns its description; raises when there is none.
#[pyfunction]
#[pyo3(signature = (adapter=None))]
fn gpu_info(py: Python<'_>, adapter: Option<String>) -> PyResult<String> {
    py.detach(move || gpu(adapter).map(|g| adapter_name(&g)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_syn, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    m.add("ENGINE_SOURCE", "dbs-localizer 00faf42 (Wyss Center for Bio and Neuroengineering)")?;
    Ok(())
}

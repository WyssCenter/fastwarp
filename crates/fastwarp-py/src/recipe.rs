// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! The registration recipe as a Python dict: `{"rigid": {...}, "affine": {...}, "syn": {...}}`,
//! each field of [`LinearRecipe`] / [`SynRecipe`] under its own name. A missing stage or field
//! keeps the dbs-localizer cohort value, an unknown one is an error, so a caller (leaddbs-py) can
//! state the whole schedule and a typo never passes silently. The keys, defaults and checks are
//! those of the command line's JSON recipe (`fastwarp_cli::recipe`).

use fastwarp_cli::recipe::{
    check_linear, cohort_linear, parse_units, units_name, Recipe, LINEAR_KEYS, STAGE_KEYS, SYN_KEYS,
};
use fastwarp_reg::linear::LinearRecipe;
use fastwarp_reg::pyramid::SigmaUnits;
use fastwarp_reg::recipe::SynRecipe;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

/// The cohort recipe with the stages and fields `d` names replaced, checked for `channels`
/// channels and for what the GPU kernels support.
pub fn from_dict(d: Option<&Bound<'_, PyDict>>, channels: usize) -> PyResult<Recipe> {
    let mut r = Recipe::cohort(channels);
    if let Some(d) = d {
        check_keys(d, STAGE_KEYS, "recipe")?;
        if let Some(s) = stage(d, "rigid")? {
            linear_from(&s, &mut r.linear.rigid, "recipe['rigid']")?;
        }
        if let Some(s) = stage(d, "affine")? {
            linear_from(&s, &mut r.linear.affine, "recipe['affine']")?;
        }
        if let Some(s) = stage(d, "syn")? {
            syn_from(&s, &mut r.syn)?;
        }
    }
    r.check(channels).map_err(value_err)?;
    Ok(r)
}

/// The recipe as the dict [`from_dict`] reads (every field present).
pub fn to_dict<'py>(py: Python<'py>, r: &Recipe) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("rigid", linear_dict(py, &r.linear.rigid)?)?;
    d.set_item("affine", linear_dict(py, &r.linear.affine)?)?;
    let s = PyDict::new(py);
    let r = &r.syn;
    s.set_item("gradient_step", r.gradient_step)?;
    s.set_item("update_field_variance", r.update_field_variance)?;
    s.set_item("total_field_variance", r.total_field_variance)?;
    s.set_item("cc_radius", r.cc_radius)?;
    s.set_item("channel_weights", r.channel_weights.clone())?;
    s.set_item("iterations", r.iterations.clone())?;
    s.set_item("convergence_threshold", r.convergence_threshold)?;
    s.set_item("convergence_window", r.convergence_window)?;
    s.set_item("shrink_factors", r.shrink_factors.clone())?;
    s.set_item("smoothing_sigmas", r.smoothing_sigmas.clone())?;
    s.set_item("sigma_units", units_name(r.sigma_units))?;
    d.set_item("syn", s)?;
    Ok(d)
}

/// One `Rigid` (`transform = "rigid"`) or `Affine` (`"affine"`) stage: the cohort's, with the
/// fields `d` names replaced, checked like a recipe.
pub fn linear_stage(d: Option<&Bound<'_, PyDict>>, transform: &str) -> PyResult<LinearRecipe> {
    let mut stage = cohort_linear(transform).map_err(value_err)?;
    if let Some(d) = d {
        linear_from(d, &mut stage, "stage")?;
    }
    check_linear("stage", &stage).map_err(value_err)?;
    Ok(stage)
}

/// The stage as the dict [`linear_stage`] reads.
pub fn linear_stage_dict<'py>(
    py: Python<'py>,
    stage: &LinearRecipe,
) -> PyResult<Bound<'py, PyDict>> {
    linear_dict(py, stage)
}

fn linear_dict<'py>(py: Python<'py>, l: &LinearRecipe) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("gradient_step", l.gradient_step)?;
    d.set_item("iterations", l.iterations.clone())?;
    d.set_item("convergence_threshold", l.convergence_threshold)?;
    d.set_item("convergence_window", l.convergence_window)?;
    d.set_item("shrink_factors", l.shrink_factors.clone())?;
    d.set_item("smoothing_sigmas", l.smoothing_sigmas.clone())?;
    d.set_item("sigma_units", units_name(l.sigma_units))?;
    d.set_item("bins", l.bins)?;
    d.set_item("sampling_percentage", f64::from(l.sampling_percentage))?;
    d.set_item("seed", l.seed)?;
    Ok(d)
}

fn check_keys(d: &Bound<'_, PyDict>, allowed: &[&str], what: &str) -> PyResult<()> {
    for key in d.keys() {
        let k: String = key
            .extract()
            .map_err(|_| PyValueError::new_err(format!("{what}: keys must be strings")))?;
        if !allowed.contains(&k.as_str()) {
            return Err(PyValueError::new_err(format!(
                "{what}: unknown key '{k}' (known: {})",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}

fn stage<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
    match d.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => v
            .cast_into::<PyDict>()
            .map(Some)
            .map_err(|_| PyValueError::new_err(format!("recipe['{key}'] must be a dict"))),
    }
}

/// Replaces `slot` with `d[key]` when present; a value of the wrong type is a `ValueError`
/// naming the field.
fn set<'py, T: FromPyObjectOwned<'py>>(
    d: &Bound<'py, PyDict>,
    key: &str,
    what: &str,
    slot: &mut T,
) -> PyResult<()> {
    if let Some(v) = d.get_item(key)? {
        // A string is a sequence too: refuse it where a list is meant.
        if v.is_instance_of::<pyo3::types::PyString>() && key != "sigma_units" {
            return Err(PyValueError::new_err(format!(
                "{what}['{key}']: got a string"
            )));
        }
        *slot = v.extract::<T>().map_err(|e| {
            let e: PyErr = e.into();
            PyValueError::new_err(format!("{what}['{key}']: {}", e.value(d.py())))
        })?;
    }
    Ok(())
}

fn units(d: &Bound<'_, PyDict>, what: &str, slot: &mut SigmaUnits) -> PyResult<()> {
    let mut name = units_name(*slot).to_string();
    set(d, "sigma_units", what, &mut name)?;
    *slot = parse_units(&name, what).map_err(value_err)?;
    Ok(())
}

fn linear_from(d: &Bound<'_, PyDict>, l: &mut LinearRecipe, what: &str) -> PyResult<()> {
    check_keys(d, LINEAR_KEYS, what)?;
    set(d, "gradient_step", what, &mut l.gradient_step)?;
    set(d, "iterations", what, &mut l.iterations)?;
    set(
        d,
        "convergence_threshold",
        what,
        &mut l.convergence_threshold,
    )?;
    set(d, "convergence_window", what, &mut l.convergence_window)?;
    set(d, "shrink_factors", what, &mut l.shrink_factors)?;
    set(d, "smoothing_sigmas", what, &mut l.smoothing_sigmas)?;
    units(d, what, &mut l.sigma_units)?;
    set(d, "bins", what, &mut l.bins)?;
    let mut sampling = f64::from(l.sampling_percentage);
    set(d, "sampling_percentage", what, &mut sampling)?;
    l.sampling_percentage = sampling as f32;
    set(d, "seed", what, &mut l.seed)?;
    Ok(())
}

fn syn_from(d: &Bound<'_, PyDict>, r: &mut SynRecipe) -> PyResult<()> {
    let what = "recipe['syn']";
    check_keys(d, SYN_KEYS, what)?;
    set(d, "gradient_step", what, &mut r.gradient_step)?;
    set(
        d,
        "update_field_variance",
        what,
        &mut r.update_field_variance,
    )?;
    set(d, "total_field_variance", what, &mut r.total_field_variance)?;
    set(d, "cc_radius", what, &mut r.cc_radius)?;
    set(d, "channel_weights", what, &mut r.channel_weights)?;
    set(d, "iterations", what, &mut r.iterations)?;
    set(
        d,
        "convergence_threshold",
        what,
        &mut r.convergence_threshold,
    )?;
    set(d, "convergence_window", what, &mut r.convergence_window)?;
    set(d, "shrink_factors", what, &mut r.shrink_factors)?;
    set(d, "smoothing_sigmas", what, &mut r.smoothing_sigmas)?;
    units(d, what, &mut r.sigma_units)?;
    Ok(())
}

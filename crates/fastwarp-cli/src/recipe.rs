// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! The registration recipe: `{"rigid": {...}, "affine": {...}, "syn": {...}}`, each field of
//! [`LinearRecipe`] / [`SynRecipe`] under its own name. A missing stage or field keeps the
//! dbs-localizer cohort value, an unknown one is an error, so a caller can state the whole
//! schedule and a typo never passes silently. The checks are shared by the Python dict and the
//! JSON file of the command line.

use fastwarp_reg::linear::LinearRecipe;
use fastwarp_reg::pyramid::SigmaUnits;
use fastwarp_reg::recipe::SynRecipe;
use fastwarp_reg::register::LinearPlan;
use serde_json::{json, Map, Value};

/// Histogram bins the GPU Mattes MI kernels are built for.
pub const GPU_MI_BINS: usize = 32;
/// Largest CC radius of the GPU box-sum kernels.
pub const GPU_CC_RADIUS_MAX: usize = 8;

/// The stages of a recipe.
pub const STAGE_KEYS: &[&str] = &["rigid", "affine", "syn"];

/// The fields of a `rigid` or `affine` stage.
pub const LINEAR_KEYS: &[&str] = &[
    "gradient_step",
    "iterations",
    "convergence_threshold",
    "convergence_window",
    "shrink_factors",
    "smoothing_sigmas",
    "sigma_units",
    "bins",
    "sampling_percentage",
    "seed",
];

/// The fields of the `syn` stage.
pub const SYN_KEYS: &[&str] = &[
    "gradient_step",
    "update_field_variance",
    "total_field_variance",
    "cc_radius",
    "channel_weights",
    "iterations",
    "convergence_threshold",
    "convergence_window",
    "shrink_factors",
    "smoothing_sigmas",
    "sigma_units",
];

/// The three stages that run: Rigid, Affine (Mattes MI, first channel) and SyN (CC, all channels).
#[derive(Clone, Debug)]
pub struct Recipe {
    pub linear: LinearPlan,
    pub syn: SynRecipe,
}

impl Recipe {
    /// The dbs-localizer cohort recipe for `channels` channels (weights 1, or 0.5 each for two).
    pub fn cohort(channels: usize) -> Self {
        Self {
            linear: LinearPlan::cohort(),
            syn: if channels == 2 {
                SynRecipe::cohort_multispectral()
            } else {
                SynRecipe::cohort()
            },
        }
    }

    /// Checks the recipe for `channels` channels and for what the GPU kernels support.
    pub fn check(&self, channels: usize) -> Result<(), String> {
        check_linear("recipe['rigid']", &self.linear.rigid)?;
        check_linear("recipe['affine']", &self.linear.affine)?;
        self.syn
            .validate()
            .map_err(|e| format!("recipe['syn']: {e}"))?;
        if self.syn.cc_radius > GPU_CC_RADIUS_MAX {
            return Err(format!(
                "recipe['syn']['cc_radius']: the GPU kernels support 1 to {GPU_CC_RADIUS_MAX}, \
                 not {}",
                self.syn.cc_radius
            ));
        }
        if self.syn.channel_weights.len() != channels {
            return Err(format!(
                "recipe['syn']['channel_weights'] has {} weights for {channels} channel(s)",
                self.syn.channel_weights.len()
            ));
        }
        Ok(())
    }

    /// The cohort recipe with the stages and fields of the JSON object `v` replaced, checked
    /// like [`Recipe::check`].
    pub fn from_json(v: &Value, channels: usize) -> Result<Self, String> {
        let mut r = Self::cohort(channels);
        let obj = v
            .as_object()
            .ok_or_else(|| "recipe: must be a JSON object".to_string())?;
        check_json_keys(obj, STAGE_KEYS, "recipe")?;
        for key in STAGE_KEYS {
            let stage = match obj.get(*key) {
                None | Some(Value::Null) => continue,
                Some(Value::Object(m)) => m,
                Some(_) => return Err(format!("recipe['{key}'] must be an object")),
            };
            let what = format!("recipe['{key}']");
            match *key {
                "rigid" => linear_from_json(stage, &mut r.linear.rigid, &what)?,
                "affine" => linear_from_json(stage, &mut r.linear.affine, &what)?,
                _ => syn_from_json(stage, &mut r.syn)?,
            }
        }
        r.check(channels)?;
        Ok(r)
    }

    /// The recipe as the JSON object [`Recipe::from_json`] reads (every field present), equal to
    /// the dict `fastwarp.cohort_recipe()` returns.
    pub fn to_json(&self) -> Value {
        let r = &self.syn;
        json!({
            "rigid": linear_to_json(&self.linear.rigid),
            "affine": linear_to_json(&self.linear.affine),
            "syn": {
                "gradient_step": r.gradient_step,
                "update_field_variance": r.update_field_variance,
                "total_field_variance": r.total_field_variance,
                "cc_radius": r.cc_radius,
                "channel_weights": r.channel_weights,
                "iterations": r.iterations,
                "convergence_threshold": r.convergence_threshold,
                "convergence_window": r.convergence_window,
                "shrink_factors": r.shrink_factors,
                "smoothing_sigmas": r.smoothing_sigmas,
                "sigma_units": units_name(r.sigma_units),
            },
        })
    }
}

/// The cohort `Rigid` (`transform = "rigid"`) or `Affine` (`"affine"`) stage.
pub fn cohort_linear(transform: &str) -> Result<LinearRecipe, String> {
    match transform {
        "rigid" => Ok(LinearRecipe::cohort_rigid()),
        "affine" => Ok(LinearRecipe::cohort_affine()),
        other => Err(format!(
            "transform must be 'rigid' or 'affine', not {other:?}"
        )),
    }
}

/// Checks one linear stage, `what` naming it in the message.
pub fn check_linear(what: &str, l: &LinearRecipe) -> Result<(), String> {
    l.validate().map_err(|e| format!("{what}: {e}"))?;
    if l.bins != GPU_MI_BINS {
        return Err(format!(
            "{what}['bins']: the GPU MI kernels support {GPU_MI_BINS} bins, not {}",
            l.bins
        ));
    }
    Ok(())
}

/// One linear stage: the cohort's for `transform`, with the fields of the JSON object `v`
/// replaced, checked like a recipe.
pub fn linear_stage_from_json(v: Option<&Value>, transform: &str) -> Result<LinearRecipe, String> {
    let mut stage = cohort_linear(transform)?;
    match v {
        None | Some(Value::Null) => {}
        Some(Value::Object(m)) => linear_from_json(m, &mut stage, "stage")?,
        Some(_) => return Err("stage: must be a JSON object".to_string()),
    }
    check_linear("stage", &stage)?;
    Ok(stage)
}

/// The name of a sigma unit in a recipe (`vox` or `mm`).
pub fn units_name(u: SigmaUnits) -> &'static str {
    match u {
        SigmaUnits::Voxels => "vox",
        SigmaUnits::Millimetres => "mm",
    }
}

/// The sigma unit of a recipe name, `what` naming the field in the message.
pub fn parse_units(name: &str, what: &str) -> Result<SigmaUnits, String> {
    match name {
        "vox" => Ok(SigmaUnits::Voxels),
        "mm" => Ok(SigmaUnits::Millimetres),
        other => Err(format!(
            "{what}['sigma_units']: 'vox' or 'mm', not '{other}'"
        )),
    }
}

/// A linear stage as JSON, the fields [`linear_stage_from_json`] reads.
pub fn linear_to_json(l: &LinearRecipe) -> Value {
    json!({
        "gradient_step": l.gradient_step,
        "iterations": l.iterations,
        "convergence_threshold": l.convergence_threshold,
        "convergence_window": l.convergence_window,
        "shrink_factors": l.shrink_factors,
        "smoothing_sigmas": l.smoothing_sigmas,
        "sigma_units": units_name(l.sigma_units),
        "bins": l.bins,
        "sampling_percentage": f64::from(l.sampling_percentage),
        "seed": l.seed,
    })
}

fn check_json_keys(m: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<(), String> {
    for k in m.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(format!(
                "{what}: unknown key '{k}' (known: {})",
                allowed.join(", ")
            ));
        }
    }
    Ok(())
}

fn field_err(what: &str, key: &str, want: &str, v: &Value) -> String {
    format!("{what}['{key}']: expected {want}, got {v}")
}

fn f64_of(what: &str, key: &str, v: &Value) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| field_err(what, key, "a number", v))
}

fn uint_of(what: &str, key: &str, v: &Value, max: u64) -> Result<u64, String> {
    v.as_u64()
        .filter(|n| *n <= max)
        .ok_or_else(|| field_err(what, key, "a non-negative integer", v))
}

fn list_of<'a>(what: &str, key: &str, v: &'a Value) -> Result<&'a Vec<Value>, String> {
    v.as_array()
        .ok_or_else(|| field_err(what, key, "a list", v))
}

fn set_f64(m: &Map<String, Value>, what: &str, key: &str, slot: &mut f64) -> Result<(), String> {
    if let Some(v) = m.get(key) {
        *slot = f64_of(what, key, v)?;
    }
    Ok(())
}

fn set_usize(
    m: &Map<String, Value>,
    what: &str,
    key: &str,
    slot: &mut usize,
) -> Result<(), String> {
    if let Some(v) = m.get(key) {
        *slot = usize::try_from(uint_of(what, key, v, u64::MAX)?)
            .map_err(|_| field_err(what, key, "a smaller integer", v))?;
    }
    Ok(())
}

fn set_u32(m: &Map<String, Value>, what: &str, key: &str, slot: &mut u32) -> Result<(), String> {
    if let Some(v) = m.get(key) {
        *slot = uint_of(what, key, v, u64::from(u32::MAX))? as u32;
    }
    Ok(())
}

fn set_u32s(
    m: &Map<String, Value>,
    what: &str,
    key: &str,
    slot: &mut Vec<u32>,
) -> Result<(), String> {
    if let Some(v) = m.get(key) {
        *slot = list_of(what, key, v)?
            .iter()
            .map(|x| uint_of(what, key, x, u64::from(u32::MAX)).map(|n| n as u32))
            .collect::<Result<_, _>>()?;
    }
    Ok(())
}

fn set_f64s(
    m: &Map<String, Value>,
    what: &str,
    key: &str,
    slot: &mut Vec<f64>,
) -> Result<(), String> {
    if let Some(v) = m.get(key) {
        *slot = list_of(what, key, v)?
            .iter()
            .map(|x| f64_of(what, key, x))
            .collect::<Result<_, _>>()?;
    }
    Ok(())
}

fn set_units(m: &Map<String, Value>, what: &str, slot: &mut SigmaUnits) -> Result<(), String> {
    if let Some(v) = m.get("sigma_units") {
        let name = v
            .as_str()
            .ok_or_else(|| field_err(what, "sigma_units", "'vox' or 'mm'", v))?;
        *slot = parse_units(name, what)?;
    }
    Ok(())
}

fn linear_from_json(
    m: &Map<String, Value>,
    l: &mut LinearRecipe,
    what: &str,
) -> Result<(), String> {
    check_json_keys(m, LINEAR_KEYS, what)?;
    set_f64(m, what, "gradient_step", &mut l.gradient_step)?;
    set_u32s(m, what, "iterations", &mut l.iterations)?;
    set_f64(
        m,
        what,
        "convergence_threshold",
        &mut l.convergence_threshold,
    )?;
    set_usize(m, what, "convergence_window", &mut l.convergence_window)?;
    set_u32s(m, what, "shrink_factors", &mut l.shrink_factors)?;
    set_f64s(m, what, "smoothing_sigmas", &mut l.smoothing_sigmas)?;
    set_units(m, what, &mut l.sigma_units)?;
    set_usize(m, what, "bins", &mut l.bins)?;
    let mut sampling = f64::from(l.sampling_percentage);
    set_f64(m, what, "sampling_percentage", &mut sampling)?;
    l.sampling_percentage = sampling as f32;
    set_u32(m, what, "seed", &mut l.seed)?;
    Ok(())
}

fn syn_from_json(m: &Map<String, Value>, r: &mut SynRecipe) -> Result<(), String> {
    let what = "recipe['syn']";
    check_json_keys(m, SYN_KEYS, what)?;
    set_f64(m, what, "gradient_step", &mut r.gradient_step)?;
    set_f64(
        m,
        what,
        "update_field_variance",
        &mut r.update_field_variance,
    )?;
    set_f64(m, what, "total_field_variance", &mut r.total_field_variance)?;
    set_usize(m, what, "cc_radius", &mut r.cc_radius)?;
    set_f64s(m, what, "channel_weights", &mut r.channel_weights)?;
    set_u32s(m, what, "iterations", &mut r.iterations)?;
    set_f64(
        m,
        what,
        "convergence_threshold",
        &mut r.convergence_threshold,
    )?;
    set_usize(m, what, "convergence_window", &mut r.convergence_window)?;
    set_u32s(m, what, "shrink_factors", &mut r.shrink_factors)?;
    set_f64s(m, what, "smoothing_sigmas", &mut r.smoothing_sigmas)?;
    set_units(m, what, &mut r.sigma_units)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_cohort_round_trips_through_json() {
        for channels in [1, 2] {
            let r = Recipe::cohort(channels);
            let back = Recipe::from_json(&r.to_json(), channels).unwrap();
            assert_eq!(back.to_json(), r.to_json());
        }
    }

    #[test]
    fn an_empty_object_is_the_cohort() {
        let r = Recipe::from_json(&json!({}), 1).unwrap();
        assert_eq!(r.to_json(), Recipe::cohort(1).to_json());
    }

    #[test]
    fn a_field_replaces_only_itself() {
        let r = Recipe::from_json(&json!({"syn": {"iterations": [5, 3]}, "rigid": null}), 1);
        let r = r.unwrap_err();
        assert!(r.contains("recipe['syn']"), "{r}");
        let r = Recipe::from_json(
            &json!({"syn": {"iterations": [5, 3], "shrink_factors": [2, 1],
                            "smoothing_sigmas": [1, 0]}}),
            1,
        )
        .unwrap();
        assert_eq!(r.syn.iterations, vec![5, 3]);
        assert_eq!(r.syn.smoothing_sigmas, vec![1.0, 0.0]);
        assert_eq!(
            r.linear.rigid.iterations,
            Recipe::cohort(1).linear.rigid.iterations
        );
    }

    #[test]
    fn unknown_keys_and_wrong_types_are_refused() {
        let e = Recipe::from_json(&json!({"rigd": {}}), 1).unwrap_err();
        assert!(e.contains("unknown key 'rigd'"), "{e}");
        let e = Recipe::from_json(&json!({"rigid": {"iteration": [1]}}), 1).unwrap_err();
        assert!(
            e.contains("recipe['rigid']: unknown key 'iteration'"),
            "{e}"
        );
        let e = Recipe::from_json(&json!({"rigid": {"iterations": "100x50"}}), 1).unwrap_err();
        assert!(e.contains("recipe['rigid']['iterations']"), "{e}");
        let e = Recipe::from_json(&json!({"rigid": {"seed": -1}}), 1).unwrap_err();
        assert!(e.contains("recipe['rigid']['seed']"), "{e}");
        let e = Recipe::from_json(&json!({"syn": []}), 1).unwrap_err();
        assert!(e.contains("must be an object"), "{e}");
        let e = Recipe::from_json(&json!([1]), 1).unwrap_err();
        assert!(e.contains("JSON object"), "{e}");
        let e = Recipe::from_json(&json!({"syn": {"sigma_units": "cm"}}), 1).unwrap_err();
        assert!(e.contains("'vox' or 'mm'"), "{e}");
    }

    #[test]
    fn gpu_limits_are_checked() {
        let e = Recipe::from_json(&json!({"affine": {"bins": 64}}), 1).unwrap_err();
        assert!(e.contains("32 bins"), "{e}");
        let e = Recipe::from_json(&json!({"syn": {"cc_radius": 9}}), 1).unwrap_err();
        assert!(e.contains("cc_radius"), "{e}");
        let e = Recipe::from_json(&json!({}), 2);
        assert!(e.is_ok());
        let e = Recipe::from_json(&json!({"syn": {"channel_weights": [1.0]}}), 2).unwrap_err();
        assert!(e.contains("1 weights for 2 channel(s)"), "{e}");
    }

    #[test]
    fn a_linear_stage_reads_its_fields() {
        let s = linear_stage_from_json(Some(&json!({"iterations": [10, 5, 0, 0]})), "affine");
        assert_eq!(s.unwrap().iterations, vec![10, 5, 0, 0]);
        assert!(linear_stage_from_json(None, "similarity").is_err());
        assert!(linear_stage_from_json(Some(&json!(3)), "rigid").is_err());
    }
}

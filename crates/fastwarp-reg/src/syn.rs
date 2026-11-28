// SPDX-FileCopyrightText: 1999-2019 Insight Software Consortium
// SPDX-FileCopyrightText: 2020-present NumFOCUS
// SPDX-FileCopyrightText: ANTs developers
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0
//
// Copyright NumFOCUS (Insight Software Consortium)
// Copyright 2026 Wyss Center for Bio and Neuroengineering
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions derived from ITK `itkSyNImageRegistrationMethod.hxx` (`InitializeRegistrationAtEachLevel`,
// `StartOptimization`, `ComputeUpdateField`, `ComputeMetricGradientField` with
// `DownsampleImagesForMetricDerivatives`, `GenerateData`), `itkObjectToObjectMultiMetricv4.hxx`
// (`GetValueAndDerivative`, `GetWeightedValue`, weight normalisation), `itkImageRegistrationMethodv4.hxx`
// (`InitializeRegistrationAtEachLevel`: shrunk virtual domain, per-level smoothing, iteration state)
// and from ANTs `itkantsRegistrationHelper.hxx` (the SyN stage set-up, the earlier stages as the
// moving initial transform, `CollapseDisplacementFieldTransforms`). See
// `docs/THIRD_PARTY_PORTS.md`.

//! ITK's symmetric normalisation (`SyNImageRegistrationMethod`) as `antsRegistration` drives it,
//! written against a [`Backend`].
//!
//! Four fields live on the virtual grid of the current level: fixed-to-middle `F`, moving-to-middle
//! `M` and their inverses. One iteration:
//!
//! 1. every channel's smoothed full-resolution fixed image is resampled onto the level grid
//!    through `F^-1`, and the moving image through `affine(M^-1(x))` (after earlier SyN stages
//!    `affine(W(M^-1(x)))`, see [`SynStart`]);
//! 2. each channel's metric (ANTs CC, or Mattes MI with dense sampling, whose histogram ranges
//!    are measured anew on the two resampled images) runs on those two images (identity
//!    transforms, so its derivative is a per-voxel image-space gradient), once with the
//!    fixed-side image as "fixed" and once with the roles swapped; with masks, the fixed mask is
//!    resampled through `F^-1` and the moving mask through the same map as the moving image
//!    (nearest neighbour); CC uses only the voxels inside both, MI as well and measures each
//!    intensity range over the voxels inside the image's own mask; several channels combine as
//!    ITK's multi-metric `(sum |d_j| / J) sum w_j d_j / |d_j|`; the value is the first metric's;
//! 3. each gradient is Gaussian-smoothed (update variance) and scaled so its largest voxel step is
//!    the gradient step;
//! 4. `F <- F + u_F(x + F(x))` (likewise `M`), optionally smoothed with the total-field variance;
//! 5. both inverses are re-estimated from the previous inverse and each field is replaced by the
//!    inverse of its inverse: `Finv = invert(F_tmp, init = Finv)`, `F = invert(Finv, init = F_tmp)`;
//! 6. the energy `0.5 * (value_a + value_b)` enters the window convergence monitor.
//!
//! The result is `SyN = Minv o F` composed with the initial affine into a [`CompositePair`].
//! After earlier SyN stages the result is collapsed with their field `W` as
//! `antsRegistration --collapse-output-transforms 1` collapses consecutive displacement fields
//! (`CollapseDisplacementFieldTransforms`): forward `SyN(x) + W(x + SyN(x))`, inverse
//! `Winv(x) + SyN^-1(x + Winv(x))`, both on the grid of the last level.

use std::borrow::Cow;
use std::sync::atomic::AtomicBool;

use crate::backend::{Backend, MetricPair};
use crate::convergence::WindowConvergenceMonitor;
use crate::error::{check_cancel, invalid, Result};
use crate::exec::block_ready;
use crate::field_ops::{InvertParams, InvertStats, VectorField};
use crate::grid::Grid;
use crate::linear::SmoothCache;
use crate::pyramid::{
    level_virtual_grid, shrink_factors_per_dimension, smooth_for_level_cow, SigmaUnits,
};
use crate::recipe::{SynMetric, SynRecipe};
use crate::transform::{AffineTransform, CompositePair, DisplacementField};
use crate::volume::Volume;

/// One metric channel: preprocessed full-resolution fixed and moving images (ANTs windowed them
/// to `[0, 1]` before the stage). The fixed images of all channels must share a grid.
#[derive(Clone, Debug)]
pub struct SynChannel {
    /// The fixed image.
    pub fixed: Volume<f64>,
    /// The moving image.
    pub moving: Volume<f64>,
}

/// Options outside the `antsRegistration` command line: `cancel`, polled at the start of every
/// level and every iteration (and by the linear stages of [`crate::register::register_full`]);
/// once it is set the run ends with [`crate::error::RegError::Cancelled`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SynOptions<'a> {
    pub cancel: Option<&'a AtomicBool>,
}

/// What one iteration measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynIteration {
    /// Energy given to the monitor: `0.5 * (a + b)` of the first-metric values.
    pub energy: f64,
    /// `0.5 * (a + b)` of the first-metric values (what ITK uses).
    pub energy_first_metric: f64,
    /// `0.5 * (a + b)` of the weighted values.
    pub energy_weighted: f64,
    /// Monitor value after this iteration (`f64::MAX` until the window is full).
    pub convergence_value: f64,
    /// Scale applied to the fixed-side update (gradient step over its largest voxel step).
    pub fixed_update_scale: f64,
    /// Scale applied to the moving-side update.
    pub moving_update_scale: f64,
    /// Inversion statistics in call order: `Finv`, `F`, `Minv`, `M`.
    pub inversions: [InvertStats; 4],
}

/// The trace of one level.
#[derive(Clone, Debug, PartialEq)]
pub struct SynLevelTrace {
    /// Level index.
    pub level: usize,
    /// Shrink factors applied to the fixed grid.
    pub shrink_factors: [u32; 3],
    /// Voxels of the level grid.
    pub grid_size: [usize; 3],
    /// One entry per executed iteration.
    pub iterations: Vec<SynIteration>,
    /// Whether the monitor stopped the level before its iteration limit.
    pub converged: bool,
}

/// Output of [`run_syn`] and [`run_syn_from`].
#[derive(Clone, Debug)]
pub struct SynResult {
    /// `Composite = [Affine, Warp]`, `InverseComposite = [InverseWarp, Affine^-1]`.
    pub pair: CompositePair,
    /// `SyN(x) = F(x) + Minv(x + F(x))` on the last level grid, `f32` like `1Warp.nii.gz`;
    /// after earlier SyN stages, collapsed with their field: `SyN(x) + W(x + SyN(x))`.
    pub warp: DisplacementField,
    /// `SyN^-1(x) = M(x) + Finv(x + M(x))` on the last level grid; after earlier SyN stages,
    /// `Winv(x) + SyN^-1(x + Winv(x))`.
    pub inverse_warp: DisplacementField,
    /// Per-level traces.
    pub levels: Vec<SynLevelTrace>,
}

/// The displacement fields of the SyN stages that ran before a SyN stage, collapsed into one
/// field per direction (what `antsRegistration --collapse-output-transforms 1` writes for
/// them). Both live on the grid of the last level of the stages, which a following stage must
/// share.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorWarp {
    /// Forward field `W`: a fixed point `x` goes to `x + W(x)`, which the affine then maps
    /// into the moving image.
    pub warp: VectorField,
    /// Inverse field `Winv`.
    pub inverse_warp: VectorField,
}

/// Where a SyN stage starts on the moving side: the linear transform of the earlier stages
/// (fixed points to moving points) and, after earlier SyN stages, their collapsed field. As
/// `antsRegistration` passes the composite of all earlier stages to a stage as its moving initial
/// transform, the moving image and the moving mask are sampled at `affine(W(M^-1(x)))`, `W`
/// evaluated by linear interpolation on its own grid (zero displacement outside it), with one
/// interpolation of the image.
#[derive(Clone, Copy, Debug)]
pub struct SynStart<'a> {
    /// The collapsed linear transform of the start and the linear stages.
    pub affine: &'a AffineTransform,
    /// The collapsed fields of the earlier SyN stages; `None` for the first SyN stage.
    pub prior: Option<&'a PriorWarp>,
}

impl<'a> SynStart<'a> {
    /// A start from `affine` alone (the first SyN stage).
    pub fn affine(affine: &'a AffineTransform) -> Self {
        Self {
            affine,
            prior: None,
        }
    }
}

/// The two collapsed fields of a SyN stage and its per-level traces.
pub(crate) struct SynFields {
    pub(crate) warp: VectorField,
    pub(crate) inverse_warp: VectorField,
    pub(crate) levels: Vec<SynLevelTrace>,
}

struct State<B: Backend> {
    f2m: B::Field,
    f2m_inv: B::Field,
    m2m: B::Field,
    m2m_inv: B::Field,
}

struct SideUpdate<B: Backend> {
    update: B::Field,
    scale: f64,
    first_value: f64,
    weighted_value: f64,
}

struct SideAccumulator<B: Backend> {
    values: Vec<f64>,
    gradient: Option<B::Field>,
    total_magnitude: f64,
}

impl<B: Backend> SideAccumulator<B> {
    fn new(channels: usize) -> Self {
        Self {
            values: Vec::with_capacity(channels),
            gradient: None,
            total_magnitude: 0.0,
        }
    }

    async fn add(
        &mut self,
        backend: &B,
        weight: f64,
        value: f64,
        derivative: B::Field,
        magnitude: Option<f64>,
    ) -> Result<()> {
        self.values.push(value);
        let Some(magnitude) = magnitude else {
            self.gradient = Some(derivative);
            return Ok(());
        };
        let acc = match self.gradient.as_mut() {
            Some(acc) => acc,
            None => {
                let grid = backend.field_grid(&derivative);
                self.gradient.insert(backend.zero_field(&grid).await?)
            }
        };
        if !magnitude.is_finite() {
            return Err(invalid("syn", "the metric gradient is not finite"));
        }
        self.total_magnitude += magnitude;
        if magnitude > f64::EPSILON {
            backend.axpy(acc, weight / magnitude, &derivative).await?;
        }
        Ok(())
    }

    async fn smoothed(
        self,
        backend: &B,
        recipe: &SynRecipe,
        normalised: &[f64],
    ) -> Result<SmoothedSide<B>> {
        let channels = self.values.len();
        let first_value = self.values[0];
        let mut gradient = self
            .gradient
            .ok_or_else(|| invalid("syn", "no channel produced a gradient"))?;
        let weighted_value = if channels == 1 {
            first_value
        } else {
            backend
                .scale(&mut gradient, self.total_magnitude / channels as f64)
                .await?;
            self.values.iter().zip(normalised).map(|(v, w)| v * w).sum()
        };
        let gradient = backend
            .smooth_field(gradient, recipe.update_field_variance)
            .await?;
        Ok(SmoothedSide {
            gradient,
            first_value,
            weighted_value,
        })
    }
}

struct SmoothedSide<B: Backend> {
    gradient: B::Field,
    first_value: f64,
    weighted_value: f64,
}

impl<B: Backend> SmoothedSide<B> {
    async fn finish(self, backend: &B, recipe: &SynRecipe, max_norm: f64) -> Result<SideUpdate<B>> {
        let mut scale = recipe.gradient_step;
        if max_norm > 0.0 {
            scale /= max_norm;
        }
        let mut gradient = self.gradient;
        backend.scale(&mut gradient, scale).await?;
        Ok(SideUpdate {
            update: gradient,
            scale,
            first_value: self.first_value,
            weighted_value: self.weighted_value,
        })
    }
}

async fn mattes_pair<B: Backend>(
    backend: &B,
    fixed: &B::Image,
    moving: &B::Image,
    masks: Option<MaskPair<'_, B::Image>>,
    bins: usize,
    single: bool,
) -> Result<(MetricPair<B::Field>, Option<[f64; 2]>)> {
    Ok(match (single, masks) {
        (true, None) => (
            backend
                .mi_value_derivative_pair(fixed, moving, bins)
                .await?,
            None,
        ),
        (true, Some((fm, mm))) => (
            backend
                .mi_value_derivative_pair_masked(fixed, moving, fm, mm, bins)
                .await?,
            None,
        ),
        (false, None) => {
            let (pair, norms) = backend
                .mi_value_derivative_pair_with_norms(fixed, moving, bins)
                .await?;
            (pair, Some(norms))
        }
        (false, Some((fm, mm))) => {
            let (pair, norms) = backend
                .mi_value_derivative_pair_masked_with_norms(fixed, moving, fm, mm, bins)
                .await?;
            (pair, Some(norms))
        }
    })
}

async fn side_updates<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    fixed_mid: &[B::Image],
    moving_mid: &[B::Image],
    masks: &LevelMasks<B>,
) -> Result<(SideUpdate<B>, SideUpdate<B>)> {
    let channels = fixed_mid.len();
    let sum: f64 = recipe.channel_weights.iter().sum();
    let normalised: Vec<f64> = recipe.channel_weights.iter().map(|w| w / sum).collect();
    let mut fixed_side = SideAccumulator::<B>::new(channels);
    let mut moving_side = SideAccumulator::<B>::new(channels);
    let valid = masks.valid.as_ref();
    for c in 0..channels {
        let (fixed, moving, radius) = (&fixed_mid[c], &moving_mid[c], recipe.cc_radius);
        let (((fixed_value, fixed_derivative), (moving_value, moving_derivative)), norms) =
            match (recipe.metric(c), channels == 1, valid) {
                (SynMetric::Mattes, single, _) => {
                    mattes_pair(
                        backend,
                        fixed,
                        moving,
                        masks.mattes(),
                        recipe.mi_bins,
                        single,
                    )
                    .await?
                }
                (SynMetric::Cc, true, None) => (
                    backend
                        .cc_value_derivative_pair(fixed, moving, radius)
                        .await?,
                    None,
                ),
                (SynMetric::Cc, true, Some(valid)) => (
                    backend
                        .cc_value_derivative_pair_masked(fixed, moving, valid, radius)
                        .await?,
                    None,
                ),
                (SynMetric::Cc, false, None) => {
                    let (pair, norms) = backend
                        .cc_value_derivative_pair_with_norms(fixed, moving, radius)
                        .await?;
                    (pair, Some(norms))
                }
                (SynMetric::Cc, false, Some(valid)) => {
                    let (pair, norms) = backend
                        .cc_value_derivative_pair_masked_with_norms(fixed, moving, valid, radius)
                        .await?;
                    (pair, Some(norms))
                }
            };
        fixed_side
            .add(
                backend,
                normalised[c],
                fixed_value,
                fixed_derivative,
                norms.map(|n| n[0]),
            )
            .await?;
        moving_side
            .add(
                backend,
                normalised[c],
                moving_value,
                moving_derivative,
                norms.map(|n| n[1]),
            )
            .await?;
    }
    let fixed_side = fixed_side.smoothed(backend, recipe, &normalised).await?;
    let moving_side = moving_side.smoothed(backend, recipe, &normalised).await?;
    let (fixed_max, moving_max) = backend
        .max_scaled_norm_pair(&fixed_side.gradient, &moving_side.gradient)
        .await?;
    if !(fixed_max.is_finite() && moving_max.is_finite()) {
        return Err(invalid("syn", "the metric gradient is not finite"));
    }
    let fixed_update = fixed_side.finish(backend, recipe, fixed_max).await?;
    let moving_update = moving_side.finish(backend, recipe, moving_max).await?;
    Ok((fixed_update, moving_update))
}

async fn update_total_field<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    update: B::Field,
    field: B::Field,
    inverse: B::Field,
) -> Result<(B::Field, B::Field, InvertStats, InvertStats)> {
    let composed = backend.compose(&update, &field).await?;
    drop(update);
    drop(field);
    let tmp = backend
        .smooth_field(composed, recipe.total_field_variance)
        .await?;
    let ((new_inverse, stats_inverse), (new_field, stats_field)) = backend
        .invert_pair(tmp, inverse, &InvertParams::SYN)
        .await?;
    if [stats_inverse, stats_field]
        .iter()
        .any(|s| !(s.mean_error_norm.is_finite() && s.max_error_norm.is_finite()))
    {
        return Err(invalid("syn", "the displacement field is not finite"));
    }
    Ok((new_field, new_inverse, stats_inverse, stats_field))
}

/// The fixed and the moving mask of an MI channel; `None` for a mask that was not given.
type MaskPair<'a, I> = (Option<&'a I>, Option<&'a I>);

/// The masks of one iteration on the level grid: `valid`, the voxels inside both masks (for CC);
/// `fixed` and `moving`, each given mask on its own (for MI).
struct LevelMasks<B: Backend> {
    valid: Option<B::Image>,
    fixed: Option<B::Image>,
    moving: Option<B::Image>,
}

impl<B: Backend> LevelMasks<B> {
    fn none() -> Self {
        Self {
            valid: None,
            fixed: None,
            moving: None,
        }
    }

    fn mattes(&self) -> Option<MaskPair<'_, B::Image>> {
        if self.fixed.is_none() && self.moving.is_none() {
            return None;
        }
        Some((self.fixed.as_ref(), self.moving.as_ref()))
    }
}

struct StageMasks<B: Backend> {
    fixed: Option<B::Image>,
    moving: Option<B::Image>,
}

impl<B: Backend> StageMasks<B> {
    async fn upload(backend: &B, recipe: &SynRecipe) -> Result<Self> {
        Ok(Self {
            fixed: Self::upload_one(backend, &recipe.fixed_mask).await?,
            moving: Self::upload_one(backend, &recipe.moving_mask).await?,
        })
    }

    async fn upload_one(
        backend: &B,
        mask: &Option<crate::mask::MaskVolume>,
    ) -> Result<Option<B::Image>> {
        match mask {
            Some(m) => Ok(Some(
                backend.upload_image_owned(m.to_binary_volume()?).await?,
            )),
            None => Ok(None),
        }
    }

    fn given(&self) -> bool {
        self.fixed.is_some() || self.moving.is_some()
    }

    /// The masks of one iteration on the level grid, as ITK resamples them onto the virtual
    /// domain in `ComputeMetricGradientField`: the voxels inside both when `cc`, each given mask
    /// on its own when `mattes`.
    async fn resolve(
        &self,
        backend: &B,
        state: &State<B>,
        initial: &AffineTransform,
        prior: Option<&B::Field>,
        cc: bool,
        mattes: bool,
    ) -> Result<LevelMasks<B>> {
        if !self.given() {
            return Ok(LevelMasks::none());
        }
        let valid = if cc {
            self.validity(backend, state, initial, prior).await?
        } else {
            None
        };
        if !mattes {
            return Ok(LevelMasks {
                valid,
                fixed: None,
                moving: None,
            });
        }
        let fixed = match &self.fixed {
            Some(m) => Some(
                backend
                    .warp_mask_through_field(m, &state.f2m_inv, None, None, None)
                    .await?,
            ),
            None => None,
        };
        let moving = match &self.moving {
            Some(m) => Some(
                backend
                    .warp_mask_through_field(m, &state.m2m_inv, prior, Some(initial), None)
                    .await?,
            ),
            None => None,
        };
        Ok(LevelMasks {
            valid,
            fixed,
            moving,
        })
    }

    /// The voxels of the level grid inside both masks, as ITK resamples the masks onto the
    /// virtual domain in `ComputeMetricGradientField`; `None` without masks.
    async fn validity(
        &self,
        backend: &B,
        state: &State<B>,
        initial: &AffineTransform,
        prior: Option<&B::Field>,
    ) -> Result<Option<B::Image>> {
        let fixed = match &self.fixed {
            Some(m) => Some(
                backend
                    .warp_mask_through_field(m, &state.f2m_inv, None, None, None)
                    .await?,
            ),
            None => None,
        };
        match &self.moving {
            Some(m) => Ok(Some(
                backend
                    .warp_mask_through_field(m, &state.m2m_inv, prior, Some(initial), fixed)
                    .await?,
            )),
            None => Ok(fixed),
        }
    }
}

async fn upload_smoothed<B: Backend>(
    backend: &B,
    vol: &Volume<f64>,
    sigma: f64,
    units: SigmaUnits,
) -> Result<B::Image> {
    match smooth_for_level_cow(vol, sigma, units)? {
        Cow::Borrowed(v) => backend.upload_image(v).await,
        Cow::Owned(v) => backend.upload_image_owned(v).await,
    }
}

/// Runs the SyN stage described by `recipe`.
///
/// `initial` is the linear transform of the earlier stages (maps fixed points to moving points);
/// it enters the moving side of every metric evaluation and the final composite. Level
/// transitions resample all four fields onto the next virtual grid (linear, zero outside,
/// values unscaled).
pub fn run_syn<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    initial: &AffineTransform,
    options: &SynOptions<'_>,
) -> Result<SynResult> {
    block_ready(run_syn_async(backend, recipe, channels, initial, options))
}

/// [`run_syn`] as a future, for a backend whose operations wait for their device.
pub async fn run_syn_async<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    initial: &AffineTransform,
    options: &SynOptions<'_>,
) -> Result<SynResult> {
    run_syn_cached_async(backend, recipe, channels, initial, options, None).await
}

/// [`run_syn`] from `start`: after earlier SyN stages, their collapsed field enters the moving
/// side of every metric evaluation, and the result is collapsed with it (see [`SynResult`]).
/// The prior fields must lie on the grid of this stage's last level.
pub fn run_syn_from<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    start: SynStart<'_>,
    options: &SynOptions<'_>,
) -> Result<SynResult> {
    block_ready(run_syn_from_async(
        backend, recipe, channels, start, options,
    ))
}

/// [`run_syn_from`] as a future, for a backend whose operations wait for their device.
pub async fn run_syn_from_async<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    start: SynStart<'_>,
    options: &SynOptions<'_>,
) -> Result<SynResult> {
    let out = Box::pin(syn_fields(
        backend,
        recipe,
        channels,
        start,
        None,
        options.cancel,
    ))
    .await?;
    let warp = out.warp.to_displacement_field()?;
    let inverse_warp = out.inverse_warp.to_displacement_field()?;
    let pair = CompositePair::new(start.affine, warp.clone(), inverse_warp.clone())?;
    Ok(SynResult {
        pair,
        warp,
        inverse_warp,
        levels: out.levels,
    })
}

/// [`run_syn`] taking the smoothed copies of the first channel from `cache` (filled by the
/// linear stages on the same images, [`crate::linear::run_linear_chain_cached`]) instead of
/// smoothing them again; every pair it uses is removed from the cache.
pub fn run_syn_cached<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    initial: &AffineTransform,
    options: &SynOptions<'_>,
    cache: Option<&mut SmoothCache>,
) -> Result<SynResult> {
    block_ready(run_syn_cached_async(
        backend, recipe, channels, initial, options, cache,
    ))
}

/// [`run_syn_cached`] as a future, for a backend whose operations wait for their device.
pub async fn run_syn_cached_async<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    initial: &AffineTransform,
    options: &SynOptions<'_>,
    cache: Option<&mut SmoothCache>,
) -> Result<SynResult> {
    let out = Box::pin(syn_fields(
        backend,
        recipe,
        channels,
        SynStart::affine(initial),
        cache,
        options.cancel,
    ))
    .await?;
    let warp = out.warp.to_displacement_field()?;
    let inverse_warp = out.inverse_warp.to_displacement_field()?;
    let pair = CompositePair::new(initial, warp.clone(), inverse_warp.clone())?;
    Ok(SynResult {
        pair,
        warp,
        inverse_warp,
        levels: out.levels,
    })
}

pub(crate) async fn run_syn_pair_cached_async<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    initial: &AffineTransform,
    options: &SynOptions<'_>,
    cache: Option<&mut SmoothCache>,
) -> Result<(CompositePair, Vec<SynLevelTrace>)> {
    let out = Box::pin(syn_fields(
        backend,
        recipe,
        channels,
        SynStart::affine(initial),
        cache,
        options.cancel,
    ))
    .await?;
    let pair = CompositePair::new(
        initial,
        out.warp.to_displacement_field()?,
        out.inverse_warp.to_displacement_field()?,
    )?;
    Ok((pair, out.levels))
}

/// One SyN stage of a stage list from `start`, its fields collapsed with the earlier ones and
/// kept in `f64` for the next stage.
pub(crate) async fn run_syn_stage_cached_async<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    start: SynStart<'_>,
    options: &SynOptions<'_>,
    cache: Option<&mut SmoothCache>,
) -> Result<SynFields> {
    Box::pin(syn_fields(
        backend,
        recipe,
        channels,
        start,
        cache,
        options.cancel,
    ))
    .await
}

/// The fields of a SyN stage (`later`, `later_inverse`) collapsed with those of the SyN stages
/// before it (`earlier` on the backend, `earlier_inverse` on the host), as ANTs'
/// `CollapseDisplacementFieldTransforms` composes consecutive displacement fields: forward
/// `compose(displacement = earlier, warping = later)`, `later(x) + earlier(x + later(x))`;
/// inverse `compose(displacement = later_inverse, warping = earlier_inverse)`. The earlier
/// inverse is uploaded only once the forward field is done, so one field less is resident.
async fn collapse<B: Backend>(
    backend: &B,
    earlier: B::Field,
    earlier_inverse: &VectorField,
    later: B::Field,
    later_inverse: B::Field,
) -> Result<(B::Field, B::Field)> {
    let total = backend.compose(&earlier, &later).await?;
    drop(later);
    drop(earlier);
    let earlier_inverse = backend.upload_field(earlier_inverse).await?;
    let total_inverse = backend.compose(&later_inverse, &earlier_inverse).await?;
    Ok((total, total_inverse))
}

async fn syn_fields<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    start: SynStart<'_>,
    cache: Option<&mut SmoothCache>,
    cancel: Option<&AtomicBool>,
) -> Result<SynFields> {
    let (forward, inverse, levels) = Box::pin(syn_device_fields(
        backend, recipe, channels, start, cache, cancel,
    ))
    .await?;
    let warp = backend.download_field(&forward).await?;
    let inverse_warp = backend.download_field(&inverse).await?;
    Ok(SynFields {
        warp,
        inverse_warp,
        levels,
    })
}

/// The last SyN stage of a stage list from `start`: [`run_syn_stage_cached_async`] with its
/// collapsed fields brought back as `f32` displacement fields, one at a time; the forward field
/// is released on the backend before the inverse one is read.
pub(crate) async fn run_syn_last_stage_cached_async<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    start: SynStart<'_>,
    options: &SynOptions<'_>,
    cache: Option<&mut SmoothCache>,
) -> Result<(DisplacementField, DisplacementField, Vec<SynLevelTrace>)> {
    let (forward, inverse, levels) = Box::pin(syn_device_fields(
        backend,
        recipe,
        channels,
        start,
        cache,
        options.cancel,
    ))
    .await?;
    let warp = backend.download_displacement_field(&forward).await?;
    drop(forward);
    let inverse_warp = backend.download_displacement_field(&inverse).await?;
    Ok((warp, inverse_warp, levels))
}

async fn syn_device_fields<B: Backend>(
    backend: &B,
    recipe: &SynRecipe,
    channels: &[SynChannel],
    start: SynStart<'_>,
    mut cache: Option<&mut SmoothCache>,
    cancel: Option<&AtomicBool>,
) -> Result<(B::Field, B::Field, Vec<SynLevelTrace>)> {
    recipe.validate()?;
    if channels.is_empty() {
        return Err(invalid("syn", "no channels"));
    }
    if channels.len() != recipe.channel_weights.len() {
        return Err(invalid(
            "syn",
            "the number of channels differs from the number of channel weights",
        ));
    }
    let fixed_grid: Grid = *channels[0].fixed.grid();
    for c in channels {
        if !c.fixed.grid().approx_eq(&fixed_grid, 1e-9) {
            return Err(invalid(
                "syn",
                "fixed images of the channels differ in grid",
            ));
        }
    }
    let initial = start.affine;
    if let Some(prior) = start.prior {
        let factors = shrink_factors_per_dimension(
            recipe.shrink_factors[recipe.levels() - 1],
            fixed_grid.spacing(),
        );
        let last = level_virtual_grid(&fixed_grid, factors)?;
        if prior.warp.grid() != &last || prior.inverse_warp.grid() != &last {
            return Err(invalid(
                "syn",
                "the field of the earlier SyN stages is not on the grid of this stage's last \
                 level: every SyN stage must end with the same shrink factor",
            ));
        }
    }
    let prior = match start.prior {
        Some(p) => Some(backend.upload_field(&p.warp).await?),
        None => None,
    };
    let masks = StageMasks::<B>::upload(backend, recipe).await?;
    let mut state: Option<State<B>> = None;
    let mut levels = Vec::with_capacity(recipe.levels());
    for level in 0..recipe.levels() {
        check_cancel(cancel)?;
        let factors =
            shrink_factors_per_dimension(recipe.shrink_factors[level], fixed_grid.spacing());
        let grid = level_virtual_grid(&fixed_grid, factors)?;
        let sigma = recipe.smoothing_sigmas[level];
        let mut fixed_smooth = Vec::with_capacity(channels.len());
        let mut moving_smooth = Vec::with_capacity(channels.len());
        for (index, c) in channels.iter().enumerate() {
            let cached = match cache.as_deref_mut() {
                Some(cache) if index == 0 && sigma > 0.0 => {
                    cache.take(&c.fixed, &c.moving, sigma, recipe.sigma_units)
                }
                _ => None,
            };
            match cached {
                Some((f, m)) => {
                    fixed_smooth.push(backend.upload_image_owned(f).await?);
                    moving_smooth.push(backend.upload_image_owned(m).await?);
                }
                None => {
                    fixed_smooth
                        .push(upload_smoothed(backend, &c.fixed, sigma, recipe.sigma_units).await?);
                    moving_smooth.push(
                        upload_smoothed(backend, &c.moving, sigma, recipe.sigma_units).await?,
                    );
                }
            }
        }
        let mut st = match state.take() {
            None => State::<B> {
                f2m: backend.zero_field(&grid).await?,
                f2m_inv: backend.zero_field(&grid).await?,
                m2m: backend.zero_field(&grid).await?,
                m2m_inv: backend.zero_field(&grid).await?,
            },
            Some(prev) => {
                let State {
                    f2m,
                    f2m_inv,
                    m2m,
                    m2m_inv,
                } = prev;
                let next_f2m = backend.resample_field(&f2m, &grid).await?;
                drop(f2m);
                let next_f2m_inv = backend.resample_field(&f2m_inv, &grid).await?;
                drop(f2m_inv);
                let next_m2m = backend.resample_field(&m2m, &grid).await?;
                drop(m2m);
                let next_m2m_inv = backend.resample_field(&m2m_inv, &grid).await?;
                drop(m2m_inv);
                State::<B> {
                    f2m: next_f2m,
                    f2m_inv: next_f2m_inv,
                    m2m: next_m2m,
                    m2m_inv: next_m2m_inv,
                }
            }
        };
        let mattes = recipe.uses(SynMetric::Mattes);
        let mut monitor = WindowConvergenceMonitor::new(recipe.convergence_window);
        let mut converged = false;
        let mut trace = Vec::new();
        let limit = recipe.iterations[level];
        let mut current = 0u32;
        while current < limit && !converged {
            check_cancel(cancel)?;
            current += 1;
            let mut fixed_mid = Vec::with_capacity(channels.len());
            let mut moving_mid = Vec::with_capacity(channels.len());
            for c in 0..channels.len() {
                fixed_mid.push(
                    backend
                        .warp_through_field(&fixed_smooth[c], &st.f2m_inv, None, None)
                        .await?,
                );
                moving_mid.push(
                    backend
                        .warp_through_field(
                            &moving_smooth[c],
                            &st.m2m_inv,
                            prior.as_ref(),
                            Some(initial),
                        )
                        .await?,
                );
            }
            let level_masks = masks
                .resolve(
                    backend,
                    &st,
                    initial,
                    prior.as_ref(),
                    recipe.uses(SynMetric::Cc),
                    mattes,
                )
                .await?;
            let (fixed_side, moving_side) =
                side_updates(backend, recipe, &fixed_mid, &moving_mid, &level_masks).await?;
            drop(level_masks);
            let State {
                f2m,
                f2m_inv,
                m2m,
                m2m_inv,
            } = st;
            let (f2m, f2m_inv, inv_f_inv, inv_f) =
                update_total_field(backend, recipe, fixed_side.update, f2m, f2m_inv).await?;
            let (m2m, m2m_inv, inv_m_inv, inv_m) =
                update_total_field(backend, recipe, moving_side.update, m2m, m2m_inv).await?;
            st = State {
                f2m,
                f2m_inv,
                m2m,
                m2m_inv,
            };
            let energy_first = 0.5 * (fixed_side.first_value + moving_side.first_value);
            let energy_weighted = 0.5 * (fixed_side.weighted_value + moving_side.weighted_value);
            let energy = energy_first;
            monitor.add_energy_value(energy);
            let convergence_value = monitor.convergence_value();
            if convergence_value < recipe.convergence_threshold {
                converged = true;
            }
            trace.push(SynIteration {
                energy,
                energy_first_metric: energy_first,
                energy_weighted,
                convergence_value,
                fixed_update_scale: fixed_side.scale,
                moving_update_scale: moving_side.scale,
                inversions: [inv_f_inv, inv_f, inv_m_inv, inv_m],
            });
        }
        levels.push(SynLevelTrace {
            level,
            shrink_factors: factors,
            grid_size: grid.size(),
            iterations: trace,
            converged,
        });
        state = Some(st);
    }
    let st = state.ok_or_else(|| invalid("syn", "no levels were run"))?;
    let forward = backend.compose(&st.m2m_inv, &st.f2m).await?;
    let inverse = backend.compose(&st.f2m_inv, &st.m2m).await?;
    let (forward, inverse) = match (prior, start.prior) {
        (Some(w), Some(p)) => {
            drop(st);
            drop(masks);
            collapse(backend, w, &p.inverse_warp, forward, inverse).await?
        }
        _ => (forward, inverse),
    };
    Ok((forward, inverse, levels))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::error::RegError;
    use crate::exec::testing::{drive, YieldingCpu};
    use crate::linalg::IDENTITY3;
    use crate::transform::{TransformChain, TransformElem};

    fn smooth(grid: Grid, amp: f64, phase: f64) -> VectorField {
        VectorField::from_fn(grid, |p| {
            [
                amp * (0.21 * p[1] + phase).sin(),
                0.7 * amp * (0.17 * p[2] - phase).cos(),
                0.5 * amp * (0.13 * p[0] + 0.5 * phase).sin(),
            ]
        })
    }

    #[test]
    fn the_collapse_maps_grid_points_as_the_two_fields_in_sequence() {
        let grid = Grid::new(
            [14, 12, 11],
            [2.0, 1.5, 1.8],
            [-13.0, -8.0, -9.0],
            IDENTITY3,
        )
        .unwrap();
        let (w1, w1_inv) = (smooth(grid, 2.5, 0.3), smooth(grid, -2.0, 1.1));
        let (w2, w2_inv) = (smooth(grid, 1.5, 2.0), smooth(grid, -1.2, -0.4));
        let b = CpuBackend;
        let (total, total_inv) = block_ready(collapse(
            &b,
            w1.clone(),
            &w1_inv,
            w2.clone(),
            w2_inv.clone(),
        ))
        .unwrap();
        let field =
            |f: &VectorField| TransformElem::DisplacementField(f.to_displacement_field().unwrap());
        let forward = TransformChain::from_queue(vec![field(&w1), field(&w2)]);
        let inverse = TransformChain::from_queue(vec![field(&w2_inv), field(&w1_inv)]);
        let collapsed = TransformChain::from_queue(vec![field(&total)]);
        let collapsed_inv = TransformChain::from_queue(vec![field(&total_inv)]);
        let mut worst = 0.0f64;
        for idx in 0..grid.num_voxels() {
            let [i, j, k] = grid.ijk(idx);
            let x = grid.index_to_point([i as f64, j as f64, k as f64]);
            for (a, c) in [(&forward, &collapsed), (&inverse, &collapsed_inv)] {
                let (p, q) = (a.apply(x), c.apply(x));
                worst = (0..3).fold(worst, |w, n| w.max((p[n] - q[n]).abs()));
            }
        }
        assert!(worst < 1e-5, "{worst}");
        let swapped = b.compose(&w2, &w1).unwrap();
        let differs = swapped
            .data()
            .iter()
            .zip(total.data())
            .any(|(s, t)| (0..3).any(|n| (s[n] - t[n]).abs() > 1e-3));
        assert!(differs, "the order of the two fields matters");
    }

    #[test]
    fn between_voxels_the_collapsed_field_follows_the_two_fields_to_interpolation_error() {
        let grid = Grid::new([24, 22, 20], [2.0; 3], [-23.0, -21.0, -19.0], IDENTITY3).unwrap();
        let (w1, w1_inv) = (smooth(grid, 2.0, 0.3), smooth(grid, -2.0, 1.1));
        let (w2, w2_inv) = (smooth(grid, 1.0, 2.0), smooth(grid, -1.0, -0.4));
        let (total, _) = block_ready(collapse(
            &CpuBackend,
            w1.clone(),
            &w1_inv,
            w2.clone(),
            w2_inv,
        ))
        .unwrap();
        let mut worst = 0.0f64;
        let [nx, ny, nz] = grid.size();
        for idx in 0..grid.num_voxels() {
            let [i, j, k] = grid.ijk(idx);
            if i < 3 || j < 3 || k < 3 || i + 4 > nx || j + 4 > ny || k + 4 > nz {
                continue;
            }
            let c = [i as f64 + 0.37, j as f64 + 0.61, k as f64 + 0.5];
            let x = grid.index_to_point(c);
            let (Some(a), Some(b)) = (total.interpolate(x), w2.interpolate(x)) else {
                continue;
            };
            let q = [x[0] + b[0], x[1] + b[1], x[2] + b[2]];
            let d = w1.interpolate(q).unwrap_or([0.0; 3]);
            worst = (0..3).fold(worst, |w, n| w.max((a[n] - (b[n] + d[n])).abs()));
        }
        eprintln!(
            "largest difference between the collapsed and the chained fields: {worst:.3e} mm"
        );
        assert!(worst < 0.1, "{worst}");
    }

    fn blob(grid: Grid, shift: f64) -> Volume<f64> {
        let data = (0..grid.num_voxels())
            .map(|idx| {
                let [i, j, k] = grid.ijk(idx);
                let r2 = (i as f64 - 5.5 - shift).powi(2)
                    + (j as f64 - 5.5).powi(2)
                    + (k as f64 - 5.0).powi(2);
                (-r2 / 12.0).exp() + 0.01 * ((idx * 7919) % 13) as f64
            })
            .collect();
        Volume::from_vec(grid, data).unwrap()
    }

    fn same_bits<T: std::fmt::Debug>(a: &T, b: &T) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    fn not_ready(e: &RegError) -> bool {
        matches!(
            e,
            RegError::Invalid {
                what: "backend",
                ..
            }
        )
    }

    #[test]
    fn the_async_entry_points_on_a_backend_that_waits_equal_the_synchronous_ones() {
        let grid = Grid::new([12, 12, 11], [2.0; 3], [0.0; 3], IDENTITY3).unwrap();
        let channels = [SynChannel {
            fixed: blob(grid, 0.0),
            moving: blob(grid, 1.5),
        }];
        let recipe = SynRecipe {
            iterations: vec![3, 2],
            shrink_factors: vec![2, 1],
            smoothing_sigmas: vec![1.0, 0.0],
            ..SynRecipe::cohort()
        };
        let initial = AffineTransform::identity();
        let options = SynOptions::default();
        let b = CpuBackend;
        let sizes = [
            std::mem::size_of_val(&run_syn_async(&b, &recipe, &channels, &initial, &options)),
            std::mem::size_of_val(&run_syn_from_async(
                &b,
                &recipe,
                &channels,
                SynStart::affine(&initial),
                &options,
            )),
            std::mem::size_of_val(&run_syn_cached_async(
                &b, &recipe, &channels, &initial, &options, None,
            )),
        ];
        assert!(sizes.iter().all(|&s| s < 1024), "{sizes:?}");

        let plain = run_syn(&b, &recipe, &channels, &initial, &options).unwrap();
        assert!(plain.levels.iter().all(|l| !l.iterations.is_empty()));
        let (waited, pending) = drive(run_syn_async(
            &YieldingCpu,
            &recipe,
            &channels,
            &initial,
            &options,
        ));
        assert!(pending > 0);
        same_bits(&plain, &waited.unwrap());
        let e = run_syn(&YieldingCpu, &recipe, &channels, &initial, &options).unwrap_err();
        assert!(not_ready(&e), "{e}");

        let stage = block_ready(run_syn_stage_cached_async(
            &b,
            &recipe,
            &channels,
            SynStart::affine(&initial),
            &options,
            None,
        ))
        .unwrap();
        let prior = PriorWarp {
            warp: stage.warp,
            inverse_warp: stage.inverse_warp,
        };
        let start = SynStart {
            affine: &initial,
            prior: Some(&prior),
        };
        let from = run_syn_from(&b, &recipe, &channels, start, &options).unwrap();
        let (waited, pending) = drive(run_syn_from_async(
            &YieldingCpu,
            &recipe,
            &channels,
            start,
            &options,
        ));
        assert!(pending > 0);
        same_bits(&from, &waited.unwrap());
        let e = run_syn_from(&YieldingCpu, &recipe, &channels, start, &options).unwrap_err();
        assert!(not_ready(&e), "{e}");
    }

    fn mattes_recipe(metrics: Vec<SynMetric>) -> SynRecipe {
        SynRecipe {
            gradient_step: 0.2,
            iterations: vec![20, 10],
            shrink_factors: vec![2, 1],
            smoothing_sigmas: vec![1.0, 0.0],
            channel_weights: vec![1.0 / metrics.len() as f64; metrics.len()],
            channel_metrics: metrics,
            ..SynRecipe::cohort()
        }
    }

    fn inverted(v: &Volume<f64>) -> Volume<f64> {
        Volume::from_vec(*v.grid(), v.data().iter().map(|x| 2.0 - 1.5 * x).collect()).unwrap()
    }

    fn centre_shift(r: &SynResult, grid: &Grid) -> f64 {
        let c = grid.index_to_point([5.5, 5.5, 5.0]);
        let p = r.pair.forward.apply(c);
        p[0] - c[0]
    }

    #[test]
    fn mattes_syn_follows_a_shift_between_images_of_inverted_contrast() {
        let grid = Grid::new([12, 12, 11], [2.0; 3], [0.0; 3], IDENTITY3).unwrap();
        let channels = [SynChannel {
            fixed: blob(grid, 0.0),
            moving: inverted(&blob(grid, 1.5)),
        }];
        let recipe = mattes_recipe(vec![SynMetric::Mattes]);
        let initial = AffineTransform::identity();
        let r = run_syn(
            &CpuBackend,
            &recipe,
            &channels,
            &initial,
            &SynOptions::default(),
        )
        .unwrap();
        let shift = centre_shift(&r, &grid);
        let first = r.levels[0].iterations[0].energy;
        let last = r.levels[1].iterations.last().unwrap().energy;
        eprintln!("centre shift {shift:.3} mm of 3 mm; energy {first:.4} -> {last:.4}");
        assert!(first < 0.0 && last < first, "{first} {last}");
        assert!((shift - 3.0).abs() < 1.5, "{shift}");
    }

    #[test]
    fn masks_of_ones_leave_a_mattes_run_unchanged_and_mixed_channels_run_with_masks() {
        let grid = Grid::new([12, 12, 11], [2.0; 3], [0.0; 3], IDENTITY3).unwrap();
        let channels = [
            SynChannel {
                fixed: blob(grid, 0.0),
                moving: blob(grid, 1.5),
            },
            SynChannel {
                fixed: blob(grid, 0.0),
                moving: inverted(&blob(grid, 1.5)),
            },
        ];
        let initial = AffineTransform::identity();
        let options = SynOptions::default();
        let ones = crate::mask::MaskVolume::new(grid, vec![1.0; grid.num_voxels()]).unwrap();
        let mattes = mattes_recipe(vec![SynMetric::Mattes]);
        let plain = run_syn(&CpuBackend, &mattes, &channels[1..], &initial, &options).unwrap();
        for (fixed, moving) in [(true, false), (false, true), (true, true)] {
            let masked = SynRecipe {
                fixed_mask: fixed.then(|| ones.clone()),
                moving_mask: moving.then(|| ones.clone()),
                ..mattes.clone()
            };
            let r = run_syn(&CpuBackend, &masked, &channels[1..], &initial, &options).unwrap();
            same_bits(&plain, &r);
        }
        let half = crate::mask::MaskVolume::new(
            grid,
            (0..grid.num_voxels())
                .map(|idx| f64::from(u8::from(grid.ijk(idx)[2] < 8)))
                .collect(),
        )
        .unwrap();
        let mixed = mattes_recipe(vec![SynMetric::Cc, SynMetric::Mattes]);
        let open = run_syn(&CpuBackend, &mixed, &channels, &initial, &options).unwrap();
        let masked = SynRecipe {
            fixed_mask: Some(half.clone()),
            moving_mask: Some(half),
            ..mixed.clone()
        };
        let closed = run_syn(&CpuBackend, &masked, &channels, &initial, &options).unwrap();
        for r in [&open, &closed] {
            assert!(r.warp.data().iter().all(|v| v.is_finite()));
            assert!(
                (centre_shift(r, &grid) - 3.0).abs() < 1.5,
                "{}",
                centre_shift(r, &grid)
            );
        }
        assert_ne!(format!("{:?}", open.warp), format!("{:?}", closed.warp));
        let (waited, pending) = drive(run_syn_async(
            &YieldingCpu,
            &masked,
            &channels,
            &initial,
            &options,
        ));
        assert!(pending > 0);
        same_bits(&closed, &waited.unwrap());
    }
}

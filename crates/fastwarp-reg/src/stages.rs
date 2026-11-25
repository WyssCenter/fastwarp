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
// The stage chaining generalises `run_linear_chain_cached` in `linear.rs` (itself after ITK
// `itkImageRegistrationMethodv4.hxx`, `InitializeCenterOfLinearOutputTransform`, and ANTs
// `itkantsRegistrationHelper.hxx`, `AddLinearTransformToCompositeTransform` /
// `CollapseLinearTransforms`) from a fixed Rigid-then-Affine pair to any list of linear stages
// after any initial transform, followed by any number of SyN stages chained and collapsed as
// `itkantsRegistrationHelper.hxx` chains them (`CollapseDisplacementFieldTransforms`). See
// `docs/THIRD_PARTY_PORTS.md`.

//! Stage-list orchestration: an initial transform, any number of `Rigid` / `Affine` stages in any
//! order, then any number of SyN stages, as `antsRegistration` runs them with
//! `--collapse-output-transforms 1`, and the ANTs-named outputs of such a run.
//!
//! [`register_stages`] with [`StagePlan::cohort`] runs exactly what
//! [`crate::register::register_full`] runs (bit for bit; a test proves it).

use fastwarp_core::Stopwatch;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use rayon::prelude::*;

use crate::apply::{apply_chain, Interpolation};
use crate::backend::Backend;
use crate::error::{invalid, RegError, Result};
use crate::exec::block_ready;
use crate::grid::Grid;
use crate::init::{initial_transform, InitFeature};
use crate::linalg::IDENTITY3;
use crate::linear::{
    run_linear_stage_cached_async, with_center, LinearEngine, LinearKind, LinearRecipe,
    LinearStageResult, SmoothCache,
};
use crate::recipe::SynRecipe;
use crate::register::{
    check_winsorize, prepare_channels_with, LinearPlan, PreparedChannels, RegisterTimings,
    WINSORIZE_LOWER, WINSORIZE_UPPER,
};
use crate::syn::{run_syn_stage_cached_async, PriorWarp, SynLevelTrace, SynOptions, SynStart};
use crate::transform::ants_io::{
    with_suffix, AFFINE_SUFFIX, COMPOSITE_SUFFIX, INVERSE_COMPOSITE_SUFFIX, INVERSE_WARP_SUFFIX,
    WARP_SUFFIX,
};
use crate::transform::field_nifti::write_field_nifti_to;
use crate::transform::itk_h5::format_h5;
use crate::transform::itk_mat::format_mat;
use crate::transform::{
    AffineTransform, CompositePair, DisplacementField, ParametricTransform, TransformChain,
    TransformElem,
};
use crate::volume::Volume;

/// Where the chain starts (`--initial-moving-transform`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitialTransform {
    /// `[fixed,moving,1]`: the translation between the centres of intensity mass of the raw
    /// (resampled, not preprocessed) first channel.
    CentreOfMass,
    /// `Identity`: the identity, centred at the origin, so a first linear stage rotates about
    /// the origin as in ANTs.
    Identity,
    /// A given affine (fixed points to moving points), e.g. read from a `.mat` file; the first
    /// linear stage is centred at its centre.
    Affine(AffineTransform),
}

/// What runs: the start, the linear stages in order, the SyN stages after them, and the
/// intensity preprocessing of every metric image.
#[derive(Clone, Debug, PartialEq)]
pub struct StagePlan {
    /// Start of the chain.
    pub initial: InitialTransform,
    /// `Rigid` / `Affine` stages, run in this order on the first channel (any number, also none).
    pub linear: Vec<LinearRecipe>,
    /// The SyN stages on every channel, in this order after the linear stages (each starts from
    /// the collapsed result of everything before it, see [`crate::syn::SynStart`]); empty for a
    /// linear-only run. They must end with the same shrink factor, so their fields share a grid.
    pub syn: Vec<SynRecipe>,
    /// `Some((lower, upper))`: `--winsorize-image-intensities [lower,upper]`; `None`: ANTs'
    /// default rescale to `[0, 1]` with the quantile window `(0, 1)`.
    pub winsorize: Option<(f32, f32)>,
}

impl StagePlan {
    /// The cohort chain with `plan`'s linear stages and `syn`: centre of mass, `Rigid`, `Affine`,
    /// SyN, winsorised to `[0.005, 0.995]`.
    pub fn cohort(plan: &LinearPlan, syn: &SynRecipe) -> Self {
        Self {
            initial: InitialTransform::CentreOfMass,
            linear: vec![plan.rigid.clone(), plan.affine.clone()],
            syn: vec![syn.clone()],
            winsorize: Some((WINSORIZE_LOWER, WINSORIZE_UPPER)),
        }
    }

    /// Checks every stage and the preprocessing window; a plan needs at least one stage.
    pub fn validate(&self) -> Result<()> {
        if self.linear.is_empty() && self.syn.is_empty() {
            return Err(invalid("stage plan", "the plan has no stage"));
        }
        for stage in &self.linear {
            stage.validate()?;
        }
        for syn in &self.syn {
            syn.validate()?;
        }
        let last = |s: &SynRecipe| s.shrink_factors.last().copied();
        if self
            .syn
            .iter()
            .any(|s| last(s) != self.syn.first().and_then(last))
        {
            return Err(invalid(
                "stage plan",
                "the SyN stages end with different shrink factors: their fields must share the \
                 grid of the last level, so give them the same last shrink factor",
            ));
        }
        if let InitialTransform::Affine(a) = &self.initial {
            let finite = a
                .parameters()
                .iter()
                .chain(&a.center)
                .all(|v| v.is_finite());
            if !finite {
                return Err(invalid(
                    "stage plan",
                    "the initial transform has non-finite values",
                ));
            }
            a.inverse()
                .map_err(|_| invalid("stage plan", "the initial transform is singular"))?;
        }
        check_winsorize(self.winsorize)
    }

    /// Whether the plan has a SyN stage.
    pub fn is_deformable(&self) -> bool {
        !self.syn.is_empty()
    }
}

/// Runs `stages` one after another from `initial`: each stage optimises a fresh transform of its
/// kind, started as the identity centred where the previous linear transform is centred and
/// applied before everything earlier; the result is collapsed into one affine centred like the
/// last stage. With no stage, the result is `initial` itself.
///
/// `fixed` / `moving` are the preprocessed images of the first channel.
#[allow(clippy::too_many_arguments)]
pub fn run_linear_stages<E: LinearEngine>(
    engine: &E,
    initial: AffineTransform,
    stages: &[LinearRecipe],
    fixed: &Volume<f64>,
    moving: &Volume<f64>,
    cache: &mut SmoothCache,
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<LinearStageResult>, AffineTransform)> {
    block_ready(run_linear_stages_async(
        engine, initial, stages, fixed, moving, cache, cancel,
    ))
}

/// [`run_linear_stages`] as a future, for an engine that waits for its device.
#[allow(clippy::too_many_arguments)]
pub async fn run_linear_stages_async<E: LinearEngine>(
    engine: &E,
    initial: AffineTransform,
    stages: &[LinearRecipe],
    fixed: &Volume<f64>,
    moving: &Volume<f64>,
    cache: &mut SmoothCache,
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<LinearStageResult>, AffineTransform)> {
    let mut current = initial;
    let mut center = initial.center;
    let mut results = Vec::with_capacity(stages.len());
    for recipe in stages {
        let start = match recipe.kind {
            LinearKind::Rigid => ParametricTransform::euler3d([0.0; 6], center),
            LinearKind::Affine => {
                ParametricTransform::affine(&AffineTransform::new(IDENTITY3, [0.0; 3], center))
            }
        };
        let stage = run_linear_stage_cached_async(
            engine, recipe, fixed, moving, current, start, cache, cancel,
        )
        .await?;
        current = AffineTransform::compose(&current, &stage.affine);
        center = stage.affine.center;
        results.push(stage);
    }
    cache.clear_samples();
    let total = if results.is_empty() {
        initial
    } else {
        with_center(&current, center)
    };
    Ok((results, total))
}

/// Result of [`register_stages`].
#[derive(Clone, Debug)]
pub struct StagesOutput {
    /// With SyN: `Composite = [Affine, Warp]`, `InverseComposite = [InverseWarp, Affine^-1]`;
    /// linear-only: `[Affine]` and `[Affine^-1]` (what ANTs 2.6.5 writes for such a run with
    /// `--write-composite-transform 1`).
    pub pair: CompositePair,
    /// The first moving channel through the forward chain onto the registration grid.
    pub warped: Volume<f64>,
    /// The registration grid (grid of the fields).
    pub grid: Grid,
    /// The start of the chain.
    pub initial: AffineTransform,
    /// One result per linear stage, in order.
    pub linear: Vec<LinearStageResult>,
    /// The start and every linear stage collapsed into one affine (`0GenericAffine.mat`).
    pub total: AffineTransform,
    /// Per-level traces of each SyN stage, in order; empty without one.
    pub syn_levels: Vec<Vec<SynLevelTrace>>,
    /// Whether a SyN stage ran (the pair holds displacement fields).
    pub deformable: bool,
    /// Phase timings.
    pub timings: RegisterTimings,
}

/// Runs `plan`: resampling to `iso_mm` and preprocessing every channel, the start, the linear
/// stages on `engine` (first channel), then the SyN stages on `backend` (every channel), the first
/// from the collapsed affine, each later one from the affine and the collapsed field of the SyN
/// stages before it. After each SyN stage its field is collapsed with the earlier ones as
/// `antsRegistration --collapse-output-transforms 1` does, so the result is one affine and one
/// field per direction. Before every SyN stage that follows another stage the backend frees its
/// idle buffers ([`Backend::release_idle_buffers`]), so the device peak is that of one stage
/// (a later SyN stage also holds the earlier field). A plan without a SyN stage takes one
/// channel. The GPU capacity of the grid is the caller's to check.
///
/// The linear stages share one cache of smoothed full-resolution images (one pair per distinct
/// sigma, when the stages keep `cache_smoothed_images`) on the host until the SyN stages have
/// taken what they reuse; a long stage list with many different sigmas holds that many pairs.
#[allow(clippy::too_many_arguments)]
pub fn register_stages<B: Backend, E: LinearEngine>(
    backend: &B,
    engine: &E,
    plan: &StagePlan,
    fixed: &[Volume<f64>],
    moving: &[Volume<f64>],
    iso_mm: Option<f64>,
    options: &SynOptions<'_>,
) -> Result<StagesOutput> {
    block_ready(register_stages_async(
        backend, engine, plan, fixed, moving, iso_mm, options,
    ))
}

/// [`register_stages`] as a future, for a backend and an engine that wait for their device.
#[allow(clippy::too_many_arguments)]
pub async fn register_stages_async<B: Backend, E: LinearEngine>(
    backend: &B,
    engine: &E,
    plan: &StagePlan,
    fixed: &[Volume<f64>],
    moving: &[Volume<f64>],
    iso_mm: Option<f64>,
    options: &SynOptions<'_>,
) -> Result<StagesOutput> {
    plan.validate()?;
    if plan
        .syn
        .iter()
        .any(|syn| syn.channel_weights.len() != fixed.len())
    {
        return Err(invalid(
            "stage plan",
            "the number of channels differs from the number of channel weights",
        ));
    }
    if plan.syn.is_empty() && fixed.len() > 1 {
        return Err(invalid(
            "stage plan",
            "a plan without a SyN stage registers one channel; the linear stages use the first",
        ));
    }
    let mut timings = RegisterTimings::default();
    let PreparedChannels {
        channels,
        fixed_raw,
        moving_raw,
    } = prepare_channels_with(fixed, moving, iso_mm, plan.winsorize, &mut timings)?;
    let t = Stopwatch::start();
    let initial = match plan.initial {
        InitialTransform::CentreOfMass => {
            initial_transform(&fixed_raw[0], &moving_raw[0], InitFeature::CentreOfMass)?
        }
        InitialTransform::Identity => AffineTransform::identity(),
        InitialTransform::Affine(a) => a,
    };
    let mut cache = SmoothCache::default();
    let (linear, total) = run_linear_stages_async(
        engine,
        initial,
        &plan.linear,
        &channels[0].fixed,
        &channels[0].moving,
        &mut cache,
        options.cancel,
    )
    .await?;
    timings.linear_s = t.elapsed().as_secs_f64();
    drop(fixed_raw);
    let moving_first = moving_raw
        .into_iter()
        .next()
        .ok_or_else(|| invalid("register", "no moving channel"))?;
    let t = Stopwatch::start();
    let mut prior: Option<PriorWarp> = None;
    let mut syn_levels = Vec::with_capacity(plan.syn.len());
    for (i, recipe) in plan.syn.iter().enumerate() {
        if i > 0 || !plan.linear.is_empty() {
            backend.release_idle_buffers();
        }
        let start = SynStart {
            affine: &total,
            prior: prior.as_ref(),
        };
        let out = run_syn_stage_cached_async(
            backend,
            recipe,
            &channels,
            start,
            options,
            Some(&mut cache),
        )
        .await?;
        prior = Some(PriorWarp {
            warp: out.warp,
            inverse_warp: out.inverse_warp,
        });
        syn_levels.push(out.levels);
    }
    drop(cache);
    let pair = match prior {
        Some(p) => {
            timings.syn_s = t.elapsed().as_secs_f64();
            CompositePair::new(
                &total,
                p.warp.to_displacement_field()?,
                p.inverse_warp.to_displacement_field()?,
            )?
        }
        None => CompositePair::linear(&total)?,
    };
    let grid = *channels[0].fixed.grid();
    drop(channels);
    let t = Stopwatch::start();
    let warped = apply_chain(
        &moving_first,
        &grid,
        &pair.forward,
        Interpolation::Linear,
        0.0,
    )?;
    timings.warp_s = t.elapsed().as_secs_f64();
    Ok(StagesOutput {
        pair,
        warped,
        grid,
        initial,
        linear,
        total,
        syn_levels,
        deformable: plan.is_deformable(),
        timings,
    })
}

/// Which transform files [`write_stage_outputs`] writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputLayout {
    /// `Composite.h5`, `InverseComposite.h5` and `0GenericAffine.mat` (what `register_syn` has
    /// always written; ANTs `--write-composite-transform 1` writes the two composites).
    Composite,
    /// ANTs' `--collapse-output-transforms 1 --write-composite-transform 0` files:
    /// `0GenericAffine.mat`, plus `1Warp.nii.gz` and `1InverseWarp.nii.gz` when a SyN stage ran.
    Split,
    /// Both sets.
    Both,
}

fn field_at(chain: &TransformChain, index: usize) -> Result<&DisplacementField> {
    match chain.queue().get(index) {
        Some(TransformElem::DisplacementField(f)) => Ok(f),
        _ => Err(invalid(
            "write outputs",
            "the composite pair holds no displacement field where one was expected",
        )),
    }
}

/// Writes the transforms of `out` under `prefix` in `layout` (ANTs naming, the prefix taken as
/// text) and, with `write_warped`, `<prefix>_Warped.nii.gz`; returns the paths written, in the
/// order composites, `0GenericAffine.mat`, warp, inverse warp, warped image. The files are
/// written in parallel: when one fails, the others are still written and one of the errors is
/// returned.
pub fn write_stage_outputs(
    prefix: &Path,
    out: &StagesOutput,
    layout: OutputLayout,
    write_warped: bool,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    visit_stage_outputs(prefix, out, layout, write_warped, |path, content| {
        files.push((path, content));
        Ok(())
    })?;
    files
        .into_par_iter()
        .map(|(path, content)| {
            let file = std::fs::File::create(&path).map_err(|source| RegError::Io {
                path: path.clone(),
                source,
            })?;
            content.write_to(&path, std::io::BufWriter::with_capacity(1 << 20, file))?;
            Ok(path)
        })
        .collect()
}

/// One file of [`write_stage_outputs`], to be serialised into a writer.
#[derive(Clone, Copy)]
pub enum StageContent<'a> {
    /// An ITK HDF5 composite transform.
    Composite(&'a TransformChain),
    /// The collapsed affine as an ITK `.mat` file.
    Affine(&'a AffineTransform),
    /// A displacement field as a gzip-compressed NIfTI file.
    Field(&'a DisplacementField),
    /// The warped image as a gzip-compressed `f32` NIfTI file.
    Warped(&'a Volume<f64>),
}

impl StageContent<'_> {
    /// Serialises the file into `out` and flushes it; `label` is what errors report. Compressed
    /// NIfTI content streams into `out` without a whole-file copy.
    pub fn write_to<W: Write>(&self, label: &Path, mut out: W) -> Result<W> {
        let io = |source| RegError::Io {
            path: label.to_path_buf(),
            source,
        };
        match *self {
            StageContent::Composite(chain) => {
                out.write_all(&format_h5(chain)?).map_err(io)?;
            }
            StageContent::Affine(total) => {
                out.write_all(&format_mat(&[TransformElem::Affine(*total)])?)
                    .map_err(io)?;
            }
            StageContent::Field(field) => {
                out = write_field_nifti_to(label, field, true, out)?;
            }
            StageContent::Warped(warped) => {
                let f32_data: Vec<f32> = warped.data().iter().map(|v| *v as f32).collect();
                out =
                    Volume::from_vec(*warped.grid(), f32_data)?.write_nifti_to(label, out, true)?;
            }
        }
        out.flush().map_err(io)?;
        Ok(out)
    }
}

/// Calls `visit` for each file [`write_stage_outputs`] writes, in the same order, with its path
/// (`prefix` with the ANTs suffix appended) and its content. Nothing is serialised until the
/// visitor asks, so a visitor that writes and drops each file holds one file at a time.
pub fn visit_stage_outputs<'a>(
    prefix: &Path,
    out: &'a StagesOutput,
    layout: OutputLayout,
    write_warped: bool,
    mut visit: impl FnMut(PathBuf, StageContent<'a>) -> Result<()>,
) -> Result<()> {
    if matches!(layout, OutputLayout::Composite | OutputLayout::Both) {
        visit(
            with_suffix(prefix, COMPOSITE_SUFFIX),
            StageContent::Composite(&out.pair.forward),
        )?;
        visit(
            with_suffix(prefix, INVERSE_COMPOSITE_SUFFIX),
            StageContent::Composite(&out.pair.inverse),
        )?;
    }
    visit(
        with_suffix(prefix, AFFINE_SUFFIX),
        StageContent::Affine(&out.total),
    )?;
    if out.deformable && matches!(layout, OutputLayout::Split | OutputLayout::Both) {
        visit(
            with_suffix(prefix, WARP_SUFFIX),
            StageContent::Field(field_at(&out.pair.forward, 1)?),
        )?;
        visit(
            with_suffix(prefix, INVERSE_WARP_SUFFIX),
            StageContent::Field(field_at(&out.pair.inverse, 0)?),
        )?;
    }
    if write_warped {
        visit(
            with_suffix(prefix, "_Warped.nii.gz"),
            StageContent::Warped(&out.warped),
        )?;
    }
    Ok(())
}

/// The files [`write_stage_outputs`] writes as `(name, content)` pairs in the same order, the
/// name being the path as text. Nothing touches the file system.
pub fn stage_output_files(
    prefix: &Path,
    out: &StagesOutput,
    layout: OutputLayout,
    write_warped: bool,
) -> Result<Vec<(String, Vec<u8>)>> {
    let mut files = Vec::new();
    visit_stage_outputs(prefix, out, layout, write_warped, |path, content| {
        let name = path.to_string_lossy().into_owned();
        let bytes = content.write_to(Path::new(&name), Vec::new())?;
        files.push((name, bytes));
        Ok(())
    })?;
    Ok(files)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::linear::CpuLinearEngine;
    use crate::register::register_full;
    use crate::transform::{read_composite_pair, read_transform_file};

    fn blob_volume(n: usize, spacing: f64, shift: [f64; 3]) -> Volume<f64> {
        let g = Grid::new([n, n, n], [spacing; 3], [0.0; 3], IDENTITY3).unwrap();
        let c = (n as f64 - 1.0) * spacing / 2.0;
        let data = (0..g.num_voxels())
            .map(|lin| {
                let [i, j, k] = g.ijk(lin);
                let p = g.index_to_point([i as f64, j as f64, k as f64]);
                let r2 = (p[0] - c - shift[0]).powi(2)
                    + (p[1] - c - shift[1]).powi(2)
                    + (p[2] - c - shift[2]).powi(2);
                let lobe = (p[0] - c - shift[0] - 6.0).powi(2)
                    + (p[1] - c - shift[1] + 3.0).powi(2)
                    + (p[2] - c - shift[2]).powi(2);
                let noise = ((lin * 2654435761) % 1000) as f64 / 20000.0;
                (-r2 / (2.0 * (0.22 * n as f64 * spacing).powi(2))).exp()
                    + 0.6 * (-lobe / 18.0).exp()
                    + noise
            })
            .collect();
        Volume::from_vec(g, data).unwrap()
    }

    fn small(r: LinearRecipe) -> LinearRecipe {
        LinearRecipe {
            iterations: vec![12, 8],
            shrink_factors: vec![2, 1],
            smoothing_sigmas: vec![1.0, 0.0],
            ..r
        }
    }

    fn small_syn() -> SynRecipe {
        SynRecipe {
            iterations: vec![6, 4],
            shrink_factors: vec![2, 1],
            smoothing_sigmas: vec![1.0, 0.0],
            ..SynRecipe::cohort()
        }
    }

    fn pair() -> (Vec<Volume<f64>>, Vec<Volume<f64>>) {
        (
            vec![blob_volume(24, 2.0, [0.0; 3])],
            vec![blob_volume(24, 2.0, [4.0, -2.0, 1.0])],
        )
    }

    fn tmp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("fastwarp_reg_stages_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_cohort_plan_reproduces_register_full_bit_for_bit() {
        let (fx, mv) = pair();
        let linear = LinearPlan {
            rigid: small(LinearRecipe::cohort_rigid()),
            affine: small(LinearRecipe::cohort_affine()),
        };
        let syn = small_syn();
        let options = SynOptions::default();
        let full = register_full(
            &CpuBackend,
            &CpuLinearEngine,
            &linear,
            &syn,
            &fx,
            &mv,
            Some(2.0),
            &options,
        )
        .unwrap();
        let staged = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &StagePlan::cohort(&linear, &syn),
            &fx,
            &mv,
            Some(2.0),
            &options,
        )
        .unwrap();
        let chain = full.linear.as_ref().unwrap();
        assert_eq!(staged.initial, chain.initial);
        assert_eq!(staged.total, chain.total);
        assert_eq!(staged.linear.len(), 2);
        assert_eq!(staged.linear[0].affine, chain.rigid.affine);
        assert_eq!(staged.linear[1].affine, chain.affine.affine);
        assert_eq!(staged.pair, full.pair);
        assert_eq!(staged.warped, full.warped);
        assert_eq!(staged.syn_levels, vec![full.levels.clone()]);
        assert!(staged.deformable);
    }

    #[test]
    fn stage_output_files_and_written_files_equal_the_legacy_writers() {
        use crate::register::write_linear_output;
        use crate::transform::field_nifti::write_field_nifti;
        use crate::transform::write_composite_pair;
        let out = run_plan(vec![small_syn()]);
        for (layout, warped) in [
            (OutputLayout::Composite, false),
            (OutputLayout::Split, true),
            (OutputLayout::Both, true),
        ] {
            let reference = tmp("reference");
            let rp = reference.join("sub_");
            if matches!(layout, OutputLayout::Composite | OutputLayout::Both) {
                write_composite_pair(&rp, &out.pair).unwrap();
            }
            write_linear_output(&rp, &out.total).unwrap();
            if matches!(layout, OutputLayout::Split | OutputLayout::Both) {
                write_field_nifti(
                    &with_suffix(&rp, WARP_SUFFIX),
                    field_at(&out.pair.forward, 1).unwrap(),
                )
                .unwrap();
                write_field_nifti(
                    &with_suffix(&rp, INVERSE_WARP_SUFFIX),
                    field_at(&out.pair.inverse, 0).unwrap(),
                )
                .unwrap();
            }
            if warped {
                let data: Vec<f32> = out.warped.data().iter().map(|v| *v as f32).collect();
                Volume::from_vec(*out.warped.grid(), data)
                    .unwrap()
                    .save_nifti(&with_suffix(&rp, "_Warped.nii.gz"))
                    .unwrap();
            }

            let dir = tmp("files");
            let prefix = dir.join("sub_");
            let files = stage_output_files(&prefix, &out, layout, warped).unwrap();
            let written = write_stage_outputs(&prefix, &out, layout, warped).unwrap();
            assert_eq!(files.len(), written.len());
            let mut count = 0;
            for entry in std::fs::read_dir(&reference).unwrap() {
                let entry = entry.unwrap();
                let expected = std::fs::read(entry.path()).unwrap();
                let file_name = entry.file_name();
                let in_files = files
                    .iter()
                    .find(|(n, _)| Path::new(n).file_name() == Some(file_name.as_os_str()))
                    .unwrap();
                assert_eq!(in_files.1, expected, "{file_name:?}");
                let on_disk = std::fs::read(dir.join(&file_name)).unwrap();
                assert_eq!(on_disk, expected, "{file_name:?}");
                count += 1;
            }
            assert_eq!(count, files.len());
            for ((name, _), path) in files.iter().zip(&written) {
                assert_eq!(Path::new(name), path.as_path());
            }
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&reference);
        }
    }

    fn run_plan(syn: Vec<SynRecipe>) -> StagesOutput {
        let (fx, mv) = pair();
        let plan = StagePlan {
            initial: InitialTransform::Identity,
            linear: Vec::new(),
            syn,
            winsorize: Some((WINSORIZE_LOWER, WINSORIZE_UPPER)),
        };
        register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &fx,
            &mv,
            None,
            &SynOptions::default(),
        )
        .unwrap()
    }

    fn fields(out: &StagesOutput) -> (DisplacementField, DisplacementField) {
        (
            field_at(&out.pair.forward, 1).unwrap().clone(),
            field_at(&out.pair.inverse, 0).unwrap().clone(),
        )
    }

    fn max_diff(a: &DisplacementField, b: &DisplacementField) -> f64 {
        a.data()
            .iter()
            .zip(b.data())
            .fold(0.0f64, |m, (x, y)| m.max(f64::from((x - y).abs())))
    }

    fn fine_syn(iterations: u32) -> SynRecipe {
        SynRecipe {
            iterations: vec![iterations],
            shrink_factors: vec![1],
            smoothing_sigmas: vec![0.0],
            convergence_threshold: -1.0,
            ..SynRecipe::cohort()
        }
    }

    #[test]
    fn a_second_syn_stage_without_iterations_keeps_the_first_stage_s_fields() {
        let one = run_plan(vec![small_syn()]);
        let two = run_plan(vec![small_syn(), fine_syn(0)]);
        let ((f1, i1), (f2, i2)) = (fields(&one), fields(&two));
        assert!(max_diff(&f1, &f2) < 1e-6, "{}", max_diff(&f1, &f2));
        assert_eq!(i1, i2);
        assert_eq!(two.syn_levels.len(), 2);
        assert_eq!(two.syn_levels[0], one.syn_levels[0]);
        assert!(two.syn_levels[1][0].iterations.is_empty());
        assert_eq!(two.total, one.total);
        let worst = one
            .warped
            .data()
            .iter()
            .zip(two.warped.data())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn a_second_syn_stage_refines_and_a_zero_prior_changes_nothing() {
        let one = run_plan(vec![small_syn()]);
        let two = run_plan(vec![small_syn(), fine_syn(6)]);
        let (fx, _) = pair();
        let err = |out: &StagesOutput| {
            fx[0]
                .data()
                .iter()
                .zip(out.warped.data())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        };
        assert!(err(&two) < err(&one), "{} {}", err(&two), err(&one));
        assert_eq!(two.syn_levels[1][0].iterations.len(), 6);
        let (f1, _) = fields(&one);
        let (f2, _) = fields(&two);
        assert!(max_diff(&f1, &f2) > 1e-2);

        let (fx, mv) = pair();
        let mut t = RegisterTimings::default();
        let prepared = prepare_channels_with(
            &fx,
            &mv,
            None,
            Some((WINSORIZE_LOWER, WINSORIZE_UPPER)),
            &mut t,
        )
        .unwrap();
        let identity = AffineTransform::identity();
        let recipe = fine_syn(4);
        let alone = crate::syn::run_syn(
            &CpuBackend,
            &recipe,
            &prepared.channels,
            &identity,
            &SynOptions::default(),
        )
        .unwrap();
        let grid = *prepared.channels[0].fixed.grid();
        let zero = PriorWarp {
            warp: crate::field_ops::VectorField::zeros(grid),
            inverse_warp: crate::field_ops::VectorField::zeros(grid),
        };
        let after_zero = crate::syn::run_syn_from(
            &CpuBackend,
            &recipe,
            &prepared.channels,
            SynStart {
                affine: &identity,
                prior: Some(&zero),
            },
            &SynOptions::default(),
        )
        .unwrap();
        assert_eq!(after_zero.warp, alone.warp);
        assert_eq!(after_zero.levels, alone.levels);
        assert!(max_diff(&after_zero.inverse_warp, &alone.inverse_warp) < 1e-6);
    }

    #[test]
    fn a_fixed_mask_on_the_second_syn_stage_changes_the_field_near_the_mask_only() {
        let (fx, _) = pair();
        let g = *fx[0].grid();
        let centre = [30.0, 20.0, 23.0];
        let radius = 7.0;
        let dist = |p: [f64; 3]| {
            (0..3)
                .map(|a| (p[a] - centre[a]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mask = crate::mask::MaskVolume::from_volume(
            Volume::from_vec(
                g,
                (0..g.num_voxels())
                    .map(|idx| {
                        let [i, j, k] = g.ijk(idx);
                        f64::from(u8::from(
                            dist(g.index_to_point([i as f64, j as f64, k as f64])) < radius,
                        ))
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let one = run_plan(vec![small_syn()]);
        let second = SynRecipe {
            fixed_mask: Some(mask),
            ..fine_syn(8)
        };
        let two = run_plan(vec![small_syn(), second]);
        let unmasked = run_plan(vec![small_syn(), fine_syn(8)]);
        let (f1, _) = fields(&one);
        let (f2, _) = fields(&two);
        let (fu, _) = fields(&unmasked);
        let (mut inside, mut far, mut far_unmasked) = (0.0f64, 0.0f64, 0.0f64);
        for idx in 0..g.num_voxels() {
            let [i, j, k] = g.ijk(idx);
            let d = dist(g.index_to_point([i as f64, j as f64, k as f64]));
            let change = |f: &DisplacementField| {
                (0..3)
                    .map(|a| f64::from(f.data()[3 * idx + a] - f1.data()[3 * idx + a]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            if d < radius {
                inside = inside.max(change(&f2));
            } else if d > radius + 14.0 {
                far = far.max(change(&f2));
                far_unmasked = far_unmasked.max(change(&fu));
            }
        }
        eprintln!("largest change inside the mask {inside:.3e} mm, far from it {far:.3e} mm (unmasked {far_unmasked:.3e} mm)");
        assert!(inside > 0.05, "{inside}");
        assert!(far_unmasked > 0.05, "{far_unmasked}");
        assert!(far < 1e-3 * inside, "{far} {inside}");
    }

    #[test]
    fn a_third_syn_stage_collapses_onto_the_first_two() {
        let two = run_plan(vec![small_syn(), fine_syn(4)]);
        let three = run_plan(vec![small_syn(), fine_syn(4), fine_syn(0)]);
        let refined = run_plan(vec![small_syn(), fine_syn(4), fine_syn(3)]);
        assert_eq!(three.syn_levels.len(), 3);
        assert_eq!(three.syn_levels[..2], two.syn_levels[..]);
        let ((f2, i2), (f3, i3)) = (fields(&two), fields(&three));
        assert!(max_diff(&f2, &f3) < 1e-6, "{}", max_diff(&f2, &f3));
        assert_eq!(i2, i3);
        let (fr, _) = fields(&refined);
        assert!(max_diff(&fr, &f2) > 1e-3);
        assert_eq!(refined.syn_levels[2][0].iterations.len(), 3);
    }

    #[test]
    fn syn_stages_must_end_on_one_grid() {
        let (fx, mv) = pair();
        let coarse_end = SynRecipe {
            shrink_factors: vec![2],
            ..fine_syn(2)
        };
        let plan = StagePlan {
            initial: InitialTransform::Identity,
            linear: Vec::new(),
            syn: vec![small_syn(), coarse_end],
            winsorize: None,
        };
        let err = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &fx,
            &mv,
            None,
            &SynOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("same last shrink factor"), "{err}");
    }

    #[test]
    fn a_linear_only_plan_writes_the_collapsed_affine_and_linear_composites() {
        let (fx, mv) = pair();
        let plan = StagePlan {
            initial: InitialTransform::CentreOfMass,
            linear: vec![
                small(LinearRecipe::cohort_rigid()),
                small(LinearRecipe::cohort_affine()),
                small(LinearRecipe::cohort_rigid()),
            ],
            syn: Vec::new(),
            winsorize: None,
        };
        let out = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &fx,
            &mv,
            None,
            &SynOptions::default(),
        )
        .unwrap();
        assert!(!out.deformable && out.syn_levels.is_empty());
        assert_eq!(out.linear.len(), 3);
        assert_eq!(out.total.center, out.linear[2].affine.center);
        let c = 23.0;
        let mapped = out.total.apply([c, c, c]);
        let want = [c + 4.0, c - 2.0, c + 1.0];
        for a in 0..3 {
            assert!((mapped[a] - want[a]).abs() < 1.0, "{mapped:?}");
        }
        let d = tmp("linear");
        let prefix = d.join("o_");
        let files = write_stage_outputs(&prefix, &out, OutputLayout::Both, true).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "o_Composite.h5",
                "o_InverseComposite.h5",
                "o_0GenericAffine.mat",
                "o__Warped.nii.gz"
            ]
        );
        let back = read_composite_pair(&prefix).unwrap();
        assert_eq!(back.forward.len(), 1);
        assert_eq!(back.inverse.len(), 1);
        let p = [5.0, 9.0, 11.0];
        let q = back.inverse.apply(back.forward.apply(p));
        for a in 0..3 {
            assert!((q[a] - p[a]).abs() < 1e-9);
        }
        let mat = read_transform_file(&with_suffix(&prefix, AFFINE_SUFFIX)).unwrap();
        assert_eq!(mat.queue(), &[TransformElem::Affine(out.total)]);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_syn_only_plan_from_the_identity_writes_the_identity_affine_and_split_fields() {
        let (fx, mv) = pair();
        let plan = StagePlan {
            initial: InitialTransform::Identity,
            linear: Vec::new(),
            syn: vec![small_syn()],
            winsorize: Some((WINSORIZE_LOWER, WINSORIZE_UPPER)),
        };
        let out = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &fx,
            &mv,
            None,
            &SynOptions::default(),
        )
        .unwrap();
        assert!(out.deformable && out.linear.is_empty());
        assert_eq!(out.total, AffineTransform::identity());
        assert_eq!(out.syn_levels.len(), 1);
        assert_eq!(out.syn_levels[0].len(), 2);
        let d = tmp("synonly");
        let prefix = d.join("o_");
        let files = write_stage_outputs(&prefix, &out, OutputLayout::Split, false).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "o_0GenericAffine.mat",
                "o_1Warp.nii.gz",
                "o_1InverseWarp.nii.gz"
            ]
        );
        assert!(!d.join("o_Composite.h5").exists());
        let warp = read_transform_file(&d.join("o_1Warp.nii.gz")).unwrap();
        let mat = read_transform_file(&d.join("o_0GenericAffine.mat")).unwrap();
        let mut chain = mat;
        chain.extend(warp);
        let err = |a: &Volume<f64>| {
            a.data()
                .iter()
                .zip(fx[0].data())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
        };
        let resampled =
            apply_chain(&mv[0], fx[0].grid(), &chain, Interpolation::Linear, 0.0).unwrap();
        assert!(err(&resampled) < err(&mv[0]));
        let probe = [20.0, 21.0, 25.0];
        let a = chain.apply(probe);
        let b = out.pair.forward.apply(probe);
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-5, "{a:?} {b:?}");
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_given_affine_starts_the_chain_and_centres_the_first_stage() {
        let (fx, mv) = pair();
        let start = AffineTransform::new(IDENTITY3, [3.0, -1.5, 0.5], [10.0, 12.0, 14.0]);
        let plan = StagePlan {
            initial: InitialTransform::Affine(start),
            linear: vec![small(LinearRecipe::cohort_affine())],
            syn: Vec::new(),
            winsorize: Some((WINSORIZE_LOWER, WINSORIZE_UPPER)),
        };
        let out = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &fx,
            &mv,
            None,
            &SynOptions::default(),
        )
        .unwrap();
        assert_eq!(out.initial, start);
        assert_eq!(out.linear[0].affine.center, start.center);
        assert_eq!(out.total.center, start.center);
        let none = StagePlan {
            linear: Vec::new(),
            ..plan.clone()
        };
        let err = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &none,
            &fx,
            &mv,
            None,
            &SynOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("no stage"), "{err}");
    }

    #[test]
    fn the_preprocessing_window_is_the_plan_s() {
        let (fx, mv) = pair();
        let mut t = RegisterTimings::default();
        let default = crate::register::prepare_channels_full(&fx, &mv, None, &mut t).unwrap();
        let explicit = prepare_channels_with(
            &fx,
            &mv,
            None,
            Some((WINSORIZE_LOWER, WINSORIZE_UPPER)),
            &mut t,
        )
        .unwrap();
        assert_eq!(default.channels[0].fixed, explicit.channels[0].fixed);
        let plain = prepare_channels_with(&fx, &mv, None, None, &mut t).unwrap();
        let direct = crate::preprocess::preprocess_volume_no_winsorize(&fx[0]).unwrap();
        assert_eq!(plain.channels[0].fixed, direct);
        assert_ne!(plain.channels[0].fixed, default.channels[0].fixed);
        for bad in [(0.5, 0.5), (-0.1, 0.9), (0.1, 1.5), (f32::NAN, 0.9)] {
            assert!(prepare_channels_with(&fx, &mv, None, Some(bad), &mut t).is_err());
        }
    }

    #[test]
    fn a_plan_checks_its_stages_and_channels_before_any_work() {
        let (fx, mv) = pair();
        let mut syn = small_syn();
        syn.channel_weights = vec![0.5, 0.5];
        let plan = StagePlan {
            initial: InitialTransform::Identity,
            linear: Vec::new(),
            syn: vec![syn],
            winsorize: None,
        };
        let err = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &fx,
            &mv,
            None,
            &SynOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("channel weights"), "{err}");
        let two = StagePlan {
            initial: InitialTransform::Identity,
            linear: vec![small(LinearRecipe::cohort_rigid())],
            syn: Vec::new(),
            winsorize: None,
        };
        let fx2 = vec![fx[0].clone(), fx[0].clone()];
        let mv2 = vec![mv[0].clone(), mv[0].clone()];
        let err = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &two,
            &fx2,
            &mv2,
            None,
            &SynOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("one channel"), "{err}");
        let singular = StagePlan {
            initial: InitialTransform::Affine(AffineTransform::from_offset(
                [[0.0; 3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
            )),
            linear: vec![small(LinearRecipe::cohort_rigid())],
            syn: Vec::new(),
            winsorize: None,
        };
        assert!(singular.validate().is_err());
    }

    #[test]
    fn register_stages_async_on_a_backend_and_engine_that_wait_equals_register_stages() {
        use crate::exec::testing::{drive, linear_bits, YieldingCpu, YieldingLinear};
        let (fx, mv) = pair();
        let linear = LinearPlan {
            rigid: small(LinearRecipe::cohort_rigid()),
            affine: small(LinearRecipe::cohort_affine()),
        };
        let plan = StagePlan::cohort(&linear, &small_syn());
        let options = SynOptions::default();
        let plain = register_stages(
            &CpuBackend,
            &CpuLinearEngine,
            &plan,
            &fx,
            &mv,
            Some(2.0),
            &options,
        )
        .unwrap();
        let (waited, pending) = drive(register_stages_async(
            &YieldingCpu,
            &YieldingLinear,
            &plan,
            &fx,
            &mv,
            Some(2.0),
            &options,
        ));
        assert!(pending > 0);
        let waited = waited.unwrap();
        let bits = |v: &dyn std::fmt::Debug| format!("{v:?}");
        assert_eq!(bits(&plain.pair), bits(&waited.pair));
        assert_eq!(bits(&plain.warped), bits(&waited.warped));
        assert_eq!(bits(&plain.grid), bits(&waited.grid));
        assert_eq!(bits(&plain.initial), bits(&waited.initial));
        assert_eq!(bits(&plain.total), bits(&waited.total));
        assert_eq!(bits(&plain.syn_levels), bits(&waited.syn_levels));
        assert_eq!(plain.linear.len(), waited.linear.len());
        for (a, b) in plain.linear.iter().zip(&waited.linear) {
            assert_eq!(linear_bits(a), linear_bits(b));
        }
        let e = register_stages(
            &CpuBackend,
            &YieldingLinear,
            &plan,
            &fx,
            &mv,
            Some(2.0),
            &options,
        )
        .unwrap_err();
        assert!(
            matches!(
                e,
                RegError::Invalid {
                    what: "backend",
                    ..
                }
            ),
            "{e}"
        );
    }
}

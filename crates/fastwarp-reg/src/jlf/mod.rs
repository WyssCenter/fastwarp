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
// Portions derived from ANTs `itkWeightedVotingFusionImageFilter.hxx` (`BeforeThreadedGenerateData`,
// `ThreadedGenerateDataForWeightedAveraging`, `ThreadedGenerateDataForReconstruction`,
// `AfterThreadedGenerateData`: label set, distance-ordered search offsets, joint error matrix,
// weights, intensity fusion and patch voting) and from ITKAdaptiveDenoising
// `itkNonLocalPatchBasedImageFilter.hxx` (`VectorizeImagePatch`,
// `ComputeNeighborhoodPatchSimilarity`), as `antsJointFusion.cxx` (ANTs v2.6.5) runs them. Re-expressed
// as plain loops over flat buffers; centres computed in parallel and accumulated in raster order;
// equal search distances ordered by neighbourhood index. See `docs/THIRD_PARTY_PORTS.md`.

//! Joint label fusion as `antsJointFusion` (ANTs 2.6.5) computes it, on the CPU: for every voxel
//! of the target, the best-matching patch of each atlas within a search window, the atlas weights
//! from the joint error matrix of the z-scored patch differences (Wang et al. 2013), the fused
//! intensities and the weighted label votes of overlapping patches.
//!
//! The search and the z-scores run in `f32` and the matrix, the solve and the estimates in `f64`,
//! with the types and summation order of the ANTs filter. The centres are processed in parallel
//! and their contributions added in raster order, so the result does not depend on the number of
//! threads.

mod patch;
mod solve;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use rayon::prelude::*;

use crate::error::{check_cancel, invalid, Result};

pub use solve::{nonnegative_least_squares, solve_weights};

use patch::{search_offsets, Dims, Patch};

const CHUNK: usize = 4096;

/// The patch similarity of the search (`antsJointFusion -m`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchMetric {
    /// `PC`: Pearson correlation of the atlas patch with the z-scored target patch (the default).
    PearsonCorrelation,
    /// `MSQ`: mean squared difference of the raw patches.
    MeanSquares,
}

/// The search window around each voxel (`antsJointFusion -s`).
#[derive(Clone, Debug, PartialEq)]
pub enum SearchRadius {
    /// The same box radius at every voxel (`3` or `3x3x3`).
    Box([usize; 3]),
    /// An isotropic radius per voxel (x fastest); a voxel with radius 0 is skipped.
    Image(Vec<u32>),
}

/// The parameters of a fusion: start from [`JlfOptions::default`] and set the fields.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct JlfOptions {
    /// Added to the diagonal of the joint error matrix (`-a`); the default is `0.1` rounded to
    /// `f32`, as `antsJointFusion` reads it.
    pub alpha: f64,
    /// Exponent of the patch difference products (`-b`).
    pub beta: f64,
    /// Patch radius per axis (`-p`).
    pub patch_radius: [usize; 3],
    /// Search window (`-s`).
    pub search: SearchRadius,
    /// Patch similarity of the search (`-m`).
    pub metric: PatchMetric,
    /// Non-negative least squares for the weights (`-c 1`).
    pub nonnegative: bool,
    /// Return the label posterior images, divided by the vote sum where it is at least 0.1.
    pub posteriors: bool,
    /// Return the atlas voting weight images, divided the same way.
    pub voting_weights: bool,
}

impl Default for JlfOptions {
    fn default() -> Self {
        Self {
            alpha: f64::from(0.1f32),
            beta: 2.0,
            patch_radius: [2; 3],
            search: SearchRadius::Box([3; 3]),
            metric: PatchMetric::PearsonCorrelation,
            nonnegative: false,
            posteriors: false,
            voting_weights: false,
        }
    }
}

/// The images of a fusion, all of the target's size and flat with x fastest. Images are matched
/// by voxel index, as in ANTs; their physical geometry is not compared. Made by
/// [`JlfInput::new`]; set the optional fields afterwards.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct JlfInput {
    /// Voxels per axis.
    pub size: [usize; 3],
    /// Voxel spacing of the target in mm; orders the search offsets by distance.
    pub spacing: [f64; 3],
    /// The target modalities: one, or as many as each atlas has.
    pub target: Vec<Vec<f32>>,
    /// The atlases (at least two), each with the same number of modalities.
    pub atlases: Vec<Vec<Vec<f32>>>,
    /// One label image per atlas for label fusion, or none for intensity fusion only.
    pub labels: Vec<Vec<u32>>,
    /// Voxels with value 0 are not fused.
    pub mask: Option<Vec<u32>>,
    /// `(label, image)`: the label cannot win where the image is not 0; a later entry for the same
    /// label replaces an earlier one.
    pub exclusions: Vec<(u32, Vec<u32>)>,
}

/// The result of a fusion.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct JlfOutput {
    /// The fused labels (0 outside the mask and where no label has a positive vote); `None`
    /// without label images.
    pub labels: Option<Vec<u32>>,
    /// The fused intensities, one image per atlas modality.
    pub intensity: Vec<Vec<f32>>,
    /// The labels found in the atlas label images inside the mask, ascending.
    pub label_set: Vec<u32>,
    /// Per label of [`Self::label_set`] (label 0 included) its posterior image, when asked for.
    pub posteriors: Vec<(u32, Vec<f32>)>,
    /// Per atlas its voting weight image, when asked for.
    pub voting_weights: Vec<Vec<f32>>,
}

impl JlfInput {
    /// The target modalities and the atlases on a grid of `size` voxels and `spacing` mm, without
    /// label images, mask or exclusions.
    pub fn new(
        size: [usize; 3],
        spacing: [f64; 3],
        target: Vec<Vec<f32>>,
        atlases: Vec<Vec<Vec<f32>>>,
    ) -> Self {
        Self {
            size,
            spacing,
            target,
            atlases,
            ..Self::default()
        }
    }
}

/// The largest patch radius per axis: a patch of `2 r + 1` voxels per axis is held per atlas and
/// modality for every centre in a block, so larger patches would need gigabytes.
pub const MAX_PATCH_RADIUS: usize = 5;

/// The largest search radius per axis (21^3 = 9261 search positions).
pub const MAX_SEARCH_RADIUS: usize = 10;

struct Centre {
    at: [isize; 3],
    offsets: Vec<[isize; 3]>,
    weights: Vec<f64>,
    estimate: Vec<f64>,
}

fn check(input: &JlfInput, opts: &JlfOptions) -> Result<usize> {
    let n = input.size.iter().product::<usize>();
    if n == 0 {
        return Err(invalid("joint fusion", "the target has no voxels"));
    }
    if input.atlases.len() < 2 {
        return Err(invalid("joint fusion", "at least 2 atlases are required"));
    }
    let m = input.atlases[0].len();
    if m == 0 || input.atlases.iter().any(|a| a.len() != m) {
        return Err(invalid(
            "joint fusion",
            "every atlas needs the same number of modalities (at least one)",
        ));
    }
    let t = input.target.len();
    if t != 1 && t != m {
        return Err(invalid(
            "joint fusion",
            format!("the number of target images ({t}) must be 1 or the number of atlas modalities ({m})"),
        ));
    }
    if !input.labels.is_empty() && input.labels.len() != input.atlases.len() {
        return Err(invalid(
            "joint fusion",
            format!(
                "{} label images for {} atlases; give one per atlas or none",
                input.labels.len(),
                input.atlases.len()
            ),
        ));
    }
    let sized = |what: &str, len: usize| -> Result<()> {
        if len == n {
            Ok(())
        } else {
            Err(invalid(
                "joint fusion",
                format!("{what} has {len} voxels, the target {n}"),
            ))
        }
    };
    for v in &input.target {
        sized("a target image", v.len())?;
    }
    for a in &input.atlases {
        for v in a {
            sized("an atlas image", v.len())?;
        }
    }
    for v in &input.labels {
        sized("an atlas label image", v.len())?;
    }
    if let Some(v) = &input.mask {
        sized("the mask", v.len())?;
    }
    for (_, v) in &input.exclusions {
        sized("an exclusion image", v.len())?;
    }
    if let SearchRadius::Image(v) = &opts.search {
        sized("the search radius image", v.len())?;
    }
    let largest = match &opts.search {
        SearchRadius::Box(r) => r.iter().copied().max().unwrap_or(0),
        SearchRadius::Image(radii) => radii.iter().copied().max().unwrap_or(0) as usize,
    };
    if largest > MAX_SEARCH_RADIUS {
        return Err(invalid(
            "joint fusion",
            format!(
                "a search radius of {largest} voxels is not supported: at most \
                 {MAX_SEARCH_RADIUS} per axis (antsJointFusion's default is 3)"
            ),
        ));
    }
    let patch = opts.patch_radius.iter().copied().max().unwrap_or(0);
    if patch > MAX_PATCH_RADIUS {
        return Err(invalid(
            "joint fusion",
            format!(
                "a patch radius of {patch} voxels is not supported: at most {MAX_PATCH_RADIUS} \
                 per axis (antsJointFusion's default is 2)"
            ),
        ));
    }
    if !opts.alpha.is_finite() || !opts.beta.is_finite() {
        return Err(invalid("joint fusion", "alpha and beta must be finite"));
    }
    if input.spacing.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err(invalid("joint fusion", "the spacing must be positive"));
    }
    Ok(n)
}

fn label_set(input: &JlfInput) -> Vec<u32> {
    let mut set = std::collections::BTreeSet::new();
    for labels in &input.labels {
        for (v, &l) in labels.iter().enumerate() {
            if input.mask.as_ref().is_none_or(|m| m[v] != 0) {
                set.insert(l);
            }
        }
    }
    set.into_iter().collect()
}

fn joint_error(diffs: &[f64], atlases: usize, width: usize, patch: usize, beta: f64) -> Vec<f64> {
    let mut mx = vec![0.0; atlases * atlases];
    for i in 0..atlases {
        for j in 0..=i {
            let mut v = 0.0f64;
            for k in 0..width {
                v += diffs[i * width + k] * diffs[j * width + k];
            }
            v /= (patch as f64) - 1.0;
            if !solve::float_almost_equal(beta, 1.0) {
                if solve::float_almost_equal(beta, 2.0) {
                    v *= v;
                } else {
                    v = v.powf(beta);
                }
            }
            if !v.is_finite() {
                v = 0.0;
            }
            mx[i * atlases + j] = v;
            mx[j * atlases + i] = v;
        }
    }
    mx
}

struct Fusion<'a> {
    input: &'a JlfInput,
    opts: &'a JlfOptions,
    dims: Dims,
    patch: Patch,
    lists: BTreeMap<u32, Vec<[isize; 3]>>,
    box_list: Vec<[isize; 3]>,
}

impl Fusion<'_> {
    fn offsets_at(&self, voxel: usize) -> Option<&[[isize; 3]]> {
        match &self.opts.search {
            SearchRadius::Box(_) => Some(&self.box_list),
            SearchRadius::Image(r) => {
                if r[voxel] == 0 {
                    None
                } else {
                    self.lists.get(&r[voxel]).map(Vec::as_slice)
                }
            }
        }
    }

    fn centre(&self, voxel: usize) -> Option<Centre> {
        if self.input.mask.as_ref().is_some_and(|m| m[voxel] == 0) {
            return None;
        }
        let offsets = self.offsets_at(voxel).filter(|o| !o.is_empty())?;
        let at = self.dims.ijk(voxel);
        let input = self.input;
        let p = self.patch.len();
        let t = input.target.len();
        let m = input.atlases[0].len();
        let n = input.atlases.len();
        let same = t == m;
        let normalized = self
            .patch
            .vectorize_list(&input.target, &self.dims, at, true);
        let raw = match self.opts.metric {
            PatchMetric::MeanSquares => {
                self.patch
                    .vectorize_list(&input.target, &self.dims, at, false)
            }
            PatchMetric::PearsonCorrelation => Vec::new(),
        };
        let y = match self.opts.metric {
            PatchMetric::MeanSquares => &raw,
            PatchMetric::PearsonCorrelation => &normalized,
        };
        let used = if same { m } else { 1 };
        let width = p * t;
        let mut diffs = vec![0.0f64; n * width];
        let mut original = vec![0.0f64; n * p * m];
        let mut best_offsets = Vec::with_capacity(n);
        for (i, atlas) in input.atlases.iter().enumerate() {
            let mut best = f64::MAX;
            let mut best_j = 0;
            for (j, off) in offsets.iter().enumerate() {
                let s = [at[0] + off[0], at[1] + off[1], at[2] + off[2]];
                if !self.dims.inside(s) {
                    continue;
                }
                let sim = f64::from(self.patch.similarity(
                    &atlas[..used],
                    &self.dims,
                    s,
                    y,
                    self.opts.metric,
                ));
                if sim < best {
                    best = sim;
                    best_j = j;
                }
            }
            let off = offsets[best_j];
            let s = [at[0] + off[0], at[1] + off[1], at[2] + off[2]];
            let a = self
                .patch
                .vectorize_list(&atlas[..used], &self.dims, s, true);
            for (k, (av, tv)) in a.iter().zip(normalized.iter()).enumerate() {
                diffs[i * width + k] = f64::from((av - tv).abs());
            }
            let o = self.patch.vectorize_list(atlas, &self.dims, s, false);
            for (k, v) in o.iter().enumerate() {
                original[i * p * m + k] = f64::from(*v);
            }
            best_offsets.push(off);
        }
        let mut mxbar = joint_error(&diffs, n, width, p, self.opts.beta);
        for i in 0..n {
            mxbar[i * n + i] += self.opts.alpha;
        }
        let weights = solve_weights(&mxbar, n, self.opts.nonnegative);
        let mut estimate = vec![0.0f64; p * m];
        for (k, e) in estimate.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (i, w) in weights.iter().enumerate() {
                sum += w * original[i * p * m + k];
            }
            *e = sum;
        }
        Some(Centre {
            at,
            offsets: best_offsets,
            weights,
            estimate,
        })
    }
}

struct Accumulators {
    intensity: Vec<Vec<f32>>,
    count: Vec<u32>,
    posteriors: Vec<Vec<f32>>,
    weight_sum: Vec<f32>,
    voting: Vec<Vec<f32>>,
}

impl Fusion<'_> {
    fn add(&self, c: &Centre, acc: &mut Accumulators, set: &[u32]) {
        let input = self.input;
        let p = self.patch.len();
        for (mi, image) in acc.intensity.iter_mut().enumerate() {
            for (j, off) in self.patch.offsets().iter().enumerate() {
                let u = [c.at[0] + off[0], c.at[1] + off[1], c.at[2] + off[2]];
                if !self.dims.inside(u) {
                    continue;
                }
                let v = self.dims.index(u);
                if input.mask.as_ref().is_some_and(|m| m[v] == 0) {
                    continue;
                }
                let mut value = c.estimate[mi * p + j] + f64::from(image[v]);
                if !value.is_finite() {
                    value = 0.0;
                }
                image[v] = value as f32;
                if mi == 0 {
                    acc.count[v] += 1;
                }
            }
        }
        if input.labels.is_empty() {
            return;
        }
        for off in self.patch.offsets() {
            let u = [c.at[0] + off[0], c.at[1] + off[1], c.at[2] + off[2]];
            if !self.dims.inside(u) {
                continue;
            }
            let v = self.dims.index(u);
            for (i, labels) in input.labels.iter().enumerate() {
                let s = c.offsets[i];
                let w = [u[0] + s[0], u[1] + s[1], u[2] + s[2]];
                if !self.dims.inside(w) {
                    continue;
                }
                let Ok(slot) = set.binary_search(&labels[self.dims.index(w)]) else {
                    continue;
                };
                let weight = c.weights[i] as f32;
                acc.posteriors[slot][v] += weight;
                acc.weight_sum[v] += weight;
                if let Some(image) = acc.voting.get_mut(i) {
                    image[v] += weight;
                }
            }
        }
    }
}

/// Fuses the atlases onto the target as `antsJointFusion` does. `cancel` is checked between
/// blocks of voxels.
pub fn joint_fusion(
    input: &JlfInput,
    opts: &JlfOptions,
    cancel: Option<&AtomicBool>,
) -> Result<JlfOutput> {
    let n = check(input, opts)?;
    let dims = Dims::new(input.size);
    let patch = Patch::new(opts.patch_radius, &dims);
    let mut lists = BTreeMap::new();
    let box_list = match &opts.search {
        SearchRadius::Box(r) => search_offsets(*r, input.spacing),
        SearchRadius::Image(radii) => {
            for &r in radii {
                if r > 0 && !lists.contains_key(&r) {
                    let r_usize = r as usize;
                    lists.insert(r, search_offsets([r_usize; 3], input.spacing));
                }
            }
            Vec::new()
        }
    };
    let fusion = Fusion {
        input,
        opts,
        dims,
        patch,
        lists,
        box_list,
    };
    let set = label_set(input);
    let atlases = input.atlases.len();
    let modalities = input.atlases[0].len();
    let mut acc = Accumulators {
        intensity: vec![vec![0.0f32; n]; modalities],
        count: vec![0u32; n],
        posteriors: vec![vec![0.0f32; n]; set.len()],
        weight_sum: vec![0.0f32; n],
        voting: if opts.voting_weights {
            vec![vec![0.0f32; n]; atlases]
        } else {
            Vec::new()
        },
    };
    let mut start = 0;
    while start < n {
        check_cancel(cancel)?;
        let end = (start + CHUNK).min(n);
        let centres: Vec<Option<Centre>> = (start..end)
            .into_par_iter()
            .map(|v| fusion.centre(v))
            .collect();
        for c in centres.iter().flatten() {
            fusion.add(c, &mut acc, &set);
        }
        start = end;
    }
    check_cancel(cancel)?;
    let labels = if input.labels.is_empty() {
        None
    } else {
        let mut out = vec![0u32; n];
        for (v, label) in out.iter_mut().enumerate() {
            if input.mask.as_ref().is_some_and(|m| m[v] == 0) {
                continue;
            }
            let mut max = 0.0f64;
            for (slot, &l) in set.iter().enumerate() {
                let excluded = input
                    .exclusions
                    .iter()
                    .rev()
                    .find(|(x, _)| *x == l)
                    .is_some_and(|(_, image)| image[v] != 0);
                if excluded {
                    continue;
                }
                let p = f64::from(acc.posteriors[slot][v]);
                if max < p {
                    max = p;
                    *label = l;
                }
            }
        }
        Some(out)
    };
    if opts.posteriors || opts.voting_weights {
        for v in 0..n {
            let sum = acc.weight_sum[v];
            if sum < 0.1f32 {
                continue;
            }
            if opts.posteriors {
                for image in &mut acc.posteriors {
                    image[v] /= sum;
                }
            }
            for image in &mut acc.voting {
                image[v] /= sum;
            }
        }
    }
    for image in &mut acc.intensity {
        for (value, &count) in image.iter_mut().zip(&acc.count) {
            if count > 0 {
                *value /= count as f32;
            }
        }
    }
    let posteriors = if opts.posteriors {
        set.iter().copied().zip(acc.posteriors).collect()
    } else {
        Vec::new()
    };
    Ok(JlfOutput {
        labels,
        intensity: acc.intensity,
        label_set: set,
        posteriors,
        voting_weights: acc.voting,
    })
}

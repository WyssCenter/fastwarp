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
#[doc(hidden)]
pub mod prof;
mod search;
mod solve;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use rayon::prelude::*;

use crate::error::{check_cancel, invalid, Result};
use crate::exec::block_ready;

pub use search::{cpu_search, CpuSearch, PatchSearch, SearchBlock};
pub use solve::{float_almost_equal, nonnegative_least_squares, solve_weights};

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

/// Where the patch search runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum JlfDevice {
    /// Everything on the CPU (the default).
    #[default]
    Cpu,
    /// The patch search on the GPU in `f32`, the rest on the CPU in `f64` (`fastwarp_gpu::jlf`).
    Hybrid,
    /// The search, the weights (with a CPU `f64` solve where the GPU's `f32` one is not reliable)
    /// and the votes and intensities on the GPU (`fastwarp_gpu::jlf`).
    Gpu,
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
    /// Where the patch search runs; [`joint_fusion`] runs [`JlfDevice::Cpu`] only.
    pub device: JlfDevice,
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
            device: JlfDevice::Cpu,
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

/// The building blocks of a fusion, for a device implementation that runs more than the search
/// (`fastwarp_gpu::jlf`): the checked inputs, the search lists, the target patches, the weights of
/// one centre in `f64`, the accumulators and the last step. Not a stable API.
#[doc(hidden)]
pub struct FusionParts<'a> {
    input: &'a JlfInput,
    opts: &'a JlfOptions,
    dims: Dims,
    patch: Patch,
    lists: Vec<Vec<[isize; 3]>>,
    list_of_radius: BTreeMap<u32, u32>,
    set: Vec<u32>,
    voxels: usize,
}

impl<'a> FusionParts<'a> {
    /// Checks the inputs and prepares the search lists and the label set.
    pub fn new(input: &'a JlfInput, opts: &'a JlfOptions) -> Result<Self> {
        let voxels = check(input, opts)?;
        let dims = Dims::new(input.size);
        let patch = Patch::new(opts.patch_radius, &dims);
        let mut list_of_radius = BTreeMap::new();
        let lists = match &opts.search {
            SearchRadius::Box(r) => vec![search_offsets(*r, input.spacing)],
            SearchRadius::Image(radii) => {
                let mut lists = Vec::new();
                for &r in radii {
                    if r > 0 && !list_of_radius.contains_key(&r) {
                        list_of_radius.insert(r, lists.len() as u32);
                        lists.push(search_offsets([r as usize; 3], input.spacing));
                    }
                }
                lists
            }
        };
        Ok(Self {
            input,
            opts,
            dims,
            patch,
            lists,
            list_of_radius,
            set: label_set(input),
            voxels,
        })
    }

    /// Voxels of the images.
    pub fn voxels(&self) -> usize {
        self.voxels
    }

    /// The search offset lists.
    pub fn lists(&self) -> &[Vec<[isize; 3]>] {
        &self.lists
    }

    /// The labels found inside the mask, ascending.
    pub fn label_set(&self) -> &[u32] {
        &self.set
    }

    /// The offset list of a centre, or `None` for a voxel that is not fused (masked, radius 0).
    pub fn list_at(&self, voxel: usize) -> Option<u32> {
        if self.input.mask.as_ref().is_some_and(|m| m[voxel] == 0) {
            return None;
        }
        let list = match &self.opts.search {
            SearchRadius::Box(_) => 0,
            SearchRadius::Image(r) => *self.list_of_radius.get(&r[voxel])?,
        };
        (!self.lists[list as usize].is_empty()).then_some(list)
    }

    /// The atlas modalities the search and the weights compare.
    pub fn used(&self) -> usize {
        let m = self.input.atlases[0].len();
        if self.input.target.len() == m {
            m
        } else {
            1
        }
    }

    /// Voxels per patch.
    pub fn patch_len(&self) -> usize {
        self.patch.len()
    }

    /// The target patch the search compares (z-scored for PC, raw for MSQ).
    pub fn search_patch(&self, voxel: usize) -> Vec<f32> {
        let normalize = self.opts.metric == PatchMetric::PearsonCorrelation;
        self.patch.vectorize_list(
            &self.input.target,
            &self.dims,
            self.dims.ijk(voxel),
            normalize,
        )
    }

    /// The z-scored target patch of every target modality, as the weights use it.
    pub fn target_patch(&self, voxel: usize) -> Vec<f32> {
        self.patch
            .vectorize_list(&self.input.target, &self.dims, self.dims.ijk(voxel), true)
    }

    /// A search block over `centres` (with their lists and search patches).
    pub fn block<'b>(
        &'b self,
        centres: &'b [usize],
        list_of: &'b [u32],
        patches: &'b [f32],
    ) -> SearchBlock<'b> {
        SearchBlock {
            size: self.input.size,
            centres,
            list_of,
            lists: &self.lists,
            patch_radius: self.opts.patch_radius,
            patches,
            atlases: &self.input.atlases,
            modalities: self.used(),
            metric: self.opts.metric,
        }
    }

    /// The normalised atlas weights of the centre at `voxel` with offset list `list` and the best
    /// offsets `best` (one per atlas), in `f64` as the CPU fusion computes them.
    pub fn weights(&self, voxel: usize, list: u32, best: &[u32]) -> Vec<f64> {
        self.weights_at(self.dims.ijk(voxel), &self.lists[list as usize], best)
    }

    fn weights_at(&self, at: [isize; 3], list: &[[isize; 3]], best: &[u32]) -> Vec<f64> {
        let input = self.input;
        let p = self.patch.len();
        let t = input.target.len();
        let n = input.atlases.len();
        let used = self.used();
        let normalized = self
            .patch
            .vectorize_list(&input.target, &self.dims, at, true);
        let width = p * t;
        let mut diffs = vec![0.0f64; n * width];
        for (i, atlas) in input.atlases.iter().enumerate() {
            let off = list[best[i] as usize];
            let s = [at[0] + off[0], at[1] + off[1], at[2] + off[2]];
            let a = self
                .patch
                .vectorize_list(&atlas[..used], &self.dims, s, true);
            for (k, (av, tv)) in a.iter().zip(normalized.iter()).enumerate() {
                diffs[i * width + k] = f64::from((av - tv).abs());
            }
        }
        let mut mxbar = joint_error(&diffs, n, width, p, self.opts.beta);
        for i in 0..n {
            mxbar[i * n + i] += self.opts.alpha;
        }
        solve_weights(&mxbar, n, self.opts.nonnegative)
    }

    fn centre(&self, voxel: usize, list: &[[isize; 3]], best: &[u32]) -> Centre {
        let at = self.dims.ijk(voxel);
        let input = self.input;
        let p = self.patch.len();
        let m = input.atlases[0].len();
        let weights = self.weights_at(at, list, best);
        let mut estimate = vec![0.0f64; p * m];
        let mut best_offsets = Vec::with_capacity(input.atlases.len());
        let mut original = vec![0.0f64; input.atlases.len() * p * m];
        for (i, atlas) in input.atlases.iter().enumerate() {
            let off = list[best[i] as usize];
            let s = [at[0] + off[0], at[1] + off[1], at[2] + off[2]];
            let o = self.patch.vectorize_list(atlas, &self.dims, s, false);
            for (k, v) in o.iter().enumerate() {
                original[i * p * m + k] = f64::from(*v);
            }
            best_offsets.push(off);
        }
        for (k, e) in estimate.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (i, w) in weights.iter().enumerate() {
                sum += w * original[i * p * m + k];
            }
            *e = sum;
        }
        Centre {
            at,
            offsets: best_offsets,
            weights,
            estimate,
        }
    }

    /// The fused voxels of `start..end` (ascending) with their offset lists.
    pub fn active(&self, start: usize, end: usize) -> (Vec<usize>, Vec<u32>) {
        (start..end.min(self.voxels))
            .into_par_iter()
            .filter_map(|v| self.list_at(v).map(|l| (v, l)))
            .unzip()
    }

    /// [`Self::search_patch`] of each centre, concatenated.
    pub fn search_patches(&self, centres: &[usize]) -> Vec<f32> {
        centres
            .par_iter()
            .flat_map_iter(|&v| self.search_patch(v))
            .collect()
    }

    /// [`Self::target_patch`] of each centre, concatenated.
    pub fn target_patches(&self, centres: &[usize]) -> Vec<f32> {
        centres
            .par_iter()
            .flat_map_iter(|&v| self.target_patch(v))
            .collect()
    }

    /// [`Self::weights`] of each centre (`best` holds one offset per atlas and centre),
    /// concatenated.
    pub fn weights_many(&self, centres: &[usize], list_of: &[u32], best: &[u32]) -> Vec<f64> {
        let n = self.input.atlases.len();
        (0..centres.len())
            .into_par_iter()
            .flat_map_iter(|c| self.weights(centres[c], list_of[c], &best[c * n..(c + 1) * n]))
            .collect()
    }

    /// The number of fused voxels in each z plane.
    pub fn active_per_plane(&self) -> Vec<usize> {
        let [nx, ny, nz] = self.input.size;
        let plane = nx * ny;
        (0..nz)
            .into_par_iter()
            .map(|z| {
                (z * plane..(z + 1) * plane)
                    .filter(|&v| self.list_at(v).is_some())
                    .count()
            })
            .collect()
    }

    /// Per atlas label image, the index of each voxel's label in [`Self::label_set`], or
    /// `u32::MAX` for a label outside it.
    pub fn label_slots(&self) -> Vec<Vec<u32>> {
        self.input
            .labels
            .par_iter()
            .map(|l| {
                l.par_iter()
                    .map(|x| self.set.binary_search(x).map_or(u32::MAX, |s| s as u32))
                    .collect()
            })
            .collect()
    }

    /// Empty accumulators for this fusion.
    pub fn accumulators(&self) -> Accumulators {
        let n = self.voxels;
        Accumulators {
            intensity: vec![vec![0.0f32; n]; self.input.atlases[0].len()],
            count: vec![0u32; n],
            posteriors: vec![vec![0.0f32; n]; self.set.len()],
            weight_sum: vec![0.0f32; n],
            voting: if self.opts.voting_weights {
                vec![vec![0.0f32; n]; self.input.atlases.len()]
            } else {
                Vec::new()
            },
        }
    }
}

/// The sums of a fusion before its last step, per voxel (x fastest): the intensity sums of each
/// atlas modality and how many patches added to them, the vote of each label of the label set, the
/// sum of the votes and, when asked for, the vote of each atlas.
#[doc(hidden)]
#[non_exhaustive]
pub struct Accumulators {
    pub intensity: Vec<Vec<f32>>,
    pub count: Vec<u32>,
    pub posteriors: Vec<Vec<f32>>,
    pub weight_sum: Vec<f32>,
    pub voting: Vec<Vec<f32>>,
}

impl FusionParts<'_> {
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

/// Fuses the atlases onto the target as `antsJointFusion` does, on the CPU. `cancel` is checked
/// between blocks of voxels. [`JlfDevice::Hybrid`] is refused here: `fastwarp_gpu::jlf` runs it.
pub fn joint_fusion(
    input: &JlfInput,
    opts: &JlfOptions,
    cancel: Option<&AtomicBool>,
) -> Result<JlfOutput> {
    if opts.device != JlfDevice::Cpu {
        return Err(invalid(
            "joint fusion",
            "the hybrid and gpu modes run on a GPU: use fastwarp_gpu::jlf::joint_fusion",
        ));
    }
    block_ready(joint_fusion_with(input, opts, &mut CpuSearch, cancel))
}

/// [`joint_fusion`] with the patch search done by `search` (whatever [`JlfOptions::device`]
/// says): the centres of each block are handed to it in raster order, and the weights, votes and
/// intensities are computed on the CPU from the offsets it returns.
pub async fn joint_fusion_with<S: PatchSearch>(
    input: &JlfInput,
    opts: &JlfOptions,
    search: &mut S,
    cancel: Option<&AtomicBool>,
) -> Result<JlfOutput> {
    search.begin();
    let guard = Finish(search);
    fuse(input, opts, &mut *guard.0, cancel).await
}

struct Finish<'a, S: PatchSearch>(&'a mut S);

impl<S: PatchSearch> Drop for Finish<'_, S> {
    fn drop(&mut self) {
        self.0.finish();
    }
}

async fn fuse<S: PatchSearch>(
    input: &JlfInput,
    opts: &JlfOptions,
    search: &mut S,
    cancel: Option<&AtomicBool>,
) -> Result<JlfOutput> {
    let sp = prof::span("h.parts_acc");
    let fusion = FusionParts::new(input, opts)?;
    let n = fusion.voxels;
    let set = fusion.set.clone();
    let atlases = input.atlases.len();
    let mut acc = fusion.accumulators();
    drop(sp);
    let width = fusion.patch.len() * fusion.used();
    let block_voxels = search.block_voxels(input.size, width).max(1);
    let mut start = 0;
    while start < n {
        check_cancel(cancel)?;
        let end = start.saturating_add(block_voxels).min(n);
        let sp = prof::span("h.active_patches");
        let (centres, list_of): (Vec<usize>, Vec<u32>) = (start..end)
            .into_par_iter()
            .filter_map(|v| fusion.list_at(v).map(|l| (v, l)))
            .unzip();
        let patches: Vec<f32> = centres
            .par_iter()
            .flat_map_iter(|&v| fusion.search_patch(v))
            .collect();
        let block = fusion.block(&centres, &list_of, &patches);
        drop(sp);
        let sp = prof::span("h.search");
        let best = search.search(&block).await?;
        drop(sp);
        let sp = prof::span("h.cpu_rest");
        drop(patches);
        if best.len() != centres.len() * atlases
            || best.chunks(atlases.max(1)).zip(&list_of).any(|(b, l)| {
                b.iter()
                    .any(|j| *j as usize >= fusion.lists[*l as usize].len())
            })
        {
            return Err(invalid(
                "joint fusion",
                "the patch search returned offsets that do not fit its block",
            ));
        }
        for part in (0..centres.len()).collect::<Vec<_>>().chunks(CHUNK) {
            check_cancel(cancel)?;
            let done: Vec<Centre> = part
                .par_iter()
                .map(|&c| {
                    let list = &fusion.lists[list_of[c] as usize];
                    fusion.centre(centres[c], list, &best[c * atlases..(c + 1) * atlases])
                })
                .collect();
            for c in &done {
                fusion.add(c, &mut acc, &set);
            }
        }
        drop(sp);
        start = end;
    }
    check_cancel(cancel)?;
    let _sp = prof::span("h.finish");
    Ok(fusion.finish(acc))
}

impl FusionParts<'_> {
    /// The last step: the labels from the votes (with the exclusions), the posteriors and voting
    /// weights divided by the vote sum where it is at least 0.1, the intensities divided by their
    /// counts.
    pub fn finish(&self, mut acc: Accumulators) -> JlfOutput {
        let input = self.input;
        let opts = self.opts;
        let n = self.voxels;
        let set = self.set.clone();
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
        JlfOutput {
            labels,
            intensity: acc.intensity,
            label_set: set,
            posteriors,
            voting_weights: acc.voting,
        }
    }
}

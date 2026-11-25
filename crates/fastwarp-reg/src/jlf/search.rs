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
// Portions derived from ANTs `itkWeightedVotingFusionImageFilter.hxx` (the per-atlas search loop of
// `ThreadedGenerateDataForWeightedAveraging`: candidates in distance order, inside the image,
// strict `<` against a `double` initialised to its maximum). Split out as an interface so that a
// device can run the search on a block of centres. See `docs/THIRD_PARTY_PORTS.md`.

//! The patch search of joint label fusion as an exchangeable step: [`PatchSearch`] finds, for a
//! block of centres, the best candidate offset of every atlas; [`CpuSearch`] is the reference.

use rayon::prelude::*;

use super::patch::{box_offsets, Dims, Patch};
use super::PatchMetric;
use crate::error::Result;

/// The centres of one search block, with what the search needs to score their candidates.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SearchBlock<'a> {
    /// Voxels per axis of the images.
    pub size: [usize; 3],
    /// The centre voxels (x fastest), ascending.
    pub centres: &'a [usize],
    /// For each centre, the index of its offset list in [`Self::lists`].
    pub list_of: &'a [u32],
    /// The candidate offset lists, each in search order (distance, then neighbourhood index).
    pub lists: &'a [Vec<[isize; 3]>],
    /// Patch radius per axis.
    pub patch_radius: [usize; 3],
    /// For each centre, the target patch of each searched modality (`modalities` x patch
    /// length values): z-scored for [`PatchMetric::PearsonCorrelation`], raw for
    /// [`PatchMetric::MeanSquares`]; NaN or infinite where a voxel does not count.
    pub patches: &'a [f32],
    /// The atlases, each with its modalities (flat, x fastest).
    pub atlases: &'a [Vec<Vec<f32>>],
    /// The atlas modalities the search compares (the first ones).
    pub modalities: usize,
    /// The patch similarity.
    pub metric: PatchMetric,
}

impl SearchBlock<'_> {
    /// Voxels per patch.
    pub fn patch_len(&self) -> usize {
        self.patch_radius.iter().map(|r| 2 * r + 1).product()
    }

    /// The patch offsets in the order of [`Self::patches`].
    pub fn patch_offsets(&self) -> Vec<[isize; 3]> {
        box_offsets(self.patch_radius)
    }

    /// The candidate offsets of centre `c`.
    pub fn list(&self, c: usize) -> &[[isize; 3]] {
        &self.lists[self.list_of[c] as usize]
    }

    /// The similarity (lower is better) of candidate `offset` of centre `c` in atlas `atlas`,
    /// as the CPU search computes it; `None` when the candidate lies outside the image.
    pub fn score(&self, c: usize, atlas: usize, offset: usize) -> Option<f32> {
        let dims = Dims::new(self.size);
        let patch = Patch::new(self.patch_radius, &dims);
        score_with(self, &dims, &patch, c, atlas, offset)
    }
}

fn score_with(
    block: &SearchBlock<'_>,
    dims: &Dims,
    patch: &Patch,
    c: usize,
    atlas: usize,
    offset: usize,
) -> Option<f32> {
    let at = dims.ijk(block.centres[c]);
    let off = block.list(c)[offset];
    let s = [at[0] + off[0], at[1] + off[1], at[2] + off[2]];
    if !dims.inside(s) {
        return None;
    }
    let width = patch.len() * block.modalities;
    let y = &block.patches[c * width..(c + 1) * width];
    Some(patch.similarity(
        &block.atlases[atlas][..block.modalities],
        dims,
        s,
        y,
        block.metric,
    ))
}

/// A search of the best candidate offset per centre and atlas.
///
/// The future need not be `Send`; the fusion drives it on the calling thread, and a native
/// implementation's future is ready when it is first polled.
#[allow(async_fn_in_trait)]
pub trait PatchSearch {
    /// Called by [`super::joint_fusion_with`] before the first block of a fusion: until
    /// [`Self::finish`], every block holds the same atlases and size, so an implementation may keep
    /// what it uploaded. Does nothing by default.
    fn begin(&mut self) {}

    /// Called by [`super::joint_fusion_with`] after the last block of a fusion, also when it fails.
    /// Does nothing by default.
    fn finish(&mut self) {}

    /// Voxels per block (raster order) for images of `size` with `patch_values` target values per
    /// centre.
    fn block_voxels(&self, size: [usize; 3], patch_values: usize) -> usize;

    /// For centre `c` and atlas `i` of `block`, at `c * atlases + i`, the index into the centre's
    /// offset list of the first candidate inside the image with the lowest similarity (strict
    /// `<`, a candidate whose similarity is not a number never wins, a first candidate with any
    /// finite similarity does).
    async fn search(&mut self, block: &SearchBlock<'_>) -> Result<Vec<u32>>;
}

/// The search on the CPU, in parallel over centres: the reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuSearch;

impl PatchSearch for CpuSearch {
    fn block_voxels(&self, _size: [usize; 3], _patch_values: usize) -> usize {
        super::CHUNK
    }

    async fn search(&mut self, block: &SearchBlock<'_>) -> Result<Vec<u32>> {
        Ok(cpu_search(block))
    }
}

/// [`CpuSearch`] as a plain function.
pub fn cpu_search(block: &SearchBlock<'_>) -> Vec<u32> {
    let dims = Dims::new(block.size);
    let patch = Patch::new(block.patch_radius, &dims);
    let atlases = block.atlases.len();
    (0..block.centres.len())
        .into_par_iter()
        .flat_map_iter(|c| {
            let dims = &dims;
            let patch = &patch;
            (0..atlases).map(move |i| {
                let mut best = f64::MAX;
                let mut best_j = 0u32;
                for j in 0..block.list(c).len() {
                    let Some(sim) = score_with(block, dims, patch, c, i, j) else {
                        continue;
                    };
                    let sim = f64::from(sim);
                    if sim < best {
                        best = sim;
                        best_j = j as u32;
                    }
                }
                best_j
            })
        })
        .collect()
}

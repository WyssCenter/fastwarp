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
// Portions derived from ITKAdaptiveDenoising `itkNonLocalPatchBasedImageFilter.hxx`
// (`VectorizeImagePatch`, `GetMeanAndStandardDeviationOfVectorizedImagePatch`,
// `ComputeNeighborhoodPatchSimilarity`) and ANTs `itkWeightedVotingFusionImageFilter.hxx` (the
// distance-ordered search offsets of `BeforeThreadedGenerateData`), with ITK's `Neighborhood`
// offset order. `f32` arithmetic as the filter's `RealType`; a direct path for patches inside the
// image; equal distances kept in neighbourhood order (stable sort). See `docs/THIRD_PARTY_PORTS.md`.

//! Patches: the neighbourhood offsets, patch vectors with out-of-image voxels as NaN, their
//! z-scores, and the patch similarity of the search.

use super::PatchMetric;

pub(crate) struct Dims {
    size: [isize; 3],
}

impl Dims {
    pub(crate) fn new(size: [usize; 3]) -> Self {
        Self {
            size: size.map(|s| s as isize),
        }
    }

    pub(crate) fn inside(&self, p: [isize; 3]) -> bool {
        (0..3).all(|d| p[d] >= 0 && p[d] < self.size[d])
    }

    pub(crate) fn index(&self, p: [isize; 3]) -> usize {
        (p[0] + self.size[0] * (p[1] + self.size[1] * p[2])) as usize
    }

    pub(crate) fn ijk(&self, v: usize) -> [isize; 3] {
        let v = v as isize;
        let nx = self.size[0];
        let ny = self.size[1];
        [v % nx, (v / nx) % ny, v / (nx * ny)]
    }
}

pub(crate) fn box_offsets(radius: [usize; 3]) -> Vec<[isize; 3]> {
    let r = radius.map(|v| v as isize);
    let mut out = Vec::new();
    for z in -r[2]..=r[2] {
        for y in -r[1]..=r[1] {
            for x in -r[0]..=r[0] {
                out.push([x, y, z]);
            }
        }
    }
    out
}

pub(crate) fn search_offsets(radius: [usize; 3], spacing: [f64; 3]) -> Vec<[isize; 3]> {
    let mut keyed: Vec<(f64, [isize; 3])> = box_offsets(radius)
        .into_iter()
        .map(|o| {
            let mut d = 0.0f64;
            for a in 0..3 {
                let t = o[a] as f64 * spacing[a];
                d += t * t;
            }
            (d, o)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().map(|(_, o)| o).collect()
}

pub(crate) struct Patch {
    radius: [isize; 3],
    offsets: Vec<[isize; 3]>,
    linear: Vec<isize>,
}

impl Patch {
    pub(crate) fn new(radius: [usize; 3], dims: &Dims) -> Self {
        let offsets = box_offsets(radius);
        let linear = offsets
            .iter()
            .map(|o| o[0] + dims.size[0] * (o[1] + dims.size[1] * o[2]))
            .collect();
        Self {
            radius: radius.map(|v| v as isize),
            offsets,
            linear,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.offsets.len()
    }

    pub(crate) fn offsets(&self) -> &[[isize; 3]] {
        &self.offsets
    }

    fn fully_inside(&self, dims: &Dims, c: [isize; 3]) -> bool {
        (0..3).all(|d| c[d] - self.radius[d] >= 0 && c[d] + self.radius[d] < dims.size[d])
    }

    fn vectorize_into(&self, image: &[f32], dims: &Dims, c: [isize; 3], out: &mut [f32]) {
        if self.fully_inside(dims, c) {
            let base = dims.index(c) as isize;
            for (o, l) in out.iter_mut().zip(&self.linear) {
                *o = image[(base + l) as usize];
            }
            return;
        }
        for (o, off) in out.iter_mut().zip(&self.offsets) {
            let q = [c[0] + off[0], c[1] + off[1], c[2] + off[2]];
            *o = if dims.inside(q) {
                image[dims.index(q)]
            } else {
                f32::NAN
            };
        }
    }

    pub(crate) fn vectorize_list(
        &self,
        images: &[Vec<f32>],
        dims: &Dims,
        c: [isize; 3],
        normalize: bool,
    ) -> Vec<f32> {
        let p = self.len();
        let mut out = vec![0.0f32; p * images.len()];
        for (image, chunk) in images.iter().zip(out.chunks_mut(p)) {
            self.vectorize_into(image, dims, c, chunk);
            if normalize {
                zscore(chunk);
            }
        }
        out
    }

    pub(crate) fn similarity(
        &self,
        images: &[Vec<f32>],
        dims: &Dims,
        c: [isize; 3],
        y: &[f32],
        metric: PatchMetric,
    ) -> f32 {
        let mut sum_x = 0.0f32;
        let mut sum_xx = 0.0f32;
        let mut sum_dd = 0.0f32;
        let mut sum_xy = 0.0f32;
        let mut n = 0.0f32;
        let inside = self.fully_inside(dims, c);
        let base = dims.index(c) as isize;
        let mut count = 0;
        for image in images {
            for (k, off) in self.offsets.iter().enumerate() {
                let yv = y[count];
                count += 1;
                let index = if inside {
                    (base + self.linear[k]) as usize
                } else {
                    let q = [c[0] + off[0], c[1] + off[1], c[2] + off[2]];
                    if !dims.inside(q) {
                        continue;
                    }
                    dims.index(q)
                };
                if !yv.is_finite() {
                    continue;
                }
                let x = image[index];
                sum_x += x;
                sum_xx += x * x;
                sum_xy += x * yv;
                let d = yv - x;
                sum_dd += d * d;
                n += 1.0;
            }
        }
        if n < 3.0 {
            return f32::MAX;
        }
        match metric {
            PatchMetric::PearsonCorrelation => {
                let variance = sum_xx - sum_x * sum_x / n;
                let measure = if variance > f32::EPSILON {
                    sum_xy * sum_xy / variance
                } else {
                    0.0
                };
                if sum_xy > 0.0 {
                    -measure
                } else {
                    measure
                }
            }
            PatchMetric::MeanSquares => sum_dd / n,
        }
    }
}

pub(crate) fn zscore(v: &mut [f32]) {
    let mut sum = 0.0f32;
    let mut squares = 0.0f32;
    let mut count = 0.0f32;
    for &x in v.iter() {
        if x.is_finite() {
            sum += x;
            squares += x * x;
            count += 1.0;
        }
    }
    let mean = sum / count;
    let sd = ((squares - count * (mean * mean)) / (count - 1.0)).sqrt();
    if sd < f32::EPSILON {
        v.fill(0.0);
    }
    let inverse = 1.0f64 / f64::from(sd);
    for x in v.iter_mut() {
        *x = (f64::from(*x - mean) * inverse) as f32;
    }
}

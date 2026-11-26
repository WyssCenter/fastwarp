// SPDX-FileCopyrightText: 1999-2019 Insight Software Consortium
// SPDX-FileCopyrightText: 2020-present NumFOCUS
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
// Portions derived from ITK `itkMattesMutualInformationImageToImageMetricv4.hxx` (`Initialize` for
// dense sampling with masks, `ComputeResults` with its local-support branch) and
// `itkMattesMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader.hxx` (`ProcessPoint`).
// The GPU evaluation (per-workgroup ranges, fixed-point integer histogram, ratio table applied per
// voxel) is this crate's own. See `docs/THIRD_PARTY_PORTS.md`.

//! Dense Mattes mutual information of two images on one grid, in both directions
//! (`k_mi_dense_range`, `k_mi_dense_hist`, `k_mi_dense_deriv` and their masked forms).
//!
//! One pair evaluation is three submissions: the intensity ranges of both images (four `f32`
//! read back), the two joint histograms (weights in `2^-32` fixed point summed in 64 bits per
//! workgroup, merged in two levels in workgroup order and read back together), and the two per-voxel derivatives after the host has formed the ratio tables with
//! the CPU reference's `finish_histogram`. Ranges are minima and maxima and the histogram is an
//! integer sum, so every result is independent of the order of the work.

use bytemuck::{Pod, Zeroable};
use fastwarp_reg::backend::MetricPair;
use fastwarp_reg::error::{RegError, Result as RegResult};
use fastwarp_reg::metric::cc::NO_VOXEL_IN_MASKS;
use fastwarp_reg::metric::mi::{empty_range_message, finish_histogram, MiBinning};

use super::{MergeU, BINS, MAX_STRIPES, STRIDE};
use crate::capacity::mattes_chunk;
use crate::device::GpuContext;
use crate::error::GpuResult;
use crate::syn_kernels::CcFinalU;
use crate::wgpu_backend::{bad, cc_uniform, GpuField, GpuImage, WgpuBackend};

const BSPLINE: &str = include_str!("../../shaders/mi_bspline.wgsl");
const SAMPLE: &str = include_str!("../../shaders/syn_sample.wgsl");
const COMMON: &str = include_str!("../../shaders/mi_dense_common.wgsl");
const UNMASKED: &str = include_str!("../../shaders/mi_dense_unmasked.wgsl");
const MASKED: &str = include_str!("../../shaders/mi_dense_masked.wgsl");
const RANGE: &str = include_str!("../../shaders/mi_dense_range.wgsl");
const RANGE_FINAL: &str = include_str!("../../shaders/mi_dense_range_final.wgsl");
const HIST: &str = include_str!("../../shaders/mi_dense_hist.wgsl");
const DERIV: &str = include_str!("../../shaders/mi_dense_deriv.wgsl");
const MERGE: &str = include_str!("../../shaders/mi_dense_merge.wgsl");

const THRESHOLDS: usize = BINS - 4;

type Direction<'a> = (
    &'a GpuImage,
    &'a GpuImage,
    Option<(&'a GpuImage, &'a GpuImage)>,
);
const FIXED_POINT: f64 = 4_294_967_296.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DenseMiU {
    dims: [u32; 4],
    cfg: [u32; 4],
    frng: [f32; 4],
    mrng: [f32; 4],
    o: [[f32; 4]; 3],
    sp: [f32; 4],
    fthr: [[f32; 4]; THRESHOLDS / 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RangeFinalU {
    count: u32,
    pad: [u32; 3],
}

pub(super) struct DenseMiKernels {
    range: wgpu::ComputePipeline,
    range_masked: wgpu::ComputePipeline,
    range_final: wgpu::ComputePipeline,
    hist: wgpu::ComputePipeline,
    hist_masked: wgpu::ComputePipeline,
    deriv: wgpu::ComputePipeline,
    deriv_masked: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    merge_final: wgpu::ComputePipeline,
}

fn source(body: &str, outputs: u32, masked: bool) -> String {
    let prelude = if masked {
        MASKED
            .replace("BINDING_FIXED_MASK", &outputs.to_string())
            .replace("BINDING_MOVING_MASK", &(outputs + 1).to_string())
            .replace("BINDING_UNIFORM", &(outputs + 2).to_string())
    } else {
        UNMASKED.replace("BINDING_UNIFORM", &outputs.to_string())
    };
    format!("{BSPLINE}\n{SAMPLE}\n{COMMON}\n{prelude}\n{body}")
}

impl DenseMiKernels {
    pub(super) async fn new(ctx: &GpuContext) -> GpuResult<Self> {
        let compile = |label: &'static str, body: &str, outputs: u32, masked: bool, entry| {
            let code = source(body, outputs, masked);
            async move { ctx.compile_async(label, &code, entry).await }
        };
        Ok(Self {
            range: compile("k_mi_dense_range", RANGE, 3, false, "k_mi_dense_range").await?,
            range_masked: compile(
                "k_mi_dense_range_masked",
                RANGE,
                3,
                true,
                "k_mi_dense_range",
            )
            .await?,
            range_final: ctx
                .compile_async(
                    "k_mi_dense_range_final",
                    RANGE_FINAL,
                    "k_mi_dense_range_final",
                )
                .await?,
            hist: compile("k_mi_dense_hist", HIST, 3, false, "k_mi_dense_hist").await?,
            hist_masked: compile("k_mi_dense_hist_masked", HIST, 3, true, "k_mi_dense_hist")
                .await?,
            deriv: compile("k_mi_dense_deriv", DERIV, 4, false, "k_mi_dense_deriv").await?,
            deriv_masked: compile(
                "k_mi_dense_deriv_masked",
                DERIV,
                4,
                true,
                "k_mi_dense_deriv",
            )
            .await?,
            merge: ctx
                .compile_async("k_mi_dense_merge_stripe", MERGE, "k_mi_dense_merge_stripe")
                .await?,
            merge_final: ctx
                .compile_async("k_mi_dense_merge_final", MERGE, "k_mi_dense_merge_final")
                .await?,
        })
    }
}

const ABSENT_FIXED: u32 = 1;
const ABSENT_MOVING: u32 = 2;

fn uniform(grid_u: &CcFinalU, bins: u32, chunk: u32, absent: u32, binning: &MiBinning) -> DenseMiU {
    DenseMiU {
        dims: grid_u.dims,
        cfg: [bins, chunk, absent, 0],
        frng: [
            binning.fixed_min as f32,
            binning.fixed_max as f32,
            binning.fixed_bin_size as f32,
            binning.fixed_normalized_min as f32,
        ],
        mrng: [
            binning.moving_min as f32,
            binning.moving_max as f32,
            binning.moving_bin_size as f32,
            binning.moving_normalized_min as f32,
        ],
        o: grid_u.o,
        sp: grid_u.sp,
        fthr: fixed_thresholds(binning),
    }
}

fn order_key(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn from_order_key(key: u32) -> f32 {
    f32::from_bits(if key >> 31 == 1 {
        key & 0x7fff_ffff
    } else {
        !key
    })
}

/// The smallest `f32` intensity whose fixed-axis bin (`MiBinning::fixed_bin`, in `f64`) is at
/// least `k`, for `k` from 3 to `bins - 3`; the remaining slots are `+inf`. Each threshold is a
/// binary search over the `f32` values in their order (at most 32 bin evaluations). The kernels
/// count the thresholds an intensity reaches, so an `f32` image gets the bins of the `f64`
/// formula exactly.
fn fixed_thresholds(binning: &MiBinning) -> [[f32; 4]; THRESHOLDS / 4] {
    let mut out = [f32::INFINITY; THRESHOLDS];
    let bin = |key: u32| binning.fixed_bin(f64::from(from_order_key(key)));
    for (slot, k) in out.iter_mut().zip(3..=binning.bins - 3) {
        let mut lo = order_key(f32::NEG_INFINITY);
        let mut hi = order_key(f32::INFINITY);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if bin(mid) >= k {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        *slot = from_order_key(hi);
    }
    let mut packed = [[0.0f32; 4]; THRESHOLDS / 4];
    for (i, v) in out.iter().enumerate() {
        packed[i / 4][i % 4] = *v;
    }
    packed
}

fn decode(raw: &[u32], bins: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let cell = |i: usize| -> u64 { u64::from(raw[2 * i]) | (u64::from(raw[2 * i + 1]) << 32) };
    let joint = (0..bins * bins)
        .map(|i| cell(i) as f64 / FIXED_POINT)
        .collect();
    let marginal = (0..bins).map(|i| cell(BINS * BINS + i) as f64).collect();
    let valid = (0..bins).map(|i| cell(BINS * BINS + i)).sum::<u64>() as usize;
    (joint, marginal, valid)
}

impl WgpuBackend {
    /// Both directions of the dense Mattes metric of `fixed` and `moving`, with the fixed and
    /// the moving mask when given (a mask that is not given has every voxel inside), and with the
    /// squared-norm reductions of both derivatives when `norms` is set.
    pub(in crate::wgpu_backend) async fn mi_dense_pair(
        &self,
        fixed: &GpuImage,
        moving: &GpuImage,
        fixed_mask: Option<&GpuImage>,
        moving_mask: Option<&GpuImage>,
        bins: usize,
        norms: bool,
    ) -> RegResult<(MetricPair<GpuField>, Option<[f64; 2]>)> {
        if let Some(why) = &self.mi_dense_failure {
            return Err(bad(format!(
                "the dense Mattes MI kernels are not available on this device, so SyN stages with \
                 MI cannot run on its GPU: {why}"
            )));
        }
        let grid = fixed.grid;
        self.require_same_grid("mattes", &grid, &moving.grid)?;
        for mask in [fixed_mask, moving_mask].into_iter().flatten() {
            self.require_same_grid("mattes", &grid, &mask.grid)?;
        }
        let (masks, absent) = match (fixed_mask, moving_mask) {
            (None, None) => (None, 0),
            (Some(f), Some(m)) => (Some((f, m)), 0),
            (Some(f), None) => (Some((f, f)), ABSENT_MOVING),
            (None, Some(m)) => (Some((m, m)), ABSENT_FIXED),
        };
        let swapped = ((absent & ABSENT_FIXED) << 1) | ((absent & ABSENT_MOVING) >> 1);
        let flags = [absent, swapped];
        if !(5..=BINS).contains(&bins) {
            return Err(bad(format!(
                "the GPU Mattes metric supports 5 to {BINS} histogram bins, not {bins}"
            )));
        }
        let k = self
            .mi_kernels
            .dense
            .as_ref()
            .map_err(|e| bad(format!("the dense Mattes MI kernels did not compile: {e}")))?;
        let n = grid.num_voxels() as u64;
        let chunk = u32::try_from(mattes_chunk(n))
            .map_err(|_| bad("the grid has too many voxels for the Mattes histogram"))?;
        let groups = n.div_ceil(u64::from(chunk)) as u32;
        let per_stripe = groups.div_ceil(MAX_STRIPES);
        let stripes = groups.div_ceil(per_stripe);
        let merge = MergeU {
            nwg: groups,
            per_stripe,
            stripes,
            pad: 0,
        };
        let geometry = cc_uniform(&grid, 0)?;
        let placeholder = MiBinning::from_ranges(bins, (0.0, 1.0), (0.0, 1.0))?;
        let range_parts = self
            .acquire("mi range partials", u64::from(groups) * 16)
            .await?;
        let range_out = self.acquire("mi range", 16).await?;
        let (range_kernel, extra): (_, Vec<&wgpu::Buffer>) = match masks {
            Some((fm, mm)) => (&k.range_masked, vec![fm.buf.buffer(), mm.buf.buffer()]),
            None => (&k.range, Vec::new()),
        };
        let base = uniform(&geometry, bins as u32, chunk, absent, &placeholder);
        self.run_pass("mi_dense_range", |pass| {
            let mut buffers = vec![
                fixed.buf.buffer(),
                moving.buf.buffer(),
                range_parts.buffer(),
            ];
            buffers.extend(&extra);
            self.launch(pass, range_kernel, &buffers, &base, [groups, 1, 1])?;
            self.launch(
                pass,
                &k.range_final,
                &[range_parts.buffer(), range_out.buffer()],
                &RangeFinalU {
                    count: groups,
                    pad: [0; 3],
                },
                [1, 1, 1],
            )
        })
        .await?;
        let r = self.read_scalars(range_out.buffer(), 4).await?;
        drop(range_parts);
        let fixed_range = (f64::from(r[0]), f64::from(r[1]));
        let moving_range = (f64::from(r[2]), f64::from(r[3]));
        if fixed_range.0 > fixed_range.1 {
            return Err(bad_metric(&empty_range_message("fixed", masks.is_some())));
        }
        if moving_range.0 > moving_range.1 {
            return Err(bad_metric(&empty_range_message("moving", masks.is_some())));
        }
        let binnings = [
            MiBinning::from_ranges(bins, fixed_range, moving_range)?,
            MiBinning::from_ranges(bins, moving_range, fixed_range)?,
        ];

        let partials = self
            .acquire("mi partials", u64::from(groups) * STRIDE as u64 * 8)
            .await?;
        let stripe_sums = self
            .acquire("mi stripe sums", u64::from(stripes) * STRIDE as u64 * 8)
            .await?;
        let merged = [
            self.acquire("mi merged", (STRIDE * 8) as u64).await?,
            self.acquire("mi merged", (STRIDE * 8) as u64).await?,
        ];
        let directions: [Direction<'_>; 2] = [
            (fixed, moving, masks),
            (moving, fixed, masks.map(|(f, m)| (m, f))),
        ];
        let bound = |d: usize, middle: &[&wgpu::Buffer]| -> Vec<wgpu::Buffer> {
            let (a, b, m) = directions[d];
            let mut out = vec![a.buf.buffer().clone(), b.buf.buffer().clone()];
            out.extend(middle.iter().map(|x| (*x).clone()));
            if let Some((ma, mb)) = m {
                out.push(ma.buf.buffer().clone());
                out.push(mb.buf.buffer().clone());
            }
            out
        };
        let hist_kernel = if masks.is_some() {
            &k.hist_masked
        } else {
            &k.hist
        };
        self.run_pass("mi_dense_hist", |pass| {
            for (d, binning) in binnings.iter().enumerate() {
                let buffers = bound(d, &[partials.buffer()]);
                let refs: Vec<&wgpu::Buffer> = buffers.iter().collect();
                self.launch(
                    pass,
                    hist_kernel,
                    &refs,
                    &uniform(&geometry, bins as u32, chunk, flags[d], binning),
                    [groups, 1, 1],
                )?;
                self.launch(
                    pass,
                    &k.merge,
                    &[partials.buffer(), stripe_sums.buffer()],
                    &merge,
                    [stripes, 1, 1],
                )?;
                self.launch(
                    pass,
                    &k.merge_final,
                    &[stripe_sums.buffer(), merged[d].buffer()],
                    &merge,
                    [1, 1, 1],
                )?;
            }
            Ok(())
        })
        .await?;
        let bytes = (STRIDE * 8) as u64;
        let raw: Vec<u32> = bytemuck::pod_collect_to_vec(
            &self
                .read_sources(&[(merged[0].buffer(), bytes), (merged[1].buffer(), bytes)])
                .await?,
        );
        drop((partials, stripe_sums, merged));
        let mut values = [0.0f64; 2];
        let ratios = [
            self.acquire("mi ratio", (BINS * BINS * 4) as u64).await?,
            self.acquire("mi ratio", (BINS * BINS * 4) as u64).await?,
        ];
        for (d, binning) in binnings.iter().enumerate() {
            let (joint, marginal, valid) = decode(&raw[d * 2 * STRIDE..], bins);
            if valid == 0 && masks.is_some() {
                return Err(bad_metric(NO_VOXEL_IN_MASKS));
            }
            let fin =
                finish_histogram(bins, binning.moving_bin_size, valid, joint, marginal, true)?;
            values[d] = fin.value;
            let ratio: Vec<f32> = fin.ratio.iter().map(|v| *v as f32).collect();
            self.ctx
                .queue
                .write_buffer(ratios[d].buffer(), 0, bytemuck::cast_slice(&ratio));
        }

        let outs = [
            self.field_buffer(&grid).await?,
            self.field_buffer(&grid).await?,
        ];
        let reduction = if norms {
            let part = self.reduction_partials(&grid).await?;
            let scalars = [
                self.acquire("reduction scalar", 16).await?,
                self.acquire("reduction scalar", 16).await?,
            ];
            Some((part, scalars))
        } else {
            None
        };
        let deriv_kernel = if masks.is_some() {
            &k.deriv_masked
        } else {
            &k.deriv
        };
        let g3 = self.groups_3d(&grid);
        self.run_pass("mi_dense_deriv", |pass| {
            for (d, binning) in binnings.iter().enumerate() {
                let buffers = bound(d, &[ratios[d].buffer(), outs[d].buffer()]);
                let refs: Vec<&wgpu::Buffer> = buffers.iter().collect();
                self.launch(
                    pass,
                    deriv_kernel,
                    &refs,
                    &uniform(&geometry, bins as u32, chunk, flags[d], binning),
                    g3,
                )?;
            }
            if let Some((part, scalars)) = &reduction {
                for (out, scalar) in outs.iter().zip(scalars) {
                    self.encode_reduce(
                        pass,
                        out.buffer(),
                        &grid,
                        1,
                        part.buffer(),
                        scalar.buffer(),
                    )?;
                }
            }
            Ok(())
        })
        .await?;
        let norms = match &reduction {
            Some((_, scalars)) => {
                let v: Vec<f32> = bytemuck::pod_collect_to_vec(
                    &self
                        .read_sources(&[(scalars[0].buffer(), 4), (scalars[1].buffer(), 4)])
                        .await?,
                );
                Some([f64::from(v[0]).sqrt(), f64::from(v[1]).sqrt()])
            }
            None => None,
        };
        let [o1, o2] = outs;
        Ok((
            (
                (values[0], GpuField { buf: o1, grid }),
                (values[1], GpuField { buf: o2, grid }),
            ),
            norms,
        ))
    }
}

fn bad_metric(message: &str) -> RegError {
    RegError::Invalid {
        what: "mattes",
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_thresholds_reproduce_the_cpu_bins_with_an_edge_at_zero_and_stay_fast() {
        let start = std::time::Instant::now();
        for (bins, fixed) in [
            (32, (-1024.0, 3072.0)),
            (32, (-1.0, 1.0)),
            (12, (0.0, 1000.0)),
        ] {
            let binning = MiBinning::from_ranges(bins, fixed, (0.0, 1.0)).unwrap();
            let thresholds: Vec<f32> = fixed_thresholds(&binning).into_iter().flatten().collect();
            for (i, t) in thresholds.iter().enumerate() {
                let k = i + 3;
                if k > bins - 3 {
                    assert_eq!(*t, f32::INFINITY);
                    continue;
                }
                assert!(
                    binning.fixed_bin(f64::from(*t)) >= k,
                    "{bins} {fixed:?} {k}"
                );
                assert!(
                    binning.fixed_bin(f64::from(t.next_down())) < k,
                    "{bins} {fixed:?} {k}"
                );
            }
            for v in [-1024.0f32, -0.5, 0.0, 0.25, 3.0, 250.0, 3072.0] {
                let gpu = 2 + thresholds.iter().filter(|t| v >= **t).count();
                assert_eq!(gpu, binning.fixed_bin(f64::from(v)), "{v}");
            }
        }
        let ct = MiBinning::from_ranges(32, (-1024.0, 3072.0), (0.0, 1.0)).unwrap();
        let edge = fixed_thresholds(&ct)[1][2];
        assert!(edge.abs() < 1e-3, "{edge}");
        assert!(start.elapsed().as_secs_f64() < 1.0, "{:?}", start.elapsed());
    }
}

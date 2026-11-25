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
// Portions derived from ANTs `itkWeightedVotingFusionImageFilter.hxx` (the patch search of
// `ThreadedGenerateDataForWeightedAveraging`) and ITKAdaptiveDenoising
// `itkNonLocalPatchBasedImageFilter.hxx` (`ComputeNeighborhoodPatchSimilarity`), as run by
// `shaders/jlf_search.wgsl`. The host splits a block of centres into z ranges whose atlas slab (with
// a halo of search and patch radius) fits the device, and bounds the work of one dispatch. See
// `docs/THIRD_PARTY_PORTS.md`.

//! Joint label fusion with the patch search on the GPU (the hybrid mode): the device scores every
//! candidate offset of every centre and atlas in `f32` and returns the best one; the CPU computes the
//! target patches, the joint error matrix, the weights, the votes and the intensities in `f64`, as
//! [`fastwarp_reg::jlf::joint_fusion`] does.
//!
//! The device follows the CPU's arithmetic and order, but WGSL allows a less exact division and lets
//! compilers fuse a multiplication and an addition, so two candidates whose scores differ by about
//! an `f32` rounding can be ranked the other way; exact ties are kept in the CPU's order.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use fastwarp_reg::jlf::prof;
use fastwarp_reg::jlf::{
    float_almost_equal, joint_fusion_with, Accumulators, FusionParts, JlfDevice, JlfInput,
    JlfOptions, JlfOutput, PatchMetric, PatchSearch, SearchBlock,
};
use fastwarp_reg::{RegError, Result as RegResult};

use crate::device::{block_gpu, GpuContext};
use crate::error::{GpuError, GpuResult};
use crate::wgpu_backend::block_reg;

const SRC: &str = include_str!("shaders/jlf_search.wgsl");
const VARIANT_SRC: [&str; 11] = [
    include_str!("shaders/jlf_search_k1.wgsl"),
    include_str!("shaders/jlf_search_k2.wgsl"),
    include_str!("shaders/jlf_search_k4.wgsl"),
    include_str!("shaders/jlf_search_t32.wgsl"),
    include_str!("shaders/jlf_search_t64.wgsl"),
    include_str!("shaders/jlf_search_ck2.wgsl"),
    include_str!("shaders/jlf_search_ck4.wgsl"),
    include_str!("shaders/jlf_search_ck8.wgsl"),
    include_str!("shaders/jlf_search_gk1.wgsl"),
    include_str!("shaders/jlf_search_gk2.wgsl"),
    include_str!("shaders/jlf_search_gk4.wgsl"),
];

#[derive(Clone, Copy, Debug)]
struct Variant {
    kernel: usize,
    dynamic: bool,
    legacy: bool,
    boxsum: bool,
}

impl Variant {
    fn from_env() -> Self {
        let mut v = Variant {
            kernel: 10,
            dynamic: true,
            legacy: false,
            boxsum: false,
        };
        if let Ok(s) = std::env::var("FASTWARP_JLF_VARIANT") {
            for tok in s.split(',') {
                match tok.trim() {
                    "orig" => v.kernel = 0,
                    "fixed" => v.dynamic = false,
                    "k1" => v.kernel = 1,
                    "k2" => v.kernel = 2,
                    "k4" => v.kernel = 3,
                    "t32" => v.kernel = 4,
                    "t64" => v.kernel = 5,
                    "ck2" => v.kernel = 6,
                    "ck4" => v.kernel = 7,
                    "ck8" => v.kernel = 8,
                    "gk1" => v.kernel = 9,
                    "gk2" => v.kernel = 10,
                    "gk4" => v.kernel = 11,
                    "dyn" => v.dynamic = true,
                    "legacy" => {
                        v.kernel = 0;
                        v.dynamic = false;
                        v.legacy = true;
                    }
                    "box" => v.boxsum = true,
                    _ => {}
                }
            }
        }
        v
    }

    fn transposed(&self) -> bool {
        self.kernel >= 9
    }

    fn wg(&self) -> u32 {
        if self.kernel == 4 {
            32
        } else {
            WG
        }
    }
}
const WEIGH_SRC: &str = include_str!("shaders/jlf_weights.wgsl");
const WEIGH_TILED_SRC: &str = include_str!("shaders/jlf_weights_tiled.wgsl");
const TILED_FROM: usize = 11;

/// The most atlases the GPU weights and votes take (`JlfDevice::Gpu` runs as `Hybrid` above).
pub const MAX_GPU_ATLASES: usize = 32;
const WG: u32 = 64;
const PATCH_BYTES: u64 = 64 << 20;
const ATLAS_BYTES: u64 = 256 << 20;
const WORK_PER_DISPATCH: u64 = 1 << 30;
const UNSET: u32 = u32::MAX;
const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SearchU {
    dims: [u32; 4],
    slab: [u32; 4],
    patch: [u32; 4],
    pad: [u32; 4],
    ext: [i32; 4],
}

#[derive(PartialEq, Eq)]
struct SlabKey {
    images: Vec<(usize, usize)>,
    size: [usize; 3],
    modalities: usize,
}

impl SlabKey {
    fn of(block: &SearchBlock<'_>) -> Self {
        SlabKey {
            images: block
                .atlases
                .iter()
                .flat_map(|a| a[..block.modalities].iter())
                .map(|v| (v.as_ptr() as usize, v.len()))
                .collect(),
            size: block.size,
            modalities: block.modalities,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WeighU {
    dims: [u32; 4],
    slab: [u32; 4],
    patch: [u32; 4],
    par: [f32; 4],
}

/// Counters of a [`GpuJlfSearch`]: atlas slabs uploaded for the search, z slabs of the vote
/// gather, voxels whose votes were redone on the CPU (more than 16 labels), dispatches of the
/// workgroup-per-voxel weights kernel (11 atlases or more) and the time in each kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct GpuJlfStats {
    pub slab_uploads: u64,
    pub gather_slabs: u64,
    pub vote_redos: u64,
    pub tiled_weight_dispatches: u64,
    /// Seconds in the search dispatches, each measured on the host until the device is idle: the
    /// uniform and bind group setup of every dispatch included, the atlas slab and target patch
    /// uploads and the readback of the offsets excluded. Only slabs that completed count; 0 on
    /// wasm32-unknown-unknown.
    pub search_seconds: f64,
    /// Seconds in the weights dispatches of the GPU mode, measured as the search seconds; the
    /// readback of the weights (`wout`) excluded. Only slabs that completed count; 0 on
    /// wasm32-unknown-unknown.
    pub weight_seconds: f64,
    /// Seconds in the gather dispatches of the GPU mode, measured as the search seconds; the
    /// readback of the gathered votes excluded. Only slabs that completed count; 0 on
    /// wasm32-unknown-unknown.
    pub gather_seconds: f64,
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn stamp() -> Option<std::time::Instant> {
    Some(std::time::Instant::now())
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn stamp() -> Option<std::time::Instant> {
    None
}

fn seconds(since: Option<std::time::Instant>) -> f64 {
    since.map_or(0.0, |s| s.elapsed().as_secs_f64())
}

/// What the device weights need besides the search block: the z-scored target patches of every
/// centre (searched modalities x patch length each), `alpha` and `beta`.
#[derive(Clone, Copy, Debug)]
pub struct WeighArgs<'a> {
    normalized: &'a [f32],
    alpha: f32,
    beta: f32,
    beta_mode: u32,
    cond_limit: f32,
}

impl<'a> WeighArgs<'a> {
    /// The arguments for `normalized` target patches and the fusion's `alpha` and `beta` (used
    /// in `f32`); a centre whose matrix has a 1-norm condition number above 1000 is flagged for the
    /// CPU.
    pub fn new(normalized: &'a [f32], alpha: f64, beta: f64) -> Self {
        WeighArgs {
            normalized,
            alpha: alpha as f32,
            beta: beta as f32,
            beta_mode: beta_mode(beta),
            cond_limit: COND_LIMIT,
        }
    }
}

struct AtlasSlab {
    key: SlabKey,
    zlo: usize,
    planes: usize,
    buffer: wgpu::Buffer,
    target: Option<wgpu::Buffer>,
    finite: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoxU {
    dims: [u32; 4],
    slab: [u32; 4],
    org: [i32; 4],
    ext: [u32; 4],
    pr: [i32; 4],
    sr: [i32; 4],
    jb: [u32; 4],
    sh: [f32; 4],
}

const BOX_SRC: &str = include_str!("shaders/jlf_box.wgsl");

struct BoxDims {
    lo: [i32; 3],
    b: [u64; 3],
    ds: u64,
    dp: u64,
    dq: u64,
}

fn box_dims(block: &SearchBlock<'_>, eligible: &[[u32; 2]], ext: [i32; 4]) -> BoxDims {
    let [nx, ny, _] = block.size;
    let plane = nx * ny;
    let pr = block.patch_radius.map(|r| r as u64);
    let sr = [0, 1, 2].map(|d| (ext[d] as u64).saturating_sub(pr[d]));
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    for e in eligible {
        let v = e[0] as usize;
        let q = [(v % nx) as i32, ((v / nx) % ny) as i32, (v / plane) as i32];
        for d in 0..3 {
            lo[d] = lo[d].min(q[d]);
            hi[d] = hi[d].max(q[d]);
        }
    }
    let b = [0, 1, 2].map(|d| (hi[d] - lo[d] + 1).max(0) as u64);
    BoxDims {
        lo,
        b,
        ds: (b[0] + 2 * sr[0]) * (b[1] + 2 * sr[1]) * (b[2] + 2 * sr[2]),
        dp: b[0] * (b[1] + 2 * pr[1]) * (b[2] + 2 * pr[2]),
        dq: b[0] * b[1] * (b[2] + 2 * pr[2]),
    }
}

fn gerr(e: GpuError) -> RegError {
    RegError::Invalid {
        what: "joint fusion GPU search",
        message: e.to_string(),
    }
}

fn too_large(what: &str) -> GpuError {
    GpuError::Invalid(format!("{what} is too large for a GPU buffer"))
}

/// The patch search on the device. Between [`PatchSearch::begin`] and [`PatchSearch::finish`] (one
/// fusion) it keeps the atlas slab it uploaded for the next block; a block searched outside them
/// uploads its own and keeps nothing.
pub struct GpuJlfSearch {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    max_groups: u32,
    binding: u64,
    atlas_budget: u64,
    patch_budget: u64,
    reuse: bool,
    slab: Option<AtlasSlab>,
    uploads: u64,
    gathers: u64,
    vote_redos: u64,
    tiled_dispatches: u64,
    times: [f64; 3],
    weigh: Option<(bool, wgpu::ComputePipeline)>,
    gather_budget: u64,
    variant: Variant,
    run: u32,
    wrun: u32,
    grun: u32,
    boxpipe: Option<wgpu::ComputePipeline>,
    transpose: Option<wgpu::ComputePipeline>,
}

const MAX_SCALE: u32 = 64;

fn adapt(scale: u32, full: bool, dt: f64) -> u32 {
    if dt > 0.25 {
        (scale / 2).max(1)
    } else if full && dt < 0.05 {
        (scale * 2).min(MAX_SCALE)
    } else {
        scale
    }
}

fn scaled(base: u64, scale: u32, unit: u64) -> u64 {
    (base * u64::from(scale)).min(1 << 24) / unit * unit
}

impl GpuJlfSearch {
    /// Compiles the search kernel on `ctx`; an error on wasm32 (use [`Self::new_async`]).
    pub fn new(ctx: Arc<GpuContext>) -> GpuResult<Self> {
        block_gpu("GpuJlfSearch::new", Self::new_async(ctx))
    }

    /// Async [`Self::new`].
    pub async fn new_async(ctx: Arc<GpuContext>) -> GpuResult<Self> {
        let variant = Variant::from_env();
        let src = if variant.kernel == 0 {
            SRC
        } else {
            VARIANT_SRC[variant.kernel - 1]
        };
        let sp = prof::span("g.compile_search");
        let pipeline = ctx
            .compile_async("k_jlf_search", src, "k_jlf_search")
            .await?;
        let transpose = if variant.transposed() {
            Some(
                ctx.compile_async(
                    "k_jlf_transpose",
                    include_str!("shaders/jlf_transpose.wgsl"),
                    "k_jlf_transpose",
                )
                .await?,
            )
        } else {
            None
        };
        let boxpipe = if variant.boxsum {
            Some(ctx.compile_async("k_jlf_box", BOX_SRC, "k_jlf_box").await?)
        } else {
            None
        };
        drop(sp);
        let limits = ctx.device.limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        Ok(Self {
            max_groups: limits.max_compute_workgroups_per_dimension.max(1),
            binding,
            atlas_budget: ATLAS_BYTES.min(binding),
            patch_budget: PATCH_BYTES.min(binding),
            reuse: false,
            slab: None,
            uploads: 0,
            gathers: 0,
            vote_redos: 0,
            tiled_dispatches: 0,
            times: [0.0; 3],
            weigh: None,
            gather_budget: GATHER_OUT_BYTES.min(binding),
            variant,
            run: 1,
            wrun: 1,
            grun: 1,
            boxpipe,
            transpose,
            pipeline,
            ctx,
        })
    }

    /// Caps the bytes of the atlas slab held on the device (default 256 MiB, at most the binding
    /// limit); a smaller cap splits a block into more z ranges.
    pub fn with_atlas_budget(mut self, bytes: u64) -> Self {
        self.atlas_budget = bytes.clamp(4, self.binding);
        self
    }

    /// Caps the bytes of target patches per block (default 64 MiB, at most the binding limit); a
    /// smaller cap makes smaller blocks.
    pub fn with_patch_budget(mut self, bytes: u64) -> Self {
        self.patch_budget = bytes.clamp(4, self.binding);
        self
    }

    /// Caps the bytes of the vote records of one gather of the GPU mode (default 64 MiB, at most
    /// the binding limit); a smaller cap gathers fewer planes at a time.
    pub fn with_gather_budget(mut self, bytes: u64) -> Self {
        self.gather_budget = bytes.clamp(4, self.binding);
        self
    }

    /// How many atlas slabs this search has uploaded.
    pub fn slab_uploads(&self) -> u64 {
        self.uploads
    }

    /// What this search has done so far.
    pub fn stats(&self) -> GpuJlfStats {
        GpuJlfStats {
            slab_uploads: self.uploads,
            gather_slabs: self.gathers,
            vote_redos: self.vote_redos,
            tiled_weight_dispatches: self.tiled_dispatches,
            search_seconds: self.times[0],
            weight_seconds: self.times[1],
            gather_seconds: self.times[2],
        }
    }

    /// The device context.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    async fn storage<T: Pod>(&self, label: &str, data: &[T]) -> GpuResult<wgpu::Buffer> {
        let bytes = (std::mem::size_of_val(data) as u64).max(4);
        if bytes > self.binding {
            return Err(too_large(label));
        }
        let buffer = self
            .ctx
            .scoped_async(label, bytes, || {
                self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage: STORAGE,
                    mapped_at_creation: false,
                })
            })
            .await?;
        if !data.is_empty() {
            self.ctx.upload(&buffer, data);
        }
        Ok(buffer)
    }

    async fn scratch(&self, label: &str, bytes: u64) -> GpuResult<wgpu::Buffer> {
        let bytes = bytes.max(4);
        if bytes > self.binding {
            return Err(too_large(label));
        }
        self.ctx
            .scoped_async(label, bytes, || {
                self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage: STORAGE,
                    mapped_at_creation: false,
                })
            })
            .await
    }

    async fn ensure_slab(
        &mut self,
        block: &SearchBlock<'_>,
        lo: usize,
        hi: usize,
        budget_planes: usize,
    ) -> GpuResult<()> {
        let key = SlabKey::of(block);
        if self
            .slab
            .as_ref()
            .is_some_and(|s| s.key == key && s.zlo <= lo && hi <= s.zlo + s.planes)
        {
            return Ok(());
        }
        self.slab = None;
        let nz = block.size[2];
        let planes = budget_planes.min(nz - lo).max(hi - lo);
        let plane = block.size[0] * block.size[1];
        let mut data = Vec::with_capacity(block.atlases.len() * block.modalities * planes * plane);
        for atlas in block.atlases {
            for image in &atlas[..block.modalities] {
                data.extend_from_slice(&image[lo * plane..(lo + planes) * plane]);
            }
        }
        let buffer = self.storage("jlf atlas slab", &data).await?;
        let (target, finite) = if self.boxpipe.is_some() && !block.target.is_empty() {
            let t = &block.target[0][lo * plane..(lo + planes) * plane];
            let finite = data.iter().all(|v| v.is_finite()) && t.iter().all(|v| v.is_finite());
            (Some(self.storage("jlf target slab", t).await?), finite)
        } else {
            (None, false)
        };
        self.uploads += 1;
        self.slab = Some(AtlasSlab {
            key,
            zlo: lo,
            planes,
            buffer,
            target,
            finite,
        });
        Ok(())
    }

    /// [`PatchSearch::search`] with the device's error.
    pub async fn search_async(&mut self, block: &SearchBlock<'_>) -> GpuResult<Vec<u32>> {
        if !self.reuse {
            self.slab = None;
        }
        let out = self.search_blocks(block, None).await.map(|(best, _)| best);
        if !self.reuse || out.is_err() {
            self.slab = None;
        }
        out
    }

    /// [`Self::search_async`] followed, on the device, by the weights of every centre: for centre
    /// `c`, `atlases + 1` values at `c * (atlases + 1)`, the normalised weights and then a flag (the
    /// bits of a `u32`, 1 where the `f32` solve was not reliable and the weights are 0). At most
    /// [`MAX_GPU_ATLASES`] atlases.
    pub async fn search_and_weigh_async(
        &mut self,
        block: &SearchBlock<'_>,
        weigh: &WeighArgs<'_>,
    ) -> GpuResult<(Vec<u32>, Vec<f32>)> {
        if block.atlases.len() > MAX_GPU_ATLASES {
            return Err(GpuError::Invalid(format!(
                "the GPU weights take at most {MAX_GPU_ATLASES} atlases"
            )));
        }
        let tiled = block.atlases.len() >= TILED_FROM;
        if self.weigh.as_ref().is_none_or(|(t, _)| *t != tiled) {
            let pipeline = if tiled {
                self.ctx
                    .compile_async(
                        "k_jlf_weights_tiled",
                        WEIGH_TILED_SRC,
                        "k_jlf_weights_tiled",
                    )
                    .await?
            } else {
                self.ctx
                    .compile_async("k_jlf_weights", WEIGH_SRC, "k_jlf_weights")
                    .await?
            };
            self.weigh = Some((tiled, pipeline));
        }
        if !self.reuse {
            self.slab = None;
        }
        let out = self.search_blocks(block, Some(weigh)).await;
        if !self.reuse || out.is_err() {
            self.slab = None;
        }
        out
    }

    async fn search_blocks(
        &mut self,
        block: &SearchBlock<'_>,
        weigh: Option<&WeighArgs<'_>>,
    ) -> GpuResult<(Vec<u32>, Vec<f32>)> {
        let atlases = block.atlases.len();
        let centres = block.centres.len();
        let mut best = Vec::with_capacity(centres * atlases);
        let mut weights = Vec::new();
        if centres == 0 || atlases == 0 {
            return Ok((best, weights));
        }
        let [nx, ny, nz] = block.size;
        let plane = nx * ny;
        let p = block.patch_len();
        let width = p * block.modalities;
        let reach = block
            .lists
            .iter()
            .flatten()
            .map(|o| o[2].unsigned_abs())
            .max()
            .unwrap_or(0)
            + block.patch_radius[2];
        let plane_bytes = (atlases * block.modalities * plane * 4) as u64;
        let budget_planes = (self.atlas_budget / plane_bytes.max(1)) as usize;
        let mut starts = Vec::with_capacity(block.lists.len());
        let mut flat: Vec<[i32; 4]> = Vec::new();
        for list in block.lists {
            starts.push(flat.len() as u32);
            flat.extend(
                list.iter()
                    .map(|o| [o[0] as i32, o[1] as i32, o[2] as i32, 0]),
            );
        }
        let longest = block.lists.iter().map(Vec::len).max().unwrap_or(1) as u64;
        let offsets_buf = self.storage("jlf search offsets", &flat).await?;
        let patch: Vec<[i32; 4]> = block
            .patch_offsets()
            .iter()
            .map(|o| [o[0] as i32, o[1] as i32, o[2] as i32, 0])
            .collect();
        let patch_buf = self.storage("jlf patch offsets", &patch).await?;
        let per_item = (longest * width as u64).max(1);
        let vwg = self.variant.wg();
        let items_per_dispatch =
            (WORK_PER_DISPATCH / per_item).clamp(u64::from(WG), 1 << 24) as u32 / vwg * vwg;
        let ext = {
            let mut e = [0i32; 4];
            for d in 0..3 {
                e[d] = (block
                    .lists
                    .iter()
                    .flatten()
                    .map(|o| o[d].unsigned_abs())
                    .max()
                    .unwrap_or(0)
                    + block.patch_radius[d]) as i32;
            }
            e
        };
        let max_count = [
            self.binding / (atlases as u64 * 4),
            self.binding / 16,
            self.binding / (width.max(1) as u64 * 4),
            self.binding / ((atlases as u64 + 1) * 4),
            u64::from(u32::MAX) / (atlases as u64 + 1),
        ]
        .into_iter()
        .min()
        .unwrap_or(1)
        .max(1) as usize;
        let max_count = if self.variant.kernel != 0 && max_count >= vwg as usize {
            max_count / vwg as usize * vwg as usize
        } else {
            max_count
        };
        let mut first = 0;
        while first < centres {
            let z0 = block.centres[first] / plane;
            let mut last = first + 1;
            while last < centres
                && last - first < max_count
                && block.centres[last] / plane - z0 + 2 * reach < budget_planes
            {
                last += 1;
            }
            if 2 * reach + 1 > budget_planes {
                return Err(GpuError::Invalid(format!(
                    "the atlas slab of one plane with a halo of {reach} planes needs {} bytes, more \
                     than the {} the GPU search may hold",
                    plane_bytes * (2 * reach as u64 + 1),
                    self.atlas_budget
                )));
            }
            let z1 = block.centres[last - 1] / plane;
            let lo = z0.saturating_sub(reach);
            let hi = (z1 + reach + 1).min(nz);
            let sp = prof::span("s.slab");
            self.ensure_slab(block, lo, hi, budget_planes).await?;
            drop(sp);
            let sp = prof::span("s.upload");
            let slab = self
                .slab
                .as_ref()
                .ok_or_else(|| GpuError::Internal("jlf atlas slab missing".into()))?;
            let (zlo, planes) = (slab.zlo, slab.planes);
            let recs: Vec<[u32; 4]> = (first..last)
                .map(|c| {
                    let l = block.list_of[c] as usize;
                    [
                        block.centres[c] as u32,
                        starts[l],
                        block.lists[l].len() as u32,
                        0,
                    ]
                })
                .collect();
            let count = last - first;
            let items = count
                .checked_mul(atlases)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| too_large("the offsets of one block"))?;
            let pcount = if self.variant.kernel == 0 {
                count
            } else {
                count.next_multiple_of(vwg as usize)
            };
            let pitems = (pcount * atlases) as u32;
            let centre_buf = self.storage("jlf centres", &recs).await?;
            let boxed = self.boxpipe.is_some()
                && block.metric == PatchMetric::PearsonCorrelation
                && block.lists.len() == 1
                && block.modalities == 1
                && self.slab.as_ref().is_some_and(|s| s.finite);
            let mut eligible: Vec<[u32; 2]> = Vec::new();
            let search_buf = if boxed {
                let mut srecs = recs.clone();
                for (k, r) in srecs.iter_mut().enumerate() {
                    let v = r[0] as usize;
                    let q = [v % nx, (v / nx) % ny, v / plane];
                    let interior = (0..3).all(|d| {
                        q[d] >= ext[d] as usize && q[d] + (ext[d] as usize) < block.size[d]
                    });
                    let c = first + k;
                    if interior && block.patches[c * width..(c + 1) * width].iter().all(|x| x.is_finite()) {
                        r[2] = 0;
                        eligible.push([v as u32, k as u32]);
                    }
                }
                let d = box_dims(block, &eligible, ext);
                if eligible.is_empty() || d.dp * 4 > self.binding || d.ds * 8 > self.binding {
                    eligible.clear();
                    None
                } else {
                    Some(self.storage("jlf search centres", &srecs).await?)
                }
            } else {
                None
            };
            let patch_values = &block.patches[first * width..last * width];
            let plain_buf = self.storage("jlf target patches", patch_values).await?;
            let target_buf = match self.transpose.as_ref() {
                Some(tp) => {
                    let total = (pcount * width) as u32;
                    let tr = self.scratch("jlf transposed target patches", u64::from(total) * 4).await?;
                    let (gx, gy) = self.groups(total);
                    let uniform = [count as u32, width as u32, total, gx * WG];
                    self.dispatch(tp, &[&plain_buf, &tr], bytemuck::bytes_of(&uniform), [gx, gy])
                        .await?;
                    tr
                }
                None => plain_buf.clone(),
            };
            let shared = weigh.is_some_and(|a| {
                std::ptr::eq(&a.normalized[first * width..last * width], patch_values)
            });
            let plain_buf = shared.then_some(plain_buf);
            let out_buf = self
                .storage("jlf best offsets", &vec![UNSET; items as usize])
                .await?;
            drop(sp);
            let search_start = stamp();
            let slab = self
                .slab
                .as_ref()
                .ok_or_else(|| GpuError::Internal("jlf atlas slab missing".into()))?;
            let buffers = [
                &slab.buffer,
                &target_buf,
                search_buf.as_ref().unwrap_or(&centre_buf),
                &offsets_buf,
                &patch_buf,
                &out_buf,
            ];
            let mut base = 0u32;
            while base < pitems {
                let want = if self.variant.dynamic {
                    scaled(u64::from(items_per_dispatch), self.run, u64::from(vwg)) as u32
                } else {
                    items_per_dispatch
                };
                let run = want.min(pitems - base);
                let (gx, gy) = self.groups_of(run, vwg);
                let started = stamp();
                let uniform = SearchU {
                    ext,
                    dims: [nx as u32, ny as u32, nz as u32, base + run],
                    slab: [
                        zlo as u32,
                        planes as u32,
                        atlases as u32,
                        block.modalities as u32,
                    ],
                    patch: [
                        p as u32,
                        u32::from(block.metric == PatchMetric::MeanSquares),
                        gx * vwg,
                        base,
                    ],
                    pad: [count as u32, pcount as u32, 0, 0],
                };
                self.dispatch(
                    &self.pipeline,
                    &buffers,
                    bytemuck::bytes_of(&uniform),
                    [gx, gy],
                )
                .await?;
                if let (true, Some(s)) = (self.variant.dynamic, started) {
                    self.run = adapt(self.run, run == want, s.elapsed().as_secs_f64());
                }
                base += run;
            }
            if !eligible.is_empty() {
                let slab = self
                    .slab
                    .as_ref()
                    .ok_or_else(|| GpuError::Internal("jlf atlas slab missing".into()))?;
                self.box_search(block, slab, &eligible, &offsets_buf, &out_buf, ext, atlases)
                    .await?;
                prof::add("n.box_centres", eligible.len() as f64);
            }
            let search_s = seconds(search_start);
            prof::add("s.kernel", search_s);
            prof::add("n.search_items", f64::from(items));
            let mut weight_s = 0.0;
            let mut tiled_runs = 0u64;
            if let (Some(args), Some((tiled, pipeline))) = (weigh, self.weigh.as_ref()) {
                let normalized = &args.normalized[first * width..last * width];
                let tn_buf = if let Some(b) = plain_buf.as_ref() {
                    b.clone()
                } else {
                    self.storage("jlf normalised target patches", normalized)
                        .await?
                };
                let wout = self
                    .storage("jlf weights", &vec![0f32; count * (atlases + 1)])
                    .await?;
                let per_centre = (width * atlases * (atlases + 2)) as u64;
                let per_dispatch =
                    (WORK_PER_DISPATCH / per_centre.max(1)).clamp(u64::from(WG), 1 << 22) as u32;
                let total = count as u32;
                let weight_start = stamp();
                let mut base = 0u32;
                while base < total {
                    let want = if self.variant.dynamic {
                        scaled(u64::from(per_dispatch), self.wrun, 1) as u32
                    } else {
                        per_dispatch
                    };
                    let run = want.min(total - base);
                    let started = stamp();
                    let tiled = *tiled;
                    tiled_runs += u64::from(tiled);
                    let (gx, gy, stride) = if tiled {
                        let gx = run.min(self.max_groups);
                        (gx, run.div_ceil(gx), gx)
                    } else {
                        let (gx, gy) = self.groups(run);
                        (gx, gy, gx * WG)
                    };
                    let uniform = WeighU {
                        dims: [nx as u32, ny as u32, nz as u32, base + run],
                        slab: [
                            zlo as u32,
                            planes as u32,
                            atlases as u32,
                            block.modalities as u32,
                        ],
                        patch: [p as u32, args.beta_mode, stride, base],
                        par: [args.alpha, args.beta, args.cond_limit, 0.0],
                    };
                    self.dispatch(
                        pipeline,
                        &[
                            &slab.buffer,
                            &tn_buf,
                            &centre_buf,
                            &offsets_buf,
                            &patch_buf,
                            &out_buf,
                            &wout,
                        ],
                        bytemuck::bytes_of(&uniform),
                        [gx, gy],
                    )
                    .await?;
                    if let (true, Some(s)) = (self.variant.dynamic, started) {
                        self.wrun = adapt(self.wrun, run == want, s.elapsed().as_secs_f64());
                    }
                    base += run;
                }
                weight_s = seconds(weight_start);
                prof::add("w.kernel", weight_s);
                let _sp = prof::span("w.readback");
                weights.extend(
                    self.ctx
                        .read_f32_async(&wout, count * (atlases + 1))
                        .await?,
                );
            }
            let sp = prof::span("s.readback");
            let got = self.ctx.read_u32_async(&out_buf, items as usize).await?;
            drop(sp);
            for (k, j) in got.iter().enumerate() {
                let c = first + k / atlases;
                if *j as usize >= block.list(c).len() {
                    return Err(GpuError::Verification(format!(
                        "the GPU search returned offset {j} for centre {} (of {} candidates)",
                        block.centres[c],
                        block.list(c).len()
                    )));
                }
            }
            prof::hash(&got);
            best.extend(got);
            self.tiled_dispatches += tiled_runs;
            self.times[0] += search_s;
            self.times[1] += weight_s;
            first = last;
        }
        Ok((best, weights))
    }

    #[allow(clippy::too_many_arguments)]
    async fn box_search(
        &self,
        block: &SearchBlock<'_>,
        slab: &AtlasSlab,
        eligible: &[[u32; 2]],
        offsets_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        ext: [i32; 4],
        atlases: usize,
    ) -> GpuResult<()> {
        let Some(pipe) = self.boxpipe.as_ref() else {
            return Ok(());
        };
        let Some(tbuf) = slab.target.as_ref() else {
            return Ok(());
        };
        let [nx, ny, nz] = block.size;
        let pr = block.patch_radius.map(|r| r as i32);
        let sr = [ext[0] - pr[0], ext[1] - pr[1], ext[2] - pr[2]];
        let BoxDims { lo, b, ds, dp, dq } = box_dims(block, eligible, ext);
        let total = block.lists[0].len() as u64;
        let limit = self.binding.min(256 << 20) / 4;
        let jn = (limit / dp.max(1)).clamp(1, total);
        let ncent = eligible.len();
        let n = block.patch_len() as f32;
        let cy = (eligible
            .iter()
            .map(|e| f64::from(block.target[0][e[0] as usize]))
            .sum::<f64>()
            / ncent as f64) as f32;
        let ecent = self.storage("jlf box centres", eligible).await?;
        let cstat = self.scratch("jlf box stats", ncent as u64 * 8).await?;
        let bx = self.scratch("jlf box sums", 2 * ds * 4).await?;
        let px = self.scratch("jlf box px", jn * dp * 4).await?;
        let pxy = self.scratch("jlf box pxy", jn * dq * 4).await?;
        let state = self.scratch("jlf box state", ncent as u64 * 16).await?;
        let buffers = [
            &slab.buffer,
            tbuf,
            offsets_buf,
            &ecent,
            &cstat,
            &bx,
            &px,
            &pxy,
            &state,
            out_buf,
        ];
        let mut calls: Vec<(BoxU, [u32; 2])> = Vec::new();
        let mut call = |mode: u32, a: usize, cx: f32, j0: u64, jcount: u64, items: u64| {
            let (gx, gy) = self.groups_of(items as u32, WG);
            calls.push((
                BoxU {
                    dims: [nx as u32, ny as u32, nz as u32, mode],
                    slab: [slab.zlo as u32, slab.planes as u32, a as u32, 0],
                    org: [lo[0], lo[1], lo[2], atlases as i32],
                    ext: [b[0] as u32, b[1] as u32, b[2] as u32, ncent as u32],
                    pr: [pr[0], pr[1], pr[2], 0],
                    sr: [sr[0], sr[1], sr[2], 0],
                    jb: [j0 as u32, jcount as u32, total as u32, gx * WG],
                    sh: [cx, cy, n, 0.0],
                },
                [gx, gy],
            ));
        };
        call(0, 0, 0.0, 0, 0, ncent as u64);
        for a in 0..atlases {
            let img = &block.atlases[a][0];
            let cx = (eligible
                .iter()
                .map(|e| f64::from(img[e[0] as usize]))
                .sum::<f64>()
                / ncent as f64) as f32;
            call(1, a, cx, 0, 0, ds);
            let mut j0 = 0;
            while j0 < total {
                let jc = jn.min(total - j0);
                call(2, a, cx, j0, jc, jc * dp);
                call(3, a, cx, j0, jc, jc * dq);
                call(4, a, cx, j0, jc, ncent as u64);
                j0 += jc;
            }
        }
        let per_atlas = 1 + 3 * total.div_ceil(jn) as usize;
        let ends: Vec<usize> = (1..=atlases).map(|a| 1 + a * per_atlas).collect();
        let list: Vec<(Vec<u8>, [u32; 2])> = calls
            .iter()
            .map(|(u, g)| (bytemuck::bytes_of(u).to_vec(), *g))
            .collect();
        let mut start = 0;
        for end in ends {
            self.dispatch_batch(pipe, &buffers, &list[start..end]).await?;
            start = end;
        }
        Ok(())
    }

    async fn dispatch_batch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        calls: &[(Vec<u8>, [u32; 2])],
    ) -> GpuResult<()> {
        let layout = pipeline.get_bind_group_layout(0);
        let mut groups = Vec::with_capacity(calls.len());
        for (bytes, _) in calls {
            let ubuf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jlf box uniform"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&ubuf, 0, bytes);
            let mut entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: buffers.len() as u32,
                resource: ubuf.as_entire_binding(),
            });
            let group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jlf box"),
                layout: &layout,
                entries: &entries,
            });
            groups.push((group, ubuf));
        }
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jlf box"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jlf box"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for ((group, _), (_, g)) in groups.iter().zip(calls) {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(g[0], g[1], 1);
            }
        }
        self.ctx
            .scoped_async("jlf box", 0, || {
                self.ctx.queue.submit(Some(enc.finish()));
            })
            .await?;
        self.ctx.sync_async().await
    }

    fn groups(&self, items: u32) -> (u32, u32) {
        self.groups_of(items, WG)
    }

    fn groups_of(&self, items: u32, wg: u32) -> (u32, u32) {
        let groups = items.div_ceil(wg);
        if groups <= self.max_groups {
            (groups, 1)
        } else {
            (self.max_groups, groups.div_ceil(self.max_groups))
        }
    }

    async fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        bytes: &[u8],
        groups: [u32; 2],
    ) -> GpuResult<()> {
        let ubuf = self
            .ctx
            .scoped_async("jlf search uniform", bytes.len() as u64, || {
                self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jlf search uniform"),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .await?;
        self.ctx.queue.write_buffer(&ubuf, 0, bytes);
        let layout = pipeline.get_bind_group_layout(0);
        let group = self
            .ctx
            .scoped_async("jlf search bind group", 0, || {
                let mut entries: Vec<wgpu::BindGroupEntry> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect();
                entries.push(wgpu::BindGroupEntry {
                    binding: buffers.len() as u32,
                    resource: ubuf.as_entire_binding(),
                });
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("jlf search"),
                        layout: &layout,
                        entries: &entries,
                    })
            })
            .await?;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jlf search"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jlf search"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        self.ctx
            .scoped_async("jlf search", 0, || {
                self.ctx.queue.submit(Some(enc.finish()));
            })
            .await?;
        self.ctx.sync_async().await
    }
}

impl PatchSearch for GpuJlfSearch {
    fn begin(&mut self) {
        self.slab = None;
        self.reuse = true;
    }

    fn finish(&mut self) {
        self.slab = None;
        self.reuse = false;
    }

    fn block_voxels(&self, size: [usize; 3], patch_values: usize) -> usize {
        let plane = (size[0] * size[1]).max(1);
        let budget = self.patch_budget;
        let voxels = (budget / (patch_values.max(1) as u64 * 4)).max(1) as usize;
        if voxels >= plane {
            voxels / plane * plane
        } else {
            voxels
        }
    }

    async fn search(&mut self, block: &SearchBlock<'_>) -> RegResult<Vec<u32>> {
        self.search_async(block).await.map_err(gerr)
    }
}

/// Joint label fusion on the GPU of `ctx`: with [`JlfDevice::Gpu`] the search, the weights and the
/// votes and intensities (see [`MAX_GPU_ATLASES`] and the fallbacks below), otherwise the patch
/// search only, with everything else on the CPU ([`JlfDevice::Hybrid`]). In the GPU mode the
/// weights of a centre whose `f32` Cholesky solve is not reliable (a pivot that is not positive, a
/// 1-norm condition number of the matrix above 1000, weights that do not sum to a positive number)
/// are computed on the CPU in `f64`, as are all weights with `nonnegative`; a voxel with more than 16
/// voted labels gets its votes on the CPU; more than [`MAX_GPU_ATLASES`] atlases run as
/// [`JlfDevice::Hybrid`]. An error on wasm32 (use [`joint_fusion_async`]).
pub fn joint_fusion(
    ctx: Arc<GpuContext>,
    input: &JlfInput,
    opts: &JlfOptions,
    cancel: Option<&AtomicBool>,
) -> RegResult<JlfOutput> {
    block_reg(
        "jlf::joint_fusion",
        joint_fusion_async(ctx, input, opts, cancel),
    )
}

/// Async [`joint_fusion`].
pub async fn joint_fusion_async(
    ctx: Arc<GpuContext>,
    input: &JlfInput,
    opts: &JlfOptions,
    cancel: Option<&AtomicBool>,
) -> RegResult<JlfOutput> {
    if opts.device == JlfDevice::Gpu {
        return gpu_fusion(ctx, input, opts, cancel).await;
    }
    let mut search = GpuJlfSearch::new_async(ctx).await.map_err(gerr)?;
    joint_fusion_with(input, opts, &mut search, cancel).await
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherU {
    dims: [u32; 4],
    a: [u32; 4],
    b: [u32; 4],
    c: [u32; 4],
    e: [u32; 4],
    f: [u32; 4],
}

const GATHER_SRC: &str = include_str!("shaders/jlf_gather.wgsl");
const SLOTS: usize = 16;
const COND_LIMIT: f32 = 1.0e3;
const GATHER_OUT_BYTES: u64 = 64 << 20;

struct PlaneCentres {
    voxels: Vec<usize>,
    weights: Vec<f32>,
    offsets: Vec<u32>,
}

fn pack(o: [isize; 3]) -> u32 {
    ((o[0] + 128) as u32) | (((o[1] + 128) as u32) << 8) | (((o[2] + 128) as u32) << 16)
}

fn unpack(o: u32) -> [isize; 3] {
    [
        (o & 0xff) as isize - 128,
        ((o >> 8) & 0xff) as isize - 128,
        ((o >> 16) & 0xff) as isize - 128,
    ]
}

struct Full<'a> {
    parts: FusionParts<'a>,
    input: &'a JlfInput,
    opts: &'a JlfOptions,
    gather: wgpu::ComputePipeline,
    store: BTreeMap<usize, PlaneCentres>,
    patch: Vec<[isize; 3]>,
}

impl Full<'_> {
    fn inside(&self, p: [isize; 3]) -> bool {
        (0..3).all(|d| p[d] >= 0 && p[d] < self.input.size[d] as isize)
    }

    fn index(&self, p: [isize; 3]) -> usize {
        let [nx, ny, _] = self.input.size;
        (p[0] as usize) + nx * (p[1] as usize + ny * p[2] as usize)
    }

    fn ijk(&self, v: usize) -> [isize; 3] {
        let [nx, ny, _] = self.input.size;
        [
            (v % nx) as isize,
            ((v / nx) % ny) as isize,
            (v / (nx * ny)) as isize,
        ]
    }

    fn centre(&self, c: [isize; 3]) -> Option<(&PlaneCentres, usize)> {
        let plane = self.store.get(&(c[2] as usize))?;
        let v = self.index(c);
        plane.voxels.binary_search(&v).ok().map(|k| (plane, k))
    }

    fn redo_votes(&self, v: usize, acc: &mut Accumulators) {
        let n = self.input.atlases.len();
        let at = self.ijk(v);
        for off in self.patch.iter().rev() {
            let c = [at[0] - off[0], at[1] - off[1], at[2] - off[2]];
            if !self.inside(c) {
                continue;
            }
            let Some((plane, k)) = self.centre(c) else {
                continue;
            };
            for i in 0..n {
                let s = unpack(plane.offsets[k * n + i]);
                let q = [at[0] + s[0], at[1] + s[1], at[2] + s[2]];
                if !self.inside(q) {
                    continue;
                }
                if let Ok(slot) = self
                    .parts
                    .label_set()
                    .binary_search(&self.input.labels[i][self.index(q)])
                {
                    acc.posteriors[slot][v] += plane.weights[k * n + i];
                }
            }
        }
    }
}

impl GpuJlfSearch {
    async fn gather_planes(
        &mut self,
        full: &Full<'_>,
        acc: &mut Accumulators,
        z0: usize,
        z1: usize,
    ) -> GpuResult<()> {
        let input = full.input;
        let [nx, ny, nz] = input.size;
        let plane = nx * ny;
        let n = input.atlases.len();
        let modalities = input.atlases[0].len();
        let p = full.patch.len();
        let rp = input_reach(&full.patch);
        let rs = full
            .parts
            .lists()
            .iter()
            .flatten()
            .map(|o| o[2].unsigned_abs())
            .max()
            .unwrap_or(0);
        let labelled = !input.labels.is_empty();
        let voting = full.opts.voting_weights;
        let rsize = 3 + 2 * SLOTS + modalities + if voting { n } else { 0 };
        let plane_bytes = (plane * 4) as u64;
        let fits = |planes: usize| -> bool {
            let atlas_planes = (planes + 2 * rs).min(nz) as u64;
            let centre_planes = (planes + 2 * rp).min(nz) as u64;
            let out = plane_bytes * (rsize * planes) as u64;
            let atlas = plane_bytes * (n * modalities) as u64 * atlas_planes;
            let slots = plane_bytes * n as u64 * atlas_planes;
            let cmap = plane_bytes * centre_planes;
            let centres = plane_bytes * n as u64 * centre_planes;
            out <= self.gather_budget
                && atlas <= self.atlas_budget
                && slots.max(cmap).max(centres) <= self.binding
        };
        if !fits(1) {
            return Err(GpuError::Invalid(format!(
                "one plane of the vote gather (with a halo of {rs} planes of atlases and {rp} of \
                 centres) needs more than the GPU's buffer limit ({} bytes) or the gather and \
                 atlas budgets",
                self.binding
            )));
        }
        let reach: [usize; 3] = std::array::from_fn(|d| {
            full.patch.iter().map(|o| o[d].unsigned_abs()).max().unwrap_or(0)
        });
        let mut a = z0;
        while a < z1 {
            let mut b = a + 1;
            while b < z1 && fits(b + 1 - a) {
                b += 1;
            }
            let ca = a.saturating_sub(rp);
            let cb = (b + rp).min(nz);
            let mut bmin = [usize::MAX; 3];
            let mut bmax = [0usize; 3];
            for (_, pc) in full.store.range(ca..cb) {
                for &v in &pc.voxels {
                    let q = [v % nx, (v / nx) % ny, v / plane];
                    for d in 0..3 {
                        bmin[d] = bmin[d].min(q[d]);
                        bmax[d] = bmax[d].max(q[d]);
                    }
                }
            }
            let (r0, r1) = if self.variant.legacy {
                ([0, 0, a], [nx, ny, b])
            } else {
                if bmin[0] == usize::MAX {
                    a = b;
                    continue;
                }
                let lo = [
                    bmin[0].saturating_sub(reach[0]),
                    bmin[1].saturating_sub(reach[1]),
                    bmin[2].saturating_sub(reach[2]).max(a),
                ];
                let hi = [
                    (bmax[0] + reach[0] + 1).min(nx),
                    (bmax[1] + reach[1] + 1).min(ny),
                    (bmax[2] + reach[2] + 1).min(b),
                ];
                (lo, hi)
            };
            if r0[2] >= r1[2] {
                a = b;
                continue;
            }
            let [bw, bh, bd] = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]];
            let region = |t: usize| r0[0] + t % bw + nx * (r0[1] + (t / bw) % bh + ny * (r0[2] + t / (bw * bh)));
            self.gathers += 1;
            let sp = prof::span("gp.host_prep");
            let (t_lo, t_hi) = (0usize, bw * bh * bd);
            let ga = r0[2].saturating_sub(rs);
            let gb = (r1[2] + rs).min(nz);
            let apl = (gb - ga) * plane;
            let mut atlas = Vec::with_capacity(n * modalities * apl);
            for at in &input.atlases {
                for image in at {
                    atlas.extend_from_slice(&image[ga * plane..gb * plane]);
                }
            }
            let slots = if labelled {
                full.parts.label_slots_range(ga * plane, gb * plane)
            } else {
                Vec::new()
            };
            let mut cmap = vec![UNSET; (cb - ca) * plane];
            let mut cw = Vec::new();
            let mut co = Vec::new();
            for z in ca..cb {
                let Some(pc) = full.store.get(&z) else {
                    continue;
                };
                for (k, v) in pc.voxels.iter().enumerate() {
                    cmap[v - ca * plane] = (cw.len() / n) as u32;
                    cw.extend_from_slice(&pc.weights[k * n..(k + 1) * n]);
                    co.extend_from_slice(&pc.offsets[k * n..(k + 1) * n]);
                }
            }
            let voxels = t_hi - t_lo;
            let mask: Vec<u32> = match &input.mask {
                Some(m) => (0..voxels).map(|t| u32::from(m[region(t)] != 0)).collect(),
                None => vec![1; voxels],
            };
            let poff: Vec<[i32; 4]> = full
                .patch
                .iter()
                .map(|o| [o[0] as i32, o[1] as i32, o[2] as i32, 0])
                .collect();
            drop(sp);
            let sp = prof::span("gp.upload");
            let atlas_buf = self.storage("jlf gather atlases", &atlas).await?;
            let slots_buf = self.storage("jlf gather label slots", &slots).await?;
            let cmap_buf = self.storage("jlf gather centre map", &cmap).await?;
            let cw_buf = self.storage("jlf gather weights", &cw).await?;
            let co_buf = self.storage("jlf gather offsets", &co).await?;
            let mask_buf = self.storage("jlf gather mask", &mask).await?;
            let poff_buf = self.storage("jlf gather patch offsets", &poff).await?;
            let out_len = voxels
                .checked_mul(rsize)
                .ok_or_else(|| too_large("the gather output"))?;
            let out_buf = self
                .scratch("jlf gather output", (out_len as u64) * 4)
                .await?;
            drop(sp);
            let per_voxel = (p * n * (modalities + 1)) as u64;
            let per_dispatch =
                (WORK_PER_DISPATCH / per_voxel.max(1)).clamp(u64::from(WG), 1 << 22) as usize;
            let gather_start = stamp();
            let mut first = t_lo;
            while first < t_hi {
                let want = if self.variant.dynamic {
                    scaled(per_dispatch as u64, self.grun, u64::from(WG)) as usize
                } else {
                    per_dispatch
                };
                let run = want.min(t_hi - first);
                let started = stamp();
                let (gx, gy) = self.groups(run as u32);
                let uniform = GatherU {
                    dims: [nx as u32, ny as u32, nz as u32, (first + run) as u32],
                    a: [r0[2] as u32, ga as u32, ca as u32, n as u32],
                    b: [modalities as u32, p as u32, gx * WG, rsize as u32],
                    c: [
                        u32::from(labelled),
                        (cb - ca) as u32,
                        apl as u32,
                        u32::from(voting),
                    ],
                    e: [first as u32, t_lo as u32, 0, 0],
                    f: [r0[0] as u32, r0[1] as u32, bw as u32, bh as u32],
                };
                self.dispatch(
                    &full.gather,
                    &[
                        &atlas_buf, &slots_buf, &cmap_buf, &cw_buf, &co_buf, &mask_buf, &poff_buf,
                        &out_buf,
                    ],
                    bytemuck::bytes_of(&uniform),
                    [gx, gy],
                )
                .await?;
                if let (true, Some(s)) = (self.variant.dynamic, started) {
                    self.grun = adapt(self.grun, run == want, s.elapsed().as_secs_f64());
                }
                first += run;
            }
            self.times[2] += seconds(gather_start);
            prof::add("gp.kernel", seconds(gather_start));
            let sp = prof::span("gp.readback");
            let rec = self.ctx.read_u32_async(&out_buf, out_len).await?;
            drop(sp);
            let sp = prof::span("gp.unpack");
            for t in 0..voxels {
                let v = region(t);
                let r = &rec[t * rsize..(t + 1) * rsize];
                let used = (r[0] & 0x7fff_ffff) as usize;
                let overflow = r[0] >> 31 != 0;
                acc.count[v] = r[1];
                acc.weight_sum[v] = f32::from_bits(r[2]);
                for m in 0..modalities {
                    acc.intensity[m][v] = f32::from_bits(r[3 + 2 * SLOTS + m]);
                }
                if voting {
                    for i in 0..n {
                        acc.voting[i][v] = f32::from_bits(r[3 + 2 * SLOTS + modalities + i]);
                    }
                }
                if overflow {
                    self.vote_redos += 1;
                    full.redo_votes(v, acc);
                } else {
                    for k in 0..used.min(SLOTS) {
                        let slot = r[3 + 2 * k] as usize;
                        acc.posteriors[slot][v] = f32::from_bits(r[4 + 2 * k]);
                    }
                }
            }
            drop(sp);
            a = b;
        }
        Ok(())
    }
}

fn input_reach(patch: &[[isize; 3]]) -> usize {
    patch.iter().map(|o| o[2].unsigned_abs()).max().unwrap_or(0)
}

fn beta_mode(beta: f64) -> u32 {
    if float_almost_equal(beta, 1.0) {
        0
    } else if float_almost_equal(beta, 2.0) {
        1
    } else {
        2
    }
}

fn cancelled(cancel: Option<&AtomicBool>) -> RegResult<()> {
    if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
        Err(RegError::Cancelled)
    } else {
        Ok(())
    }
}

async fn fuse_on_gpu(
    search: &mut GpuJlfSearch,
    input: &JlfInput,
    opts: &JlfOptions,
    cancel: Option<&AtomicBool>,
) -> RegResult<JlfOutput> {
    let sp = prof::span("g.parts");
    let parts = FusionParts::new(input, opts)?;
    drop(sp);
    let sp = prof::span("g.compile_gather");
    let gather = search
        .ctx
        .compile_async("k_jlf_gather", GATHER_SRC, "k_jlf_gather")
        .await
        .map_err(gerr)?;
    drop(sp);
    let sp = prof::span("g.slots_acc");
    let [nx, ny, nz] = input.size;
    let plane = nx * ny;
    let n = input.atlases.len();
    let width = parts.patch_len() * parts.used();
    let cap = (search.patch_budget / (width.max(1) as u64 * 4)).max(1) as usize;
    let lists: Vec<Vec<[isize; 3]>> = parts.lists().to_vec();
    if lists
        .iter()
        .flatten()
        .flatten()
        .any(|c| !(-127..=127).contains(c))
    {
        return Err(RegError::Invalid {
            what: "joint fusion",
            message: "a search offset is beyond 127 voxels".into(),
        });
    }
    let mut full = Full {
        patch: box_patch(opts.patch_radius),
        parts,
        input,
        opts,
        gather,
        store: BTreeMap::new(),
    };
    let rp = input_reach(&full.patch);
    let mut acc = full.parts.accumulators();
    let per_plane = full.parts.active_per_plane();
    drop(sp);
    let pc = opts.metric == PatchMetric::PearsonCorrelation;
    let mut next = 0usize;
    let mut z0 = 0usize;
    while z0 < nz {
        cancelled(cancel)?;
        let mut z1 = z0 + 1;
        let mut total = per_plane[z0];
        while z1 < nz && total + per_plane[z1] <= cap {
            total += per_plane[z1];
            z1 += 1;
        }
        if search.variant.legacy {
            z1 = (z0 + (search.block_voxels(input.size, width) / plane).max(1)).min(nz);
        }
        let sp = prof::span("g.active");
        let (centres, list_of) = full.parts.active(z0 * plane, z1 * plane);
        drop(sp);
        let sp = prof::span("g.search_patches");
        let patches = full.parts.search_patches(&centres);
        drop(sp);
        let sp = prof::span("g.search_weigh");
        let block = full.parts.block(&centres, &list_of, &patches);
        let (best, gpu) = if opts.nonnegative {
            (search.search_async(&block).await.map_err(gerr)?, Vec::new())
        } else {
            let normalized = if pc {
                None
            } else {
                Some(full.parts.target_patches(&centres))
            };
            let args = WeighArgs::new(
                normalized.as_deref().unwrap_or(&patches),
                opts.alpha,
                opts.beta,
            );
            search
                .search_and_weigh_async(&block, &args)
                .await
                .map_err(gerr)?
        };
        drop(sp);
        prof::add("n.centres", centres.len() as f64);
        let sp = prof::span("g.redo_store");
        let mut weights = vec![0f32; centres.len() * n];
        let mut redo = Vec::new();
        for c in 0..centres.len() {
            let flagged = gpu.get(c * (n + 1) + n).is_none_or(|f| f.to_bits() != 0);
            if flagged {
                redo.push(c);
            } else {
                weights[c * n..(c + 1) * n].copy_from_slice(&gpu[c * (n + 1)..c * (n + 1) + n]);
            }
        }
        if !redo.is_empty() {
            let rc: Vec<usize> = redo.iter().map(|&c| centres[c]).collect();
            let rl: Vec<u32> = redo.iter().map(|&c| list_of[c]).collect();
            let rb: Vec<u32> = redo
                .iter()
                .flat_map(|&c| best[c * n..(c + 1) * n].iter().copied())
                .collect();
            let cpu = full.parts.weights_many(&rc, &rl, &rb);
            for (k, &c) in redo.iter().enumerate() {
                for i in 0..n {
                    weights[c * n + i] = cpu[k * n + i] as f32;
                }
            }
        }
        drop(patches);
        for (c, &v) in centres.iter().enumerate() {
            let list = &lists[list_of[c] as usize];
            let entry = full.store.entry(v / plane).or_insert_with(|| PlaneCentres {
                voxels: Vec::new(),
                weights: Vec::new(),
                offsets: Vec::new(),
            });
            entry.voxels.push(v);
            entry
                .weights
                .extend_from_slice(&weights[c * n..(c + 1) * n]);
            for j in &best[c * n..(c + 1) * n] {
                let Some(off) = list.get(*j as usize) else {
                    return Err(RegError::Invalid {
                        what: "joint fusion",
                        message: "the patch search returned offsets that do not fit its block"
                            .into(),
                    });
                };
                entry.offsets.push(pack(*off));
            }
        }
        drop(sp);
        let _sp = prof::span("g.gather");
        let ready = if z1 == nz { nz } else { z1.saturating_sub(rp) };
        if ready > next {
            search
                .gather_planes(&full, &mut acc, next, ready)
                .await
                .map_err(gerr)?;
            next = ready;
        }
        full.store.retain(|z, _| z + rp >= next);
        z0 = z1;
    }
    cancelled(cancel)?;
    let _sp = prof::span("g.finish");
    Ok(full.parts.finish(acc))
}

fn box_patch(radius: [usize; 3]) -> Vec<[isize; 3]> {
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

async fn gpu_fusion(
    ctx: Arc<GpuContext>,
    input: &JlfInput,
    opts: &JlfOptions,
    cancel: Option<&AtomicBool>,
) -> RegResult<JlfOutput> {
    let mut search = GpuJlfSearch::new_async(ctx).await.map_err(gerr)?;
    joint_fusion_gpu_with(&mut search, input, opts, cancel).await
}

/// The [`JlfDevice::Gpu`] fusion of [`joint_fusion_async`] on `search`, with its budgets
/// ([`GpuJlfSearch::with_patch_budget`], [`GpuJlfSearch::with_atlas_budget`],
/// [`GpuJlfSearch::with_gather_budget`]), whatever [`JlfOptions::device`] says.
pub async fn joint_fusion_gpu_with(
    search: &mut GpuJlfSearch,
    input: &JlfInput,
    opts: &JlfOptions,
    cancel: Option<&AtomicBool>,
) -> RegResult<JlfOutput> {
    if input.atlases.len() > MAX_GPU_ATLASES {
        return joint_fusion_with(input, opts, search, cancel).await;
    }
    search.begin();
    let guard = Finishing(search);
    fuse_on_gpu(&mut *guard.0, input, opts, cancel).await
}

struct Finishing<'a>(&'a mut GpuJlfSearch);

impl Drop for Finishing<'_> {
    fn drop(&mut self) {
        self.0.finish();
    }
}

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
use fastwarp_reg::jlf::{
    joint_fusion_with, JlfInput, JlfOptions, JlfOutput, PatchMetric, PatchSearch, SearchBlock,
};
use fastwarp_reg::{RegError, Result as RegResult};

use crate::device::{block_gpu, GpuContext};
use crate::error::{GpuError, GpuResult};
use crate::wgpu_backend::block_reg;

const SRC: &str = include_str!("shaders/jlf_search.wgsl");
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

struct AtlasSlab {
    key: SlabKey,
    zlo: usize,
    planes: usize,
    buffer: wgpu::Buffer,
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
}

impl GpuJlfSearch {
    /// Compiles the search kernel on `ctx`; an error on wasm32 (use [`Self::new_async`]).
    pub fn new(ctx: Arc<GpuContext>) -> GpuResult<Self> {
        block_gpu("GpuJlfSearch::new", Self::new_async(ctx))
    }

    /// Async [`Self::new`].
    pub async fn new_async(ctx: Arc<GpuContext>) -> GpuResult<Self> {
        let pipeline = ctx
            .compile_async("k_jlf_search", SRC, "k_jlf_search")
            .await?;
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

    /// How many atlas slabs this search has uploaded.
    pub fn slab_uploads(&self) -> u64 {
        self.uploads
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
        self.uploads += 1;
        self.slab = Some(AtlasSlab {
            key,
            zlo: lo,
            planes,
            buffer,
        });
        Ok(())
    }

    /// [`PatchSearch::search`] with the device's error.
    pub async fn search_async(&mut self, block: &SearchBlock<'_>) -> GpuResult<Vec<u32>> {
        if !self.reuse {
            self.slab = None;
        }
        let out = self.search_blocks(block).await;
        if !self.reuse || out.is_err() {
            self.slab = None;
        }
        out
    }

    async fn search_blocks(&mut self, block: &SearchBlock<'_>) -> GpuResult<Vec<u32>> {
        let atlases = block.atlases.len();
        let centres = block.centres.len();
        let mut best = Vec::with_capacity(centres * atlases);
        if centres == 0 || atlases == 0 {
            return Ok(best);
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
        let items_per_dispatch =
            (WORK_PER_DISPATCH / per_item).clamp(u64::from(WG), 1 << 24) as u32;
        let max_count = [
            self.binding / (atlases as u64 * 4),
            self.binding / 16,
            self.binding / (width.max(1) as u64 * 4),
            u64::from(u32::MAX) / atlases as u64,
        ]
        .into_iter()
        .min()
        .unwrap_or(1)
        .max(1) as usize;
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
            self.ensure_slab(block, lo, hi, budget_planes).await?;
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
            let centre_buf = self.storage("jlf centres", &recs).await?;
            let patch_values = &block.patches[first * width..last * width];
            let target_buf = self.storage("jlf target patches", patch_values).await?;
            let out_buf = self
                .storage("jlf best offsets", &vec![UNSET; items as usize])
                .await?;
            let mut base = 0u32;
            while base < items {
                let run = items_per_dispatch.min(items - base);
                let groups = run.div_ceil(WG);
                let (gx, gy) = if groups <= self.max_groups {
                    (groups, 1)
                } else {
                    (self.max_groups, groups.div_ceil(self.max_groups))
                };
                let uniform = SearchU {
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
                        gx * WG,
                        base,
                    ],
                    pad: [0; 4],
                };
                self.dispatch(
                    &[
                        &slab.buffer,
                        &target_buf,
                        &centre_buf,
                        &offsets_buf,
                        &patch_buf,
                        &out_buf,
                    ],
                    &uniform,
                    [gx, gy],
                )
                .await?;
                base += run;
            }
            let got = self.ctx.read_u32_async(&out_buf, items as usize).await?;
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
            best.extend(got);
            first = last;
        }
        Ok(best)
    }

    async fn dispatch(
        &self,
        buffers: &[&wgpu::Buffer],
        uniform: &SearchU,
        groups: [u32; 2],
    ) -> GpuResult<()> {
        let bytes = bytemuck::bytes_of(uniform);
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
        let layout = self.pipeline.get_bind_group_layout(0);
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
            pass.set_pipeline(&self.pipeline);
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

/// Joint label fusion with the patch search on the GPU of `ctx` and everything else on the CPU
/// (whatever [`JlfOptions::device`] says); an error on wasm32 (use [`joint_fusion_async`]).
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
    let mut search = GpuJlfSearch::new_async(ctx).await.map_err(gerr)?;
    joint_fusion_with(input, opts, &mut search, cancel).await
}

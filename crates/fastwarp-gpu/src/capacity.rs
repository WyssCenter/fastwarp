// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! Pre-flight capacity check for a registration grid.
//!
//! Pure and device-free: given the finest level's grid, the channel count and the device's
//! binding limit and memory budget, decide whether the level fits before any image is loaded. A
//! caller uses this to fall back to another backend (or fail with a remedy) instead of dying at
//! the finest SyN level mid-run.

use crate::caps::GpuCaps;
use crate::device::GpuContext;
use crate::pool::RETAIN_FLOOR_BYTES;

static FALLBACK_BUDGET_NOTE: std::sync::Once = std::sync::Once::new();

/// The largest single storage binding of the split layout is the three-plane displacement field,
/// `n · 3 · 4` bytes (`wgpu_backend.rs`, `field_buffer`). WP8 split the five-plane CC buffer into
/// five `n · 4` bindings, so the field is now the largest.
const LARGEST_BINDING_BYTES_PER_VOXEL: u64 = 3 * 4;
/// Peak device allocation per voxel of the finest level's grid, two channels: the buffer pool's
/// high-water mark of a two-level `run_syn` on a 128^3 grid measured 180.2 B/voxel (the pool keeps
/// each size class at its own peak: about nine fields at `n * 12` and eighteen images at `n * 4`),
/// rounded up. It includes the full-resolution smoothed images of a finest level that is not
/// shrunk (`tests/syn_parity.rs`, `the_capacity_plan_bounds_the_pool_peak_of_a_two_level_run`);
/// the idle buffers of coarser levels and of the device self-test are planned separately.
const LEVEL_BYTES_PER_VOXEL_TWO_CHANNELS: u64 = 188;
/// What each channel after the first adds (two smoothed and two warped images, with the derivative
/// fields accumulated one channel at a time); one channel measured 165.7 B/voxel.
const LEVEL_BYTES_PER_VOXEL_PER_CHANNEL: u64 = 20;
/// Each channel keeps one smoothed moving image on the moving image's own grid for the whole
/// level (`syn.rs`, `syn_fields`); the per-voxel figures above count it on the fixed grid.
const MOVING_IMAGE_BYTES_PER_VOXEL: u64 = 4;
/// What masks add per voxel of a level (`syn.rs`, `StageMasks::validity`, and
/// `wgpu_backend.rs`, `cc_evaluate_masked`): the image of valid voxels and the sixth window
/// plane with its sum (12 B/voxel), and the per-workgroup counts of valid voxels, rounded up.
/// Both masks on a 96^3 run added 21.5 B/voxel to the pool peak, the two masks' own 8 B/voxel
/// included, so 13.5 B/voxel for the level itself; the 0.5 B/voxel above this constant is
/// covered by the slack of the per-level figures (188 against 180.2 measured for two channels,
/// 168 against 165.7 for one), which the plan adds for every kept level
/// (`tests/syn_parity.rs`, `the_capacity_plan_bounds_the_pool_peak_of_a_masked_run`).
const MASK_BYTES_PER_VOXEL: u64 = 13;
/// A mask stays on its own grid for the whole stage, one `f32` per voxel.
const MASK_IMAGE_BYTES_PER_VOXEL: u64 = 4;
/// What a SyN stage after another one holds besides its own working set: the collapsed forward
/// field of the earlier SyN stages, for the whole stage (`syn.rs`, `syn_fields`). Its inverse
/// stays on the host until the stage ends; it is uploaded after the stage's four fields have
/// gone back to the buffer pool, so it reuses one of their buffers instead of adding to the peak
/// (`tests/syn_parity.rs`, `the_capacity_plan_bounds_the_pool_peak_of_two_syn_stages`). The
/// field lies on the grid of the last level; counting it on the fixed grid over-counts when the
/// stages end at a shrink factor above 1.
const PRIOR_FIELD_BYTES_PER_VOXEL: u64 = 12;
/// The share of the per-voxel figures held by the nine displacement fields of a level.
const LEVEL_FIELD_BYTES_PER_VOXEL: u64 = 9 * 12;
const SYN_WORKGROUP: [u64; 3] = [8, 8, 4];
const SYN_REDUCTION_GROUP: u64 = 256;
const SYN_REDUCTION_MAX_GROUPS_X: u64 = 32768;

/// Bytes the device self-test of `WgpuBackend::new` leaves idle in the buffer pool (631,952 on
/// the adapter measured; `tests/syn_parity.rs` checks the bound). Each buffer is below
/// [`RETAIN_FLOOR_BYTES`], so no level transition frees them.
pub const SELFTEST_POOL_BYTES: u64 = 640 << 10;

/// Environment override of the device memory budget, in mebibytes.
pub const MEMORY_BUDGET_ENV: &str = "FASTWARP_GPU_MEMORY_BUDGET_MB";
pub const VIEWER_RESERVE_BYTES: u64 = 1 << 30;
pub const FALLBACK_MEMORY_CAP_BYTES: u64 = 4 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum MemoryFigure {
    DxgiLocalBudget,
    VulkanDeviceLocalHeap,
    VulkanHeapBudget,
    MetalRecommendedWorkingSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct DeviceMemory {
    pub bytes: u64,
    pub figure: MemoryFigure,
}

/// One memory heap as `VK_EXT_memory_budget` reports it.
#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeapBudget {
    pub device_local: bool,
    pub size: u64,
    pub budget: u64,
    pub usage: u64,
}

/// The most a device-local heap has left within its budget (`budget` capped at the heap size,
/// minus `usage`), or `None` when no device-local heap reports a budget.
#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
pub(crate) fn heap_budget_bytes(heaps: &[HeapBudget]) -> Option<u64> {
    heaps
        .iter()
        .filter(|h| h.device_local && h.budget > 0 && h.size > 0)
        .map(|h| h.budget.min(h.size).saturating_sub(h.usage))
        .max()
}

/// The Vulkan memory figure: the heap budget when `heaps` has one, else `heap_size`.
#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
pub(crate) fn vulkan_figure(
    heaps: Option<&[HeapBudget]>,
    heap_size: Option<DeviceMemory>,
) -> Option<DeviceMemory> {
    match heaps.and_then(heap_budget_bytes) {
        Some(bytes) => Some(DeviceMemory {
            bytes,
            figure: MemoryFigure::VulkanHeapBudget,
        }),
        None => heap_size,
    }
}

/// The Metal memory figure: the device's `recommendedMaxWorkingSetSize` minus its
/// `currentAllocatedSize`, or `None` when the device recommends nothing.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn metal_figure(recommended: u64, allocated: u64) -> Option<DeviceMemory> {
    (recommended > 0).then_some(DeviceMemory {
        bytes: recommended.saturating_sub(allocated),
        figure: MemoryFigure::MetalRecommendedWorkingSet,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    Env,
    Device(MemoryFigure),
    Fallback,
}

pub fn memory_budget(
    env_mb: Option<&str>,
    device: Option<DeviceMemory>,
    shares_host_memory: bool,
    max_buffer_size: u64,
) -> (u64, BudgetSource) {
    if let Some(mb) = env_mb.and_then(|v| v.trim().parse::<u64>().ok()) {
        return (mb.saturating_mul(1024 * 1024), BudgetSource::Env);
    }
    let (usable, source) = match device {
        Some(m) if shares_host_memory => (m.bytes / 2, BudgetSource::Device(m.figure)),
        Some(m) => (m.bytes, BudgetSource::Device(m.figure)),
        None => (
            max_buffer_size.min(FALLBACK_MEMORY_CAP_BYTES),
            BudgetSource::Fallback,
        ),
    };
    (usable.saturating_sub(VIEWER_RESERVE_BYTES), source)
}

/// The device limits a capacity check needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityCaps {
    /// Largest single storage binding the device permits.
    pub max_storage_buffer_binding_size: u64,
    /// Total device allocation the finest level may use.
    pub memory_budget: u64,
}

impl CapacityCaps {
    pub fn from_context(ctx: &GpuContext) -> Self {
        let shares = matches!(ctx.info.device_type.as_str(), "IntegratedGpu" | "Cpu");
        Self::build(&ctx.caps, ctx.memory, shares)
    }

    /// Reads the binding limit from `caps` and the budget from [`MEMORY_BUDGET_ENV`] if set,
    /// else `caps.max_buffer_size` capped at [`FALLBACK_MEMORY_CAP_BYTES`] minus
    /// [`VIEWER_RESERVE_BYTES`] (no device-memory figure is available here; the fallback is noted
    /// once on stderr because it is not a heap figure).
    pub fn from_gpu_caps(caps: &GpuCaps) -> Self {
        Self::build(caps, None, false)
    }

    fn build(caps: &GpuCaps, device: Option<DeviceMemory>, shares_host_memory: bool) -> Self {
        let env = crate::device::env_var(MEMORY_BUDGET_ENV);
        let (budget, source) = memory_budget(
            env.as_deref(),
            device,
            shares_host_memory,
            caps.max_buffer_size,
        );
        if source == BudgetSource::Fallback {
            FALLBACK_BUDGET_NOTE.call_once(|| {
                eprintln!(
                    "fastwarp: note: the graphics API reports no device memory and \
                     {MEMORY_BUDGET_ENV} is unset; the GPU memory budget is min(max_buffer_size, \
                     {FALLBACK_MEMORY_CAP_BYTES}) minus {VIEWER_RESERVE_BYTES} bytes kept for the \
                     viewers = {budget} bytes (set {MEMORY_BUDGET_ENV} to the device heap)"
                );
            });
        }
        Self {
            max_storage_buffer_binding_size: caps.max_storage_buffer_binding_size,
            memory_budget: budget,
        }
    }
}

/// What a grid needs and what the device allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityPlan {
    /// Voxels of the finest registration level.
    pub voxels: u64,
    /// Bytes of the largest single storage binding (the three-plane displacement field, or a
    /// moving image with more than three times the voxels).
    pub largest_binding_bytes: u64,
    /// Bytes of the stage's peak device allocation: the finest level's working set and the idle
    /// buffers the pool keeps from coarser levels and from the device self-test.
    pub level_total_bytes: u64,
    /// Binding limit the plan was checked against.
    pub binding_limit: u64,
    /// Memory budget the plan was checked against.
    pub memory_budget: u64,
}

/// Why a grid cannot fit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CapacityError {
    /// The largest binding (the three-plane displacement field, or a larger moving image or
    /// mask) exceeds `max_storage_buffer_binding_size`.
    #[error(
        "the finest registration level {dims:?} needs a storage binding of {need} bytes \
         (the three-plane displacement field, or the moving image or a mask when that is larger), \
         but the device binds at most {limit} bytes per storage buffer"
    )]
    Binding {
        /// Grid dimensions of the finest level.
        dims: [usize; 3],
        /// Bytes the binding needs.
        need: u64,
        /// Limit the device permits.
        limit: u64,
    },
    /// The SyN stage's peak device allocation exceeds the memory budget.
    #[error(
        "the SyN stage on the registration grid {dims:?} needs {need} bytes of device memory, \
         but the budget is {budget} bytes"
    )]
    Memory {
        /// Grid dimensions of the finest level.
        dims: [usize; 3],
        /// Bytes the SyN stage needs.
        need: u64,
        /// Budget the device permits.
        budget: u64,
    },
    /// The largest binding of the linear stages (the Mattes MI sample records of a level, or the
    /// moving image) exceeds `max_storage_buffer_binding_size`.
    #[error(
        "the linear stages on the registration grid {dims:?} need a storage binding of {need} \
         bytes (the Mattes MI sample records of one level, or the moving image when that is \
         larger), but the device binds at most {limit} bytes per storage buffer"
    )]
    LinearBinding {
        /// Grid dimensions of the registration grid.
        dims: [usize; 3],
        /// Bytes the binding needs.
        need: u64,
        /// Limit the device permits.
        limit: u64,
    },
    /// The device allocation of the whole linear chain exceeds the memory budget.
    #[error(
        "the linear stages on the registration grid {dims:?} need {need} bytes of device memory \
         in total (the moving image and the Mattes MI sample and reduction buffers of every level \
         of every stage), but the budget is {budget} bytes"
    )]
    LinearMemory {
        /// Grid dimensions of the registration grid.
        dims: [usize; 3],
        /// Bytes the chain needs.
        need: u64,
        /// Budget the device permits.
        budget: u64,
    },
}

/// Checks whether `grid_dims` at `channels` channels fits `caps`, for a moving image on a grid of
/// the same size and a SyN stage with one full-resolution level. Pure; no device is touched.
pub fn plan_capacity(
    grid_dims: [usize; 3],
    channels: usize,
    caps: &CapacityCaps,
) -> Result<CapacityPlan, CapacityError> {
    plan_capacity_pair(grid_dims, grid_dims, channels, caps)
}

/// Like [`plan_capacity`] for a moving image whose registration grid is `moving_dims` (the moving
/// image resampled on its own, not onto the fixed grid). A moving grid with more voxels than the
/// fixed one adds its excess for every channel; a smaller one is planned as if it were the size
/// of the fixed grid.
pub fn plan_capacity_pair(
    grid_dims: [usize; 3],
    moving_dims: [usize; 3],
    channels: usize,
    caps: &CapacityCaps,
) -> Result<CapacityPlan, CapacityError> {
    plan_capacity_syn(grid_dims, moving_dims, channels, &[1], caps)
}

fn voxel_count(dims: [usize; 3]) -> u64 {
    dims.iter()
        .fold(1u64, |acc, d| acc.saturating_mul(*d as u64))
}

fn syn_partial_bytes(dims: [usize; 3], voxels: u64, masked: bool) -> u64 {
    let groups = dims.iter().zip(SYN_WORKGROUP).fold(1u64, |acc, (d, w)| {
        acc.saturating_mul((*d as u64).div_ceil(w))
    });
    let items = voxels.div_ceil(SYN_REDUCTION_GROUP).max(1);
    let x = items.min(SYN_REDUCTION_MAX_GROUPS_X);
    groups
        .saturating_mul(4 + 8 + 8 + if masked { 4 } else { 0 })
        .saturating_add(x.saturating_mul(items.div_ceil(x)).saturating_mul(4))
}

/// Like [`plan_capacity_pair`] for a SyN stage with the levels `shrink_factors` on an isotropic
/// grid: the finest level's working set, plus the buffers of every coarser level that the pool
/// keeps (those below [`RETAIN_FLOOR_BYTES`], or all of a level that follows a larger one), the
/// last field resampled from the previous level (idle until the next trim, which one channel
/// never reaches), plus [`SELFTEST_POOL_BYTES`].
pub fn plan_capacity_syn(
    grid_dims: [usize; 3],
    moving_dims: [usize; 3],
    channels: usize,
    shrink_factors: &[u32],
    caps: &CapacityCaps,
) -> Result<CapacityPlan, CapacityError> {
    plan_capacity_syn_masked(grid_dims, moving_dims, channels, shrink_factors, &[], caps)
}

/// Like [`plan_capacity_syn`] for a SyN stage with masks: `mask_dims` holds the grid size of
/// each mask given (the fixed mask, the moving mask, or both). Each mask stays on its own grid
/// for the whole stage, and every level adds the image of valid voxels and the extra window
/// plane of the masked metric.
pub fn plan_capacity_syn_masked(
    grid_dims: [usize; 3],
    moving_dims: [usize; 3],
    channels: usize,
    shrink_factors: &[u32],
    mask_dims: &[[usize; 3]],
    caps: &CapacityCaps,
) -> Result<CapacityPlan, CapacityError> {
    let voxels = voxel_count(grid_dims);
    let moving_voxels = voxel_count(moving_dims);
    let channels = channels.max(1) as u64;
    let masked = !mask_dims.is_empty();
    let mask_bytes: Vec<u64> = mask_dims
        .iter()
        .map(|d| voxel_count(*d).saturating_mul(MASK_IMAGE_BYTES_PER_VOXEL))
        .collect();
    let largest_binding_bytes = voxels
        .saturating_mul(LARGEST_BINDING_BYTES_PER_VOXEL)
        .max(moving_voxels.saturating_mul(MOVING_IMAGE_BYTES_PER_VOXEL))
        .max(mask_bytes.iter().copied().max().unwrap_or(0));
    let per_voxel = LEVEL_BYTES_PER_VOXEL_TWO_CHANNELS - LEVEL_BYTES_PER_VOXEL_PER_CHANNEL
        + LEVEL_BYTES_PER_VOXEL_PER_CHANNEL.saturating_mul(channels - 1)
        + if masked { MASK_BYTES_PER_VOXEL } else { 0 };
    let moving_excess = moving_voxels
        .saturating_sub(voxels)
        .saturating_mul(MOVING_IMAGE_BYTES_PER_VOXEL)
        .saturating_mul(channels);
    let mut coarser: Vec<([usize; 3], u64, bool)> = Vec::new();
    let mut largest_before = 0u64;
    for &factor in shrink_factors {
        let f = factor.max(1) as usize;
        let dims = grid_dims.map(|d| (d / f).max(1));
        let level = voxel_count(dims);
        let after_larger = level < largest_before;
        largest_before = largest_before.max(level);
        if level >= voxels {
            continue;
        }
        match coarser.iter_mut().find(|c| c.1 == level) {
            Some(c) => c.2 |= after_larger,
            None => coarser.push((dims, level, after_larger)),
        }
    }
    let coarser_bytes = coarser
        .iter()
        .map(|&(dims, level, whole)| {
            let kept_per_voxel = if whole || level.saturating_mul(12) < RETAIN_FLOOR_BYTES {
                per_voxel
            } else if level.saturating_mul(4) < RETAIN_FLOOR_BYTES {
                per_voxel - LEVEL_FIELD_BYTES_PER_VOXEL
            } else {
                0
            };
            level
                .saturating_mul(kept_per_voxel)
                .saturating_add(syn_partial_bytes(dims, level, masked))
        })
        .fold(0u64, u64::saturating_add);
    let stale_field_bytes = coarser
        .iter()
        .filter(|&&(_, level, whole)| !whole && level.saturating_mul(12) >= RETAIN_FLOOR_BYTES)
        .map(|&(_, level, _)| level.saturating_mul(12))
        .max()
        .unwrap_or(0);
    let level_total_bytes = voxels
        .saturating_mul(per_voxel)
        .saturating_add(moving_excess)
        .saturating_add(coarser_bytes)
        .saturating_add(stale_field_bytes)
        .saturating_add(mask_bytes.iter().fold(0u64, |a, b| a.saturating_add(*b)))
        .saturating_add(SELFTEST_POOL_BYTES);
    if largest_binding_bytes > caps.max_storage_buffer_binding_size {
        return Err(CapacityError::Binding {
            dims: grid_dims,
            need: largest_binding_bytes,
            limit: caps.max_storage_buffer_binding_size,
        });
    }
    if level_total_bytes > caps.memory_budget {
        return Err(CapacityError::Memory {
            dims: grid_dims,
            need: level_total_bytes,
            budget: caps.memory_budget,
        });
    }
    Ok(CapacityPlan {
        voxels,
        largest_binding_bytes,
        level_total_bytes,
        binding_limit: caps.max_storage_buffer_binding_size,
        memory_budget: caps.memory_budget,
    })
}

/// One SyN stage for [`plan_capacity_syn_stages`]: its levels and the grid sizes of the masks it
/// uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynStageCapacity<'a> {
    /// The shrink factor of each level.
    pub shrink_factors: &'a [u32],
    /// The grid size of each mask the stage uses (none, the fixed mask, the moving mask or both).
    pub mask_dims: &'a [[usize; 3]],
}

/// Like [`plan_capacity_syn_masked`] for SyN stages run one after the other, as
/// `register_stages` runs them: the backend frees its idle buffers before each stage, so each
/// stage is planned on its own, and every stage after the first also holds the collapsed field of
/// the earlier ones on the fixed grid (12 bytes per voxel). Returns the plan of the stage that
/// needs the most device memory (an empty list as one stage of one full-resolution level). Pure;
/// no device is touched.
pub fn plan_capacity_syn_stages(
    grid_dims: [usize; 3],
    moving_dims: [usize; 3],
    channels: usize,
    stages: &[SynStageCapacity<'_>],
    caps: &CapacityCaps,
) -> Result<CapacityPlan, CapacityError> {
    let Some((first, rest)) = stages.split_first() else {
        return plan_capacity_syn(grid_dims, moving_dims, channels, &[1], caps);
    };
    let unbudgeted = CapacityCaps {
        memory_budget: u64::MAX,
        ..*caps
    };
    let prior = voxel_count(grid_dims).saturating_mul(PRIOR_FIELD_BYTES_PER_VOXEL);
    let stage_plan = |stage: &SynStageCapacity<'_>| {
        plan_capacity_syn_masked(
            grid_dims,
            moving_dims,
            channels,
            stage.shrink_factors,
            stage.mask_dims,
            &unbudgeted,
        )
    };
    let mut plan = stage_plan(first)?;
    for stage in rest {
        let mut later = stage_plan(stage)?;
        later.level_total_bytes = later.level_total_bytes.saturating_add(prior);
        if later.level_total_bytes > plan.level_total_bytes {
            plan = later;
        }
    }
    plan.memory_budget = caps.memory_budget;
    if plan.level_total_bytes > caps.memory_budget {
        return Err(CapacityError::Memory {
            dims: grid_dims,
            need: plan.level_total_bytes,
            budget: caps.memory_budget,
        });
    }
    Ok(plan)
}

/// Bytes of the record of one Mattes MI sample (`wgpu_backend/mi.rs`).
const LINEAR_SAMPLE_RECORD_BYTES: u64 = 16;
/// Samples per workgroup chunk of the MI kernels.
const LINEAR_CHUNK: u64 = 2048;
/// Histogram entries of one partial (`STRIDE` of `wgpu_backend/mi.rs`).
const LINEAR_STRIDE: u64 = 1056;
/// Most stripes of the two-pass histogram merge.
const LINEAR_MAX_STRIPES: u64 = 64;
/// The buffers of one level that do not depend on its samples: the merged histogram (`f64`),
/// the 32 x 32 ratio table and the jacobian rows of twelve parameters.
const LINEAR_LEVEL_FIXED_BYTES: u64 = LINEAR_STRIDE * 8 + 32 * 32 * 4 + 12 * 3 * 16;

/// Device bytes of one linear level with `samples` samples, as `GpuMiLevel::new` acquires them:
/// the sample records, one histogram partial and twelve derivative partials per chunk, the stripe
/// sums (`f64`) and the fixed buffers.
fn linear_level_bytes(samples: u64) -> u64 {
    let groups = samples.div_ceil(LINEAR_CHUNK).max(1);
    let stripes = groups.min(LINEAR_MAX_STRIPES);
    samples
        .saturating_mul(LINEAR_SAMPLE_RECORD_BYTES)
        .saturating_add(groups.saturating_mul(LINEAR_STRIDE * 4 + 12 * 4))
        .saturating_add(stripes * LINEAR_STRIDE * 8)
        .saturating_add(LINEAR_LEVEL_FIXED_BYTES)
}

/// Checks whether the linear stages `stages` (Mattes MI on the GPU) fit `caps` for a fixed grid of
/// `grid_dims` and a moving grid of `moving_dims` voxels, both isotropic (every axis of a level is
/// shrunk by the level's factor). Pure; no device is touched.
///
/// A level holds the full-resolution moving image (one buffer, reused by every level) and its
/// sample records and reduction buffers; the buffer pool keeps the idle buffers of every earlier
/// level of every stage, so the plan sums them over the whole chain (an upper bound: buffers of
/// equal size, as in stages with one schedule, are in fact reused), with the level's voxel count
/// rounded up per axis and the sampling percentage applied to it. Two small read-back buffers
/// per level (the merged histogram and the derivative partials) are outside the pool and the
/// plan. The figure is the chain's total, not one level's, plus [`SELFTEST_POOL_BYTES`]
/// ([`CapacityError::LinearMemory`]). A SyN stage that follows is checked with
/// [`plan_capacity_syn`]: `register_stages` frees the idle linear buffers before it, so the
/// two checks are separate.
pub fn plan_capacity_linear(
    grid_dims: [usize; 3],
    moving_dims: [usize; 3],
    stages: &[fastwarp_reg::linear::LinearRecipe],
    caps: &CapacityCaps,
) -> Result<CapacityPlan, CapacityError> {
    let count = |dims: [usize; 3]| {
        dims.iter()
            .fold(1u64, |acc, d| acc.saturating_mul(*d as u64))
    };
    let voxels = count(grid_dims);
    let moving_bytes = count(moving_dims).saturating_mul(MOVING_IMAGE_BYTES_PER_VOXEL);
    let mut largest_binding_bytes = moving_bytes;
    let mut level_total_bytes = moving_bytes.saturating_add(SELFTEST_POOL_BYTES);
    for stage in stages {
        let fraction = f64::from(stage.sampling_percentage).clamp(0.0, 1.0);
        for &factor in &stage.shrink_factors {
            let f = u64::from(factor.max(1));
            let level = grid_dims
                .iter()
                .fold(1u64, |acc, d| acc.saturating_mul((*d as u64).div_ceil(f)));
            let samples = ((level as f64 * fraction).ceil() as u64).saturating_add(1);
            largest_binding_bytes =
                largest_binding_bytes.max(samples.saturating_mul(LINEAR_SAMPLE_RECORD_BYTES));
            level_total_bytes = level_total_bytes.saturating_add(linear_level_bytes(samples));
        }
    }
    if largest_binding_bytes > caps.max_storage_buffer_binding_size {
        return Err(CapacityError::LinearBinding {
            dims: grid_dims,
            need: largest_binding_bytes,
            limit: caps.max_storage_buffer_binding_size,
        });
    }
    if level_total_bytes > caps.memory_budget {
        return Err(CapacityError::LinearMemory {
            dims: grid_dims,
            need: level_total_bytes,
            budget: caps.memory_budget,
        });
    }
    Ok(CapacityPlan {
        voxels,
        largest_binding_bytes,
        level_total_bytes,
        binding_limit: caps.max_storage_buffer_binding_size,
        memory_budget: caps.memory_budget,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use fastwarp_reg::linear::LinearRecipe;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn arc() -> CapacityCaps {
        CapacityCaps {
            max_storage_buffer_binding_size: 1_073_741_820,
            memory_budget: 17 * GIB,
        }
    }

    fn caps(max_buffer_size: u64) -> GpuCaps {
        GpuCaps {
            max_storage_buffer_binding_size: 1_073_741_820,
            max_buffer_size,
            max_uniform_buffer_binding_size: 65_536,
            max_compute_workgroup_storage_size: 32_768,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size: [1024, 1024, 64],
            max_compute_workgroups_per_dimension: 65_535,
            max_storage_buffers_per_shader_stage: 16,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 1000,
            max_immediate_size: 0,
            timestamp_query: true,
            timestamp_query_inside_encoders: true,
            subgroup: true,
            immediates: false,
            shader_f16: false,
            shader_float32_atomic: false,
            float32_filterable: false,
            timestamp_period_ns: None,
            feature_names: Vec::new(),
        }
    }

    #[test]
    fn several_syn_stages_are_planned_one_by_one_with_the_earlier_field() {
        let dims = [96, 96, 96];
        let n = 96u64 * 96 * 96;
        let caps = arc();
        let full: &[u32] = &[4, 2, 1];
        let fine: &[u32] = &[1];
        let mask = [[64, 64, 64]];
        let stage = |shrink_factors, mask_dims| SynStageCapacity {
            shrink_factors,
            mask_dims,
        };
        let one = plan_capacity_syn_masked(dims, dims, 2, full, &[], &caps).unwrap();
        let alone =
            plan_capacity_syn_stages(dims, dims, 2, &[stage(full, &[][..])], &caps).unwrap();
        assert_eq!(alone, one);
        let twice = plan_capacity_syn_stages(
            dims,
            dims,
            2,
            &[stage(full, &[][..]), stage(full, &[][..])],
            &caps,
        )
        .unwrap();
        assert_eq!(twice.level_total_bytes, one.level_total_bytes + 12 * n);
        let masked_second = plan_capacity_syn_stages(
            dims,
            dims,
            2,
            &[stage(full, &[][..]), stage(fine, &mask[..])],
            &caps,
        )
        .unwrap();
        let fine_masked = plan_capacity_syn_masked(dims, dims, 2, fine, &mask, &caps).unwrap();
        assert_eq!(
            masked_second.level_total_bytes,
            (fine_masked.level_total_bytes + 12 * n).max(one.level_total_bytes)
        );
        let masked_first = plan_capacity_syn_stages(
            dims,
            dims,
            2,
            &[stage(full, &mask[..]), stage(fine, &[][..])],
            &caps,
        )
        .unwrap();
        let full_masked = plan_capacity_syn_masked(dims, dims, 2, full, &mask, &caps).unwrap();
        assert!(masked_first.level_total_bytes >= full_masked.level_total_bytes);
        let tight = CapacityCaps {
            memory_budget: twice.level_total_bytes - 1,
            ..caps
        };
        assert!(plan_capacity_syn_masked(dims, dims, 2, full, &[], &tight).is_ok());
        match plan_capacity_syn_stages(
            dims,
            dims,
            2,
            &[stage(full, &[][..]), stage(full, &[][..])],
            &tight,
        ) {
            Err(CapacityError::Memory { need, budget, .. }) => {
                assert_eq!(
                    (need, budget),
                    (twice.level_total_bytes, tight.memory_budget)
                );
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            plan_capacity_syn_stages(dims, dims, 2, &[], &caps).unwrap(),
            plan_capacity_pair(dims, dims, 2, &caps).unwrap()
        );
    }

    #[test]
    fn budget_defaults_to_the_capped_buffer_proxy_minus_the_reserve_and_the_env_overrides_it() {
        std::env::remove_var("DBSLOC_GPU_MEMORY_BUDGET_MB");
        std::env::remove_var(MEMORY_BUDGET_ENV);
        let small = caps(2 * GIB);
        assert_eq!(CapacityCaps::from_gpu_caps(&small).memory_budget, GIB);
        let caps = caps(64 * GIB);
        assert_eq!(CapacityCaps::from_gpu_caps(&caps).memory_budget, 3 * GIB);
        std::env::set_var(MEMORY_BUDGET_ENV, " 128 ");
        assert_eq!(
            CapacityCaps::from_gpu_caps(&caps).memory_budget,
            128 * 1024 * 1024
        );
        std::env::set_var(MEMORY_BUDGET_ENV, "not-a-number");
        assert_eq!(CapacityCaps::from_gpu_caps(&caps).memory_budget, 3 * GIB);
        std::env::set_var(MEMORY_BUDGET_ENV, "16");
        let from_env = CapacityCaps::from_gpu_caps(&caps);
        assert_eq!(from_env.memory_budget, 16 * 1024 * 1024);
        assert_eq!(
            from_env.max_storage_buffer_binding_size,
            caps.max_storage_buffer_binding_size
        );
        std::env::remove_var(MEMORY_BUDGET_ENV);
    }

    #[test]
    fn the_planner_follows_the_webgpu_default_limits() {
        let limits = wgpu::Limits::default();
        let granted = GpuCaps::from_limits(&limits);
        assert_eq!(
            granted.max_storage_buffer_binding_size,
            limits.max_storage_buffer_binding_size
        );
        assert_eq!(granted.max_buffer_size, limits.max_buffer_size);
        let cohort = [193, 229, 193];
        let derived = CapacityCaps {
            max_storage_buffer_binding_size: granted.max_storage_buffer_binding_size,
            memory_budget: 0,
        };
        match plan_capacity(cohort, 1, &derived) {
            Err(CapacityError::Memory {
                budget: 0, need, ..
            }) => assert!(need > 0),
            other => panic!("expected a memory refusal at budget 0, got {other:?}"),
        }
        let budgeted = CapacityCaps {
            memory_budget: 8 * GIB,
            ..derived
        };
        let plan = plan_capacity(cohort, 1, &budgeted).unwrap();
        assert!(plan.largest_binding_bytes <= limits.max_storage_buffer_binding_size);
        assert_eq!(plan.binding_limit, limits.max_storage_buffer_binding_size);
        match plan_capacity([256, 256, 256], 1, &budgeted) {
            Err(CapacityError::Binding { limit, need, .. }) => {
                assert_eq!(limit, limits.max_storage_buffer_binding_size);
                assert!(need > limit);
            }
            other => panic!("expected a binding refusal, got {other:?}"),
        }
    }

    fn vulkan(bytes: u64) -> Option<DeviceMemory> {
        Some(DeviceMemory {
            bytes,
            figure: MemoryFigure::VulkanDeviceLocalHeap,
        })
    }

    #[cfg(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    fn heap(device_local: bool, size: u64, budget: u64, usage: u64) -> HeapBudget {
        HeapBudget {
            device_local,
            size,
            budget,
            usage,
        }
    }

    #[cfg(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    #[test]
    fn the_heap_budget_is_the_largest_device_local_budget_left_after_the_usage() {
        let heaps = [
            heap(false, 32 * GIB, 30 * GIB, 0),
            heap(true, 8 * GIB, 7 * GIB, 2 * GIB),
            heap(true, 256 << 20, 200 << 20, 0),
        ];
        assert_eq!(heap_budget_bytes(&heaps), Some(5 * GIB));
        assert_eq!(
            heap_budget_bytes(&[heap(true, 8 * GIB, 9 * GIB, GIB)]),
            Some(7 * GIB)
        );
        assert_eq!(
            heap_budget_bytes(&[heap(true, 8 * GIB, 7 * GIB, 8 * GIB)]),
            Some(0)
        );
    }

    #[cfg(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    #[test]
    fn without_a_device_local_heap_reporting_a_budget_there_is_no_heap_budget() {
        assert_eq!(heap_budget_bytes(&[]), None);
        assert_eq!(
            heap_budget_bytes(&[heap(false, 32 * GIB, 30 * GIB, 0)]),
            None
        );
        assert_eq!(heap_budget_bytes(&[heap(true, 8 * GIB, 0, 0)]), None);
        assert_eq!(heap_budget_bytes(&[heap(true, 0, 7 * GIB, 0)]), None);
    }

    #[cfg(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    #[test]
    fn the_vulkan_figure_falls_back_to_the_heap_size_without_a_budget() {
        let size = vulkan(8 * GIB);
        let heaps = [heap(true, 8 * GIB, 7 * GIB, GIB)];
        assert_eq!(
            vulkan_figure(Some(&heaps), size),
            Some(DeviceMemory {
                bytes: 6 * GIB,
                figure: MemoryFigure::VulkanHeapBudget
            })
        );
        assert_eq!(vulkan_figure(None, size), size);
        assert_eq!(vulkan_figure(Some(&[]), size), size);
        assert_eq!(
            vulkan_figure(Some(&[heap(true, 8 * GIB, 0, 0)]), size),
            size
        );
        assert_eq!(vulkan_figure(None, None), None);
    }

    #[test]
    fn the_metal_figure_is_the_recommended_working_set_left_after_the_allocations() {
        let metal = |bytes| {
            Some(DeviceMemory {
                bytes,
                figure: MemoryFigure::MetalRecommendedWorkingSet,
            })
        };
        assert_eq!(metal_figure(12 * GIB, GIB), metal(11 * GIB));
        assert_eq!(metal_figure(12 * GIB, 0), metal(12 * GIB));
        assert_eq!(metal_figure(GIB, 2 * GIB), metal(0));
        assert_eq!(metal_figure(0, 0), None);
    }

    #[test]
    fn a_discrete_device_keeps_its_reported_memory_minus_the_viewer_reserve() {
        assert_eq!(
            memory_budget(None, vulkan(8 * GIB), false, 64 * GIB),
            (
                7 * GIB,
                BudgetSource::Device(MemoryFigure::VulkanDeviceLocalHeap)
            )
        );
        let dxgi = Some(DeviceMemory {
            bytes: 7 * GIB,
            figure: MemoryFigure::DxgiLocalBudget,
        });
        assert_eq!(
            memory_budget(None, dxgi, false, GIB),
            (6 * GIB, BudgetSource::Device(MemoryFigure::DxgiLocalBudget))
        );
    }

    #[test]
    fn a_device_sharing_host_memory_gets_half_its_reported_memory_minus_the_reserve() {
        assert_eq!(
            memory_budget(None, vulkan(18 * GIB), true, 64 * GIB).0,
            8 * GIB
        );
        assert_eq!(memory_budget(None, vulkan(GIB), true, 64 * GIB).0, 0);
    }

    #[test]
    fn the_env_override_wins_verbatim_over_the_device_figure() {
        assert_eq!(
            memory_budget(Some(" 2048 "), vulkan(8 * GIB), true, 64 * GIB),
            (2 * GIB, BudgetSource::Env)
        );
        assert_eq!(
            memory_budget(Some("lots"), vulkan(8 * GIB), false, 64 * GIB),
            (
                7 * GIB,
                BudgetSource::Device(MemoryFigure::VulkanDeviceLocalHeap)
            )
        );
    }

    #[test]
    fn without_a_figure_the_buffer_proxy_is_capped_and_the_reserve_saturates() {
        assert_eq!(
            memory_budget(None, None, false, 64 * GIB),
            (
                FALLBACK_MEMORY_CAP_BYTES - VIEWER_RESERVE_BYTES,
                BudgetSource::Fallback
            )
        );
        assert_eq!(memory_budget(None, None, true, 2 * GIB).0, GIB);
        assert_eq!(memory_budget(None, None, false, 256 << 20).0, 0);
    }

    #[test]
    fn the_shared_arc_heap_rejects_point_five_mm_and_fits_point_eight_mm() {
        let caps = CapacityCaps {
            memory_budget: memory_budget(None, vulkan(18 * GIB), true, 64 * GIB).0,
            ..arc()
        };
        assert!(matches!(
            plan_capacity([394, 466, 378], 2, &caps),
            Err(CapacityError::Memory { .. })
        ));
        assert!(plan_capacity([246, 291, 236], 2, &caps).is_ok());
        assert!(plan_capacity([197, 233, 189], 2, &caps).is_ok());
    }

    #[test]
    fn one_mm_fits_on_the_arc_binding() {
        let plan = plan_capacity([197, 233, 189], 2, &arc()).expect("1.0 mm must fit");
        assert!(plan.largest_binding_bytes < GIB, "{plan:?}");
        assert_eq!(
            plan.level_total_bytes,
            197 * 233 * 189 * 188 + SELFTEST_POOL_BYTES
        );
        assert!(plan.level_total_bytes < 17 * GIB, "{plan:?}");
        let one = plan_capacity([197, 233, 189], 1, &arc()).expect("one channel must fit");
        assert_eq!(
            one.level_total_bytes,
            197 * 233 * 189 * 168 + SELFTEST_POOL_BYTES
        );
    }

    #[test]
    fn every_channel_past_the_first_adds_the_same_bytes() {
        let caps = CapacityCaps {
            max_storage_buffer_binding_size: u64::MAX,
            memory_budget: u64::MAX,
        };
        let per_voxel = |channels| {
            (plan_capacity([10, 10, 10], channels, &caps)
                .unwrap()
                .level_total_bytes
                - SELFTEST_POOL_BYTES)
                / 1000
        };
        assert_eq!(per_voxel(1), 168);
        assert_eq!(per_voxel(2), 188);
        assert_eq!(per_voxel(3), 208);
        assert_eq!(per_voxel(4), 228);
        assert_eq!(per_voxel(0), 168);
    }

    #[test]
    fn point_five_mm_fits_the_split_layout() {
        let plan = plan_capacity([394, 466, 378], 2, &arc()).expect("0.5 mm must fit");
        assert_eq!(plan.largest_binding_bytes, 394 * 466 * 378 * 12);
        assert!(plan.largest_binding_bytes < GIB, "{plan:?}");
        assert_eq!(
            plan.level_total_bytes,
            394 * 466 * 378 * 188 + SELFTEST_POOL_BYTES
        );
        assert!(plan.level_total_bytes < 17 * GIB, "{plan:?}");
    }

    #[test]
    fn the_split_layout_fits_a_binding_the_packed_planes_would_not() {
        let caps = CapacityCaps {
            max_storage_buffer_binding_size: 15_000_000,
            memory_budget: u64::MAX,
        };
        let plan = plan_capacity([100, 100, 100], 2, &caps).expect("split layout must fit");
        assert_eq!(plan.largest_binding_bytes, 12_000_000);
        assert!(plan.largest_binding_bytes <= caps.max_storage_buffer_binding_size);
        assert!(1_000_000u64 * 20 > caps.max_storage_buffer_binding_size);
    }

    #[test]
    fn the_boundary_is_near_the_field_binding_limit() {
        let caps = CapacityCaps {
            memory_budget: u64::MAX,
            ..arc()
        };
        assert!(plan_capacity([447, 447, 447], 2, &caps).is_ok());
        assert!(matches!(
            plan_capacity([448, 447, 447], 2, &caps),
            Err(CapacityError::Binding { .. })
        ));
    }

    #[test]
    fn a_tiny_grid_fits_a_tiny_budget_and_a_big_one_fails_it() {
        let caps = CapacityCaps {
            max_storage_buffer_binding_size: 1 << 30,
            memory_budget: 1 << 20,
        };
        assert!(plan_capacity([4, 4, 4], 1, &caps).is_ok());
        assert!(matches!(
            plan_capacity([256, 256, 256], 1, &caps),
            Err(CapacityError::Memory { .. })
        ));
    }

    fn unlimited() -> CapacityCaps {
        CapacityCaps {
            max_storage_buffer_binding_size: u64::MAX,
            memory_budget: u64::MAX,
        }
    }

    #[test]
    fn a_moving_grid_of_the_fixed_size_or_smaller_plans_like_the_fixed_grid_alone() {
        for channels in [1, 2, 3, 4] {
            let alone = plan_capacity([197, 233, 189], channels, &arc()).unwrap();
            let same =
                plan_capacity_pair([197, 233, 189], [197, 233, 189], channels, &arc()).unwrap();
            let smaller =
                plan_capacity_pair([197, 233, 189], [100, 120, 90], channels, &arc()).unwrap();
            assert_eq!(same, alone);
            assert_eq!(smaller, alone);
        }
    }

    #[test]
    fn a_larger_moving_grid_adds_its_excess_image_bytes_for_every_channel() {
        let fixed = 10 * 10 * 10;
        let moving = 20 * 20 * 20;
        for channels in [1u64, 2, 3, 4] {
            let plan =
                plan_capacity_pair([10, 10, 10], [20, 20, 20], channels as usize, &unlimited())
                    .unwrap();
            let alone = plan_capacity([10, 10, 10], channels as usize, &unlimited()).unwrap();
            assert_eq!(plan.voxels, fixed);
            assert_eq!(
                plan.level_total_bytes,
                alone.level_total_bytes + (moving - fixed) * 4 * channels
            );
        }
    }

    #[test]
    fn a_larger_moving_grid_can_exceed_a_budget_the_fixed_grid_alone_fits() {
        let fixed = [128, 128, 128];
        let moving = [256, 256, 256];
        let need = plan_capacity(fixed, 2, &unlimited())
            .unwrap()
            .level_total_bytes;
        let caps = CapacityCaps {
            max_storage_buffer_binding_size: u64::MAX,
            memory_budget: need,
        };
        assert!(plan_capacity(fixed, 2, &caps).is_ok());
        assert!(matches!(
            plan_capacity_pair(fixed, moving, 2, &caps),
            Err(CapacityError::Memory { .. })
        ));
    }

    #[test]
    fn a_moving_image_with_over_three_times_the_voxels_is_the_largest_binding() {
        let plan = plan_capacity_pair([100, 100, 100], [200, 200, 100], 1, &unlimited()).unwrap();
        assert_eq!(plan.largest_binding_bytes, 200 * 200 * 100 * 4);
        let caps = CapacityCaps {
            max_storage_buffer_binding_size: 100 * 100 * 100 * 12,
            memory_budget: u64::MAX,
        };
        assert!(plan_capacity([100, 100, 100], 1, &caps).is_ok());
        assert!(matches!(
            plan_capacity_pair([100, 100, 100], [200, 200, 100], 1, &caps),
            Err(CapacityError::Binding { .. })
        ));
    }

    #[test]
    fn the_linear_plan_sums_the_samples_of_every_level_of_every_stage() {
        let rigid = LinearRecipe::cohort_rigid();
        let one =
            plan_capacity_linear([64; 3], [64; 3], std::slice::from_ref(&rigid), &unlimited())
                .unwrap();
        let level = |f: u64| (64u64.div_ceil(f)).pow(3);
        let levels_bytes: u64 = [8u64, 4, 2, 1]
            .iter()
            .map(|f| linear_level_bytes((level(*f) as f64 * 0.25).ceil() as u64 + 1))
            .sum();
        assert_eq!(
            linear_level_bytes(2049),
            2049 * 16 + 2 * (1056 * 4 + 48) + 2 * 1056 * 8 + LINEAR_LEVEL_FIXED_BYTES
        );
        assert_eq!(
            linear_level_bytes(1_000_000),
            1_000_000 * 16 + 489 * (1056 * 4 + 48) + 64 * 1056 * 8 + LINEAR_LEVEL_FIXED_BYTES
        );
        assert_eq!(
            one.level_total_bytes,
            64u64.pow(3) * 4 + levels_bytes + SELFTEST_POOL_BYTES
        );
        let two = plan_capacity_linear(
            [64; 3],
            [64; 3],
            &[rigid.clone(), LinearRecipe::cohort_affine()],
            &unlimited(),
        )
        .unwrap();
        let fixed = 64u64.pow(3) * 4 + SELFTEST_POOL_BYTES;
        assert_eq!(
            two.level_total_bytes - fixed,
            2 * (one.level_total_bytes - fixed)
        );
        let dense = LinearRecipe {
            sampling_percentage: 1.0,
            ..rigid.clone()
        };
        let d = plan_capacity_linear([64; 3], [64; 3], &[dense], &unlimited()).unwrap();
        assert!(d.level_total_bytes > one.level_total_bytes);
        assert_eq!(d.largest_binding_bytes, (64u64.pow(3) + 1) * 16);
        let none = plan_capacity_linear([64; 3], [80; 3], &[], &unlimited()).unwrap();
        assert_eq!(
            none.level_total_bytes,
            80u64.pow(3) * 4 + SELFTEST_POOL_BYTES
        );
    }

    #[test]
    fn the_linear_plan_is_far_below_the_syn_plan_and_refuses_a_tiny_budget() {
        let stages = [LinearRecipe::cohort_rigid(), LinearRecipe::cohort_affine()];
        let lin = plan_capacity_linear([197, 233, 189], [197, 233, 189], &stages, &arc()).unwrap();
        let syn = plan_capacity([197, 233, 189], 1, &arc()).unwrap();
        assert!(lin.level_total_bytes * 4 < syn.level_total_bytes);
        let caps = CapacityCaps {
            max_storage_buffer_binding_size: 1 << 30,
            memory_budget: 1 << 20,
        };
        assert!(matches!(
            plan_capacity_linear([128; 3], [128; 3], &stages, &caps),
            Err(CapacityError::LinearMemory { .. })
        ));
        let text = plan_capacity_linear([128; 3], [128; 3], &stages, &caps)
            .unwrap_err()
            .to_string();
        assert!(
            text.contains("linear stages") && text.contains("in total"),
            "{text}"
        );
        let binding = CapacityCaps {
            max_storage_buffer_binding_size: 1 << 20,
            memory_budget: u64::MAX,
        };
        assert!(matches!(
            plan_capacity_linear([128; 3], [128; 3], &stages, &binding),
            Err(CapacityError::LinearBinding { .. })
        ));
        let text = plan_capacity_linear([128; 3], [128; 3], &stages, &binding)
            .unwrap_err()
            .to_string();
        assert!(text.contains("Mattes MI sample records"), "{text}");
    }

    fn syn_total(dims: [usize; 3], channels: usize, factors: &[u32]) -> u64 {
        plan_capacity_syn(dims, dims, channels, factors, &unlimited())
            .unwrap()
            .level_total_bytes
    }

    #[test]
    fn a_single_full_resolution_level_plans_like_the_pair() {
        for channels in [1, 2, 3, 4] {
            let pair = plan_capacity_pair([48; 3], [60; 3], channels, &unlimited()).unwrap();
            let syn = plan_capacity_syn([48; 3], [60; 3], channels, &[1], &unlimited()).unwrap();
            assert_eq!(syn, pair);
            assert_eq!(
                syn_total([48; 3], channels, &[1, 1]),
                plan_capacity([48; 3], channels, &unlimited())
                    .unwrap()
                    .level_total_bytes
            );
        }
    }

    #[test]
    fn a_coarse_level_below_the_retain_floor_is_kept_whole() {
        let partials = syn_partial_bytes([24; 3], 24u64.pow(3), false);
        assert_eq!(partials, 3 * 3 * 6 * 20 + 54 * 4);
        assert_eq!(
            syn_total([48; 3], 1, &[2, 1]),
            (48u64.pow(3) + 24u64.pow(3)) * 168 + partials + SELFTEST_POOL_BYTES
        );
        assert_eq!(
            syn_total([48; 3], 2, &[2, 1]) - syn_total([48; 3], 2, &[1]),
            24u64.pow(3) * 188 + partials
        );
        assert_eq!(
            syn_total([48; 3], 1, &[2, 2, 1]),
            syn_total([48; 3], 1, &[2, 1])
        );
    }

    #[test]
    fn a_coarse_level_keeps_its_images_below_the_floor_and_only_partials_above_it() {
        let images_only = 50u64.pow(3);
        assert!(images_only * 4 < RETAIN_FLOOR_BYTES && images_only * 12 >= RETAIN_FLOOR_BYTES);
        assert_eq!(
            syn_total([100; 3], 1, &[2, 1]) - syn_total([100; 3], 1, &[1]),
            images_only * 60 + syn_partial_bytes([50; 3], images_only, false) + images_only * 12
        );
        let above = 128u64.pow(3);
        assert_eq!(
            syn_total([256; 3], 2, &[2, 1]) - syn_total([256; 3], 2, &[1]),
            syn_partial_bytes([128; 3], above, false) + above * 12
        );
        let small = 24u64.pow(3);
        assert_eq!(
            syn_total([48; 3], 1, &[2, 1]) - syn_total([48; 3], 1, &[1]),
            small * 168 + syn_partial_bytes([24; 3], small, false)
        );
    }

    #[test]
    fn the_memory_error_names_the_syn_stage() {
        let caps = CapacityCaps {
            max_storage_buffer_binding_size: u64::MAX,
            memory_budget: GIB,
        };
        let text = plan_capacity_syn([197, 233, 189], [197, 233, 189], 2, &[8, 4, 2, 1], &caps)
            .unwrap_err()
            .to_string();
        assert_eq!(
            text,
            "the SyN stage on the registration grid [197, 233, 189] needs 1658255740 bytes of device \
             memory, but the budget is 1073741824 bytes"
        );
    }

    #[test]
    fn only_the_largest_stale_coarse_field_is_added() {
        let f4 = 64u64.pow(3);
        assert_eq!(
            syn_total([256; 3], 1, &[4, 2, 1]) - syn_total([256; 3], 1, &[2, 1]),
            syn_partial_bytes([64; 3], f4, false)
        );
    }

    #[test]
    fn a_level_that_follows_a_larger_one_is_kept_whole() {
        let level = 128u64.pow(3);
        assert_eq!(
            syn_total([256; 3], 1, &[1, 2]) - syn_total([256; 3], 1, &[1]),
            level * 168 + syn_partial_bytes([128; 3], level, false)
        );
        assert_eq!(
            syn_total([256; 3], 1, &[4, 2, 4, 1]) - syn_total([256; 3], 1, &[4, 2, 1]),
            64u64.pow(3) * 168
        );
    }

    #[test]
    fn the_cohort_pyramid_adds_little_to_a_large_grid() {
        for channels in [1, 2, 3, 4] {
            let single = syn_total([197, 233, 189], channels, &[1]);
            let cohort = syn_total([197, 233, 189], channels, &[8, 4, 2, 1]);
            assert!(cohort > single);
            assert!(cohort - single < single / 50, "{cohort} vs {single}");
        }
    }
}

// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;

use crate::device::{block_gpu, GpuContext};
use crate::error::{GpuError, GpuResult};

/// Usage of every pooled buffer: storage binding plus host transfers.
pub const STORAGE_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Idle buffers below this size are never dropped by [`BufferPool::retain_sizes`]: they are
/// reduction partials and scalars, reused at every level.
pub const RETAIN_FLOOR_BYTES: u64 = 1 << 20;

/// Exact-size buffer pool. Buffers are allocated once, returned with [`BufferPool::release`] and
/// handed out again for the same size, so a stage allocates its arena up front and reuses it.
#[derive(Default)]
pub struct BufferPool {
    free: Vec<wgpu::Buffer>,
    allocated_bytes: u64,
    peak_bytes: u64,
    allocations: u64,
}

impl BufferPool {
    /// Creates an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes the pool currently owns on the device (buffers in use and idle ones).
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Largest value [`Self::allocated_bytes`] has reached.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Starts [`Self::peak_bytes`] again from the bytes owned now.
    pub fn reset_peak(&mut self) {
        self.peak_bytes = self.allocated_bytes;
    }

    /// Bytes held by idle buffers.
    pub fn idle_bytes(&self) -> u64 {
        self.free.iter().map(wgpu::Buffer::size).sum()
    }

    /// Number of device allocations made so far.
    pub fn allocations(&self) -> u64 {
        self.allocations
    }

    /// Returns a storage buffer of exactly `size` bytes (rounded up to a multiple of 4), reusing a
    /// released buffer when one fits.
    pub fn acquire(&mut self, ctx: &GpuContext, label: &str, size: u64) -> GpuResult<wgpu::Buffer> {
        block_gpu("BufferPool::acquire", self.acquire_async(ctx, label, size))
    }

    /// Async [`Self::acquire`]: the allocation's error scopes and the out-of-memory retry are
    /// awaited.
    pub async fn acquire_async(
        &mut self,
        ctx: &GpuContext,
        label: &str,
        size: u64,
    ) -> GpuResult<wgpu::Buffer> {
        acquire_from(self, ctx, label, size).await
    }

    fn take_free(&mut self, size: u64) -> Option<wgpu::Buffer> {
        let pos = self.free.iter().position(|b| b.size() == size)?;
        Some(self.free.swap_remove(pos))
    }

    fn note_allocation(&mut self, size: u64) {
        self.allocated_bytes += size;
        self.peak_bytes = self.peak_bytes.max(self.allocated_bytes);
        self.allocations += 1;
    }

    /// Returns `buffer` to the pool.
    pub fn release(&mut self, buffer: wgpu::Buffer) {
        self.free.push(buffer);
    }

    /// Drops every idle buffer, returning the bytes freed.
    pub fn release_idle(&mut self) -> u64 {
        let freed = self.idle_bytes();
        self.free.clear();
        self.allocated_bytes -= freed;
        freed
    }

    /// Drops every idle buffer of [`RETAIN_FLOOR_BYTES`] or more whose size is not in `keep`,
    /// returning the bytes freed. Called at a level transition so the buffers of a coarser grid
    /// do not stay allocated next to the finer one's working set.
    pub fn retain_sizes(&mut self, keep: &[u64]) -> u64 {
        let mut freed = 0;
        self.free.retain(|b| {
            let size = b.size();
            let keep_it = size < RETAIN_FLOOR_BYTES || keep.contains(&size);
            if !keep_it {
                freed += size;
            }
            keep_it
        });
        self.allocated_bytes -= freed;
        freed
    }
}

pub(crate) trait PoolAccess {
    fn with<R>(&mut self, f: impl FnOnce(&mut BufferPool) -> R) -> R;
}

impl PoolAccess for &mut BufferPool {
    fn with<R>(&mut self, f: impl FnOnce(&mut BufferPool) -> R) -> R {
        f(self)
    }
}

impl PoolAccess for &Mutex<BufferPool> {
    fn with<R>(&mut self, f: impl FnOnce(&mut BufferPool) -> R) -> R {
        f(&mut self.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

pub(crate) async fn acquire_from(
    mut pool: impl PoolAccess,
    ctx: &GpuContext,
    label: &str,
    size: u64,
) -> GpuResult<wgpu::Buffer> {
    let size = size.max(4).next_multiple_of(4);
    if let Some(buffer) = pool.with(|p| p.take_free(size)) {
        return Ok(buffer);
    }
    let create = || {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: STORAGE_USAGE,
            mapped_at_creation: false,
        })
    };
    let buffer = match ctx.scoped_async(label, size, create).await {
        Err(GpuError::OutOfMemory { .. }) if pool.with(|p| !p.free.is_empty()) => {
            pool.with(BufferPool::release_idle);
            ctx.sync_async().await?;
            ctx.scoped_async(label, size, create).await?
        }
        other => other?,
    };
    pool.with(|p| p.note_allocation(size));
    Ok(buffer)
}

// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used)]

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use fastwarp_gpu::parity::{check_mi_async, field, image, Case};
use fastwarp_gpu::{BackendSelect, BufferPool, GpuContext, GpuError, WgpuBackend};
use fastwarp_reg::backend::Backend;
use fastwarp_reg::linalg::IDENTITY3;
use fastwarp_reg::Grid;

fn require_adapter() -> bool {
    fastwarp_gpu::device::env_var("FASTWARP_GPU_REQUIRE_ADAPTER")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn context() -> Option<GpuContext> {
    match GpuContext::new(&BackendSelect::from_env().unwrap_or_default()) {
        Ok(ctx) => Some(ctx),
        Err(GpuError::NoAdapter(why)) => {
            assert!(
                !require_adapter(),
                "no GPU adapter with FASTWARP_GPU_REQUIRE_ADAPTER set: {why}"
            );
            eprintln!("skip: no adapter ({why})");
            None
        }
        Err(e) => panic!("context creation failed: {e}"),
    }
}

fn ready<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("the future was not ready on its first poll"),
    }
}

fn known_buffer(ctx: &GpuContext, pool: &mut BufferPool, n: usize) -> (wgpu::Buffer, Vec<f32>) {
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
    let buffer = pool
        .acquire(ctx, "async test buffer", (n * 4) as u64)
        .unwrap();
    ctx.upload(&buffer, &data);
    (buffer, data)
}

#[test]
fn async_readbacks_match_the_blocking_ones() {
    let Some(ctx) = context() else { return };
    let mut pool = BufferPool::new();
    let (buffer, data) = known_buffer(&ctx, &mut pool, 1000);
    let blocking = ctx.read_f32(&buffer, data.len()).unwrap();
    let asynchronous = ready(ctx.read_f32_async(&buffer, data.len())).unwrap();
    assert_eq!(blocking, data);
    assert_eq!(asynchronous, blocking);
    let bytes = ctx.read_bytes(&buffer, 16, 64).unwrap();
    assert_eq!(ready(ctx.read_bytes_async(&buffer, 16, 64)).unwrap(), bytes);
    let as_u32 = ctx.read_u32(&buffer, 100).unwrap();
    assert_eq!(ready(ctx.read_u32_async(&buffer, 100)).unwrap(), as_u32);
    let staging = ready(ctx.staging_buffer_async(256)).unwrap();
    let staged = ready(ctx.read_f32_staged_async(&buffer, 8, &staging)).unwrap();
    assert_eq!(staged, ctx.read_f32_staged(&buffer, 8, &staging).unwrap());
    assert_eq!(staged, data[..8]);
}

#[test]
fn async_context_creation_selects_the_same_adapter() {
    let Some(blocking) = context() else { return };
    let sel = BackendSelect::from_env().unwrap_or_default();
    let asynchronous = ready(GpuContext::new_async(&sel)).unwrap();
    assert_eq!(asynchronous.info.name, blocking.info.name);
    assert_eq!(asynchronous.info.backend, blocking.info.backend);
    assert_eq!(asynchronous.features, blocking.features);
    assert_eq!(
        asynchronous.device.limits().max_buffer_size,
        blocking.device.limits().max_buffer_size
    );
    ready(asynchronous.sync_async()).unwrap();
}

#[test]
fn async_scoped_captures_the_same_errors() {
    let Some(ctx) = context() else { return };
    let too_big = ctx.device.limits().max_buffer_size + 4;
    let make = || {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("too big"),
            size: too_big,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let blocking = ctx.scoped("too big", too_big, make).err().unwrap();
    let asynchronous = ready(ctx.scoped_async("too big", too_big, make))
        .err()
        .unwrap();
    assert!(
        matches!(
            blocking,
            GpuError::Validation(_) | GpuError::OutOfMemory { .. }
        ),
        "{blocking}"
    );
    assert_eq!(blocking.to_string(), asynchronous.to_string());
    assert_eq!(ready(ctx.scoped_async("fine", 0, || 7)).unwrap(), 7);
    ctx.check_errors().unwrap();
}

#[test]
fn async_pool_acquire_matches_the_blocking_one() {
    let Some(ctx) = context() else { return };
    let mut blocking = BufferPool::new();
    let mut asynchronous = BufferPool::new();
    let a = blocking.acquire(&ctx, "pool a", 1001).unwrap();
    let b = ready(asynchronous.acquire_async(&ctx, "pool b", 1001)).unwrap();
    assert_eq!(a.size(), b.size());
    assert_eq!(blocking.allocated_bytes(), asynchronous.allocated_bytes());
    blocking.release(a);
    asynchronous.release(b);
    let reused = ready(asynchronous.acquire_async(&ctx, "pool b again", 1001)).unwrap();
    assert_eq!(reused.size(), 1004);
    assert_eq!(asynchronous.allocations(), 1);
}

#[test]
fn async_timing_runs() {
    let Some(ctx) = context() else { return };
    let timing = ready(ctx.timed_async(true, |_| {})).unwrap();
    assert!(timing.wall_seconds >= 0.0);
}

#[test]
fn backend_operations_are_ready_on_their_first_poll() {
    let Some(ctx) = context() else { return };
    let backend = ready(WgpuBackend::from_context_async(ctx)).unwrap();
    let grid = Grid::new([20, 18, 16], [1.5, 1.2, 1.0], [-3.0, 2.0, 1.0], IDENTITY3).unwrap();
    let f = ready(Backend::upload_field(&backend, &field(&grid, 2.0, 3))).unwrap();
    assert!(ready(Backend::l2_norm(&backend, &f)).unwrap() > 0.0);
    assert!(ready(Backend::max_scaled_norm(&backend, &f)).unwrap() > 0.0);
    let fixed = ready(Backend::upload_image(&backend, &image(&grid, 4, 0.1))).unwrap();
    let moving = ready(Backend::upload_image(&backend, &image(&grid, 5, 0.1))).unwrap();
    let (value, derivative) =
        ready(Backend::cc_value_derivative(&backend, &fixed, &moving, 2)).unwrap();
    assert!(value.is_finite());
    assert!(ready(Backend::l2_norm(&backend, &derivative)).unwrap() > 0.0);
    let mi = ready(check_mi_async(&backend, &Case::selftest_mi())).unwrap();
    assert!(mi.iter().all(|r| r.ok), "{mi:?}");
}

#[test]
fn async_backend_creation_matches_the_blocking_one() {
    let Some(ctx) = context() else { return };
    let blocking = WgpuBackend::from_context(ctx).unwrap();
    let sel = BackendSelect::from_env().unwrap_or_default();
    let asynchronous = ready(WgpuBackend::new_async(&sel)).unwrap();
    assert_eq!(Backend::name(&asynchronous), Backend::name(&blocking));
    let (a, b) = (asynchronous.selftest_record(), blocking.selftest_record());
    assert!(a.attempts >= 1 && b.attempts >= 1);
    assert_eq!(a.lines.len(), b.lines.len());
}

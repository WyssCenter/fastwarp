// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used)]

mod common;

use fastwarp_gpu::{BackendChoice, BackendSelect, GpuContext, GpuError, GpuN4, WgpuBackend};

fn native_backends() -> Vec<BackendChoice> {
    let mut v = Vec::new();
    if cfg!(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    )) {
        v.push(BackendChoice::Vulkan);
    }
    if cfg!(windows) {
        v.push(BackendChoice::Dx12);
    }
    if cfg!(target_vendor = "apple") {
        v.push(BackendChoice::Metal);
    }
    v
}

fn check(sel: &BackendSelect) -> Result<(), GpuError> {
    let defaults = wgpu::Limits::default();
    GpuContext::new_with_webgpu_default_limits(sel, |ctx| {
        let granted = ctx.device.limits();
        assert_eq!(
            granted.max_storage_buffer_binding_size,
            defaults.max_storage_buffer_binding_size
        );
        assert_eq!(
            granted.max_compute_workgroups_per_dimension,
            defaults.max_compute_workgroups_per_dimension
        );
        assert_eq!(
            ctx.caps.max_storage_buffer_binding_size,
            defaults.max_storage_buffer_binding_size
        );
        assert_eq!(ctx.caps.max_buffer_size, defaults.max_buffer_size);
        assert!(!ctx.features.contains(wgpu::Features::TIMESTAMP_QUERY));
        WgpuBackend::from_context(ctx).map(|b| {
            let record = b.selftest_record();
            assert!(
                record.mi_dense_failure.is_none(),
                "the dense Mattes kernels fail under the WebGPU default limits: {:?}",
                record.mi_dense_failure
            );
        })
    })?;
    GpuContext::new_with_webgpu_default_limits(sel, GpuN4::from_context).map(drop)
}

fn native(sel: &BackendSelect) -> Result<(), GpuError> {
    GpuContext::new_with(sel, WgpuBackend::from_context).map(drop)?;
    GpuContext::new_with(sel, GpuN4::from_context).map(drop)
}

#[test]
fn kernels_fit_webgpu_default_limits() {
    let base = BackendSelect::from_env().unwrap_or_default();
    let mut ran = 0;
    for backend in native_backends() {
        let sel = BackendSelect {
            backend,
            ..base.clone()
        };
        match check(&sel) {
            Ok(()) => {
                eprintln!("{backend:?}: kernels fit the WebGPU default limits");
                ran += 1;
            }
            Err(GpuError::NoAdapter(why)) => eprintln!("skip {backend:?}: no adapter ({why})"),
            Err(e) => match native(&sel) {
                Err(native_error) if native_error.to_string() == e.to_string() => {
                    eprintln!("skip {backend:?}: fails with the native limits too ({e})");
                }
                Err(native_error) => panic!(
                    "{backend:?}: fails under the WebGPU default limits ({e}) and differently \
                     with the native limits ({native_error})"
                ),
                Ok(()) => panic!("{backend:?}: fails only under the WebGPU default limits: {e}"),
            },
        }
    }
    if ran == 0 && common::require_adapter() {
        panic!("no GPU adapter on any backend with FASTWARP_GPU_REQUIRE_ADAPTER set");
    }
}

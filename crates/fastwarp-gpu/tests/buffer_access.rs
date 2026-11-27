#![allow(clippy::unwrap_used, clippy::expect_used)]
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use fastwarp_gpu::pool::STORAGE_USAGE;
use fastwarp_gpu::{BackendSelect, GpuError, GpuField, WgpuBackend};
use fastwarp_reg::backend::Backend;
use fastwarp_reg::field_ops::{InvertParams, VectorField};
use fastwarp_reg::linalg::IDENTITY3;
use fastwarp_reg::{Grid, Volume};
use pollster::block_on;

const SCALE_BY_TWO: &str = "
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&data)) {
        data[i] = data[i] * 2.0;
    }
}
";

fn require_adapter() -> bool {
    fastwarp_gpu::device::env_var("FASTWARP_GPU_REQUIRE_ADAPTER")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn backend() -> Option<WgpuBackend> {
    match WgpuBackend::new(&BackendSelect::from_env().unwrap_or_default()) {
        Ok(b) => {
            eprintln!("using {}", b.name());
            Some(b)
        }
        Err(GpuError::NoAdapter(why)) => {
            if require_adapter() {
                panic!("no GPU adapter with FASTWARP_GPU_REQUIRE_ADAPTER set: {why}");
            }
            eprintln!("skip: no adapter ({why})");
            None
        }
        Err(e) => panic!("backend creation failed: {e}"),
    }
}

fn grid() -> Grid {
    Grid::new([7, 5, 4], [1.2, 0.9, 1.5], [3.0, -2.0, 1.0], IDENTITY3).unwrap()
}

fn sample_field(grid: Grid) -> VectorField {
    VectorField::from_fn(grid, |p| {
        [
            0.3 * p[0] - 0.1 * p[1],
            0.25 * (p[2] - 1.0),
            -0.2 * p[0] + 0.05 * p[1] * p[2],
        ]
    })
}

fn planes(field: &VectorField) -> Vec<f32> {
    let n = field.grid().num_voxels();
    let mut out = vec![0.0f32; 3 * n];
    for (i, v) in field.data().iter().enumerate() {
        out[i] = v[0] as f32;
        out[n + i] = v[1] as f32;
        out[2 * n + i] = v[2] as f32;
    }
    out
}

fn new_buffer(b: &WgpuBackend, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    b.context().device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("caller buffer"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    })
}

fn scale_by_two(b: &WgpuBackend, field: &GpuField) {
    let device = &b.context().device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scale by two"),
        source: wgpu::ShaderSource::Wgsl(SCALE_BY_TWO.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scale by two"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scale by two"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: field.buffer().as_entire_binding(),
        }],
    });
    let values = 3 * field.grid().num_voxels() as u32;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scale by two"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scale by two"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(values.div_ceil(64), 1, 1);
    }
    b.context().queue.submit(Some(enc.finish()));
}

fn message(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[test]
fn custom_kernel_scales_a_backend_field_like_the_cpu() {
    let Some(b) = backend() else { return };
    let host = sample_field(grid());
    let mut field = block_on(b.upload_field(&host)).unwrap();
    assert_eq!(field.buffer().size(), 12 * grid().num_voxels() as u64);
    assert!(field.buffer().usage().contains(STORAGE_USAGE));
    scale_by_two(&b, &field);
    let doubled = block_on(b.download_field(&field)).unwrap();
    for (got, v) in doubled.data().iter().zip(host.data()) {
        for c in 0..3 {
            assert_eq!(got[c], 2.0 * f64::from(v[c] as f32));
            assert!((got[c] - 2.0 * v[c]).abs() <= 1e-6 * v[c].abs().max(1.0));
        }
    }
    block_on(b.scale(&mut field, 0.5)).unwrap();
    let back = block_on(b.download_field(&field)).unwrap();
    for (got, v) in back.data().iter().zip(host.data()) {
        for c in 0..3 {
            assert_eq!(got[c], f64::from(v[c] as f32));
        }
    }
}

#[test]
fn wrapped_buffers_round_trip_and_stay_out_of_the_pool() {
    let Some(b) = backend() else { return };
    let g = grid();
    let n = g.num_voxels();
    let host = sample_field(g);
    let before = b.allocated_bytes();
    let buffer = new_buffer(&b, 12 * n as u64, STORAGE_USAGE);
    b.context()
        .queue
        .write_buffer(&buffer, 0, bytemuck::cast_slice(&planes(&host)));
    let mut field = b.field_from_buffer(buffer.clone(), &g).unwrap();
    assert_eq!(field.buffer(), &buffer);
    assert_eq!(*field.grid(), g);
    assert_eq!(b.allocated_bytes(), before);
    let down = block_on(b.download_field(&field)).unwrap();
    for (got, v) in down.data().iter().zip(host.data()) {
        for c in 0..3 {
            assert_eq!(got[c], f64::from(v[c] as f32));
        }
    }
    scale_by_two(&b, &field);
    block_on(b.scale(&mut field, 0.25)).unwrap();
    let copy = block_on(b.copy_field(&field)).unwrap();
    drop(field);
    let halved = block_on(b.download_field(&copy)).unwrap();
    for (got, v) in halved.data().iter().zip(host.data()) {
        for c in 0..3 {
            assert_eq!(got[c], 0.5 * f64::from(v[c] as f32));
        }
    }
    drop(copy);
    b.release_idle();
    assert_eq!(b.allocated_bytes(), 0);

    let values: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.37).sin() * 50.0 + 100.0)
        .collect();
    let volume = Volume::from_vec(g, values.clone()).unwrap();
    let image_buffer = new_buffer(&b, 4 * n as u64, STORAGE_USAGE);
    let data: Vec<f32> = values.iter().map(|v| *v as f32).collect();
    b.context()
        .queue
        .write_buffer(&image_buffer, 0, bytemuck::cast_slice(&data));
    let wrapped = b.image_from_buffer(image_buffer, &g).unwrap();
    let uploaded = block_on(b.upload_image(&volume)).unwrap();
    assert_eq!(*wrapped.grid(), g);
    assert_eq!(
        b.download_image(&wrapped).unwrap(),
        b.download_image(&uploaded).unwrap()
    );
}

#[test]
fn wrapping_rejects_a_wrong_size_usage_or_device() {
    let Some(b) = backend() else { return };
    let g = grid();
    let n = g.num_voxels() as u64;

    let short = new_buffer(&b, 12 * n - 4, STORAGE_USAGE);
    let e = message(b.field_from_buffer(short, &g).err().unwrap());
    assert!(e.contains("needs exactly"), "{e}");
    let long = new_buffer(&b, 4 * n + 4, STORAGE_USAGE);
    let e = message(b.image_from_buffer(long, &g).err().unwrap());
    assert!(e.contains("needs exactly"), "{e}");
    let field_sized = new_buffer(&b, 12 * n, STORAGE_USAGE);
    let e = message(b.image_from_buffer(field_sized, &g).err().unwrap());
    assert!(e.contains("needs exactly"), "{e}");

    let no_copy_src = new_buffer(
        &b,
        12 * n,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let e = message(b.field_from_buffer(no_copy_src, &g).err().unwrap());
    assert!(e.contains("COPY_SRC"), "{e}");
    let uniform = new_buffer(
        &b,
        4 * n,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let e = message(b.image_from_buffer(uniform, &g).err().unwrap());
    assert!(e.contains("STORAGE"), "{e}");

    let adapter = &b.context().adapter;
    let (other, _queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let foreign = other.create_buffer(&wgpu::BufferDescriptor {
        label: Some("other device"),
        size: 12 * n,
        usage: STORAGE_USAGE,
        mapped_at_creation: false,
    });
    let e = message(b.field_from_buffer(foreign, &g).err().unwrap());
    assert!(e.contains("another device"), "{e}");

    let field = b
        .field_from_buffer(new_buffer(&b, 12 * n, STORAGE_USAGE), &g)
        .unwrap();
    assert!(block_on(b.l2_norm(&field)).is_ok());
}

#[test]
fn wrapped_fields_smooth_and_invert_like_uploaded_ones() {
    let Some(b) = backend() else { return };
    let g = grid();
    let host = sample_field(g);
    let wrap = |f: &VectorField| {
        let buffer = new_buffer(&b, 12 * g.num_voxels() as u64, STORAGE_USAGE);
        b.context()
            .queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&planes(f)));
        b.field_from_buffer(buffer, &g).unwrap()
    };
    let down = |f: &GpuField| block_on(b.download_field(f)).unwrap();

    let smoothed_wrapped = block_on(b.smooth_field(wrap(&host), 1.5)).unwrap();
    let smoothed_uploaded =
        block_on(b.smooth_field(block_on(b.upload_field(&host)).unwrap(), 1.5)).unwrap();
    assert_eq!(down(&smoothed_wrapped), down(&smoothed_uploaded));

    let small = VectorField::from_vec(
        g,
        host.data()
            .iter()
            .map(|v| [0.1 * v[0], 0.1 * v[1], 0.1 * v[2]])
            .collect(),
    )
    .unwrap();
    let zero = VectorField::from_vec(g, vec![[0.0; 3]; g.num_voxels()]).unwrap();
    let (inv_wrapped, _) =
        block_on(b.invert(&wrap(&small), wrap(&zero), &InvertParams::SYN)).unwrap();
    let (inv_uploaded, _) = block_on(b.invert(
        &block_on(b.upload_field(&small)).unwrap(),
        block_on(b.zero_field(&g)).unwrap(),
        &InvertParams::SYN,
    ))
    .unwrap();
    assert_eq!(down(&inv_wrapped), down(&inv_uploaded));
}

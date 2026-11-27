# API reference

Everything below except the sections [ANTsPy compatibility](#antspy-compatibility) and
[Rust: GPU buffers](#rust-gpu-buffers) is imported from the package itself, `import fastwarp`,
and listed in `fastwarp.__all__`; the first of those is the module `fastwarp.ants`, the second
the Rust crate `fastwarp-gpu`. Modules whose names start with an underscore are private.

An *image* is an `ANTsImage` or a tuple `(array, spacing, origin, direction)` (see
[Images](guide/images.md)); a *reference grid* is an `ANTsImage` or a tuple
`(shape, spacing, origin, direction)`. Lengths are in millimetres and coordinates in
{term}`LPS`.

## Registration

Full normalization and linear fits on the GPU. Guides: [Normalization](guide/normalization.md),
[Linear fits](guide/linear.md).

```{eval-rst}
.. autoapifunction:: fastwarp.register_syn
```

```{eval-rst}
.. autoapidata:: fastwarp.OUTPUT_LAYOUTS
```

```{eval-rst}
.. autoapifunction:: fastwarp.linear_session
```

```{eval-rst}
.. autoapiclass:: fastwarp.LinearSession
   :members:
```

```{eval-rst}
.. autoapiclass:: fastwarp.CancelToken
   :members:
```

## Recipe

The registration schedule. Guide: [Recipes](guide/recipes.md).

```{eval-rst}
.. autoapifunction:: fastwarp.cohort_recipe
```

## Resampling

Applying ANTs transform files on the CPU. Guide: [Resampling](guide/resampling.md).

```{eval-rst}
.. autoapifunction:: fastwarp.apply_transforms
```

```{eval-rst}
.. autoapifunction:: fastwarp.compose_field
```

```{eval-rst}
.. autoapidata:: fastwarp.INTERPOLATORS
```

## Label fusion

Joint label fusion on the CPU, as `antsJointFusion` runs it. Guide:
[Label fusion](guide/label-fusion.md).

```{eval-rst}
.. autoapifunction:: fastwarp.joint_label_fusion
```

```{eval-rst}
.. autoapifunction:: fastwarp.register_atlases
```

## ANTsPy compatibility

The module `fastwarp.ants`: ANTsPy's `ants.registration`, `ants.apply_transforms` and
`ants.joint_label_fusion`. Guide: [ANTsPy compatibility](guide/ants.md).

```{eval-rst}
.. autoapifunction:: fastwarp.ants.registration
```

```{eval-rst}
.. autoapifunction:: fastwarp.ants.apply_transforms
```

```{eval-rst}
.. autoapifunction:: fastwarp.ants.joint_label_fusion
```

```{eval-rst}
.. autoapiexception:: fastwarp.ants.UnsupportedArgument
```

## GPU checks

Guide: [GPU and memory](guide/gpu.md).

```{eval-rst}
.. autoapifunction:: fastwarp.gpu_available
```

```{eval-rst}
.. autoapifunction:: fastwarp.gpu_info
```

```{eval-rst}
.. autoapifunction:: fastwarp.fits_gpu
```

```{eval-rst}
.. autoapifunction:: fastwarp.release_idle_gpus
```

```{eval-rst}
.. autoapifunction:: fastwarp.iso_grid_dims
```

## Exceptions

When each one is raised: [Errors](errors.md).

```{eval-rst}
.. autoapiexception:: fastwarp.GpuUnavailableError
```

```{eval-rst}
.. autoapiexception:: fastwarp.AdapterNotFoundError
```

```{eval-rst}
.. autoapiexception:: fastwarp.GpuCapacityError
```

```{eval-rst}
.. autoapiexception:: fastwarp.HostMemoryError
```

```{eval-rst}
.. autoapiexception:: fastwarp.RegistrationError
```

```{eval-rst}
.. autoapiexception:: fastwarp.RegistrationCancelled
```

## Package information

```{eval-rst}
.. py:data:: fastwarp.__version__
   :type: str

   The installed version of FastWarp, for example ``"0.1.0"``, read from the package metadata.
   It is ``"0+unknown"`` when FastWarp runs from a source tree that was not installed.
```

```{eval-rst}
.. autoapidata:: fastwarp.CITATION
```

```{eval-rst}
.. autoapidata:: fastwarp.ENGINE_SOURCE
```

## Rust: GPU buffers

The Rust crate `fastwarp-gpu` lets a program that builds its own registration loop on
`WgpuBackend` run its own WGSL compute kernels, such as a metric or a regulariser of its own, on
the images and fields the backend holds, without copying them to the host and back.

- `GpuImage::buffer()` and `GpuField::buffer()` return the `wgpu::Buffer` that holds the data;
  `GpuImage::grid()` and `GpuField::grid()` return its grid.
- `backend.context().device` and `backend.context().queue` are the device and queue of the
  backend. Create pipelines and buffers on that device and submit on that queue, so that the
  kernels run in order with the backend's own operations.
- `WgpuBackend::image_from_buffer` and `WgpuBackend::field_from_buffer` (and their `_async`
  forms, for wasm32) wrap a buffer that the program created as an image or a field, without a
  copy.

### Layout

| | `GpuImage` | `GpuField` |
| --- | --- | --- |
| Element type | `f32` (WGSL `array<f32>`) | `f32` (WGSL `array<f32>`) |
| Values | `n = grid.num_voxels()` intensities | `3 * n` displacement components in LPS millimetres |
| Components | one per voxel | three planes, not interleaved: x in `0..n`, y in `n..2n`, z in `2n..3n` |
| Voxel order | voxel `(i, j, k)` is element `i + nx * (j + ny * k)` (x fastest) | the same within each plane |
| Size | exactly `4 * n` bytes | exactly `12 * n` bytes |
| Usage | at least `STORAGE \| COPY_SRC \| COPY_DST` (`fastwarp_gpu::pool::STORAGE_USAGE`) | the same |

There is no header, padding or stride in the buffer. The grid (size, spacing, origin and
direction) is not stored on the device; it stays with the `GpuImage` or `GpuField` on the host.

### Wrapping a buffer

`image_from_buffer` and `field_from_buffer` check that the buffer has exactly the size the grid
needs, has the three usages, fits the largest storage binding of the device and can be bound on the
backend's device. A buffer that fails a check, or that was destroyed or created on another device of
the same `wgpu::Instance`, is refused with an error. Some rules are not checked and are up to the
program:

- Create the buffer with `backend.context().device`: wgpu cannot tell apart the buffers of a
  separate `wgpu::Instance`.
- Do not wrap a buffer that is mapped.
- Do not wrap the buffer of an image or field the backend made (from `buffer()`), and do not wrap
  one buffer as two images or fields.

### Ownership and memory

- A buffer the backend allocated belongs to the backend's buffer pool. When its image or field is
  dropped, it returns to the pool and is reused for other data, so a clone of it must not be used
  after that.
- A wrapped buffer belongs to its image or field and never enters the pool. Dropping the image or
  field drops its handle; wgpu frees the memory once the program's own clones are gone too.
- `WgpuBackend::allocated_bytes`, `peak_allocated_bytes` and the capacity plans of
  `fastwarp_gpu::capacity` count the pool only, so a program budgets for its wrapped buffers
  itself.
- Operations that work in place (`scale`, `axpy`) write into the buffer of the field they are
  given, wrapped or not. Operations that take a field by value (`smooth_field`, `invert`,
  `invert_pair`) may write into that field and return it, or return a pooled buffer and drop it.
  Results computed from a borrowed field or image, such as a composition or a warped image, are
  new pooled buffers.
- The CC metric subtracts the most frequent intensity of an image when the image is uploaded, for
  precision; it subtracts nothing from a wrapped image, and a kernel that overwrites an uploaded
  image does not change the value it subtracts. Either way the CC changes only by `f32` rounding.

### Example

A kernel that scales a field by 2 on the backend's device:

```rust
let field = backend.upload_field(&host_field).await?;
let device = &backend.context().device;
let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("scale"),
    source: wgpu::ShaderSource::Wgsl(
        "@group(0) @binding(0) var<storage, read_write> data: array<f32>;
         @compute @workgroup_size(64)
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {
             if (id.x < arrayLength(&data)) { data[id.x] = data[id.x] * 2.0; }
         }"
        .into(),
    ),
});
let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("scale"),
    layout: None,
    module: &module,
    entry_point: Some("main"),
    compilation_options: Default::default(),
    cache: None,
});
let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("scale"),
    layout: &pipeline.get_bind_group_layout(0),
    entries: &[wgpu::BindGroupEntry { binding: 0, resource: field.buffer().as_entire_binding() }],
});
let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups((3 * field.grid().num_voxels() as u32).div_ceil(64), 1, 1);
}
backend.context().queue.submit(Some(encoder.finish()));
```

The test `crates/fastwarp-gpu/tests/buffer_access.rs` runs this kernel, wraps buffers and checks
the refusals.

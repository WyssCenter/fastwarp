# API reference

Everything below is imported from the package itself, `import fastwarp`, and listed in
`fastwarp.__all__`. Modules whose names start with an underscore are private.

## Registration

Full normalization and linear fits on the GPU. Guides: [Normalization](guide/normalization.md),
[Linear fits](guide/linear.md).

```{eval-rst}
.. autoapifunction:: fastwarp.register_syn
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
.. autoapifunction:: fastwarp.iso_grid_dims
```

## Exceptions

When each one is raised: [Errors](errors.md).

```{eval-rst}
.. autoapiexception:: fastwarp.GpuUnavailableError
```

```{eval-rst}
.. autoapiexception:: fastwarp.GpuCapacityError
```

```{eval-rst}
.. autoapiexception:: fastwarp.RegistrationError
```

```{eval-rst}
.. autoapiexception:: fastwarp.RegistrationCancelled
```

## Package information

```{eval-rst}
.. autoapidata:: fastwarp.__version__
```

```{eval-rst}
.. autoapidata:: fastwarp.CITATION
```

```{eval-rst}
.. autoapidata:: fastwarp.ENGINE_SOURCE
```

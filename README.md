# dbsloc-gpureg

GPU image registration for Python: the SyN normalization engine of **Wyss DBS Localizer** by
**Richard Köhler** (Wyss Center for Bio and Neuroengineering), packaged as a wheel.

The engine is a port of ANTs/ITK's SyN registration (cross-correlation and Mattes mutual
information metrics, Gaussian field regularisation, field inversion) to GPU compute shaders via
[wgpu](https://wgpu.rs), so it runs on Vulkan (Linux, Windows), Metal (macOS) and DX12
(Windows). Outputs are ANTs-compatible `Composite.h5` / `InverseComposite.h5` files.

> **Status: private prototype.** Licence terms with the Wyss Center are not yet agreed. Do not
> redistribute. The ported ANTs/ITK files keep their Apache-2.0 notices (`THIRD_PARTY_PORTS.md`).

## Install (prototype)

```bash
pip install "git+ssh://git@github.com/JPPayonk/dbsloc-gpureg.git"   # builds from source: needs Rust
```

## Use

```python
import ants, dbsloc_gpureg

fixed = [ants.image_read("t1_template.nii.gz"), ants.image_read("t2_template.nii.gz")]
moving = [ants.image_read("sub_T1w.nii.gz"), ants.image_read("sub_T2w_in_T1w.nii.gz")]
info = dbsloc_gpureg.register_syn(fixed, moving, "out/antsout", iso_mm=1.0)
# out/antsoutComposite.h5, out/antsoutInverseComposite.h5 -> ants.apply_transforms(...)
print(info["adapter"], info["total_s"])
```

`dbsloc_gpureg.gpu_available()` reports whether a hardware GPU opens (no software fallback).

## Recipe

dbs-localizer cohort recipe (`specs/04` §5.1–5.2 there): centre-of-mass initialisation; Rigid and
Affine with `MI[32 bins, regular 25 %]`, `1000x500x250x100`, shrink `8x4x2x1`, smoothing
`3x2x1x0 vox`; SyN `[0.1, 3, 0]` with `CC[radius 4]` (two channels at weight 0.5 each for
T1w + T2w), `100x70x50x20`; winsorize `[0.005, 0.995]`; all on an isotropic grid (`iso_mm`).

## Source

`crates/dbsloc-{core,niftiio,matfile,reg,gpu}` are copies of dbs-localizer at commit `00faf42`
(2026-10-26). Change: the VAT solver and its `dbsloc-vat` dependency are removed from
`dbsloc-gpu`. `crates/dbsloc-gpureg-py` is the Python binding (PyO3).

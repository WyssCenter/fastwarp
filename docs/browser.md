# Try FastWarp in your browser

<p class="fw-subtitle">Nothing is uploaded. Runs on your GPU.</p>

```{raw} html
<link rel="stylesheet" href="_static/web/app.css">
<div class="fw-demo" id="fw-demo">
  <p id="fw-message" class="fw-message" role="alert" hidden></p>
  <form id="fw-form">
    <fieldset class="fw-files-in">
      <label class="fw-pick" id="fw-pick-fixed">
        <span class="fw-pick-title">Fixed image</span>
        <span class="fw-hint">for example a template (.nii or .nii.gz)</span>
        <span class="fw-pick-row">
          <input type="file" id="fw-fixed" accept=".nii,.gz,application/gzip">
          <span class="fw-drop">or drop a file here</span>
        </span>
        <span id="fw-fixed-name" class="fw-name">No file chosen</span>
      </label>
      <label class="fw-pick" id="fw-pick-moving">
        <span class="fw-pick-title">Moving image</span>
        <span class="fw-hint">the scan to register onto it</span>
        <span class="fw-pick-row">
          <input type="file" id="fw-moving" accept=".nii,.gz,application/gzip">
          <span class="fw-drop">or drop a file here</span>
        </span>
        <span id="fw-moving-name" class="fw-name">No file chosen</span>
      </label>
    </fieldset>
    <p id="fw-sample-row" class="fw-sample-line" hidden>Don't have any data?
      <button type="button" id="fw-sample" class="fw-secondary">Load sample data</button>
      <span id="fw-sample-state" hidden>Sample loaded</span></p>
    <div class="fw-run-row">
      <label class="fw-grid-label">Registration grid
        <select id="fw-grid">
          <option value="1mm">1 mm</option>
          <option value="2mm">2 mm (faster, less memory)</option>
        </select>
      </label>
      <div class="fw-actions">
        <span id="fw-hint" class="fw-run-hint"></span>
        <button type="submit" id="fw-run" disabled>Run</button>
        <button type="button" id="fw-cancel" class="fw-secondary" hidden>Cancel</button>
        <a id="fw-download" class="fw-button" hidden>Download zip</a>
      </div>
    </div>
  </form>
  <div class="fw-status">
    <progress id="fw-progress" max="1" value="0" aria-label="Progress"></progress>
    <div class="fw-statusline" aria-live="polite">
      <span id="fw-stage"></span>
      <span id="fw-elapsed"></span>
    </div>
  </div>
  <details class="fw-more fw-logbox">
    <summary>Log</summary>
    <pre id="fw-log" class="fw-log" aria-live="polite"></pre>
  </details>
  <output id="fw-result" hidden></output>
  <div class="fw-meta">
    <p id="fw-support" role="status">Checking whether this browser offers WebGPU...</p>
    <details id="fw-adapter-box" class="fw-more" hidden>
      <summary id="fw-adapter-summary">WebGPU</summary>
      <dl id="fw-adapter" class="fw-adapter"></dl>
    </details>
  </div>
</div>
<script type="module" src="_static/web/app.js"></script>
```

## What runs where

FastWarp's default registration, the same rigid, affine and {term}`SyN` stages as `fastwarp register`, runs on your own graphics card through [WebGPU](https://www.w3.org/TR/webgpu/). Nothing is installed; the result is a zip file that the page hands to your browser's download folder.

- **Served by Read the Docs:** this page, its script, the WebAssembly module and the two sample
  images, as static files. The page has no server part of its own.
- **In your browser tab:** everything else. A background worker reads the two images, resamples and
  prepares them and writes the output files on one CPU core; the registration itself runs on the
  GPU through WebGPU. The engine is FastWarp's Rust engine compiled to WebAssembly, with the same
  GPU kernels as the Python package and the command line.
- **Nothing is uploaded.** The images are read from your disk into the tab and the zip file is
  made in the tab. Closing the tab discards everything.

## What it computes

The recipe is that of `fastwarp register` without options (see [Recipes](guide/recipes.md)):
centre of mass, `Rigid` and `Affine` with Mattes mutual information, then `SyN` with
neighbourhood cross-correlation, four levels each. The registration grid is isotropic over the
fixed image, 1 mm by default or 2 mm. The command line uses the fixed image's smallest voxel spacing
instead; for the same grid both give the same transforms, up to the small differences between
GPUs. One image pair only: several channels, masks and other recipes need the
[Python package](getting-started.md) or the [command line](cli.md).

## Limits

- **Browsers.** WebGPU runs in Chrome and Edge 113 or later on Windows, macOS and ChromeOS, Chrome
  121 or later on Android, Firefox 141 or later on Windows, and Safari 26 or later on macOS, iOS and
  iPadOS. Other browsers, older versions and GPUs on a browser's block list show a message
  instead (`chrome://gpu` and `about:support` say why). WebGPU needs a secure page: `https`, or
  `localhost` for a local copy.
- **GPU memory.** A browser does not tell a page how much GPU memory there is. The page checks
  the largest buffer of every stage against the browser's limits (`maxBufferSize` and
  `maxStorageBufferBindingSize`) before it starts and refuses a grid that cannot fit. The SyN stage
  needs about 170 bytes of GPU memory per voxel of the registration grid: about 1.4 GiB for the
  1 mm grid of the sample template (197 x 233 x 189 voxels) and about 190 MiB for the 2 mm grid.
  When the GPU runs out of memory, or the browser stops the GPU because a step takes too long, the
  page says so: choose the 2 mm grid, which is preselected on phones and tablets.
- **Computer memory.** The tab holds both images, their resampled and prepared copies and the
  displacement fields in double precision; WebAssembly in a browser can use at most 4 GiB. A 1 mm
  registration of the sample used 1.2 GiB of WebAssembly memory, plus the 210 MiB zip file.
- **Speed.** Slower than natively: the optimisers wait for many small results from the GPU, and
  each takes longer to come back in a browser, and the CPU parts (reading, resampling, writing
  compressed files) run on one core. On one laptop GPU (Intel Arc 140V) Chrome registered the
  sample in 13 s on the 2 mm grid and in 33 s on the 1 mm grid; `fastwarp register` took 2.5 s
  for the 2 mm grid on the same GPU.

## Output files

The zip file holds the files of an `antsRegistration` run with
`--collapse-output-transforms 1 --write-composite-transform 0`, which ANTs, ANTsPy and FastWarp
read (see [Output files](guide/outputs.md)):

| File | Content |
| --- | --- |
| `0GenericAffine.mat` | The affine part (centre of mass, Rigid and Affine in one), ITK format |
| `1Warp.nii.gz` | The SyN displacement field, on the registration grid |
| `1InverseWarp.nii.gz` | Its inverse |
| `Warped.nii.gz` | The moving image resampled onto the fixed image's grid |
| `run.json` | Settings, FastWarp version, GPU and its limits, grids, iterations and timings |
| `README.txt` | The same commands as below |

Apply them as the transforms of any ANTs registration, for example the moving image onto the fixed
image and the fixed image onto the moving image:

```bash
antsApplyTransforms -d 3 -i moving.nii.gz -r fixed.nii.gz -o warped.nii.gz \
    -t 1Warp.nii.gz -t 0GenericAffine.mat
antsApplyTransforms -d 3 -i fixed.nii.gz -r moving.nii.gz -o fixed_in_moving.nii.gz \
    -t "[0GenericAffine.mat,1]" -t 1InverseWarp.nii.gz
```

`fastwarp apply` takes the same `-i`, `-r`, `-o` and `-t` arguments
([Command line: apply](cli.md#apply)); in Python,
`fastwarp.apply_transforms(moving, fixed, ["1Warp.nii.gz", "0GenericAffine.mat"])`.

## About the sample data

The sample pair is `sample-t1w-defaced.nii.gz`, a T1-weighted MPRAGE at 1 mm of an adult volunteer who agreed to its public use, with the header reduced to the geometry and the face, eyes and ears removed (CC0-1.0), and `mni152nlin2009basym-head-1mm.nii.gz`, the MNI ICBM 2009b nonlinear asymmetric T1w template (every second voxel of the 0.5 mm original), cut the same way. Both come from the [sample-data-1](https://github.com/WyssCenter/fastwarp/releases/tag/sample-data-1) release of FastWarp.

MNI template: Copyright (C) 1993-2004 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University. Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided that the above copyright notice appear in all copies. The authors and McGill University make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

## Building this page

The WebAssembly module and the sample images are not in the repository. Read the Docs makes them
before each documentation build with `docs/_scripts/build_web_demo.sh`, which installs the Rust
toolchain of `rust-toolchain.toml` with the `wasm32-unknown-unknown` target, `wasm-bindgen-cli` at
the version in `Cargo.lock`, builds the crate `fastwarp-web`, writes the JavaScript bindings to
`docs/_static/web/pkg/` and downloads the sample images into `docs/_static/web/sample/`, checking
their SHA-256. To try the page locally:

```bash
bash docs/_scripts/build_web_demo.sh
python -m sphinx -b html docs docs/_build/html
python -m http.server -d docs/_build/html 8000
```

then open <http://localhost:8000/browser.html>. Without the script, the documentation still builds
and this page says that the module is missing.

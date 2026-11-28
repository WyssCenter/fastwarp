# Try FastWarp in your browser

<p class="fw-subtitle">Nothing is uploaded. Runs on your GPU.</p>

```{raw} html
<link rel="stylesheet" href="_static/web/app.css">
<div class="fw-demo" id="fw-demo">
  <p id="fw-message" class="fw-message" role="alert" hidden></p>
  <div id="fw-report" class="fw-report" hidden>
    <p class="fw-report-lead">Please send us the diagnostic report: it holds your browser, your GPU and its limits, the image sizes and the error, never your images or file names. Nothing is sent until you send it yourself.</p>
    <div class="fw-report-actions">
      <a id="fw-mail" class="fw-button" hidden>Send report via e-mail</a>
      <button type="button" id="fw-copy" class="fw-secondary">Copy diagnostic report</button>
      <button type="button" id="fw-dismiss" class="fw-secondary" hidden>Dismiss</button>
    </div>
    <p id="fw-copy-state" class="fw-small" role="status"></p>
    <details id="fw-report-box" class="fw-more">
      <summary>Show report</summary>
      <pre id="fw-report-text" class="fw-log"></pre>
    </details>
  </div>
  <form id="fw-form">
    <fieldset class="fw-files-in">
      <label class="fw-pick" id="fw-pick-fixed">
        <span class="fw-pick-title">Fixed image</span>
        <span class="fw-hint">for example a template (.nii or .nii.gz)</span>
        <span class="fw-pick-row">
          <input type="file" id="fw-fixed" class="fw-file-input" accept=".nii,.gz,application/gzip">
          <span class="fw-choose" aria-hidden="true">Choose file</span>
          <span class="fw-drop">or drop a file here</span>
        </span>
        <span id="fw-fixed-name" class="fw-name">No file chosen</span>
      </label>
      <label class="fw-pick" id="fw-pick-moving">
        <span class="fw-pick-title">Moving image</span>
        <span class="fw-hint">the scan to register onto it</span>
        <span class="fw-pick-row">
          <input type="file" id="fw-moving" class="fw-file-input" accept=".nii,.gz,application/gzip">
          <span class="fw-choose" aria-hidden="true">Choose file</span>
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
  <section id="fw-viewer" class="fw-viewer" aria-labelledby="fw-viewer-title" data-state="ready" hidden>
    <div class="fw-viewer-head">
      <p id="fw-viewer-title" class="fw-viewer-title">Check the result</p>
      <button type="button" id="fw-fullscreen" class="fw-secondary" aria-pressed="false">Full screen</button>
    </div>
    <div class="fw-toolbar">
      <div id="fw-modes" class="fw-seg" role="group" aria-label="Comparison">
        <button type="button" data-mode="blend" aria-pressed="true">Blend</button>
        <button type="button" data-mode="flicker" class="fw-secondary" aria-pressed="false">Flicker</button>
        <button type="button" data-mode="edges" class="fw-secondary" aria-pressed="false">Edges</button>
      </div>
      <div id="fw-when" class="fw-seg" role="group" aria-labelledby="fw-when-label" hidden>
        <span id="fw-when-label" class="fw-small-inline">Your image</span>
        <button type="button" data-when="after" aria-pressed="true">After registration</button>
        <button type="button" data-when="before" class="fw-secondary" aria-pressed="false">Before registration</button>
      </div>
    </div>
    <div class="fw-controls">
      <label id="fw-ctl-blend" class="fw-control">
        <span>Template</span>
        <input type="range" id="fw-blend" min="0" max="100" value="50" aria-label="Blend between the template and your image">
        <span>Your image</span>
      </label>
      <div id="fw-ctl-flicker" class="fw-control" hidden>
        <button type="button" id="fw-flicker-play" class="fw-secondary">Pause</button>
        <button type="button" id="fw-flicker-switch" class="fw-secondary">Switch</button>
        <span id="fw-flicker-state" class="fw-small-inline">Showing the template</span>
      </div>
      <p id="fw-ctl-edges" class="fw-control fw-small-inline" hidden>Edges of the template, in yellow, over your image.</p>
    </div>
    <div class="fw-views">
      <figure class="fw-view" data-axis="2">
        <canvas role="img" width="10" height="10"></canvas>
        <figcaption></figcaption>
        <input type="range" min="0" max="0" value="0">
      </figure>
      <span class="fw-break" aria-hidden="true"></span>
      <figure class="fw-view" data-axis="1">
        <canvas role="img" width="10" height="10"></canvas>
        <figcaption></figcaption>
        <input type="range" min="0" max="0" value="0">
      </figure>
      <figure class="fw-view" data-axis="0">
        <canvas role="img" width="10" height="10"></canvas>
        <figcaption></figcaption>
        <input type="range" min="0" max="0" value="0">
      </figure>
    </div>
    <p class="fw-note">Neurological view: the left of the head is on the left of the image.</p>
  </section>
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
      <p class="fw-device-report">
        <button type="button" id="fw-device-copy" class="fw-secondary">Copy device report</button>
        <a id="fw-device-mail" class="fw-button fw-secondary" hidden>Send device report via e-mail</a>
        <span id="fw-device-state" role="status"></span>
      </p>
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
  When the GPU runs out of memory, the page says so: choose the 2 mm grid, which is preselected on
  phones and tablets. When the browser or the GPU driver resets the GPU without a sign that memory
  ran short, which happens now and then on some phones and computers, the page says that instead:
  run it again. If it happens again on the 1 mm grid, the GPU may be too slow for its longest steps
  (the browser stops a step that takes too long): choose the 2 mm grid.
- **Computer memory.** During the registration the tab holds the prepared copies of both images
  in double precision; afterwards it holds the displacement fields in single precision, the warped
  image and the zip file. WebAssembly in a browser can use at most 4 GiB, and its memory does not
  shrink during a visit of the page. A 1 mm registration of the sample used 435 MiB of WebAssembly
  memory while the GPU worked and 836 MiB in all, with the 210 MiB zip file; phones also share
  their memory with the GPU, which needs about 1.4 GiB more for the 1 mm grid.
- **Speed.** Slower than natively: the optimisers wait for many small results from the GPU, and
  each takes longer to come back in a browser, and the CPU parts (reading, resampling, writing
  compressed files) run on one core. On one laptop GPU (Intel Arc 140V) Chrome registered the
  sample in 13 s on the 2 mm grid and in 33 s on the 1 mm grid; `fastwarp register` took 2.5 s
  for the 2 mm grid on the same GPU.

## Problems?

When the page cannot use WebGPU, or a run fails or makes no progress for three minutes, the page
says what happened and what to try, and offers a diagnostic report. When the browser
stopped a run by closing the page (phones do this when a tab needs too much memory), the page says
so the next time you open it, at the top of the demo, and after a 1 mm run selects the 2 mm grid.
Closing, reloading or leaving the page yourself during a run leaves no note. Starting a run or
**Dismiss** removes the note. A run in a tab the browser has put to sleep in the background can be
reported as stopped.

- **Send report via e-mail** opens your mail app with a short report addressed to the FastWarp
  maintainers. You can read and change it before you send it. Mail apps limit the length of such
  a message, so it holds the main points only and says what it left out.
- **Copy diagnostic report** copies the full report, to paste into an e-mail or a
  [GitHub issue](https://github.com/WyssCenter/fastwarp/issues). **Show report** shows it.
- **Copy device report** and **Send device report via e-mail**, under *WebGPU* below the demo, do
  the same when nothing went wrong: a report from a device where the demo works helps as well.

The report contains the browser and operating system (the user agent and the browser's brand
names, and in Chrome the device model, the operating system version and the processor
architecture), the number of processor cores and the browser's memory class, the address of this
page without the site name, the GPU's name, the WebGPU limits and features the browser offers and
those FastWarp asked for and got, the grid, the size and voxel spacing of both images, the time the
report was made and the run started (to the minute, in UTC), how far the run got and when it
reached each step, the memory of the WebAssembly module and of the page's JavaScript, and the
error messages of the browser, the GPU and FastWarp. From FastWarp's engine it adds the limits of
the GPU as the engine sees them, those it requested and those the device granted, the result of
the GPU self-test, and for a failed run the step it reached, the planned GPU memory, the GPU memory
in use when it failed and the errors the GPU reported. When FastWarp stops with an internal error,
the report holds its message and its place in FastWarp's source code (a file of the program, such
as `crates/fastwarp-web/src/web.rs`, never one of yours), and the size of an allocation that
failed when the tab ran out of memory.

It never contains your images or any part of them, the text fields of their headers, file names,
or anything else from your computer. The page sends nothing: the report leaves your device only
when you send the e-mail or paste the text somewhere yourself. The note about an unfinished run is
kept in your browser's local storage on your device; it holds the same kind of information and is
removed when you start the next run or dismiss it.

## Checking the result

When a run has finished, the page shows the template (the fixed image) and your image (the moving image) in three views: axial, coronal and sagittal, at the true proportions of the voxels. The left of the head is on the left of the image (neurological view), and the edges of every view are marked L, R, A, P, S and I, unless the image's orientation cannot be read, in which case the views are shown in voxel order without letters. Click or tap in a view to place the cross; the other two views follow. Each view has a slider for its slice, which the arrow keys, Page Up, Page Down, Home and End move once it has the focus. The Full screen button enlarges the views to the whole screen; Esc or the button leaves it. Nothing leaves the tab: the page keeps 8-bit copies of the images on the template's grid, on a grey scale that runs from the 1st to the 99.5th percentile of the voxels inside the head, until the next run starts.

- **Blend** mixes the two images; the slider runs from the template to your image.
- **Flicker** switches between the two images once a second; Pause stops it and Switch changes the image by hand. With the system setting *reduce motion* it starts paused.
- **Edges** draws the edges of the template in yellow over your image. The outline of the head, the ventricles and the cortex should follow the anatomy beneath.

*After registration* shows your image as registered, *Before registration* as it was when the registration started, moved only by the centre-of-mass alignment, so that the effect of the registration can be seen. The views are a quick check, not a measurement: they show the images, not the quality of any landmark.

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

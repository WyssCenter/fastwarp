# Related projects

[ANTs](https://github.com/ANTsX/ANTs)
: Advanced Normalization Tools, the image registration toolkit whose rigid, affine and SyN
  registration FastWarp runs on the GPU. FastWarp writes ANTs transform files, accepts the same
  schedule as `antsRegistration` (see [Recipes](guide/recipes.md#the-same-schedule-in-ants)),
  and its tests compare it with the programs of ANTs 2.6.5. When no GPU is available, ANTs is
  the fallback.

[ANTsPy](https://github.com/ANTsX/ANTsPy)
: The Python interface to ANTs (package `antspyx`). FastWarp accepts its `ANTsImage` objects
  directly, and {func}`~fastwarp.apply_transforms` takes the same transform lists as
  {func}`ants.apply_transforms`. ANTsPy is optional.

[ITK](https://itk.org)
: The Insight Toolkit, on which ANTs is built. FastWarp's transforms, metrics, optimizer and
  file formats are ports of ITK classes; [Third-party ports](THIRD_PARTY_PORTS.md) lists each
  ported file with its upstream source.

[wgpu](https://wgpu.rs)
: The Rust library through which FastWarp's compute shaders run on Vulkan, Metal and
  DirectX 12.

[Wyss DBS Localizer](https://github.com/WyssCenter/dbs-localizer)
: A tool for localizing deep brain stimulation (DBS) electrodes in brain images. FastWarp was
  developed inside it as its registration engine, and its default recipe is the recipe of Wyss
  DBS Localizer ({func}`~fastwarp.cohort_recipe`).

[Lead-DBS](https://www.lead-dbs.org)
: A toolbox for deep brain stimulation imaging, which can normalize images with ANTs. The
  post-processing timings on the [Performance](performance.md) page are for the steps Lead-DBS
  runs after a registration: composing the transforms into displacement fields and re-warping
  the anatomy.

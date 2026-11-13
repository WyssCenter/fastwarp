// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! The arguments of the `fastwarp` command line.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// GPU image registration (Rigid, Affine, SyN) with ANTs-compatible outputs.
///
/// Exit status: 0 success, 1 registration failed, 2 invalid arguments, 3 file could not be read
/// or written, 4 no usable GPU, 5 the grid does not fit the GPU, 6 cancelled.
#[derive(Parser, Debug)]
#[command(
    name = "fastwarp",
    bin_name = "fastwarp",
    version,
    max_term_width = 100
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Rigid, Affine and SyN registration of MOVING to FIXED (the cohort recipe by default).
    Register(RegisterArgs),
    /// One rigid or affine fit of MOVING to FIXED (Mattes mutual information).
    Linear(LinearArgs),
    /// Resample an image onto a reference grid through transforms (antsApplyTransforms).
    Apply(ApplyArgs),
    /// Write transforms as one displacement field on a reference grid
    /// (antsApplyTransforms --compose).
    Compose(ComposeArgs),
    /// Print the default registration recipe as JSON.
    Recipe(RecipeArgs),
    /// Describe the GPU FastWarp would use, as JSON.
    GpuInfo(GpuInfoArgs),
    /// Check whether a registration fits the GPU (exit 0 if it fits, 5 if not).
    Fits(FitsArgs),
    /// Time FastWarp registrations on synthetic volumes or on your images.
    Bench(BenchArgs),
}

#[derive(Args, Debug)]
pub struct RegisterArgs {
    /// Fixed image (NIfTI).
    pub fixed: PathBuf,
    /// Moving image (NIfTI).
    pub moving: PathBuf,
    /// Output prefix: writes <PREFIX>Composite.h5, <PREFIX>InverseComposite.h5 and
    /// <PREFIX>0GenericAffine.mat. A prefix ending in a path separator is a folder.
    #[arg(short = 'o', long = "output", value_name = "PREFIX")]
    pub output: PathBuf,
    /// Second fixed channel (for example a T2w image) for the SyN stage.
    #[arg(long, value_name = "F2", requires = "moving2")]
    pub fixed2: Option<PathBuf>,
    /// Second moving channel, already in the space of MOVING.
    #[arg(long, value_name = "M2", requires = "fixed2")]
    pub moving2: Option<PathBuf>,
    /// Spacing of the isotropic registration grid in millimetres. Default: the smallest voxel
    /// spacing of FIXED (the Python function register_syn defaults to 1.0 mm instead).
    #[arg(long, value_name = "MM")]
    pub iso_mm: Option<f64>,
    /// Recipe as a JSON file (the form `fastwarp recipe` prints); fields left out keep the
    /// default.
    #[arg(long, value_name = "FILE")]
    pub recipe: Option<PathBuf>,
    /// Also write <PREFIX>_Warped.nii.gz, MOVING on the registration grid.
    #[arg(long)]
    pub write_warped: bool,
    #[command(flatten)]
    pub common: Common,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearTransform {
    Rigid,
    Affine,
}

impl LinearTransform {
    pub fn name(self) -> &'static str {
        match self {
            LinearTransform::Rigid => "rigid",
            LinearTransform::Affine => "affine",
        }
    }
}

#[derive(Args, Debug)]
pub struct LinearArgs {
    /// Fixed image (NIfTI).
    pub fixed: PathBuf,
    /// Moving image (NIfTI).
    pub moving: PathBuf,
    /// Output prefix: writes <PREFIX>0GenericAffine.mat. Without it nothing is written.
    #[arg(short = 'o', long = "output", value_name = "PREFIX")]
    pub output: Option<PathBuf>,
    /// Transform to fit.
    #[arg(long, value_enum, default_value_t = LinearTransform::Rigid)]
    pub transform: LinearTransform,
    /// Fixed-image mask (NIfTI, any grid): the metric samples only its non-zero voxels.
    #[arg(long, value_name = "MASK")]
    pub mask: Option<PathBuf>,
    /// Winsorize both images at these intensity quantiles, for example 0.005,0.995.
    #[arg(long, value_name = "LOWER,UPPER", value_parser = parse_winsorize)]
    pub winsorize: Option<(f32, f32)>,
    /// Start from this affine transform file (.mat or .txt, mapping fixed points to moving
    /// points) instead of the centres of intensity mass.
    #[arg(long, value_name = "FILE")]
    pub initial: Option<PathBuf>,
    /// Stage schedule as a JSON file (the fields of the "rigid" entry of `fastwarp recipe`).
    #[arg(long, value_name = "FILE")]
    pub stage: Option<PathBuf>,
    #[command(flatten)]
    pub common: Common,
}

#[derive(Args, Debug)]
pub struct ApplyArgs {
    /// Reference image: its grid is the output grid.
    #[arg(short = 'r', long, value_name = "REFERENCE")]
    pub reference: PathBuf,
    /// Image to resample.
    #[arg(short = 'i', long, value_name = "INPUT")]
    pub input: PathBuf,
    /// Output image (NIfTI, float32).
    #[arg(short = 'o', long, value_name = "OUTPUT")]
    pub output: PathBuf,
    /// Transform file (.h5, .mat, .txt, or a .nii/.nii.gz displacement field). Repeat it as in
    /// antsApplyTransforms: the first one given maps the reference point first.
    #[arg(short = 't', long = "transform", value_name = "FILE", required = true)]
    pub transforms: Vec<PathBuf>,
    /// Interpolator: linear, nearestNeighbor or lanczosWindowedSinc.
    #[arg(short = 'n', long, value_name = "NAME", default_value = "linear")]
    pub interpolation: String,
    /// Value of points that map outside the input image.
    #[arg(
        short = 'f',
        long,
        value_name = "VALUE",
        default_value_t = 0.0,
        allow_negative_numbers = true
    )]
    pub default_value: f64,
}

#[derive(Args, Debug)]
pub struct ComposeArgs {
    /// Reference image: the field is written on its grid.
    #[arg(short = 'r', long, value_name = "REFERENCE")]
    pub reference: PathBuf,
    /// Output displacement field (.nii or .nii.gz).
    #[arg(short = 'o', long, value_name = "FIELD")]
    pub output: PathBuf,
    /// Transform file, repeated as for `fastwarp apply`.
    #[arg(short = 't', long = "transform", value_name = "FILE", required = true)]
    pub transforms: Vec<PathBuf>,
    /// Gzip level 0 to 9 of a .nii.gz output (0 stores without compressing).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(0..=9))]
    pub compression_level: Option<u32>,
}

#[derive(Args, Debug)]
pub struct RecipeArgs {
    /// Number of channels (1, or 2 for a multispectral SyN stage).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=2))]
    pub channels: u8,
}

#[derive(Args, Debug)]
pub struct GpuInfoArgs {
    /// Pick the GPU whose name contains this text (overrides FASTWARP_GPU_ADAPTER).
    #[arg(long, value_name = "NAME")]
    pub adapter: Option<String>,
    /// Accepted for symmetry with the other commands; the output is always JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct FitsArgs {
    /// Fixed image (NIfTI; only the header is read).
    pub fixed: PathBuf,
    /// Moving image (NIfTI; only the header is read). Without it the moving grid is assumed to
    /// be the fixed grid.
    pub moving: Option<PathBuf>,
    /// Spacing of the isotropic registration grid in millimetres. Default: the smallest voxel
    /// spacing of FIXED, as for `fastwarp register` (fastwarp.fits_gpu defaults to 1.0 mm).
    #[arg(long, value_name = "MM")]
    pub iso_mm: Option<f64>,
    /// Number of channels (1, or 2 for a multispectral SyN stage).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=2))]
    pub channels: u8,
    #[command(flatten)]
    pub common: Common,
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Fixed image (NIfTI) to time instead of the synthetic volumes; give MOVING too.
    #[arg(requires = "moving")]
    pub fixed: Option<PathBuf>,
    /// Moving image (NIfTI).
    pub moving: Option<PathBuf>,
    /// Synthetic size to run; repeat for several. Default: mni-1mm and mni-0.5mm.
    #[arg(
        long = "size",
        value_name = "NAME",
        value_enum,
        conflicts_with = "fixed"
    )]
    pub sizes: Vec<crate::bench::Size>,
    /// Registration grid spacing in millimetres for FIXED / MOVING. Default: the smallest voxel
    /// spacing of FIXED, as for `fastwarp register`. Synthetic volumes register at their own
    /// spacing.
    #[arg(long, value_name = "MM", requires = "fixed")]
    pub iso_mm: Option<f64>,
    /// Timed runs per size.
    #[arg(long, value_name = "N", default_value_t = 1,
          value_parser = clap::value_parser!(u32).range(1..))]
    pub repeat: u32,
    /// Untimed warm-up runs before the timed ones. Default: 1 when --repeat is above 1, else 0.
    #[arg(long, value_name = "N")]
    pub warmup: Option<u32>,
    /// CPU threads for the host-side work (default: all logical cores, or RAYON_NUM_THREADS).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub threads: Option<u32>,
    /// Recipe as a JSON file; fields left out keep the default.
    #[arg(long, value_name = "FILE")]
    pub recipe: Option<PathBuf>,
    #[command(flatten)]
    pub common: Common,
}

/// Options most commands share.
#[derive(Args, Debug, Default)]
pub struct Common {
    /// Pick the GPU whose name contains this text (overrides FASTWARP_GPU_ADAPTER).
    #[arg(long, value_name = "NAME")]
    pub adapter: Option<String>,
    /// Print the result as JSON on standard output.
    #[arg(long)]
    pub json: bool,
}

/// `LOWER,UPPER` quantiles, `0 <= LOWER < UPPER <= 1`.
pub fn parse_winsorize(s: &str) -> Result<(f32, f32), String> {
    let (lo, hi) = s
        .split_once(',')
        .ok_or_else(|| "expected LOWER,UPPER, for example 0.005,0.995".to_string())?;
    let num = |t: &str| {
        t.trim()
            .parse::<f32>()
            .map_err(|_| format!("{t:?} is not a number"))
    };
    let (lo, hi) = (num(lo)?, num(hi)?);
    if !(0.0 <= lo && lo < hi && hi <= 1.0) {
        return Err("expected quantiles with 0 <= LOWER < UPPER <= 1".into());
    }
    Ok((lo, hi))
}

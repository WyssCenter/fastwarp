// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! `antsJointFusion` arguments to a FastWarp joint label fusion plan, and its run.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use fastwarp_gpu::{BackendSelect, GpuContext};
use fastwarp_reg::jlf::{joint_fusion, JlfDevice, JlfInput, JlfOptions, PatchMetric, SearchRadius};
use fastwarp_reg::{Grid, RegError, Volume};
use rayon::prelude::*;

use super::run::{failed, refused, warn};
use super::{
    call, check_image_name, parse_bool, parse_f64, path_arg, refuse, tokenize, unsupported,
    version_text, Images, KeptData, KeptImage, OptSpec, PixelType, Refusal, Takes,
};
use crate::commands::print;
use crate::{image, CliError, ExitStatus};

const PROGRAM: &str = "antsJointFusion";
const COMMAND: &str = "ants-joint-fusion";

const fn spec(short: Option<&'static str>, long: &'static str, takes: Takes) -> OptSpec {
    OptSpec { short, long, takes }
}

const SPECS: &[OptSpec] = &[
    spec(None, "--version", Takes::Optional),
    spec(Some("-h"), "--help", Takes::Optional),
    spec(Some("-d"), "--image-dimensionality", Takes::Required),
    spec(Some("-t"), "--target-image", Takes::Required),
    spec(Some("-g"), "--atlas-image", Takes::Required),
    spec(Some("-l"), "--atlas-segmentation", Takes::Required),
    spec(Some("-a"), "--alpha", Takes::Required),
    spec(Some("-b"), "--beta", Takes::Required),
    spec(Some("-c"), "--constrain-nonnegative", Takes::Optional),
    spec(Some("-p"), "--patch-radius", Takes::Required),
    spec(Some("-m"), "--patch-metric", Takes::Required),
    spec(Some("-s"), "--search-radius", Takes::Required),
    spec(Some("-e"), "--exclusion-image", Takes::Required),
    spec(Some("-x"), "--mask-image", Takes::Required),
    spec(Some("-o"), "--output", Takes::Required),
    spec(Some("-v"), "--verbose", Takes::Optional),
];

/// What an `antsJointFusion` argument list asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum JointFusionCall {
    Help(String),
    Version(String),
    Run(Box<JointFusionPlan>),
}

/// `-s`: a box radius, or an image of isotropic radii per voxel.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchArg {
    Box([usize; 3]),
    Image(PathBuf),
}

/// The files `-o` names. The formats take one `printf` integer (`%d`, `%03d`, ...): the modality
/// or atlas number from 1, or the label.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointFusionOutputs {
    /// The fused label image (`unsigned int`), written only with atlas segmentations.
    pub labels: Option<PathBuf>,
    /// The fused intensity image of each atlas modality.
    pub intensity: Option<String>,
    /// The posterior image of each label but 0, written only with atlas segmentations.
    pub posteriors: Option<String>,
    /// The voting weight image of each atlas.
    pub voting_weights: Option<String>,
}

/// An `antsJointFusion` call FastWarp runs. The atlases are in the order `antsJointFusion`
/// numbers them: the last `-g` (and `-l`) given is atlas 1.
#[derive(Clone, Debug, PartialEq)]
pub struct JointFusionPlan {
    /// `-t`: the target modalities.
    pub target: Vec<PathBuf>,
    /// `-g`: the modalities of each atlas.
    pub atlases: Vec<Vec<PathBuf>>,
    /// `-l`: one segmentation per atlas, or none (intensity fusion only).
    pub labels: Vec<PathBuf>,
    pub mask: Option<PathBuf>,
    /// `-e`: `(label, image)` in the order `antsJointFusion` applies them (the last one for a label
    /// wins, which is the first given).
    pub exclusions: Vec<(u32, PathBuf)>,
    /// `-a` and `-b`, rounded to `float` as `antsJointFusion` reads them.
    pub alpha: f64,
    pub beta: f64,
    pub patch_radius: [usize; 3],
    pub search: SearchArg,
    pub metric: PatchMetric,
    pub nonnegative: bool,
    pub outputs: JointFusionOutputs,
    /// More than two `-o` entries: keep the posteriors (only then are they divided by the vote sum).
    pub retain_posteriors: bool,
    /// Four `-o` entries: keep the atlas voting weights.
    pub retain_voting_weights: bool,
    pub verbose: bool,
    /// Accepted arguments that change nothing or less than they appear to, printed as warnings.
    pub warnings: Vec<String>,
}

pub(crate) fn help_text() -> String {
    format!(
        "{}\
Usage: fastwarp ants-joint-fusion [antsJointFusion options]

Runs antsJointFusion (ANTs 2.6.5) on the CPU and writes what antsJointFusion writes. Options that
would change the result and are not supported are refused with exit status 2 before any work.
See docs/cli.md.

Supported:
  -d 3   -t target.nii.gz | [modality1.nii.gz,modality2.nii.gz,...]
  -g atlas.nii.gz | [modality1.nii.gz,...] (repeat; the last -g is atlas 1)
  -l atlasLabels.nii.gz (repeat, one per -g; none for intensity fusion only)
  -a 0.1   -b 2.0   -c 0|1   -p 2 | 2x2x2   -m PC|MSQ   -s 3 | 3x3x3 | radii.nii.gz
  -e label[exclusion.nii.gz] (repeat)   -x mask.nii.gz
  -o labels.nii.gz | [labels.nii.gz,intensity%d.nii.gz,posterior%04d.nii.gz,weights%02d.nii.gz]
  -v 0|1   --version   -h, --help
",
        version_text(PROGRAM)
    )
}

/// Checks a `printf` file name format with at most one integer conversion.
pub(crate) fn check_format(format: &str, what: &str) -> Result<(), Refusal> {
    let mut conversions = 0;
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            continue;
        }
        while chars.peek().is_some_and(|c| "-+ 0#".contains(*c)) {
            chars.next();
        }
        while chars.peek().is_some_and(char::is_ascii_digit) {
            chars.next();
        }
        match chars.next() {
            Some('d' | 'i' | 'u') => conversions += 1,
            other => {
                return unsupported(
                    format!("{what} {format}"),
                    format!(
                        "FastWarp formats file names with one integer conversion (%d, %i or %u \
                         with flags and a width), not %{}",
                        other.map(String::from).unwrap_or_default()
                    ),
                )
            }
        }
    }
    if conversions > 1 {
        return refuse(format!(
            "{what} {format}: a file name format takes at most one number"
        ));
    }
    Ok(())
}

/// `format` with its integer conversion replaced by `n`, as `snprintf` writes it.
pub fn series_name(format: &str, n: u32) -> String {
    let mut out = String::new();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            out.push('%');
            continue;
        }
        let mut left = false;
        let mut zero = false;
        let mut plus = false;
        let mut space = false;
        while let Some(&f) = chars.peek() {
            match f {
                '-' => left = true,
                '0' => zero = true,
                '+' => plus = true,
                ' ' => space = true,
                '#' => {}
                _ => break,
            }
            chars.next();
        }
        let mut width = 0usize;
        while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
            width = width * 10 + d as usize;
            chars.next();
        }
        let signed = matches!(chars.next(), Some('d' | 'i'));
        let value = if signed {
            i64::from(n as i32)
        } else {
            i64::from(n)
        };
        let sign = if value < 0 {
            "-"
        } else if signed && plus {
            "+"
        } else if signed && space {
            " "
        } else {
            ""
        };
        let digits = value.unsigned_abs().to_string();
        let pad = width.saturating_sub(sign.len() + digits.len());
        if left {
            out.push_str(sign);
            out.push_str(&digits);
            out.push_str(&" ".repeat(pad));
        } else if zero {
            out.push_str(sign);
            out.push_str(&"0".repeat(pad));
            out.push_str(&digits);
        } else {
            out.push_str(&" ".repeat(pad));
            out.push_str(sign);
            out.push_str(&digits);
        }
    }
    out
}

fn radius(text: &str, what: &str) -> Result<[usize; 3], Refusal> {
    let values: Vec<usize> = text
        .split('x')
        .map(|t| {
            t.parse::<u32>().map(|r| r as usize).map_err(|_| {
                Refusal::invalid(format!("{what}: {text:?} is not a radius (2 or 2x2x2)"))
            })
        })
        .collect::<Result<_, _>>()?;
    match values.as_slice() {
        [r] => Ok([*r; 3]),
        [a, b, c] => Ok([*a, *b, *c]),
        _ => refuse(format!(
            "{what}: {text:?} needs 1 or 3 values for a 3D image"
        )),
    }
}

fn image_list(value: &str, what: &str) -> Result<(Vec<PathBuf>, bool), Refusal> {
    let c = call(value)?;
    if !c.bracketed {
        let p = path_arg(value, what)?;
        check_image_name(&p)?;
        return Ok((vec![p], false));
    }
    if !c.name.is_empty() || c.args.is_empty() {
        return refuse(format!("{what}: expected an image or [image1,image2,...]"));
    }
    let paths = c
        .args
        .iter()
        .map(|a| {
            let p = path_arg(a, what)?;
            check_image_name(&p)?;
            Ok(p)
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    Ok((paths, true))
}

fn output_name(text: &str, what: &str, format: bool) -> Result<Option<String>, Refusal> {
    if text.is_empty() {
        return Ok(None);
    }
    check_image_name(Path::new(text))?;
    if format {
        check_format(text, what)?;
    }
    Ok(Some(text.to_string()))
}

fn dimensionality(value: &str, spelled: &str) -> Result<(), Refusal> {
    match value {
        "3" => Ok(()),
        "2" | "4" => unsupported(format!("{spelled} {value}"), "FastWarp fuses 3D images."),
        _ => refuse(format!("{spelled}: expected 2, 3 or 4, got {value:?}")),
    }
}

/// Parses the arguments of `antsJointFusion` (without the program name). Pure: reads no file;
/// image names are checked by their extension only. A `-s` value that is not a radius is taken
/// as a radius image. A count of `-l` that differs from the count of `-g` means intensity fusion
/// only, with a warning, as in `antsJointFusion`.
pub fn parse_joint_fusion<S: AsRef<str>>(args: &[S]) -> Result<JointFusionCall, Refusal> {
    let args: Vec<String> = args.iter().map(|a| a.as_ref().to_string()).collect();
    let opts = tokenize(&args, SPECS)?;
    if let Some(o) = opts.iter().find(|o| o.key == "--help") {
        if parse_bool(o, true)? {
            return Ok(JointFusionCall::Help(help_text()));
        }
    }
    if let Some(o) = opts.iter().find(|o| o.key == "--version") {
        if parse_bool(o, true)? {
            return Ok(JointFusionCall::Version(version_text(PROGRAM)));
        }
    }
    let repeated = ["--atlas-image", "--atlas-segmentation", "--exclusion-image"];
    let mut single = std::collections::HashMap::new();
    for o in &opts {
        if repeated.contains(&o.key) || o.key == "--help" || o.key == "--version" {
            continue;
        }
        if single.insert(o.key, o).is_some() {
            return refuse(format!(
                "{} is given more than once; give it once",
                o.spelled
            ));
        }
    }
    let get = |k: &str| single.get(k).copied();
    if let Some(o) = get("--image-dimensionality") {
        dimensionality(o.value()?, &o.spelled)?;
    }
    let mut warnings = Vec::new();
    let Some(t) = get("--target-image") else {
        return refuse("--target-image is required");
    };
    let (target, _) = image_list(t.value()?, &t.spelled)?;
    let mut atlases = Vec::new();
    let mut modalities = None;
    for o in opts.iter().filter(|o| o.key == "--atlas-image") {
        let (paths, bracketed) = image_list(o.value()?, &o.shown())?;
        if !bracketed && target.len() != 1 {
            return refuse(
                "the number of atlas modalities does not match the number of target modalities",
            );
        }
        if *modalities.get_or_insert(paths.len()) != paths.len() {
            return refuse("every --atlas-image needs the same number of modalities");
        }
        atlases.push(paths);
    }
    if atlases.len() < 2 {
        return refuse("at least 2 atlases are required (--atlas-image)");
    }
    let m = atlases[0].len();
    if target.len() != 1 && target.len() != m {
        return refuse(format!(
            "the number of target images ({}) must be 1 or the number of atlas modalities ({m})",
            target.len()
        ));
    }
    let mut labels = opts
        .iter()
        .filter(|o| o.key == "--atlas-segmentation")
        .map(|o| {
            let p = path_arg(o.value()?, &o.spelled)?;
            check_image_name(&p)?;
            Ok(p)
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    if !labels.is_empty() && labels.len() != atlases.len() {
        warnings.push(format!(
            "{} atlas segmentations for {} atlases: only performing joint intensity fusion, as \
             antsJointFusion does",
            labels.len(),
            atlases.len()
        ));
        labels.clear();
    }
    atlases.reverse();
    labels.reverse();
    let mut exclusions = Vec::new();
    for o in opts.iter().filter(|o| o.key == "--exclusion-image") {
        let v = o.value()?;
        let c = call(v)?;
        let label = c
            .name
            .parse::<u32>()
            .ok()
            .filter(|_| c.bracketed && c.args.len() == 1);
        let Some(label) = label else {
            return refuse(format!("{} {v}: expected label[exclusionImage]", o.spelled));
        };
        let p = path_arg(&c.args[0], &o.spelled)?;
        check_image_name(&p)?;
        if exclusions.iter().any(|(l, _): &(u32, PathBuf)| *l == label) {
            warnings.push(format!(
                "{} is ignored: an earlier --exclusion-image for label {label} wins, as in \
                 antsJointFusion",
                o.shown()
            ));
        }
        exclusions.push((label, p));
    }
    exclusions.reverse();
    let real = |key: &str, default: f64| -> Result<f64, Refusal> {
        match get(key) {
            Some(o) => Ok(f64::from(parse_f64(o.value()?, &o.shown())? as f32)),
            None => Ok(default),
        }
    };
    let alpha = real("--alpha", f64::from(0.1f32))?;
    let beta = real("--beta", 2.0)?;
    let nonnegative = match get("--constrain-nonnegative") {
        Some(o) => parse_bool(o, true)?,
        None => false,
    };
    let verbose = match get("--verbose") {
        Some(o) => parse_bool(o, true)?,
        None => false,
    };
    let patch_radius = match get("--patch-radius") {
        Some(o) => radius(o.value()?, &o.spelled)?,
        None => [2; 3],
    };
    let metric = match get("--patch-metric") {
        Some(o) => match o.value()?.to_ascii_lowercase().as_str() {
            "pc" => PatchMetric::PearsonCorrelation,
            "msq" => PatchMetric::MeanSquares,
            v => {
                return refuse(format!(
                    "{}: unrecognized metric {v:?}; use PC or MSQ",
                    o.spelled
                ))
            }
        },
        None => PatchMetric::PearsonCorrelation,
    };
    let search = match get("--search-radius") {
        Some(o) => {
            let v = o.value()?;
            if v.split('x')
                .all(|t| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit()))
            {
                SearchArg::Box(radius(v, &o.spelled)?)
            } else {
                let p = path_arg(v, &o.spelled)?;
                check_image_name(&p)?;
                SearchArg::Image(p)
            }
        }
        None => SearchArg::Box([3; 3]),
    };
    let mask = match get("--mask-image") {
        Some(o) => {
            let p = path_arg(o.value()?, &o.spelled)?;
            check_image_name(&p)?;
            Some(p)
        }
        None => None,
    };
    let Some(o) = get("--output") else {
        return refuse("--output is required");
    };
    let value = o.value()?;
    let what = o.spelled.clone();
    let c = call(value)?;
    let segmented = !labels.is_empty();
    let mut outputs = JointFusionOutputs::default();
    let (retain_posteriors, retain_voting_weights) = if !c.bracketed {
        if segmented {
            let p = path_arg(value, &what)?;
            check_image_name(&p)?;
            outputs.labels = Some(p);
        } else {
            outputs.intensity = output_name(value, &what, true)?;
        }
        (false, false)
    } else {
        if !c.name.is_empty() || c.args.len() > 4 {
            return refuse(format!(
                "{what} {value}: expected labels.nii.gz or [labels,intensity,posterior,weights]"
            ));
        }
        let arg = |i: usize| c.args.get(i).map(String::as_str).unwrap_or("");
        if segmented {
            outputs.labels = output_name(arg(0), &what, false)?.map(PathBuf::from);
            outputs.posteriors = output_name(arg(2), &what, true)?;
        }
        outputs.intensity = output_name(arg(1), &what, true)?;
        outputs.voting_weights = output_name(arg(3), &what, true)?;
        (c.args.len() > 2, c.args.len() > 3)
    };
    if outputs == JointFusionOutputs::default() {
        warnings.push(format!(
            "{what} {value} names no file to write for this input (label outputs need \
             --atlas-segmentation)"
        ));
    }
    Ok(JointFusionCall::Run(Box::new(JointFusionPlan {
        target,
        atlases,
        labels,
        mask,
        exclusions,
        alpha,
        beta,
        patch_radius,
        search,
        metric,
        nonnegative,
        outputs,
        retain_posteriors,
        retain_voting_weights,
        verbose,
        warnings,
    })))
}

impl JointFusionPlan {
    /// The output names and formats, for checking their folders.
    fn output_names(&self) -> Vec<PathBuf> {
        let o = &self.outputs;
        let mut out: Vec<PathBuf> = o.labels.iter().cloned().collect();
        for f in [&o.intensity, &o.posteriors, &o.voting_weights]
            .into_iter()
            .flatten()
        {
            out.push(PathBuf::from(f));
        }
        out
    }
}

/// What a joint fusion run made: the files written, the images kept in memory, the label set.
#[derive(Debug)]
pub struct JointFusionOutcome {
    pub files: Vec<PathBuf>,
    pub kept: Vec<KeptImage>,
    /// The labels found in the atlas segmentations inside the mask, ascending.
    pub label_set: Vec<u32>,
}

/// How [`run_joint_fusion_plan`] runs: where the patch search runs, which GPU (a substring of
/// its name, overriding `FASTWARP_GPU_ADAPTER`), and a flag that cancels the run.
#[derive(Clone, Debug, Default)]
pub struct JointFusionOptions {
    pub cancel: Option<Arc<AtomicBool>>,
    pub device: JlfDevice,
    pub adapter: Option<String>,
}

fn open_gpu(adapter: Option<String>) -> Result<GpuContext, CliError> {
    let unavailable = |e: fastwarp_gpu::GpuError| CliError::from(crate::engine::gpu_failure(e));
    let mut sel = BackendSelect::from_env().map_err(unavailable)?;
    if adapter.is_some() {
        sel.adapter_substring = adapter;
    }
    GpuContext::new(&sel).map_err(unavailable)
}

fn as_label(v: f64, whole: bool) -> Option<u32> {
    let in_range = v >= 0.0 && v <= f64::from(u32::MAX);
    (in_range && (!whole || v.fract() == 0.0)).then_some(v as u32)
}

fn reg_failure(e: RegError) -> CliError {
    match e {
        RegError::Cancelled => CliError::new(ExitStatus::Cancelled, "cancelled"),
        RegError::Invalid { message, .. } => CliError::usage(message),
        other => CliError::new(ExitStatus::Registration, other.to_string()),
    }
}

struct Loader<'a> {
    images: &'a mut Images,
    grid: Option<Grid>,
}

impl Loader<'_> {
    fn volume(&mut self, path: &Path, role: &str, labels: bool) -> Result<Volume<f64>, CliError> {
        let v = self.images.load_as(path, role, labels)?;
        match self.grid {
            None => self.grid = Some(*v.grid()),
            Some(g) if g.size() != v.grid().size() => {
                return Err(CliError::usage(format!(
                    "{role} {} has {:?} voxels, the target {:?}",
                    path.display(),
                    v.grid().size(),
                    g.size()
                )))
            }
            Some(_) => {}
        }
        Ok(v)
    }

    fn real(&mut self, path: &Path, role: &str) -> Result<Vec<f32>, CliError> {
        Ok(self
            .volume(path, role, false)?
            .data()
            .par_iter()
            .map(|v| *v as f32)
            .collect())
    }

    fn labels(&mut self, path: &Path, role: &str, whole: bool) -> Result<Vec<u32>, CliError> {
        let volume = self.volume(path, role, true)?;
        let data = volume.data();
        if let Some(i) = data
            .par_iter()
            .position_first(|v| as_label(*v, whole).is_none())
        {
            let kind = if whole {
                "whole numbers"
            } else {
                "numbers (truncated to whole numbers)"
            };
            return Err(CliError::usage(format!(
                "{role} {}: the value {} is not supported; its values must be {kind} \
                 from 0 to {}",
                path.display(),
                data[i],
                u32::MAX
            )));
        }
        Ok(data
            .par_iter()
            .map(|v| as_label(*v, whole).unwrap_or(0))
            .collect())
    }
}

/// Runs a parsed `antsJointFusion` plan as `fastwarp ants-joint-fusion` does, with input images
/// taken from `images` where it holds them. An output kept in memory is named by its file name or
/// format in the plan; its images are kept under the expanded names.
pub fn run_joint_fusion_plan(
    plan: &JointFusionPlan,
    mut images: Images,
    options: JointFusionOptions,
) -> Result<JointFusionOutcome, CliError> {
    let JointFusionOptions {
        cancel,
        device,
        adapter,
    } = options;
    for f in plan.output_names() {
        images.check_folder(&f)?;
    }
    let mut load = Loader {
        images: &mut images,
        grid: None,
    };
    let mut target = Vec::new();
    for (i, p) in plan.target.iter().enumerate() {
        target.push(load.real(p, &format!("target image {}", i + 1))?);
    }
    let Some(grid) = load.grid else {
        return Err(CliError::usage("--target-image is required"));
    };
    let mut atlases = Vec::new();
    for (i, a) in plan.atlases.iter().enumerate() {
        let mut modalities = Vec::new();
        for (j, p) in a.iter().enumerate() {
            modalities.push(load.real(p, &format!("atlas {} image {}", i + 1, j + 1))?);
        }
        atlases.push(modalities);
    }
    let mut labels = Vec::new();
    for (i, p) in plan.labels.iter().enumerate() {
        labels.push(load.labels(p, &format!("atlas {} segmentation", i + 1), true)?);
    }
    let mask = match &plan.mask {
        Some(p) => Some(load.labels(p, "mask image", false)?),
        None => None,
    };
    let mut exclusions = Vec::new();
    for (label, p) in &plan.exclusions {
        exclusions.push((
            *label,
            load.labels(p, &format!("exclusion image of label {label}"), false)?,
        ));
    }
    let search = match &plan.search {
        SearchArg::Box(r) => SearchRadius::Box(*r),
        SearchArg::Image(p) => {
            let v = load.volume(p, "search radius image", true)?;
            if v.data()
                .iter()
                .any(|r| !r.is_finite() || *r < 0.0 || *r > f64::from(u32::MAX))
            {
                return Err(CliError::usage(format!(
                    "search radius image {}: radii must be whole numbers from 0 to {}",
                    p.display(),
                    u32::MAX
                )));
            }
            SearchRadius::Image(v.data().iter().map(|r| *r as u32).collect())
        }
    };
    let mut input = JlfInput::new(grid.size(), grid.spacing(), target, atlases);
    input.labels = labels;
    input.mask = mask;
    input.exclusions = exclusions;
    let mut opts = JlfOptions::default();
    opts.alpha = plan.alpha;
    opts.beta = plan.beta;
    opts.patch_radius = plan.patch_radius;
    opts.search = search;
    opts.metric = plan.metric;
    opts.nonnegative = plan.nonnegative;
    opts.posteriors = plan.retain_posteriors;
    opts.voting_weights = plan.retain_voting_weights;
    opts.device = device;
    let fused = match device {
        JlfDevice::Cpu => joint_fusion(&input, &opts, cancel.as_deref()),
        _ => {
            let ctx = open_gpu(adapter)?;
            fastwarp_gpu::jlf::joint_fusion(Arc::new(ctx), &input, &opts, cancel.as_deref())
        }
    }
    .map_err(reg_failure)?;
    drop(input);
    let mut files = Vec::new();
    let mut emit = |images: &mut Images,
                    keep_as: &Path,
                    name: PathBuf,
                    data: KeptData|
     -> Result<(), CliError> {
        if images.keeps(keep_as) {
            images.keep(KeptImage { name, grid, data });
            return Ok(());
        }
        match &data {
            KeptData::Labels(l) => image::save_labels(&name, grid, l)?,
            KeptData::Real(v) => image::save_as(&name, grid, v, PixelType::Float)?,
        }
        files.push(name);
        Ok(())
    };
    let outputs = &plan.outputs;
    if let (Some(path), Some(l)) = (&outputs.labels, fused.labels) {
        emit(&mut images, path, path.clone(), KeptData::Labels(l))?;
    }
    if let Some(format) = &outputs.intensity {
        for (i, data) in fused.intensity.into_iter().enumerate() {
            let name = PathBuf::from(series_name(format, i as u32 + 1));
            emit(&mut images, Path::new(format), name, KeptData::Real(data))?;
        }
    }
    if let Some(format) = &outputs.posteriors {
        for (label, data) in fused.posteriors {
            if label == 0 {
                continue;
            }
            let name = PathBuf::from(series_name(format, label));
            emit(&mut images, Path::new(format), name, KeptData::Real(data))?;
        }
    }
    if let Some(format) = &outputs.voting_weights {
        for (i, data) in fused.voting_weights.into_iter().enumerate() {
            let name = PathBuf::from(series_name(format, i as u32 + 1));
            emit(&mut images, Path::new(format), name, KeptData::Real(data))?;
        }
    }
    Ok(JointFusionOutcome {
        files,
        kept: images.into_kept(),
        label_set: fused.label_set,
    })
}

/// `fastwarp ants-joint-fusion ARGS...`.
pub(crate) fn command(
    args: &[String],
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<ExitStatus, CliError> {
    let plan = match parse_joint_fusion(args).map_err(|r| refused(COMMAND, r))? {
        JointFusionCall::Help(t) | JointFusionCall::Version(t) => {
            print(out, &t)?;
            return Ok(ExitStatus::Ok);
        }
        JointFusionCall::Run(p) => *p,
    };
    warn(err, COMMAND, &plan.warnings);
    let done = run_joint_fusion_plan(&plan, Images::default(), JointFusionOptions::default())
        .map_err(|e| failed(COMMAND, e))?;
    if plan.verbose {
        let mut text = format!(
            "fastwarp {} antsJointFusion compatibility mode\n{} atlases, label set {:?}\n",
            env!("CARGO_PKG_VERSION"),
            plan.atlases.len(),
            done.label_set
        );
        for f in &done.files {
            text.push_str(&format!("wrote {}\n", f.display()));
        }
        print(out, &text)?;
    }
    Ok(ExitStatus::Ok)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn plan(s: &str) -> JointFusionPlan {
        match parse_joint_fusion(&words(s)) {
            Ok(JointFusionCall::Run(p)) => *p,
            other => panic!("{s}: {other:?}"),
        }
    }

    fn refused(s: &str) -> String {
        match parse_joint_fusion(&words(s)) {
            Err(e) => e.0,
            Ok(other) => panic!("{s}: accepted as {other:?}"),
        }
    }

    fn paths(list: &[&str]) -> Vec<PathBuf> {
        list.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn an_ants_call_parses_with_the_atlases_in_ants_order() {
        let p = plan(
            "-d 3 -t t.nii.gz -g a.nii.gz -l la.nii.gz -g b.nii.gz -l lb.nii.gz -g c.nii -l lc.nii \
             -a 0.2 -b 1.5 -c 1 -p 1x2x3 -m msq -s 2 -x m.nii.gz -e 3[e3.nii] -e 4[e4.nii] \
             -o [s.nii.gz,i%d.nii.gz,p%03d.nii.gz,w%02d.nii.gz] -v",
        );
        assert_eq!(p.target, paths(&["t.nii.gz"]));
        assert_eq!(
            p.atlases,
            vec![
                paths(&["c.nii"]),
                paths(&["b.nii.gz"]),
                paths(&["a.nii.gz"])
            ]
        );
        assert_eq!(p.labels, paths(&["lc.nii", "lb.nii.gz", "la.nii.gz"]));
        assert_eq!(p.alpha, f64::from(0.2f32));
        assert_eq!(p.beta, 1.5);
        assert!(p.nonnegative && p.verbose);
        assert_eq!(p.patch_radius, [1, 2, 3]);
        assert_eq!(p.search, SearchArg::Box([2; 3]));
        assert_eq!(p.metric, PatchMetric::MeanSquares);
        assert_eq!(p.mask, Some(PathBuf::from("m.nii.gz")));
        assert_eq!(
            p.exclusions,
            vec![(4, PathBuf::from("e4.nii")), (3, PathBuf::from("e3.nii"))]
        );
        assert_eq!(
            p.outputs,
            JointFusionOutputs {
                labels: Some(PathBuf::from("s.nii.gz")),
                intensity: Some("i%d.nii.gz".into()),
                posteriors: Some("p%03d.nii.gz".into()),
                voting_weights: Some("w%02d.nii.gz".into()),
            }
        );
        assert!(p.retain_posteriors && p.retain_voting_weights);
        assert!(p.warnings.is_empty());
    }

    #[test]
    fn defaults_and_the_output_forms() {
        let p = plan("-t t.nii -g a.nii -g b.nii -l la.nii -l lb.nii -o s.nii");
        assert_eq!(p.alpha, f64::from(0.1f32));
        assert_eq!(p.beta, 2.0);
        assert_eq!(p.patch_radius, [2; 3]);
        assert_eq!(p.search, SearchArg::Box([3; 3]));
        assert_eq!(p.metric, PatchMetric::PearsonCorrelation);
        assert_eq!(p.outputs.labels, Some(PathBuf::from("s.nii")));
        assert!(p.outputs.intensity.is_none());
        assert!(!p.retain_posteriors && !p.retain_voting_weights);
        let p = plan("-t t.nii -g a.nii -g b.nii -o i.nii");
        assert_eq!(p.outputs.intensity.as_deref(), Some("i.nii"));
        assert!(p.outputs.labels.is_none());
        let p = plan("-t t.nii -g a.nii -g b.nii -o [s.nii,i.nii,p%d.nii]");
        assert!(p.outputs.labels.is_none() && p.outputs.posteriors.is_none());
        assert_eq!(p.outputs.intensity.as_deref(), Some("i.nii"));
        let p = plan("-t t.nii -g a.nii -g b.nii -l la.nii -o [s.nii,i.nii] -s r.nii.gz");
        assert_eq!(p.labels, Vec::<PathBuf>::new());
        assert_eq!(p.search, SearchArg::Image(PathBuf::from("r.nii.gz")));
        assert_eq!(
            p.warnings,
            vec![
                "1 atlas segmentations for 2 atlases: only performing joint intensity fusion, \
                  as antsJointFusion does"
            ]
        );
        let p = plan("-t [t1.nii,t2.nii] -g [a1.nii,a2.nii] -g [b1.nii,b2.nii] -o i%d.nii");
        assert_eq!(p.target.len(), 2);
        assert_eq!(p.atlases[0], paths(&["b1.nii", "b2.nii"]));
        let p = plan("-t t.nii -g [a1.nii,a2.nii] -g [b1.nii,b2.nii] -o i%d.nii");
        assert_eq!(p.atlases[1], paths(&["a1.nii", "a2.nii"]));
        let p = plan("-t t.nii -g a.nii -g b.nii -l la.nii -l lb.nii -o [s.nii,,p%d.nii] -e 1[x.nii] -e 1[y.nii]");
        assert!(p.outputs.intensity.is_none());
        assert_eq!(p.exclusions[1], (1, PathBuf::from("x.nii")));
        assert_eq!(
            p.warnings,
            vec![
                "-e 1[y.nii] is ignored: an earlier --exclusion-image for label 1 wins, as in \
                  antsJointFusion"
            ]
        );
    }

    #[test]
    fn help_and_version() {
        for s in ["--help", "-h", "--version", "--version 1"] {
            assert!(
                matches!(
                    parse_joint_fusion(&words(s)),
                    Ok(JointFusionCall::Help(_) | JointFusionCall::Version(_))
                ),
                "{s}"
            );
        }
    }

    #[test]
    fn file_name_formats() {
        assert_eq!(series_name("p%d.nii", 7), "p7.nii");
        assert_eq!(series_name("p%03d.nii", 7), "p007.nii");
        assert_eq!(series_name("p%-3d|.nii", 7), "p7  |.nii");
        assert_eq!(series_name("p%3u.nii", 7), "p  7.nii");
        assert_eq!(series_name("p%%%i.nii", 12), "p%12.nii");
        assert_eq!(series_name("p.nii", 3), "p.nii");
        assert!(check_format("p%02d.nii.gz", "-o").is_ok());
        assert!(check_format("p%s.nii", "-o").unwrap_err().is_unsupported());
        assert!(!check_format("p%d%d.nii", "-o")
            .unwrap_err()
            .is_unsupported());
    }

    #[test]
    fn every_refusal_has_its_message() {
        let base = "-t t.nii -g a.nii -g b.nii -o i.nii";
        let cases: Vec<(String, &str)> = vec![
            (format!("{base} -d 2"), "-d 2 is not supported: FastWarp fuses 3D images."),
            (format!("{base} -d 5"), "-d: expected 2, 3 or 4, got \"5\""),
            ("-g a.nii -g b.nii -o i.nii".into(), "--target-image is required"),
            ("-t t.nii -g a.nii -o i.nii".into(), "at least 2 atlases are required (--atlas-image)"),
            ("-t t.nii -g a.nii -g b.nii".into(), "--output is required"),
            ("-t [t.nii,u.nii] -g a.nii -g b.nii -o i.nii".into(), "the number of atlas modalities does not match the number of target modalities"),
            ("-t [t.nii,u.nii,v.nii] -g [a.nii,b.nii] -g [c.nii,d.nii] -o i.nii".into(), "the number of target images (3) must be 1 or the number of atlas modalities (2)"),
            ("-t t.nii -g [a.nii,b.nii] -g c.nii -o i.nii".into(), "every --atlas-image needs the same number of modalities"),
            (format!("{base} -m NCC"), "-m: unrecognized metric \"ncc\"; use PC or MSQ"),
            (format!("{base} -p 1x2"), "-p: \"1x2\" needs 1 or 3 values for a 3D image"),
            (format!("{base} -p a"), "-p: \"a\" is not a radius (2 or 2x2x2)"),
            (format!("{base} -s 18446744073709551615"), "-s: \"18446744073709551615\" is not a radius (2 or 2x2x2)"),
            (format!("{base} -a x"), "-a x: \"x\" is not a number"),
            (format!("{base} -e x[e.nii]"), "-e x[e.nii]: expected label[exclusionImage]"),
            (format!("{base} -a 1 -a 2"), "-a is given more than once; give it once"),
            (format!("{base} -r 1"), "unknown option -r"),
            ("-t t.mha -g a.nii -g b.nii -o i.nii".into(), "image t.mha is not supported: FastWarp reads and writes NIfTI images (.nii, .nii.gz) only."),
            ("-t t.nii -g a.nii -g b.nii -o [s.nii,i%s.nii]".into(), "-o i%s.nii is not supported: FastWarp formats file names with one integer conversion (%d, %i or %u with flags and a width), not %s"),
            ("-t t.nii -g a.nii -g b.nii -o [s.nii,i.nii,p.nii,w.nii,x.nii]".into(), "-o [s.nii,i.nii,p.nii,w.nii,x.nii]: expected labels.nii.gz or [labels,intensity,posterior,weights]"),
        ];
        for (args, want) in cases {
            assert_eq!(refused(&args), want, "{args}");
        }
    }
}

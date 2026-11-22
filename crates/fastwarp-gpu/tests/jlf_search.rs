// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Instant;

use fastwarp_gpu::jlf::{joint_fusion, joint_fusion_gpu_with, GpuJlfSearch, WeighArgs};
use fastwarp_gpu::{BackendSelect, GpuContext, GpuError};
use fastwarp_reg::jlf::{
    cpu_search, joint_fusion as cpu_fusion, joint_fusion_with, FusionParts, JlfDevice, JlfInput,
    JlfOptions, JlfOutput, PatchMetric, PatchSearch, SearchBlock, SearchRadius,
};
use fastwarp_reg::Result as RegResult;

fn require_adapter() -> bool {
    fastwarp_gpu::device::env_var("FASTWARP_GPU_REQUIRE_ADAPTER")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn context() -> Option<Arc<GpuContext>> {
    match GpuContext::new(&BackendSelect::from_env().unwrap_or_default()) {
        Ok(ctx) => Some(Arc::new(ctx)),
        Err(GpuError::NoAdapter(why)) => {
            assert!(
                !require_adapter(),
                "no GPU adapter with FASTWARP_GPU_REQUIRE_ADAPTER set: {why}"
            );
            eprintln!("skip: no adapter ({why})");
            None
        }
        Err(e) => panic!("context creation failed: {e}"),
    }
}

#[derive(Default, Debug)]
struct Tally {
    items: usize,
    differ: usize,
    worst: f64,
}

struct Both {
    gpu: GpuJlfSearch,
    tally: Tally,
    gpu_s: f64,
    cpu_s: f64,
}

impl PatchSearch for Both {
    fn begin(&mut self) {
        self.gpu.begin();
    }

    fn finish(&mut self) {
        self.gpu.finish();
    }

    fn block_voxels(&self, size: [usize; 3], patch_values: usize) -> usize {
        self.gpu.block_voxels(size, patch_values)
    }

    async fn search(&mut self, block: &SearchBlock<'_>) -> RegResult<Vec<u32>> {
        let t = Instant::now();
        let cpu = cpu_search(block);
        self.cpu_s += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let gpu = self.gpu.search(block).await?;
        self.gpu_s += t.elapsed().as_secs_f64();
        let atlases = block.atlases.len();
        assert_eq!(gpu.len(), cpu.len());
        for (k, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            self.tally.items += 1;
            if g == c {
                continue;
            }
            self.tally.differ += 1;
            let (centre, atlas) = (k / atlases, k % atlases);
            let sc = f64::from(block.score(centre, atlas, *c as usize).unwrap());
            let sg = f64::from(
                block
                    .score(centre, atlas, *g as usize)
                    .expect("the GPU chose a candidate outside the image"),
            );
            let gap = (sg - sc) / sc.abs().max(1.0);
            self.tally.worst = self.tally.worst.max(gap);
        }
        Ok(cpu)
    }
}

fn noise(seed: u32, i: usize) -> f32 {
    let mut h = (i as u32).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503);
    h ^= h >> 13;
    h = h.wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h % 1000) as f32 / 1000.0
}

fn brain(size: [usize; 3], shift: f32, seed: u32) -> Vec<f32> {
    let c = size.map(|s| s as f32 / 2.0);
    (0..size.iter().product::<usize>())
        .map(|v| {
            let p = [
                v % size[0],
                (v / size[0]) % size[1],
                v / (size[0] * size[1]),
            ]
            .map(|x| x as f32);
            let r: f32 = (0..3)
                .map(|d| ((p[d] - c[d]) / (0.42 * size[d] as f32)).powi(2))
                .sum();
            if r > 1.0 {
                return 0.0;
            }
            let tissue = 80.0
                + 40.0 * (0.35 * p[0] + shift).sin() * (0.27 * p[1]).cos()
                + 25.0 * (0.21 * p[2] + 0.5 * shift).sin();
            tissue + 6.0 * noise(seed, v)
        })
        .collect()
}

fn labels(size: [usize; 3], image: &[f32], shift: usize) -> Vec<u32> {
    image
        .iter()
        .enumerate()
        .map(|(v, x)| {
            if *x == 0.0 {
                0
            } else {
                1 + u32::from((v % size[0]) + shift >= size[0] / 2) + 2 * u32::from(*x > 90.0)
            }
        })
        .collect()
}

fn scene(size: [usize; 3], atlases: usize, modalities: usize, targets: usize) -> JlfInput {
    let target: Vec<Vec<f32>> = (0..targets)
        .map(|m| brain(size, 0.7 * m as f32, 1 + m as u32))
        .collect();
    let atl: Vec<Vec<Vec<f32>>> = (0..atlases)
        .map(|i| {
            (0..modalities)
                .map(|m| {
                    brain(
                        size,
                        0.15 * (i as f32 + 1.0) + 0.7 * m as f32,
                        10 + i as u32,
                    )
                })
                .collect()
        })
        .collect();
    let labs = atl
        .iter()
        .enumerate()
        .map(|(i, a)| labels(size, &a[0], i))
        .collect();
    let mut input = JlfInput::new(size, [1.0, 1.1, 1.2], target, atl);
    input.labels = labs;
    input
}

fn spoil(input: &mut JlfInput) {
    let n = input.target[0].len();
    for (k, v) in [
        (n / 2 + 7, f32::NAN),
        (n / 3 + 5, f32::INFINITY),
        (n / 2 + 40, f32::NEG_INFINITY),
    ] {
        input.atlases[0][0][k] = v;
        input.atlases[1][0][k + 3] = f32::NAN;
    }
    input.target[0][n / 2 + 11] = f32::NAN;
    input.mask = Some((0..n).map(|v| u32::from(v % 7 != 0)).collect());
}

#[derive(Clone, Copy, Default)]
struct Budgets {
    atlas: Option<u64>,
    patch: Option<u64>,
}

fn compare(
    ctx: &Arc<GpuContext>,
    input: &JlfInput,
    opts: &JlfOptions,
    budget: Option<u64>,
) -> Tally {
    let budgets = Budgets {
        atlas: budget,
        patch: None,
    };
    compare_with(ctx, input, opts, budgets).0
}

fn compare_with(
    ctx: &Arc<GpuContext>,
    input: &JlfInput,
    opts: &JlfOptions,
    budgets: Budgets,
) -> (Tally, u64) {
    let mut gpu = GpuJlfSearch::new(Arc::clone(ctx)).unwrap();
    if let Some(b) = budgets.atlas {
        gpu = gpu.with_atlas_budget(b);
    }
    if let Some(b) = budgets.patch {
        gpu = gpu.with_patch_budget(b);
    }
    let mut both = Both {
        gpu,
        tally: Tally::default(),
        gpu_s: 0.0,
        cpu_s: 0.0,
    };
    pollster::block_on(joint_fusion_with(input, opts, &mut both, None)).unwrap();
    let search = match &opts.search {
        SearchRadius::Box(r) => format!("box {r:?}"),
        _ => "radius image".to_string(),
    };
    eprintln!(
        "{:?}, patch {:?}, search {search}, {} modalities: {:?}",
        opts.metric,
        opts.patch_radius,
        input.atlases[0].len(),
        both.tally
    );
    (both.tally, both.gpu.slab_uploads())
}

fn check(t: &Tally) {
    assert!(t.items > 0);
    assert!(t.differ * 200 <= t.items * 3, "{t:?}");
    assert!(t.worst <= 1e-3, "{t:?}");
}

#[test]
fn gpu_offsets_equal_the_cpu_ones_except_near_ties() {
    let Some(ctx) = context() else { return };
    let size = [26, 24, 22];
    let one = scene(size, 4, 1, 1);
    let mut opts = JlfOptions::default();
    opts.patch_radius = [1, 1, 1];
    opts.search = SearchRadius::Box([2, 2, 1]);
    check(&compare(&ctx, &one, &opts, None));
    opts.metric = PatchMetric::MeanSquares;
    check(&compare(&ctx, &one, &opts, None));
    opts.metric = PatchMetric::PearsonCorrelation;
    opts.patch_radius = [2, 2, 2];
    opts.search = SearchRadius::Image((0..one.target[0].len()).map(|v| (v % 4) as u32).collect());
    check(&compare(&ctx, &one, &opts, None));
    opts.patch_radius = [1, 1, 1];
    opts.search = SearchRadius::Box([3; 3]);
    check(&compare(&ctx, &one, &opts, Some(4 * 4 * 26 * 24 * 10)));
    opts.patch_radius = [2, 2, 2];
    let two = scene(size, 3, 2, 2);
    opts.search = SearchRadius::Box([1; 3]);
    check(&compare(&ctx, &two, &opts, None));
    let first_only = scene(size, 3, 2, 1);
    check(&compare(&ctx, &first_only, &opts, None));
}

#[test]
fn the_hybrid_fusion_matches_the_cpu_fusion() {
    let Some(ctx) = context() else { return };
    let input = scene([24, 22, 20], 3, 1, 1);
    let mut opts = JlfOptions::default();
    opts.patch_radius = [1; 3];
    opts.search = SearchRadius::Box([2; 3]);
    opts.posteriors = true;
    let cpu = cpu_fusion(&input, &opts, None).unwrap();
    let gpu = joint_fusion(Arc::clone(&ctx), &input, &opts, None).unwrap();
    let (a, b) = (cpu.labels.unwrap(), gpu.labels.unwrap());
    let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert!(differ * 100 <= a.len(), "{differ} of {}", a.len());
    assert_eq!(cpu.label_set, gpu.label_set);
}

#[test]
fn a_slab_too_small_for_the_halo_is_refused() {
    let Some(ctx) = context() else { return };
    let input = scene([20, 18, 16], 2, 1, 1);
    let mut opts = JlfOptions::default();
    opts.patch_radius = [1; 3];
    opts.search = SearchRadius::Box([1; 3]);
    let gpu = GpuJlfSearch::new(Arc::clone(&ctx))
        .unwrap()
        .with_atlas_budget(2 * 4 * 20 * 18 * 2);
    let mut search = gpu;
    let e = pollster::block_on(joint_fusion_with(&input, &opts, &mut search, None))
        .unwrap_err()
        .to_string();
    assert!(e.contains("halo"), "{e}");
}

#[test]
fn more_shapes_metrics_and_spoilt_voxels() {
    let Some(ctx) = context() else { return };
    let size = [24, 22, 20];
    let mut opts = JlfOptions::default();
    opts.patch_radius = [2, 1, 0];
    opts.search = SearchRadius::Box([2, 1, 2]);
    let one = scene(size, 3, 1, 1);
    check(&compare(&ctx, &one, &opts, None));
    opts.metric = PatchMetric::MeanSquares;
    opts.patch_radius = [1; 3];
    opts.search = SearchRadius::Box([1; 3]);
    let two = scene(size, 3, 2, 2);
    check(&compare(&ctx, &two, &opts, None));
    let t = compare(&ctx, &one, &opts, Some(3 * 4 * 24 * 22 * 6));
    check(&t);
    assert_eq!(t.differ, 0, "{t:?}");
    for metric in [PatchMetric::PearsonCorrelation, PatchMetric::MeanSquares] {
        let mut spoilt = scene(size, 3, 1, 1);
        spoil(&mut spoilt);
        opts.metric = metric;
        check(&compare(&ctx, &spoilt, &opts, None));
    }
}

#[test]
fn blocks_reuse_one_slab_and_a_reused_search_sees_new_atlases() {
    let Some(ctx) = context() else { return };
    let size = [20, 18, 30];
    let mut opts = JlfOptions::default();
    opts.patch_radius = [1; 3];
    opts.search = SearchRadius::Box([1; 3]);
    let input = scene(size, 3, 1, 1);
    let budgets = Budgets {
        atlas: None,
        patch: Some(27 * 4 * 20 * 18 * 3),
    };
    let (t, uploads) = compare_with(&ctx, &input, &opts, budgets);
    check(&t);
    assert_eq!(uploads, 1);
    let budgets = Budgets {
        atlas: Some(3 * 4 * 20 * 18 * 12),
        patch: Some(27 * 4 * 20 * 18 * 3),
    };
    let (t, uploads) = compare_with(&ctx, &input, &opts, budgets);
    check(&t);
    assert!(uploads > 1 && uploads < 10, "{uploads}");
    let other = scene(size, 3, 1, 1);
    let mut other = JlfInput::new(
        size,
        other.spacing,
        other.target.clone(),
        other
            .atlases
            .iter()
            .map(|a| {
                a.iter()
                    .map(|m| m.iter().map(|v| 200.0 - v).collect())
                    .collect()
            })
            .collect(),
    );
    other.labels = input.labels.clone();
    let mut both = Both {
        gpu: GpuJlfSearch::new(Arc::clone(&ctx)).unwrap(),
        tally: Tally::default(),
        gpu_s: 0.0,
        cpu_s: 0.0,
    };
    for i in [&input, &other, &input] {
        pollster::block_on(joint_fusion_with(i, &opts, &mut both, None)).unwrap();
    }
    check(&both.tally);
    assert_eq!(both.gpu.slab_uploads(), 3);
    let first = GpuJlfSearch::new(Arc::clone(&ctx)).unwrap();
    let mut direct = first;
    let a = pollster::block_on(joint_fusion_with(&input, &opts, &mut direct, None)).unwrap();
    let b = pollster::block_on(joint_fusion_with(&other, &opts, &mut direct, None)).unwrap();
    let c = cpu_fusion(&other, &opts, None).unwrap();
    assert_ne!(a.intensity, b.intensity);
    let agree = b.intensity[0]
        .iter()
        .zip(&c.intensity[0])
        .filter(|(x, y)| (*x - *y).abs() <= 1e-3 * y.abs().max(1.0))
        .count();
    assert!(agree * 100 >= 99 * c.intensity[0].len(), "{agree}");
}

#[derive(Debug, Default)]
struct ModeDiff {
    labels: usize,
    voxels: usize,
    voting: f64,
    voting_mean: f64,
    posterior: f64,
    intensity: f64,
}

fn mode_diff(a: &JlfOutput, b: &JlfOutput) -> ModeDiff {
    let mut d = ModeDiff::default();
    if let (Some(x), Some(y)) = (&a.labels, &b.labels) {
        d.labels = x.iter().zip(y).filter(|(p, q)| p != q).count();
        d.voxels = x.len();
    }
    let (mut sum, mut count) = (0.0f64, 0usize);
    for (x, y) in a.voting_weights.iter().zip(&b.voting_weights) {
        for (p, q) in x.iter().zip(y) {
            let rel = f64::from((p - q).abs()) / f64::from(q.abs()).max(1e-3);
            d.voting = d.voting.max(rel);
            if *q > 0.01 {
                sum += rel;
                count += 1;
            }
        }
    }
    d.voting_mean = sum / count.max(1) as f64;
    assert_eq!(a.posteriors.len(), b.posteriors.len());
    for ((la, x), (lb, y)) in a.posteriors.iter().zip(&b.posteriors) {
        assert_eq!(la, lb);
        for (p, q) in x.iter().zip(y) {
            d.posterior = d.posterior.max(f64::from((p - q).abs()));
        }
    }
    for (x, y) in a.intensity.iter().zip(&b.intensity) {
        for (p, q) in x.iter().zip(y) {
            d.intensity = d
                .intensity
                .max(f64::from((p - q).abs()) / f64::from(q.abs()).max(1.0));
        }
    }
    d
}

fn run_modes(ctx: &Arc<GpuContext>, input: &JlfInput, opts: &JlfOptions) -> (ModeDiff, ModeDiff) {
    let mut o = opts.clone();
    o.voting_weights = true;
    o.posteriors = true;
    let cpu = cpu_fusion(input, &o, None).unwrap();
    o.device = JlfDevice::Hybrid;
    let hybrid = joint_fusion(Arc::clone(ctx), input, &o, None).unwrap();
    o.device = JlfDevice::Gpu;
    let gpu = joint_fusion(Arc::clone(ctx), input, &o, None).unwrap();
    assert_eq!(gpu.label_set, cpu.label_set);
    let vs_hybrid = mode_diff(&gpu, &hybrid);
    let vs_cpu = mode_diff(&gpu, &cpu);
    eprintln!("gpu vs hybrid {vs_hybrid:?}, gpu vs cpu {vs_cpu:?}");
    (vs_hybrid, vs_cpu)
}

#[derive(Default)]
struct WeightDiff {
    centres: usize,
    flagged: usize,
    errors: Vec<f64>,
}

impl std::fmt::Debug for WeightDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} of {} centres flagged; weight relative error median {:.2e}, 99 % {:.2e}, max {:.2e}",
            self.flagged,
            self.centres,
            self.quantile(0.5),
            self.quantile(0.99),
            self.quantile(1.0)
        )
    }
}

impl WeightDiff {
    fn quantile(&self, q: f64) -> f64 {
        let mut e = self.errors.clone();
        e.sort_by(f64::total_cmp);
        e.get(((e.len() as f64 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

struct Weigh<'a> {
    gpu: GpuJlfSearch,
    parts: FusionParts<'a>,
    opts: &'a JlfOptions,
    diff: WeightDiff,
}

impl PatchSearch for Weigh<'_> {
    fn begin(&mut self) {
        self.gpu.begin();
    }

    fn finish(&mut self) {
        self.gpu.finish();
    }

    fn block_voxels(&self, size: [usize; 3], patch_values: usize) -> usize {
        self.gpu.block_voxels(size, patch_values)
    }

    async fn search(&mut self, block: &SearchBlock<'_>) -> RegResult<Vec<u32>> {
        let normalized = self.parts.target_patches(block.centres);
        let beta = self.opts.beta;
        let args = WeighArgs::new(&normalized, self.opts.alpha, beta);
        let (best, w) = self.gpu.search_and_weigh_async(block, &args).await.unwrap();
        let n = block.atlases.len();
        for (c, &v) in block.centres.iter().enumerate() {
            self.diff.centres += 1;
            if w[c * (n + 1) + n].to_bits() != 0 {
                self.diff.flagged += 1;
                continue;
            }
            let cpu = self
                .parts
                .weights(v, block.list_of[c], &best[c * n..(c + 1) * n]);
            for i in 0..n {
                if cpu[i] > 1e-3 {
                    let g = f64::from(w[c * (n + 1) + i]);
                    self.diff.errors.push((g - cpu[i]).abs() / cpu[i]);
                }
            }
        }
        Ok(best)
    }
}

fn weight_diff(ctx: &Arc<GpuContext>, input: &JlfInput, opts: &JlfOptions) -> WeightDiff {
    let mut w = Weigh {
        gpu: GpuJlfSearch::new(Arc::clone(ctx)).unwrap(),
        parts: FusionParts::new(input, opts).unwrap(),
        opts,
        diff: WeightDiff::default(),
    };
    pollster::block_on(joint_fusion_with(input, opts, &mut w, None)).unwrap();
    let d = w.diff;
    eprintln!("{} atlases: {d:?}", input.atlases.len());
    d
}

#[test]
fn gpu_weights_match_cpu_weights() {
    let Some(ctx) = context() else { return };
    let size = [24, 22, 20];
    let mut opts = JlfOptions::default();
    opts.patch_radius = [1; 3];
    opts.search = SearchRadius::Box([1; 3]);
    for atlases in [4, 12] {
        let d = weight_diff(&ctx, &scene(size, atlases, 1, 1), &opts);
        assert!(d.flagged * 10 <= d.centres, "{d:?}");
        assert!(d.quantile(0.99) <= 1e-4, "{d:?}");
    }
    opts.alpha = 1e-6;
    for atlases in [4, 12] {
        let d = weight_diff(&ctx, &scene(size, atlases, 1, 1), &opts);
        assert!(d.quantile(0.99) <= 1e-3, "{d:?}");
        let mut twins = scene(size, atlases, 1, 1);
        let first = twins.atlases[0].clone();
        for a in &mut twins.atlases {
            *a = first.clone();
        }
        let d = weight_diff(&ctx, &twins, &opts);
        assert!(d.flagged > 0 && d.quantile(0.99) <= 1e-4, "{d:?}");
    }
    opts.alpha = 0.1;
    opts.beta = 0.0;
    let d = weight_diff(&ctx, &scene(size, 4, 1, 1), &opts);
    assert!(d.quantile(0.99) <= 1e-4, "{d:?}");
}

#[test]
fn the_gpu_mode_matches_the_hybrid_mode() {
    let Some(ctx) = context() else { return };
    let size = [24, 22, 20];
    let mut opts = JlfOptions::default();
    opts.patch_radius = [1; 3];
    opts.search = SearchRadius::Box([1; 3]);
    for input in [
        scene(size, 4, 1, 1),
        scene(size, 12, 1, 1),
        scene(size, 3, 2, 2),
        scene(size, 3, 2, 1),
        scene(size, 12, 2, 2),
    ] {
        let (h, c) = run_modes(&ctx, &input, &opts);
        assert!(h.labels * 1000 <= h.voxels, "{h:?}");
        assert!(h.voting <= 1e-3, "{h:?}");
        assert!(h.voting_mean <= 1e-5, "{h:?}");
        assert!(h.posterior <= 1e-3, "{h:?}");
        assert!(h.intensity <= 1e-4, "{h:?}");
        assert!(c.labels * 100 <= c.voxels, "{c:?}");
    }
    let mut spoilt = scene(size, 3, 1, 1);
    spoil(&mut spoilt);
    opts.metric = PatchMetric::MeanSquares;
    opts.beta = 1.5;
    let (h, _) = run_modes(&ctx, &spoilt, &opts);
    assert!(h.labels * 1000 <= h.voxels, "{h:?}");
    assert!(h.intensity <= 1e-4, "{h:?}");
    opts.nonnegative = true;
    opts.patch_radius = [2, 1, 1];
    let (h, _) = run_modes(&ctx, &scene(size, 3, 1, 1), &opts);
    assert!(h.labels * 1000 <= h.voxels, "{h:?}");
}

fn many_labels(input: &mut JlfInput) {
    let n = input.target[0].len();
    for (i, labels) in input.labels.iter_mut().enumerate() {
        *labels = (0..n)
            .map(|v| 1 + ((v as u32).wrapping_mul(2_654_435_761) ^ (i as u32 * 97)) % 40)
            .collect();
    }
}

#[test]
fn gpu_mode_edge_paths_match_the_hybrid_mode() {
    let Some(ctx) = context() else { return };
    let size = [20, 18, 24];
    let mut opts = JlfOptions::default();
    opts.patch_radius = [1; 3];
    opts.search = SearchRadius::Box([1; 3]);
    opts.posteriors = true;
    opts.voting_weights = true;
    let input = scene(size, 4, 1, 1);
    let mut small = GpuJlfSearch::new(Arc::clone(&ctx))
        .unwrap()
        .with_patch_budget(27 * 4 * 20 * 18 * 2)
        .with_atlas_budget(4 * 4 * 20 * 18 * 8)
        .with_gather_budget(39 * 4 * 20 * 18 * 3);
    let sliced =
        pollster::block_on(joint_fusion_gpu_with(&mut small, &input, &opts, None)).unwrap();
    let stats = small.stats();
    assert!(
        stats.slab_uploads > 2 && stats.gather_slabs > 2,
        "{stats:?}"
    );
    opts.device = JlfDevice::Gpu;
    let whole = joint_fusion(Arc::clone(&ctx), &input, &opts, None).unwrap();
    assert_eq!(sliced, whole);
    opts.device = JlfDevice::Hybrid;
    let hybrid = joint_fusion(Arc::clone(&ctx), &input, &opts, None).unwrap();
    let d = mode_diff(&sliced, &hybrid);
    assert!(
        d.labels * 200 <= d.voxels && d.posterior <= 1e-3 && d.intensity <= 1e-4,
        "{d:?}"
    );
    let mut crowded = scene(size, 6, 1, 1);
    many_labels(&mut crowded);
    let mut search = GpuJlfSearch::new(Arc::clone(&ctx)).unwrap();
    let gpu =
        pollster::block_on(joint_fusion_gpu_with(&mut search, &crowded, &opts, None)).unwrap();
    assert!(search.stats().vote_redos > 0, "{:?}", search.stats());
    opts.device = JlfDevice::Hybrid;
    let hybrid = joint_fusion(Arc::clone(&ctx), &crowded, &opts, None).unwrap();
    assert!(gpu.label_set.len() > 16);
    let d = mode_diff(&gpu, &hybrid);
    assert!(d.labels * 100 <= d.voxels && d.posterior <= 1e-3, "{d:?}");
    let most = scene([12, 11, 10], 32, 1, 1);
    let mut search = GpuJlfSearch::new(Arc::clone(&ctx)).unwrap();
    let gpu = pollster::block_on(joint_fusion_gpu_with(&mut search, &most, &opts, None)).unwrap();
    assert!(
        search.stats().tiled_weight_dispatches > 0,
        "{:?}",
        search.stats()
    );
    opts.device = JlfDevice::Hybrid;
    let hybrid = joint_fusion(Arc::clone(&ctx), &most, &opts, None).unwrap();
    let d = mode_diff(&gpu, &hybrid);
    assert!(d.labels * 100 <= d.voxels && d.posterior <= 1e-3, "{d:?}");
    let many = scene([12, 11, 10], 33, 1, 1);
    opts.device = JlfDevice::Gpu;
    let gpu = joint_fusion(Arc::clone(&ctx), &many, &opts, None).unwrap();
    opts.device = JlfDevice::Hybrid;
    let hybrid = joint_fusion(Arc::clone(&ctx), &many, &opts, None).unwrap();
    assert_eq!(gpu, hybrid);
}

#[test]
#[ignore = "whole-fusion timing on a 1 mm brain-sized volume; run with --ignored --nocapture"]
fn whole_fusion_timing_one_millimetre() {
    let Some(ctx) = context() else { return };
    let size = [182, 218, 182];
    let counts: Vec<usize> = std::env::var("JLF_TIMING_ATLASES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![5, 20]);
    for atlases in counts {
        let mut input = scene(size, atlases, 1, 1);
        let mask: Vec<u32> = input.target[0]
            .iter()
            .map(|t| u32::from(*t != 0.0))
            .collect();
        let inside = mask.iter().filter(|m| **m != 0).count();
        input.mask = Some(mask);
        let mut opts = JlfOptions::default();
        let mut time = |device: JlfDevice| {
            opts.device = device;
            let t = Instant::now();
            let out = if device == JlfDevice::Cpu {
                cpu_fusion(&input, &opts, None).unwrap()
            } else {
                joint_fusion(Arc::clone(&ctx), &input, &opts, None).unwrap()
            };
            (out, t.elapsed().as_secs_f64())
        };
        let (gpu, gpu_s) = time(JlfDevice::Gpu);
        let (hybrid, hybrid_s) = time(JlfDevice::Hybrid);
        let (cpu, cpu_s) = time(JlfDevice::Cpu);
        let against = |a: &JlfOutput| {
            a.labels
                .as_ref()
                .unwrap()
                .iter()
                .zip(cpu.labels.as_ref().unwrap())
                .filter(|(x, y)| x != y)
                .count()
        };
        eprintln!(
            "{}: {atlases} atlases, {inside} voxels in the mask: CPU {cpu_s:.1} s, hybrid \
             {hybrid_s:.1} s, GPU {gpu_s:.1} s; labels differing from the CPU: hybrid {}, GPU {}; \
             GPU vs hybrid {:?}",
            ctx.info.name,
            against(&hybrid),
            against(&gpu),
            mode_diff(&gpu, &hybrid)
        );
    }
}

#[test]
#[ignore = "timing on a 1 mm brain-sized volume; run with --ignored --nocapture"]
fn search_timing_one_millimetre() {
    let Some(ctx) = context() else { return };
    let size = [182, 218, 182];
    let input = scene(size, 5, 1, 1);
    let mut mask = vec![0u32; input.target[0].len()];
    for (m, t) in mask.iter_mut().zip(&input.target[0]) {
        *m = u32::from(*t != 0.0);
    }
    let mut input = input;
    input.mask = Some(mask);
    let opts = JlfOptions::default();
    let mut both = Both {
        gpu: GpuJlfSearch::new(Arc::clone(&ctx)).unwrap(),
        tally: Tally::default(),
        gpu_s: 0.0,
        cpu_s: 0.0,
    };
    let t = Instant::now();
    pollster::block_on(joint_fusion_with(&input, &opts, &mut both, None)).unwrap();
    eprintln!(
        "{}: {} centres x 5 atlases, search CPU {:.1} s ({} threads), GPU {:.1} s, total {:.1} s, {:?}",
        ctx.info.name,
        both.tally.items / 5,
        both.cpu_s,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        both.gpu_s,
        t.elapsed().as_secs_f64(),
        both.tally
    );
    check(&both.tally);
}

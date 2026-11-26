#![allow(clippy::unwrap_used, clippy::needless_range_loop)]
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

mod common;

use std::time::Instant;

use common::*;
use fastwarp_gpu::capacity::SELFTEST_POOL_BYTES;
use fastwarp_gpu::{
    plan_capacity_syn, plan_capacity_syn_masked, plan_capacity_syn_stages, CapacityCaps,
    SynStageCapacity, WgpuBackend,
};
use fastwarp_reg::apply::{apply_chain, Interpolation};
use fastwarp_reg::backend::Backend;
use fastwarp_reg::cpu::CpuBackend;
use fastwarp_reg::field_ops::VectorField;
use fastwarp_reg::linalg::{Vec3, IDENTITY3};
use fastwarp_reg::recipe::{SynMetric, SynRecipe};
use fastwarp_reg::stages::{register_stages, InitialTransform, StagePlan};
use fastwarp_reg::syn::{
    run_syn, run_syn_from, PriorWarp, SynChannel, SynOptions, SynResult, SynStart,
};
use fastwarp_reg::{AffineTransform, Grid, MaskVolume, Volume};

/// A mask on `grid` that is inside everywhere except in a ball of `radius` mm around `hole`.
fn mask_without_ball(grid: Grid, hole: Vec3, radius: f64) -> MaskVolume {
    let data = (0..grid.num_voxels())
        .map(|idx| {
            let [i, j, k] = grid.ijk(idx);
            let p = grid.index_to_point([i as f64, j as f64, k as f64]);
            let d2 = (0..3).map(|a| (p[a] - hole[a]).powi(2)).sum::<f64>();
            if d2 < radius * radius {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    MaskVolume::from_volume(Volume::from_vec(grid, data).unwrap())
}

fn contrast3(v: f64) -> f64 {
    (1.5 * v).sin()
}

fn contrast4(v: f64) -> f64 {
    (v - 0.5).abs()
}

fn remapped(v: &Volume<f64>, f: fn(f64) -> f64) -> Volume<f64> {
    Volume::from_vec(*v.grid(), v.data().iter().map(|x| f(*x)).collect()).unwrap()
}

fn channels_n(p: &Phantom, n: usize) -> Vec<SynChannel> {
    let mut c = vec![SynChannel {
        fixed: p.fixed.clone(),
        moving: p.moving.clone(),
    }];
    if n >= 2 {
        c.push(SynChannel {
            fixed: p.fixed2.clone(),
            moving: p.moving2.clone(),
        });
    }
    if n >= 3 {
        c.push(SynChannel {
            fixed: remapped(&p.fixed, contrast3),
            moving: remapped(&p.moving, contrast3),
        });
    }
    if n >= 4 {
        c.push(SynChannel {
            fixed: remapped(&p.fixed, contrast4),
            moving: remapped(&p.moving, contrast4),
        });
    }
    assert_eq!(c.len(), n);
    c
}

fn channels(p: &Phantom, two: bool) -> Vec<SynChannel> {
    channels_n(p, if two { 2 } else { 1 })
}

struct Comparison {
    warp_rms_diff: f64,
    warp_rms: f64,
    inverse_rms_diff: f64,
    ncc_cpu: f64,
    ncc_gpu: f64,
    recovery_cpu: f64,
    recovery_gpu: f64,
    initial_error: f64,
    consistency_cpu: f64,
    consistency_gpu: f64,
    max_energy_diff: f64,
    iterations_cpu: Vec<usize>,
    iterations_gpu: Vec<usize>,
    inversion_steps_cpu: f64,
    inversion_steps_gpu: f64,
    inversion_steps_max_diff: u32,
    cpu_seconds: f64,
    gpu_seconds: f64,
}

fn rms_diff(
    a: &fastwarp_reg::DisplacementField,
    b: &fastwarp_reg::DisplacementField,
) -> (f64, f64) {
    let mut d = 0.0;
    let mut r = 0.0;
    for (x, y) in a.data().iter().zip(b.data()) {
        d += f64::from(x - y).powi(2);
        r += f64::from(*x).powi(2);
    }
    let n = a.data().len() as f64;
    ((d / n).sqrt(), (r / n).sqrt())
}

fn steps(r: &SynResult) -> Vec<u32> {
    r.levels
        .iter()
        .flat_map(|l| {
            l.iterations
                .iter()
                .flat_map(|i| i.inversions.iter().map(|s| s.iterations))
        })
        .collect()
}

fn compare(
    gpu: &WgpuBackend,
    p: &Phantom,
    recipe: &SynRecipe,
    two: bool,
    options: &SynOptions,
) -> Comparison {
    compare_n(gpu, p, recipe, if two { 2 } else { 1 }, options)
}

fn compare_n(
    gpu: &WgpuBackend,
    p: &Phantom,
    recipe: &SynRecipe,
    n: usize,
    options: &SynOptions,
) -> Comparison {
    let ch = channels_n(p, n);
    let t = Instant::now();
    let cpu = run_syn(&CpuBackend, recipe, &ch, &p.initial, options).unwrap();
    let cpu_seconds = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let g = run_syn(gpu, recipe, &ch, &p.initial, options).unwrap();
    let gpu_seconds = t.elapsed().as_secs_f64();
    let (warp_rms_diff, warp_rms) = rms_diff(&cpu.warp, &g.warp);
    let (inverse_rms_diff, _) = rms_diff(&cpu.inverse_warp, &g.inverse_warp);
    let (_, ncc_cpu) = warped_ncc(p, &cpu, &p.moving, &p.fixed);
    let (_, ncc_gpu) = warped_ncc(p, &g, &p.moving, &p.fixed);
    let (initial_error, recovery_cpu, _) = recovery(p, &cpu);
    let (_, recovery_gpu, _) = recovery(p, &g);
    let mut max_energy_diff = 0.0f64;
    for (a, b) in cpu.levels.iter().zip(&g.levels) {
        for (x, y) in a.iterations.iter().zip(&b.iterations) {
            max_energy_diff = max_energy_diff.max((x.energy - y.energy).abs());
        }
    }
    let sc = steps(&cpu);
    let sg = steps(&g);
    let mean = |v: &[u32]| v.iter().sum::<u32>() as f64 / v.len().max(1) as f64;
    let inversion_steps_max_diff = if sc.len() == sg.len() {
        sc.iter()
            .zip(&sg)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0)
    } else {
        u32::MAX
    };
    Comparison {
        warp_rms_diff,
        warp_rms,
        inverse_rms_diff,
        ncc_cpu,
        ncc_gpu,
        recovery_cpu,
        recovery_gpu,
        initial_error,
        consistency_cpu: inverse_consistency(p, &cpu),
        consistency_gpu: inverse_consistency(p, &g),
        max_energy_diff,
        iterations_cpu: cpu.levels.iter().map(|l| l.iterations.len()).collect(),
        iterations_gpu: g.levels.iter().map(|l| l.iterations.len()).collect(),
        inversion_steps_cpu: mean(&sc),
        inversion_steps_gpu: mean(&sg),
        inversion_steps_max_diff,
        cpu_seconds,
        gpu_seconds,
    }
}

fn report(label: &str, c: &Comparison) {
    eprintln!(
        "[{label}] warp RMS diff {:.3e} mm (warp RMS {:.3e}, rel {:.2e}); inverse RMS diff {:.3e} mm",
        c.warp_rms_diff,
        c.warp_rms,
        c.warp_rms_diff / c.warp_rms.max(1e-12),
        c.inverse_rms_diff
    );
    eprintln!(
        "[{label}] NCC after: cpu {:.6} gpu {:.6} (diff {:.2e}); recovery error: initial {:.3} mm, cpu {:.3}, gpu {:.3}; forward o inverse: cpu {:.3} gpu {:.3} voxels",
        c.ncc_cpu,
        c.ncc_gpu,
        (c.ncc_cpu - c.ncc_gpu).abs(),
        c.initial_error,
        c.recovery_cpu,
        c.recovery_gpu,
        c.consistency_cpu,
        c.consistency_gpu
    );
    eprintln!(
        "[{label}] max energy diff {:.3e}; iterations cpu {:?} gpu {:?}; mean inversion steps cpu {:.3} gpu {:.3}, max per-call difference {}; time cpu {:.2} s gpu {:.2} s",
        c.max_energy_diff,
        c.iterations_cpu,
        c.iterations_gpu,
        c.inversion_steps_cpu,
        c.inversion_steps_gpu,
        c.inversion_steps_max_diff,
        c.cpu_seconds,
        c.gpu_seconds
    );
}

#[test]
fn syn_on_the_blob_phantom_matches_the_cpu_reference() {
    let Some(gpu) = backend() else { return };
    let p = phantom(24, 2.0, 2.0);
    let mut recipe = small_recipe(&[(2, 1.0, 20), (1, 0.0, 12)], 2);
    recipe.convergence_threshold = -1.0;
    let c = compare(&gpu, &p, &recipe, false, &SynOptions::default());
    report("blob 24^3 T1", &c);
    assert!(c.warp_rms_diff / c.warp_rms < 0.05, "warp field differs");
    assert!((c.ncc_cpu - c.ncc_gpu).abs() < 2e-3);
    assert!(c.ncc_gpu > 0.97);
    assert!(c.recovery_gpu < 0.5 * c.initial_error);
    assert!(c.recovery_gpu < c.recovery_cpu + 0.05);
    assert!(c.consistency_gpu < 0.25);
    assert!(c.max_energy_diff < 2e-3);
}

#[test]
fn masked_syn_matches_the_cpu_reference() {
    let Some(gpu) = backend() else { return };
    let fixed_mask_grid = Grid::new([33; 3], [1.5; 3], [-24.0, -23.5, -24.25], IDENTITY3).unwrap();
    let moving_mask_grid = Grid::new([20; 3], [2.5; 3], [-24.0; 3], IDENTITY3).unwrap();
    let with_masks = |mut recipe: SynRecipe| {
        recipe.convergence_threshold = -1.0;
        recipe.fixed_mask = Some(mask_without_ball(fixed_mask_grid, [9.0, 3.0, -2.0], 6.0));
        recipe.moving_mask = Some(mask_without_ball(moving_mask_grid, [-7.0, 5.5, 4.5], 7.0));
        recipe
    };
    let one = with_masks(small_recipe(&[(2, 1.0, 20), (1, 0.0, 12)], 2));
    let two = SynRecipe {
        channel_weights: vec![0.5, 0.5],
        ..with_masks(small_recipe(&[(2, 1.0, 12), (1, 0.0, 6)], 2))
    };
    let p24 = phantom(24, 2.0, 2.0);
    let on_grid = SynRecipe {
        fixed_mask: Some(mask_without_ball(p24.grid, [9.0, 3.0, -2.0], 6.0)),
        moving_mask: Some(mask_without_ball(p24.grid, [-7.0, 5.5, 4.5], 7.0)),
        ..one.clone()
    };
    let cases = [
        ("masked blob 24^3 T1", phantom(24, 2.0, 2.0), one, false),
        (
            "masked blob 24^3 T1, masks on the image grid",
            p24,
            on_grid,
            false,
        ),
        (
            "masked blob 16^3 T1+T2",
            phantom(16, 2.0, 1.5),
            two.clone(),
            true,
        ),
    ];
    for (label, p, recipe, two) in &cases {
        let c = compare(&gpu, p, recipe, *two, &SynOptions::default());
        report(label, &c);
        assert!(
            c.warp_rms_diff / c.warp_rms < 0.05,
            "{label}: warp field differs"
        );
        assert!((c.ncc_cpu - c.ncc_gpu).abs() < 2e-3, "{label}");
        assert!(c.ncc_gpu > 0.95, "{label}");
        assert!(c.max_energy_diff < 2e-3, "{label}");
        assert_eq!(c.iterations_cpu, c.iterations_gpu, "{label}");
    }
    let p = &cases[1].1;
    let ch = channels(p, true);
    let a = run_syn(&gpu, &two, &ch, &p.initial, &SynOptions::default()).unwrap();
    let b = run_syn(&gpu, &two, &ch, &p.initial, &SynOptions::default()).unwrap();
    assert_eq!(digest(&a), digest(&b));
    let unmasked = SynRecipe {
        fixed_mask: None,
        moving_mask: None,
        ..two.clone()
    };
    let u = run_syn(&gpu, &unmasked, &ch, &p.initial, &SynOptions::default()).unwrap();
    assert_ne!(digest(&a), digest(&u), "the masks changed nothing");
}

#[test]
fn mattes_syn_matches_the_cpu_reference_alone_mixed_and_masked() {
    let Some(gpu) = backend() else { return };
    let mut one = small_recipe(&[(2, 1.0, 20), (1, 0.0, 12)], 2);
    one.convergence_threshold = -1.0;
    one.channel_metrics = vec![SynMetric::Mattes];
    let p24 = phantom(24, 2.0, 2.0);
    let c = compare(&gpu, &p24, &one, false, &SynOptions::default());
    report("blob 24^3 T1, MI", &c);
    assert!(c.warp_rms_diff / c.warp_rms < 0.05, "warp field differs");
    assert!((c.ncc_cpu - c.ncc_gpu).abs() < 2e-3);
    assert!(c.recovery_gpu < 0.5 * c.initial_error);
    assert!(c.max_energy_diff < 1e-2);
    assert_eq!(c.iterations_cpu, c.iterations_gpu);

    let p16 = phantom(16, 2.0, 1.5);
    let mut mixed = small_recipe(&[(2, 1.0, 12), (1, 0.0, 6)], 2);
    mixed.convergence_threshold = -1.0;
    mixed.channel_weights = vec![0.5, 0.5];
    mixed.channel_metrics = vec![SynMetric::Cc, SynMetric::Mattes];
    let fixed_only = SynRecipe {
        fixed_mask: Some(mask_without_ball(p16.grid, [5.0, 3.0, -2.0], 5.0)),
        ..mixed.clone()
    };
    let both = SynRecipe {
        moving_mask: Some(mask_without_ball(p16.grid, [-4.0, 4.5, 3.5], 5.0)),
        ..fixed_only.clone()
    };
    for (label, recipe) in [
        ("blob 16^3 T1 CC + T2 MI", &mixed),
        ("blob 16^3 T1 CC + T2 MI, fixed mask", &fixed_only),
        ("blob 16^3 T1 CC + T2 MI, both masks", &both),
    ] {
        let c = compare(&gpu, &p16, recipe, true, &SynOptions::default());
        report(label, &c);
        assert!(
            c.warp_rms_diff / c.warp_rms < 0.05,
            "{label}: warp field differs"
        );
        assert!((c.ncc_cpu - c.ncc_gpu).abs() < 2e-3, "{label}");
        assert!(c.max_energy_diff < 1e-2, "{label}");
        assert_eq!(c.iterations_cpu, c.iterations_gpu, "{label}");
    }
    let ch = channels(&p16, true);
    let a = run_syn(&gpu, &both, &ch, &p16.initial, &SynOptions::default()).unwrap();
    let b = run_syn(&gpu, &both, &ch, &p16.initial, &SynOptions::default()).unwrap();
    assert_eq!(digest(&a), digest(&b));
    let open = run_syn(&gpu, &mixed, &ch, &p16.initial, &SynOptions::default()).unwrap();
    assert_ne!(digest(&a), digest(&open), "the masks changed nothing");
}

#[test]
fn a_mask_with_no_voxel_on_the_level_grid_is_an_error_not_nan() {
    let Some(gpu) = backend() else { return };
    let p = phantom(16, 2.0, 1.0);
    let mut recipe = small_recipe(&[(1, 0.0, 3)], 2);
    let far = Grid::new([4; 3], [2.0; 3], [200.0; 3], IDENTITY3).unwrap();
    recipe.fixed_mask = Some(MaskVolume::from_volume(
        Volume::from_vec(far, vec![1.0; far.num_voxels()]).unwrap(),
    ));
    let ch = channels(&p, false);
    for name in ["gpu", "cpu"] {
        let r = if name == "gpu" {
            run_syn(&gpu, &recipe, &ch, &p.initial, &SynOptions::default())
        } else {
            run_syn(
                &CpuBackend,
                &recipe,
                &ch,
                &p.initial,
                &SynOptions::default(),
            )
        };
        let e = r.expect_err("a mask outside the grid is refused");
        assert!(e.to_string().contains("mask"), "{name}: {e}");
    }
    recipe.fixed_mask = Some(MaskVolume::from_volume(Volume::zeros(p.grid)));
    let e = run_syn(&gpu, &recipe, &ch, &p.initial, &SynOptions::default())
        .expect_err("an all-zero mask is refused");
    assert!(e.to_string().contains("fixed mask"), "{e}");
}

#[test]
fn multispectral_syn_matches_the_cpu_reference() {
    let Some(gpu) = backend() else { return };
    let p = phantom(16, 2.0, 1.5);
    let mut recipe = small_recipe(&[(2, 1.0, 12), (1, 0.0, 6)], 2);
    recipe.channel_weights = vec![0.5, 0.5];
    recipe.convergence_threshold = -1.0;
    let c = compare(&gpu, &p, &recipe, true, &SynOptions::default());
    report("blob 16^3 T1+T2", &c);
    assert!(c.warp_rms_diff / c.warp_rms < 0.05);
    assert!((c.ncc_cpu - c.ncc_gpu).abs() < 2e-3);
    assert!(c.max_energy_diff < 2e-3);
}

#[test]
fn three_channel_syn_matches_the_cpu_reference() {
    let Some(gpu) = backend() else { return };
    let p = phantom(16, 2.0, 1.5);
    let mut recipe = small_recipe(&[(2, 1.0, 12), (1, 0.0, 6)], 2);
    recipe.channel_weights = vec![1.0 / 3.0; 3];
    recipe.convergence_threshold = -1.0;
    let c = compare_n(&gpu, &p, &recipe, 3, &SynOptions::default());
    report("blob 16^3 three channels", &c);
    assert!(c.warp_rms_diff / c.warp_rms < 0.05);
    assert!((c.ncc_cpu - c.ncc_gpu).abs() < 2e-3);
    assert!(c.max_energy_diff < 2e-3);
}

#[test]
fn the_convergence_monitor_stops_the_same_levels() {
    let Some(gpu) = backend() else { return };
    let p = phantom(16, 2.0, 1.0);
    let mut recipe = small_recipe(&[(2, 1.0, 200)], 2);
    recipe.convergence_threshold = 1e-3;
    let c = compare(&gpu, &p, &recipe, false, &SynOptions::default());
    report("blob 16^3 converge", &c);
    let (a, b) = (c.iterations_cpu[0], c.iterations_gpu[0]);
    assert!(a < 200 && b < 200);
    assert!(a.abs_diff(b) <= 3, "cpu {a} gpu {b} iterations");
}

#[test]
fn gpu_syn_is_deterministic_bit_for_bit() {
    let Some(gpu) = backend() else { return };
    let p = phantom(16, 2.0, 1.5);
    let mut recipe = small_recipe(&[(2, 1.0, 4), (1, 0.0, 3)], 2);
    recipe.channel_weights = vec![0.5, 0.5];
    let ch = channels(&p, true);
    let a = run_syn(&gpu, &recipe, &ch, &p.initial, &SynOptions::default()).unwrap();
    let b = run_syn(&gpu, &recipe, &ch, &p.initial, &SynOptions::default()).unwrap();
    let bits = |r: &SynResult| -> Vec<u32> {
        r.warp
            .data()
            .iter()
            .chain(r.inverse_warp.data())
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(a.levels, b.levels);
}

#[test]
fn gpu_syn_is_deterministic_with_smoothed_total_field_and_wide_kernels() {
    let Some(gpu) = backend() else { return };
    let p = phantom(16, 2.0, 1.5);
    let variants: [(&str, bool, f64, f64); 4] = [
        ("one channel, default smoothing", false, 3.0, 0.0),
        ("two channels, default smoothing", true, 3.0, 0.0),
        ("two channels, smoothed total field", true, 3.0, 2.0),
        ("one channel, wide update kernel", false, 14.0, 6.0),
    ];
    for (label, two, update_variance, total_variance) in variants {
        let mut recipe = small_recipe(&[(2, 1.0, 3), (1, 0.0, 3)], 2);
        if two {
            recipe.channel_weights = vec![0.5, 0.5];
        }
        recipe.update_field_variance = update_variance;
        recipe.total_field_variance = total_variance;
        recipe.convergence_threshold = -1.0;
        let ch = channels(&p, two);
        let a = run_syn(&gpu, &recipe, &ch, &p.initial, &SynOptions::default()).unwrap();
        let b = run_syn(&gpu, &recipe, &ch, &p.initial, &SynOptions::default()).unwrap();
        eprintln!("DIGEST {label}: {:016x}", digest(&a));
        assert_eq!(digest(&a), digest(&b), "{label}");
        assert!(a.warp.data().iter().all(|v| v.is_finite()), "{label}");
    }
}

#[test]
fn a_syn_iteration_reads_back_four_scalar_blocks_with_one_channel_and_five_with_two() {
    let Some(gpu) = backend() else { return };
    let p = phantom(16, 2.0, 1.5);
    let mask = mask_without_ball(p.grid, [3.0, 0.0, 0.0], 5.0);
    for (two, masked, expected) in [
        (false, false, 4),
        (true, false, 5),
        (false, true, 4),
        (true, true, 5),
    ] {
        let readbacks = |iterations: u32| {
            let mut recipe = small_recipe(&[(1, 0.0, iterations)], 2);
            if two {
                recipe.channel_weights = vec![0.5, 0.5];
            }
            if masked {
                recipe.fixed_mask = Some(mask.clone());
                recipe.moving_mask = Some(mask.clone());
            }
            recipe.convergence_threshold = -1.0;
            let before = gpu.stats().scalar_readbacks;
            let r = run_syn(
                &gpu,
                &recipe,
                &channels(&p, two),
                &p.initial,
                &SynOptions::default(),
            )
            .unwrap();
            assert_eq!(r.levels[0].iterations.len(), iterations as usize);
            gpu.stats().scalar_readbacks - before
        };
        let per_iteration = (readbacks(5) - readbacks(2)) / 3;
        assert_eq!(
            per_iteration, expected,
            "two channels: {two}, masked: {masked}"
        );
    }
}

fn capacity_plan_bounds_pool_peak(n: usize, moving_n: usize, factors: &[u32], iterations: u32) {
    capacity_plan_bounds_pool_peak_with(n, moving_n, factors, iterations, false);
}

fn capacity_plan_bounds_pool_peak_with(
    n: usize,
    moving_n: usize,
    factors: &[u32],
    iterations: u32,
    masked: bool,
) {
    let n_full = (n * n * n) as u64;
    let mut h = head([n, n, n], 2.0, 3.0);
    if moving_n != n {
        let spacing = (n - 1) as f64 * 2.0 / (moving_n - 1) as f64;
        h.moving = head([moving_n; 3], spacing, 3.0).moving;
    }
    let mask_dims: Vec<[usize; 3]> = if masked {
        vec![[n; 3], [moving_n; 3]]
    } else {
        Vec::new()
    };
    let caps = CapacityCaps {
        max_storage_buffer_binding_size: u64::MAX,
        memory_budget: u64::MAX,
    };
    let mut failures = Vec::new();
    for channels in [1usize, 2, 3, 4] {
        let peak_of = |with_masks: bool| -> Option<u64> {
            let gpu = backend()?;
            let levels: Vec<(u32, f64, u32)> = factors
                .iter()
                .map(|&f| (f, if f > 1 { 1.0 } else { 0.0 }, iterations))
                .collect();
            let mut recipe = small_recipe(&levels, 2);
            let mut ch = vec![SynChannel {
                fixed: h.fixed.clone(),
                moving: h.moving.clone(),
            }];
            let contrasts: [fn(f64) -> f64; 3] = [contrast2, contrast3, contrast4];
            for contrast in contrasts.iter().take(channels - 1) {
                ch.push(SynChannel {
                    fixed: remapped(&h.fixed, *contrast),
                    moving: remapped(&h.moving, *contrast),
                });
            }
            recipe.channel_weights = vec![1.0 / channels as f64; channels];
            recipe.convergence_threshold = -1.0;
            if with_masks {
                recipe.fixed_mask = Some(mask_without_ball(h.grid, [20.0, 0.0, 10.0], 15.0));
                recipe.moving_mask = Some(mask_without_ball(
                    *h.moving.grid(),
                    [-15.0, 10.0, 0.0],
                    12.0,
                ));
            }
            run_syn(
                &gpu,
                &recipe,
                &ch,
                &AffineTransform::identity(),
                &SynOptions::default(),
            )
            .unwrap();
            Some(gpu.peak_allocated_bytes())
        };
        let Some(peak) = peak_of(masked) else { return };
        let plan =
            plan_capacity_syn_masked([n; 3], [moving_n; 3], channels, factors, &mask_dims, &caps)
                .unwrap();
        eprintln!(
            "{n}^3 {factors:?} channels {channels} masked {masked}: peak {peak} bytes = {:.1} B/voxel, planned {} bytes = {:.1} B/voxel",
            peak as f64 / n_full as f64,
            plan.level_total_bytes,
            plan.level_total_bytes as f64 / n_full as f64
        );
        failures.extend((peak > plan.level_total_bytes).then(|| {
            format!(
                "{n}^3 {factors:?} channels {channels}: peak {peak} above plan {}",
                plan.level_total_bytes
            )
        }));
        if masked {
            let Some(unmasked_peak) = peak_of(false) else {
                return;
            };
            let unmasked =
                plan_capacity_syn([n; 3], [moving_n; 3], channels, factors, &caps).unwrap();
            let added = peak.saturating_sub(unmasked_peak);
            let planned = plan.level_total_bytes - unmasked.level_total_bytes;
            eprintln!(
                "{n}^3 {factors:?} channels {channels}: the masks add {added} bytes = {:.1} B/voxel, planned {planned} bytes = {:.1} B/voxel",
                added as f64 / n_full as f64,
                planned as f64 / n_full as f64
            );
            failures.extend((added > planned).then(|| {
                format!("channels {channels}: the masks add {added} bytes, planned {planned}")
            }));
        }
        if moving_n != n {
            let fixed_only = plan_capacity_syn([n; 3], [n; 3], channels, factors, &caps).unwrap();
            failures.extend((peak <= fixed_only.level_total_bytes).then(|| {
                format!(
                    "channels {channels}: peak {peak} within the fixed-only plan {}",
                    fixed_only.level_total_bytes
                )
            }));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn the_capacity_plan_bounds_the_pool_peak_of_a_two_level_run() {
    capacity_plan_bounds_pool_peak(96, 96, &[2, 1], 1);
}

#[test]
fn the_capacity_plan_bounds_the_pool_peak_with_a_finer_moving_grid() {
    capacity_plan_bounds_pool_peak(96, 191, &[2, 1], 1);
}

#[test]
fn the_capacity_plan_bounds_the_pool_peak_of_a_masked_run() {
    capacity_plan_bounds_pool_peak_with(96, 96, &[2, 1], 1, true);
    capacity_plan_bounds_pool_peak_with(96, 191, &[2, 1], 1, true);
    for n in [24, 33, 48] {
        capacity_plan_bounds_pool_peak_with(n, n, &[8, 4, 2, 1], 1, true);
        capacity_plan_bounds_pool_peak_with(n, n, &[2, 1], 1, true);
    }
}

#[test]
#[ignore = "recalibration at 128^3; run with --ignored --release"]
fn the_capacity_plan_bounds_the_pool_peak_of_a_two_level_run_at_128() {
    capacity_plan_bounds_pool_peak(128, 128, &[2, 1], 3);
    capacity_plan_bounds_pool_peak(128, 128, &[8, 4, 2, 1], 1);
}

#[test]
fn the_capacity_plan_bounds_the_pool_peak_of_small_grids() {
    for n in [24, 33, 48] {
        capacity_plan_bounds_pool_peak(n, n, &[8, 4, 2, 1], 1);
        capacity_plan_bounds_pool_peak(n, n, &[2, 1], 1);
    }
}

#[test]
fn the_device_self_test_leaves_no_more_than_its_planned_pool_bytes() {
    let Some(gpu) = backend() else { return };
    let peak = gpu.peak_allocated_bytes();
    eprintln!("self-test pool peak {peak} bytes, planned {SELFTEST_POOL_BYTES}");
    assert!(peak > 0);
    assert!(
        peak <= SELFTEST_POOL_BYTES,
        "{peak} > {SELFTEST_POOL_BYTES}"
    );
}

#[test]
fn gpu_syn_result_applies_through_the_cpu_apply_path() {
    let Some(gpu) = backend() else { return };
    let p = phantom(20, 2.0, 1.5);
    let recipe = small_recipe(&[(2, 1.0, 3), (1, 0.0, 2)], 2);
    let ch = channels(&p, false);
    let g = run_syn(
        &gpu,
        &recipe,
        &ch,
        &AffineTransform::identity(),
        &SynOptions::default(),
    )
    .unwrap();
    let w: Volume<f64> = apply_chain(
        &p.moving,
        &p.grid,
        &g.pair.forward,
        Interpolation::Linear,
        0.0,
    )
    .unwrap();
    assert!(w.data().iter().all(|v| v.is_finite()));
    let name = gpu.name();
    assert!(name.starts_with("wgpu-f32"));
}

fn prior_of(r: &SynResult) -> PriorWarp {
    PriorWarp {
        warp: VectorField::from_displacement_field(&r.warp),
        inverse_warp: VectorField::from_displacement_field(&r.inverse_warp),
    }
}

fn two_stages<B: Backend>(
    b: &B,
    p: &Phantom,
    first: &SynRecipe,
    second: &SynRecipe,
    ch: &[SynChannel],
) -> (SynResult, SynResult) {
    let one = run_syn(b, first, ch, &p.initial, &SynOptions::default()).unwrap();
    let prior = prior_of(&one);
    let two = run_syn_from(
        b,
        second,
        ch,
        SynStart {
            affine: &p.initial,
            prior: Some(&prior),
        },
        &SynOptions::default(),
    )
    .unwrap();
    (one, two)
}

#[test]
fn two_syn_stages_match_the_cpu_reference_with_and_without_masks() {
    let Some(gpu) = backend() else { return };
    let p = phantom(24, 2.0, 2.0);
    let mut first = small_recipe(&[(2, 1.0, 12), (1, 0.0, 6)], 2);
    first.convergence_threshold = -1.0;
    let mut second = small_recipe(&[(2, 1.0, 6), (1, 0.0, 8)], 2);
    second.convergence_threshold = -1.0;
    second.gradient_step = 0.15;
    let subcortical = Grid::new([17; 3], [1.5; 3], [-6.0, -9.5, -10.25], IDENTITY3).unwrap();
    let inside_ball = |grid: Grid, centre: Vec3, radius: f64| {
        let data = (0..grid.num_voxels())
            .map(|idx| {
                let [i, j, k] = grid.ijk(idx);
                let q = grid.index_to_point([i as f64, j as f64, k as f64]);
                let d2 = (0..3).map(|a| (q[a] - centre[a]).powi(2)).sum::<f64>();
                f64::from(u8::from(d2 < radius * radius))
            })
            .collect();
        MaskVolume::from_volume(Volume::from_vec(grid, data).unwrap())
    };
    let lesion = mask_without_ball(p.grid, [-7.0, 5.5, 4.5], 5.0);
    let masked_first = SynRecipe {
        moving_mask: Some(lesion.clone()),
        ..first.clone()
    };
    let masked_second = SynRecipe {
        fixed_mask: Some(inside_ball(subcortical, [5.0, 2.5, 1.75], 11.0)),
        moving_mask: Some(lesion),
        ..second.clone()
    };
    let ch = channels(&p, false);
    for (label, a, b) in [
        ("two SyN stages", &first, &second),
        (
            "two SyN stages, lesion mask, fixed mask on the second",
            &masked_first,
            &masked_second,
        ),
    ] {
        let t = Instant::now();
        let (cpu_one, cpu) = two_stages(&CpuBackend, &p, a, b, &ch);
        let cpu_seconds = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (gpu_one, g) = two_stages(&gpu, &p, a, b, &ch);
        let gpu_seconds = t.elapsed().as_secs_f64();
        let (d1, r1) = rms_diff(&cpu_one.warp, &gpu_one.warp);
        let (warp_rms_diff, warp_rms) = rms_diff(&cpu.warp, &g.warp);
        let (inverse_rms_diff, inverse_rms) = rms_diff(&cpu.inverse_warp, &g.inverse_warp);
        let (_, change) = rms_diff(&cpu.warp, &cpu_one.warp);
        let (_, ncc_one) = warped_ncc(&p, &cpu_one, &p.moving, &p.fixed);
        let (_, ncc_cpu) = warped_ncc(&p, &cpu, &p.moving, &p.fixed);
        let (_, ncc_gpu) = warped_ncc(&p, &g, &p.moving, &p.fixed);
        let mut max_energy_diff = 0.0f64;
        for (x, y) in cpu.levels.iter().zip(&g.levels) {
            for (u, v) in x.iterations.iter().zip(&y.iterations) {
                max_energy_diff = max_energy_diff.max((u.energy - v.energy).abs());
            }
        }
        let (second_change, _) = rms_diff(&cpu.warp, &cpu_one.warp);
        eprintln!(
            "[{label}] first stage: warp RMS diff {d1:.3e} of {r1:.3e} mm; collapsed: warp RMS diff {warp_rms_diff:.3e} of {warp_rms:.3e} mm (rel {:.2e}), inverse {inverse_rms_diff:.3e} of {inverse_rms:.3e} mm; the second stage moved the warp by {second_change:.3e} mm RMS (|warp| {change:.3e}); NCC after one stage {ncc_one:.6}, two stages cpu {ncc_cpu:.6} gpu {ncc_gpu:.6}; max energy diff {max_energy_diff:.3e}; consistency cpu {:.3} gpu {:.3} voxels; time cpu {cpu_seconds:.2} s gpu {gpu_seconds:.2} s",
            warp_rms_diff / warp_rms,
            inverse_consistency(&p, &cpu),
            inverse_consistency(&p, &g),
        );
        assert!(
            warp_rms_diff / warp_rms < 0.01,
            "{label}: warp field differs"
        );
        assert!(
            inverse_rms_diff / inverse_rms < 0.01,
            "{label}: inverse field differs"
        );
        assert!((ncc_cpu - ncc_gpu).abs() < 2e-3, "{label}");
        assert!(max_energy_diff < 2e-3, "{label}");
        assert!(
            second_change > 1e-2,
            "{label}: the second stage changed nothing"
        );
        assert!(inverse_consistency(&p, &g) < 0.3, "{label}");
        let again = two_stages(&gpu, &p, a, b, &ch).1;
        assert_eq!(digest(&again), digest(&g), "{label}: not deterministic");
    }
}

fn stage_capacity<'a>(r: &'a SynRecipe, mask_dims: &'a [[usize; 3]]) -> SynStageCapacity<'a> {
    SynStageCapacity {
        shrink_factors: &r.shrink_factors,
        mask_dims,
    }
}

#[test]
fn the_capacity_plan_bounds_the_pool_peak_of_two_syn_stages() {
    let n = 96usize;
    let n_full = (n * n * n) as u64;
    let h = head([n, n, n], 2.0, 3.0);
    let caps = CapacityCaps {
        max_storage_buffer_binding_size: u64::MAX,
        memory_budget: u64::MAX,
    };
    let mask = mask_without_ball(h.grid, [20.0, 0.0, 10.0], 15.0);
    let mut failures = Vec::new();
    for (channels, masked) in [(1usize, false), (2, false), (1, true), (2, true)] {
        let mut first = small_recipe(&[(2, 1.0, 1), (1, 0.0, 1)], 2);
        first.channel_weights = vec![1.0 / channels as f64; channels];
        first.convergence_threshold = -1.0;
        let mut second = small_recipe(&[(1, 0.0, 2)], 2);
        second.channel_weights = first.channel_weights.clone();
        second.convergence_threshold = -1.0;
        if masked {
            first.moving_mask = Some(mask.clone());
            second.fixed_mask = Some(mask.clone());
            second.moving_mask = Some(mask.clone());
        }
        let mut fixed = vec![h.fixed.clone()];
        let mut moving = vec![h.moving.clone()];
        if channels == 2 {
            fixed.push(remapped(&h.fixed, contrast2));
            moving.push(remapped(&h.moving, contrast2));
        }
        let peak_of = |syn: Vec<SynRecipe>| -> Option<u64> {
            let gpu = backend()?;
            let plan = StagePlan {
                initial: InitialTransform::Identity,
                linear: Vec::new(),
                syn,
                winsorize: None,
            };
            register_stages(
                &gpu,
                &gpu,
                &plan,
                &fixed,
                &moving,
                None,
                &SynOptions::default(),
            )
            .unwrap();
            Some(gpu.peak_allocated_bytes())
        };
        let Some(one) = peak_of(vec![first.clone()]) else {
            return;
        };
        let Some(two) = peak_of(vec![first.clone(), second.clone()]) else {
            return;
        };
        let dims = |r: &SynRecipe| -> Vec<[usize; 3]> {
            r.fixed_mask
                .iter()
                .chain(&r.moving_mask)
                .map(|m| m.grid().size())
                .collect()
        };
        let (d1, d2) = (dims(&first), dims(&second));
        let plan_one = plan_capacity_syn_stages(
            [n; 3],
            [n; 3],
            channels,
            &[stage_capacity(&first, &d1)],
            &caps,
        )
        .unwrap();
        let plan_two = plan_capacity_syn_stages(
            [n; 3],
            [n; 3],
            channels,
            &[stage_capacity(&first, &d1), stage_capacity(&second, &d2)],
            &caps,
        )
        .unwrap();
        eprintln!(
            "{n}^3 channels {channels} masked {masked}: one SyN stage peak {one} bytes = {:.1} B/voxel (plan {:.1}), two SyN stages peak {two} bytes = {:.1} B/voxel (plan {:.1}), the second stage adds {:.1} B/voxel",
            one as f64 / n_full as f64,
            plan_one.level_total_bytes as f64 / n_full as f64,
            two as f64 / n_full as f64,
            plan_two.level_total_bytes as f64 / n_full as f64,
            two.saturating_sub(one) as f64 / n_full as f64,
        );
        failures.extend((two > plan_two.level_total_bytes).then(|| {
            format!(
                "channels {channels} masked {masked}: peak {two} above plan {}",
                plan_two.level_total_bytes
            )
        }));
        failures.extend((one > plan_one.level_total_bytes).then(|| {
            format!(
                "channels {channels} masked {masked}: one-stage peak {one} above plan {}",
                plan_one.level_total_bytes
            )
        }));
    }
    assert!(failures.is_empty(), "{failures:?}");
}

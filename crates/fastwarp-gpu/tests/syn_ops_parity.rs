#![allow(clippy::unwrap_used, clippy::expect_used)]
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use fastwarp_gpu::parity::{check_all, field, image, Case};
use fastwarp_gpu::{BackendSelect, GpuError, GpuField, WgpuBackend};
use fastwarp_reg::backend::Backend;
use fastwarp_reg::field_ops::{InvertParams, VectorField};
use fastwarp_reg::linalg::{Vec3, IDENTITY3};
use fastwarp_reg::metric::cc::NO_VOXEL_IN_MASKS;
use fastwarp_reg::transform::AffineTransform;
use fastwarp_reg::{Grid, Volume};
use pollster::block_on;

fn require_adapter() -> bool {
    fastwarp_gpu::device::env_var("FASTWARP_GPU_REQUIRE_ADAPTER")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn backend() -> Option<WgpuBackend> {
    match WgpuBackend::new(&BackendSelect::from_env().unwrap_or_default()) {
        Ok(b) => {
            eprintln!("using {}", b.name());
            Some(b)
        }
        Err(GpuError::NoAdapter(why)) => {
            if require_adapter() {
                panic!("no GPU adapter with FASTWARP_GPU_REQUIRE_ADAPTER set: {why}");
            }
            eprintln!("skip: no adapter ({why})");
            None
        }
        Err(e) => panic!("backend creation failed: {e}"),
    }
}

fn run_case(b: &WgpuBackend, label: &str, case: &Case) {
    let results = check_all(b, case).unwrap();
    let mut failed = Vec::new();
    for r in &results {
        eprintln!("[{label}] {}", r.line());
        if !r.ok {
            failed.push(r.line());
        }
    }
    assert!(failed.is_empty(), "{label}: {}", failed.join("\n"));
}

#[test]
fn selftest_record_is_clean_on_creation() {
    let Some(b) = backend() else { return };
    let rec = b.selftest_record();
    eprintln!(
        "attempts {} first-attempt failures {:?}",
        rec.attempts, rec.first_attempt_failures
    );
    assert!(rec.attempts >= 1);
    assert!(!rec.lines.is_empty());
}

#[test]
fn ops_match_cpu_on_an_odd_oblique_anisotropic_grid() {
    let Some(b) = backend() else { return };
    run_case(
        &b,
        "oblique 19x15x13",
        &Case::awkward([19, 15, 13], [1.3, 2.1, 1.7], true, 3),
    );
}

#[test]
fn ops_match_cpu_on_an_axis_aligned_grid_with_long_axes() {
    let Some(b) = backend() else { return };
    run_case(
        &b,
        "aligned 41x9x7",
        &Case::awkward([41, 9, 7], [1.0, 1.5, 2.5], false, 4),
    );
    run_case(
        &b,
        "aligned 7x10x45",
        &Case::awkward([7, 10, 45], [2.0, 2.0, 1.0], false, 2),
    );
    run_case(
        &b,
        "aligned 300x5x4",
        &Case::awkward([300, 5, 4], [1.0, 2.0, 2.0], false, 6),
    );
}

#[test]
fn ops_match_cpu_on_a_larger_grid_crossing_workgroup_tiles() {
    let Some(b) = backend() else { return };
    run_case(
        &b,
        "oblique 53x37x29",
        &Case::awkward([53, 37, 29], [0.9, 1.1, 1.4], true, 4),
    );
}

#[test]
fn ops_match_cpu_on_a_degenerate_thin_grid() {
    let Some(b) = backend() else { return };
    run_case(
        &b,
        "thin 12x9x3",
        &Case::awkward([12, 9, 3], [1.0, 1.0, 3.0], false, 1),
    );
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn gpu_results_are_bit_identical_across_runs() {
    let Some(b) = backend() else { return };
    let case = Case::awkward([29, 23, 19], [1.2, 1.6, 2.0], true, 3);
    let grid = case.grid().unwrap();
    let fixed = block_on(b.upload_image(&image(&grid, 3, 0.2))).unwrap();
    let moving = block_on(b.upload_image(&image(&grid, 4, 0.2))).unwrap();
    let f0 = field(&grid, 2.0, 5);
    let run = || {
        let f = block_on(b.upload_field(&f0)).unwrap();
        let w = block_on(b.warp_through_field(
            &moving,
            &f,
            None,
            Some(&AffineTransform::from_translation([0.4, -0.3, 0.2])),
        ))
        .unwrap();
        let (value, d) = block_on(b.cc_value_derivative(&fixed, &w, 3)).unwrap();
        let s = block_on(b.smooth_field(d, 3.0)).unwrap();
        let m = block_on(b.max_scaled_norm(&s)).unwrap();
        let l = block_on(b.l2_norm(&s)).unwrap();
        let c = block_on(b.compose(&s, &f)).unwrap();
        let (inv, stats) = block_on(b.invert(&c, f, &InvertParams::SYN)).unwrap();
        let down = block_on(b.download_field(&inv)).unwrap();
        let mut flat: Vec<f64> = down.data().iter().flat_map(|v| v.iter().copied()).collect();
        flat.extend([
            value,
            m,
            l,
            f64::from(stats.iterations),
            stats.mean_error_norm,
        ]);
        flat.extend(b.download_image(&w).unwrap());
        bits(&flat)
    };
    let a = run();
    let c = run();
    assert_eq!(a, c);
}

#[test]
fn an_inversion_longer_than_one_submission_runs_every_step_and_matches_the_cpu() {
    let Some(b) = backend() else { return };
    let cpu = fastwarp_reg::cpu::CpuBackend;
    let grid = Case::awkward([12, 10, 8], [1.0, 1.2, 1.5], false, 2)
        .grid()
        .unwrap();
    let f0 = field(&grid, 1.5, 7);
    let params = InvertParams {
        max_iterations: 1030,
        mean_error_tolerance: 0.0,
        max_error_tolerance: 0.0,
        enforce_boundary_condition: true,
    };
    let f = block_on(b.upload_field(&f0)).unwrap();
    let (inverse, stats) =
        block_on(b.invert(&f, block_on(b.zero_field(&grid)).unwrap(), &params)).unwrap();
    let (cpu_inverse, cpu_stats) = cpu
        .invert(&f0, cpu.zero_field(&grid).unwrap(), &params)
        .unwrap();
    assert_eq!(stats.iterations, 1030);
    assert_eq!(cpu_stats.iterations, 1030);
    let gpu_inverse = block_on(b.download_field(&inverse)).unwrap();
    let worst = gpu_inverse
        .data()
        .iter()
        .zip(cpu_inverse.data())
        .flat_map(|(g, c)| (0..3).map(move |k| (g[k] - c[k]).abs()))
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-3, "{worst}");
}

#[test]
fn an_inversion_dispatched_indirectly_matches_the_cpu() {
    let Some(b) = backend() else { return };
    let cpu = fastwarp_reg::cpu::CpuBackend;
    let grid = Case::awkward([136, 128, 72], [1.0, 1.1, 1.3], true, 5)
        .grid()
        .unwrap();
    let f0 = field(&grid, 1.5, 11);
    let every_step = InvertParams {
        max_iterations: 12,
        mean_error_tolerance: 0.0,
        max_error_tolerance: 0.0,
        enforce_boundary_condition: true,
    };
    for params in [InvertParams::SYN, every_step] {
        let f = block_on(b.upload_field(&f0)).unwrap();
        let (inverse, stats) =
            block_on(b.invert(&f, block_on(b.zero_field(&grid)).unwrap(), &params)).unwrap();
        let (cpu_inverse, cpu_stats) = cpu
            .invert(&f0, cpu.zero_field(&grid).unwrap(), &params)
            .unwrap();
        assert_eq!(stats.iterations, cpu_stats.iterations);
        assert!((stats.max_error_norm - cpu_stats.max_error_norm).abs() < 1e-3);
        assert!((stats.mean_error_norm - cpu_stats.mean_error_norm).abs() < 1e-3);
        let gpu_inverse = block_on(b.download_field(&inverse)).unwrap();
        let worst = gpu_inverse
            .data()
            .iter()
            .zip(cpu_inverse.data())
            .flat_map(|(g, c)| (0..3).map(move |k| (g[k] - c[k]).abs()))
            .fold(0.0f64, f64::max);
        assert!(worst < 2e-2, "{worst}");
    }
}

#[test]
fn the_indirect_and_direct_inversions_are_bit_identical() {
    let Some(b) = backend() else { return };
    let grid = Case::awkward([136, 128, 72], [1.0, 1.1, 1.3], true, 5)
        .grid()
        .unwrap();
    let f0 = field(&grid, 1.5, 11);
    let every_step = InvertParams {
        max_iterations: 12,
        mean_error_tolerance: 0.0,
        max_error_tolerance: 0.0,
        enforce_boundary_condition: true,
    };
    let run = |params: &InvertParams| {
        let f = block_on(b.upload_field(&f0)).unwrap();
        let (inverse, stats) =
            block_on(b.invert(&f, block_on(b.zero_field(&grid)).unwrap(), params)).unwrap();
        let down = block_on(b.download_field(&inverse)).unwrap();
        let mut flat: Vec<f64> = down.data().iter().flat_map(|v| v.iter().copied()).collect();
        flat.extend([
            f64::from(stats.iterations),
            stats.mean_error_norm,
            stats.max_error_norm,
        ]);
        bits(&flat)
    };
    for params in [InvertParams::SYN, every_step] {
        let indirect = run(&params);
        b.set_invert_indirect_min_groups(u32::MAX);
        let direct = run(&params);
        b.set_invert_indirect_min_groups(0);
        let always = run(&params);
        b.set_invert_indirect_min_groups(2048);
        assert_eq!(indirect, direct);
        assert_eq!(indirect, always);
    }
}

#[test]
fn cc_of_an_image_with_itself_is_minus_one_with_zero_derivative() {
    let Some(b) = backend() else { return };
    let grid = Case::awkward([13, 11, 9], [1.0, 1.0, 1.0], false, 2)
        .grid()
        .unwrap();
    let img = block_on(b.upload_image(&image(&grid, 9, 0.3))).unwrap();
    let (value, d) = block_on(b.cc_value_derivative(&img, &img, 2)).unwrap();
    assert!((value + 1.0).abs() < 1e-5, "{value}");
    let d = block_on(b.download_field(&d)).unwrap();
    let worst = d
        .data()
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0f64, |a, x| a.max(x.abs()));
    assert!(worst < 1e-2, "{worst}");
}

#[test]
fn the_cc_pair_matches_the_cpu_in_both_directions_and_two_gpu_calls_bit_for_bit() {
    let Some(b) = backend() else { return };
    let cpu = fastwarp_reg::cpu::CpuBackend;
    for (dims, spacing, radius) in [
        ([29, 23, 19], [1.2, 1.6, 2.0], 3usize),
        ([13, 11, 9], [1.0, 1.0, 1.0], 2usize),
    ] {
        let grid = Case::awkward(dims, spacing, true, 3).grid().unwrap();
        let (fa, fb) = (image(&grid, 3, 0.2), image(&grid, 4, 0.2));
        let (ga, gb) = (
            block_on(b.upload_image(&fa)).unwrap(),
            block_on(b.upload_image(&fb)).unwrap(),
        );
        let ((v1, d1), (v2, d2)) = block_on(b.cc_value_derivative_pair(&ga, &gb, radius)).unwrap();
        let (s1, e1) = block_on(b.cc_value_derivative(&ga, &gb, radius)).unwrap();
        let (s2, e2) = block_on(b.cc_value_derivative(&gb, &ga, radius)).unwrap();
        assert_eq!(v1.to_bits(), s1.to_bits());
        assert_eq!(v2.to_bits(), s2.to_bits());
        let down = |f| block_on(b.download_field(&f)).unwrap();
        assert_eq!(down(d1).data(), down(e1).data());
        let swapped = down(d2);
        assert_eq!(swapped.data(), down(e2).data());
        let ((cv1, _), (cv2, cd2)) = cpu.cc_value_derivative_pair(&fa, &fb, radius).unwrap();
        assert!((v1 - cv1).abs() < 1e-4 * cv1.abs().max(1e-6), "{v1} {cv1}");
        assert!((v2 - cv2).abs() < 1e-4 * cv2.abs().max(1e-6), "{v2} {cv2}");
        let range = cd2
            .data()
            .iter()
            .flat_map(|v| v.iter())
            .fold(0.0f64, |a, x| a.max(x.abs()));
        let worst = swapped
            .data()
            .iter()
            .zip(cd2.data())
            .enumerate()
            .filter(|(i, _)| {
                let [x, y, z] = grid.ijk(*i);
                let [nx, ny, nz] = grid.size();
                x > 0 && y > 0 && z > 0 && x + 1 < nx && y + 1 < ny && z + 1 < nz
            })
            .flat_map(|(_, (g, c))| (0..3).map(move |k| (g[k] - c[k]).abs()))
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-3 * range,
            "swapped derivative {worst} of {range}"
        );
    }
}

#[test]
fn paired_scalar_operations_match_the_single_ones_bit_for_bit_in_one_read_back() {
    let Some(b) = backend() else { return };
    let grid = Case::awkward([29, 23, 19], [1.2, 1.6, 2.0], true, 3)
        .grid()
        .unwrap();
    let fixed = block_on(b.upload_image(&image(&grid, 3, 0.2))).unwrap();
    let moving = block_on(b.upload_image(&image(&grid, 4, 0.2))).unwrap();
    let field_bits = |f: &GpuField| -> Vec<u64> {
        let down = block_on(b.download_field(f)).unwrap();
        let flat: Vec<f64> = down.data().iter().flat_map(|v| v.iter().copied()).collect();
        bits(&flat)
    };
    let reads = || b.stats().scalar_readbacks;

    let before = reads();
    let (((v1, d1), (v2, d2)), [n1, n2]) =
        block_on(b.cc_value_derivative_pair_with_norms(&fixed, &moving, 3)).unwrap();
    assert_eq!(reads() - before, 1);
    let ((s1, e1), (s2, e2)) = block_on(b.cc_value_derivative_pair(&fixed, &moving, 3)).unwrap();
    let (m1, m2) = (
        block_on(b.l2_norm(&e1)).unwrap(),
        block_on(b.l2_norm(&e2)).unwrap(),
    );
    assert_eq!(bits(&[v1, v2, n1, n2]), bits(&[s1, s2, m1, m2]));
    assert_eq!(field_bits(&d1), field_bits(&e1));
    assert_eq!(field_bits(&d2), field_bits(&e2));

    let fixed_side = block_on(b.smooth_field(d1, 3.0)).unwrap();
    let moving_side = block_on(b.smooth_field(d2, 3.0)).unwrap();
    let before = reads();
    let (max1, max2) = block_on(b.max_scaled_norm_pair(&fixed_side, &moving_side)).unwrap();
    assert_eq!(reads() - before, 1);
    let single = [
        block_on(b.max_scaled_norm(&fixed_side)).unwrap(),
        block_on(b.max_scaled_norm(&moving_side)).unwrap(),
    ];
    assert_eq!(bits(&[max1, max2]), bits(&single));

    let f0 = field(&grid, 2.0, 5);
    let g0 = field(&grid, 0.5, 6);
    let (f, g) = (
        block_on(b.upload_field(&f0)).unwrap(),
        block_on(b.upload_field(&g0)).unwrap(),
    );
    let before = reads();
    let ((inverse, inverse_stats), (forward, forward_stats)) =
        block_on(b.invert_pair(f, g, &InvertParams::SYN)).unwrap();
    assert_eq!(reads() - before, 1);
    let (f, g) = (
        block_on(b.upload_field(&f0)).unwrap(),
        block_on(b.upload_field(&g0)).unwrap(),
    );
    let (single_inverse, single_inverse_stats) =
        block_on(b.invert(&f, g, &InvertParams::SYN)).unwrap();
    let (single_forward, single_forward_stats) =
        block_on(b.invert(&single_inverse, f, &InvertParams::SYN)).unwrap();
    assert!(inverse_stats.iterations > 0 && forward_stats.iterations > 0);
    assert_eq!(inverse_stats, single_inverse_stats);
    assert_eq!(forward_stats, single_forward_stats);
    assert_eq!(field_bits(&inverse), field_bits(&single_inverse));
    assert_eq!(field_bits(&forward), field_bits(&single_forward));
}

fn ball_mask(grid: &Grid, centre: Vec3, radius: f64, cut: Option<(usize, f64)>) -> Volume<f64> {
    let data = (0..grid.num_voxels())
        .map(|idx| {
            let [i, j, k] = grid.ijk(idx);
            let p = grid.index_to_point([i as f64, j as f64, k as f64]);
            let d = ((p[0] - centre[0]).powi(2)
                + (p[1] - centre[1]).powi(2)
                + (p[2] - centre[2]).powi(2))
            .sqrt();
            let out = cut.is_some_and(|(axis, at)| p[axis] > at);
            if d < radius && !out {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Volume::from_vec(*grid, data).unwrap()
}

fn grid_centre(grid: &Grid) -> Vec3 {
    let [nx, ny, nz] = grid.size();
    grid.index_to_point([
        (nx - 1) as f64 / 2.0,
        (ny - 1) as f64 / 2.0,
        (nz - 1) as f64 / 2.0,
    ])
}

fn mismatches(a: &[f64], b: &[f64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
fn the_nearest_mask_warp_matches_the_cpu_and_multiplies_into_a_product() {
    let Some(b) = backend() else { return };
    let cpu = fastwarp_reg::cpu::CpuBackend;
    let case = Case::awkward([29, 23, 19], [1.2, 1.6, 2.0], true, 3);
    let grid = case.grid().unwrap();
    let mask_grid = case.other_grid().unwrap();
    let c = grid_centre(&grid);
    let first = ball_mask(&mask_grid, c, 13.0, Some((1, c[1] + 4.0)));
    let second = ball_mask(&mask_grid, [c[0] + 3.0, c[1], c[2] - 2.0], 12.0, None);
    let f0 = field(&grid, 2.0, 5);
    let g0 = field(&grid, 1.5, 8);
    let affine = AffineTransform::new(
        [[1.03, 0.02, 0.0], [-0.02, 0.98, 0.01], [0.0, 0.01, 1.01]],
        [0.7, -0.4, 0.3],
        c,
    );
    let (gf, gg) = (
        block_on(b.upload_field(&f0)).unwrap(),
        block_on(b.upload_field(&g0)).unwrap(),
    );
    let (m1, m2) = (
        block_on(b.upload_image(&first)).unwrap(),
        block_on(b.upload_image(&second)).unwrap(),
    );
    let single = block_on(b.warp_mask_through_field(&m1, &gf, None, None, None)).unwrap();
    let single_gpu = b.download_image(&single).unwrap();
    let product =
        block_on(b.warp_mask_through_field(&m2, &gg, None, Some(&affine), Some(single))).unwrap();
    let product_gpu = b.download_image(&product).unwrap();
    let single_cpu = cpu
        .warp_mask_through_field(&first, &f0, None, None, None)
        .unwrap();
    let product_cpu = cpu
        .warp_mask_through_field(&second, &g0, None, Some(&affine), Some(single_cpu.clone()))
        .unwrap();
    let n = grid.num_voxels();
    for (label, gpu, reference) in [
        ("single", &single_gpu, single_cpu.data()),
        ("product", &product_gpu, product_cpu.data()),
    ] {
        let inside = reference.iter().filter(|v| **v == 1.0).count();
        let wrong = mismatches(gpu, reference);
        eprintln!("{label}: {inside} of {n} voxels inside, {wrong} differ from the CPU");
        assert!(gpu.iter().all(|v| *v == 0.0 || *v == 1.0));
        assert!(inside > n / 10 && inside < 9 * n / 10, "{label}: {inside}");
        assert_eq!(wrong, 0, "{label}: {wrong} of {n} voxels differ");
    }
    let level = Grid::new([12, 10, 9], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
    let coarse = Grid::new([6, 5, 5], [2.0; 3], [0.0; 3], IDENTITY3).unwrap();
    let stripes = Volume::from_vec(
        coarse,
        (0..coarse.num_voxels())
            .map(|idx| {
                let [i, j, k] = coarse.ijk(idx);
                f64::from(u8::from((i + 2 * j + k) % 3 == 0))
            })
            .collect(),
    )
    .unwrap();
    let zero = VectorField::zeros(level);
    let want = cpu
        .warp_mask_through_field(&stripes, &zero, None, None, None)
        .unwrap();
    let got = block_on(b.warp_mask_through_field(
        &block_on(b.upload_image(&stripes)).unwrap(),
        &block_on(b.upload_field(&zero)).unwrap(),
        None,
        None,
        None,
    ))
    .unwrap();
    let got = b.download_image(&got).unwrap();
    assert_eq!(got, want.data(), "half-voxel points round up on both");
    assert_eq!(want.data()[level.linear_index(1, 0, 0)], 0.0);
    assert_eq!(want.data()[level.linear_index(5, 0, 0)], 1.0);
    let fine = Grid::new([40, 36, 32], [1.1; 3], [-21.37, -19.03, -17.11], IDENTITY3).unwrap();
    let coarse = Grid::new([20, 18, 16], [2.2; 3], [-20.82, -18.48, -16.56], IDENTITY3).unwrap();
    let ball = ball_mask(&fine, grid_centre(&fine), 13.0, Some((2, 4.0)));
    let zero = VectorField::zeros(coarse);
    let want = cpu
        .warp_mask_through_field(&ball, &zero, None, None, None)
        .unwrap();
    let got = block_on(b.warp_mask_through_field(
        &block_on(b.upload_image(&ball)).unwrap(),
        &block_on(b.upload_field(&zero)).unwrap(),
        None,
        None,
        None,
    ))
    .unwrap();
    let got = b.download_image(&got).unwrap();
    let wrong = mismatches(&got, want.data());
    let inside = want.data().iter().filter(|v| **v == 1.0).count();
    eprintln!(
        "ties in floating point (a 2:1 grid with inexact origins): {wrong} of {} voxels differ, {inside} inside",
        coarse.num_voxels()
    );
    assert!(inside > 500, "{inside}");
    assert!(
        wrong * 100 <= inside * 25,
        "{wrong} of {inside} mask voxels round differently"
    );
    let [nx, ny, nz] = coarse.size();
    for idx in (0..coarse.num_voxels()).filter(|i| got[*i] != want.data()[*i]) {
        let [i, j, k] = coarse.ijk(idx);
        let edge = [[1i64, 0, 0], [0, 1, 0], [0, 0, 1]].iter().any(|d| {
            [-1i64, 1].iter().any(|s| {
                let n = [
                    i as i64 + s * d[0],
                    j as i64 + s * d[1],
                    k as i64 + s * d[2],
                ];
                let inside_grid = n[0] >= 0
                    && n[1] >= 0
                    && n[2] >= 0
                    && n[0] < nx as i64
                    && n[1] < ny as i64
                    && n[2] < nz as i64;
                inside_grid
                    && want.data()[coarse.linear_index(n[0] as usize, n[1] as usize, n[2] as usize)]
                        != want.data()[idx]
            })
        });
        assert!(edge, "voxel {i},{j},{k} differs away from the mask's edge");
    }
}

#[test]
fn the_masked_cc_pair_matches_the_cpu_and_an_all_ones_mask_is_the_unmasked_pair() {
    let Some(b) = backend() else { return };
    let cpu = fastwarp_reg::cpu::CpuBackend;
    for (dims, spacing, radius) in [
        ([29, 23, 19], [1.2, 1.6, 2.0], 3usize),
        ([13, 11, 9], [1.0, 1.0, 1.0], 2usize),
    ] {
        let grid = Case::awkward(dims, spacing, true, 3).grid().unwrap();
        let n = grid.num_voxels();
        let (fa, fb) = (image(&grid, 3, 0.2), image(&grid, 4, 0.2));
        let c = grid_centre(&grid);
        let hole = ball_mask(&grid, c, 0.25 * dims[0] as f64 * spacing[0], None);
        let valid = Volume::from_vec(
            grid,
            hole.data()
                .iter()
                .enumerate()
                .map(|(idx, h)| {
                    let [i, _, _] = grid.ijk(idx);
                    if *h == 0.0 && i + 2 < dims[0] {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
        .unwrap();
        let (ga, gb, gv) = (
            block_on(b.upload_image(&fa)).unwrap(),
            block_on(b.upload_image(&fb)).unwrap(),
            block_on(b.upload_image(&valid)).unwrap(),
        );
        let field_values = |f: &GpuField| -> Vec<f64> {
            block_on(b.download_field(f))
                .unwrap()
                .data()
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect()
        };
        let before = b.stats().scalar_readbacks;
        let (((v1, d1), (v2, d2)), [n1, n2]) =
            block_on(b.cc_value_derivative_pair_masked_with_norms(&ga, &gb, &gv, radius)).unwrap();
        assert_eq!(b.stats().scalar_readbacks - before, 1);
        let ((s1, e1), (s2, e2)) =
            block_on(b.cc_value_derivative_pair_masked(&ga, &gb, &gv, radius)).unwrap();
        let (m1, m2) = (
            block_on(b.l2_norm(&e1)).unwrap(),
            block_on(b.l2_norm(&e2)).unwrap(),
        );
        assert_eq!(bits(&[v1, v2, n1, n2]), bits(&[s1, s2, m1, m2]));
        assert_eq!(bits(&field_values(&d1)), bits(&field_values(&e1)));
        assert_eq!(bits(&field_values(&d2)), bits(&field_values(&e2)));
        let ((cv1, cd1), (cv2, cd2)) = cpu
            .cc_value_derivative_pair_masked(&fa, &fb, &valid, radius)
            .unwrap();
        assert!((v1 - cv1).abs() < 1e-4 * cv1.abs().max(1e-6), "{v1} {cv1}");
        assert!((v2 - cv2).abs() < 1e-4 * cv2.abs().max(1e-6), "{v2} {cv2}");
        for (gpu, reference) in [(d1, &cd1), (d2, &cd2)] {
            let gpu = block_on(b.download_field(&gpu)).unwrap();
            let range = reference
                .data()
                .iter()
                .flat_map(|v| v.iter())
                .fold(0.0f64, |a, x| a.max(x.abs()));
            assert!(range > 0.0);
            let mut worst = 0.0f64;
            for (idx, (g, r)) in gpu.data().iter().zip(reference.data()).enumerate() {
                if valid.data()[idx] == 0.0 {
                    assert_eq!(g, &[0.0; 3], "voxel {idx} is outside the mask");
                    continue;
                }
                let [x, y, z] = grid.ijk(idx);
                let [nx, ny, nz] = grid.size();
                if x > 0 && y > 0 && z > 0 && x + 1 < nx && y + 1 < ny && z + 1 < nz {
                    for k in 0..3 {
                        worst = worst.max((g[k] - r[k]).abs());
                    }
                }
            }
            eprintln!("{dims:?}: masked derivative max diff {worst:.3e} of {range:.3e}");
            assert!(worst < 2e-4 * range, "masked derivative {worst} of {range}");
        }
        let ones =
            block_on(b.upload_image(&Volume::from_vec(grid, vec![1.0; n]).unwrap())).unwrap();
        let ((o1, od1), (o2, od2)) =
            block_on(b.cc_value_derivative_pair_masked(&ga, &gb, &ones, radius)).unwrap();
        let ((u1, ud1), (u2, ud2)) =
            block_on(b.cc_value_derivative_pair(&ga, &gb, radius)).unwrap();
        for (masked, plain) in [(o1, u1), (o2, u2)] {
            assert!(
                (masked - plain).abs() <= 1e-6 * plain.abs(),
                "{masked} {plain}"
            );
        }
        for (masked, plain) in [(od1, ud1), (od2, ud2)] {
            let (m, p) = (field_values(&masked), field_values(&plain));
            let range = p.iter().fold(0.0f64, |a, x| a.max(x.abs()));
            let worst = m
                .iter()
                .zip(&p)
                .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
            let differ = m.iter().zip(&p).filter(|(x, y)| x != y).count();
            eprintln!(
                "{dims:?}: all-ones mask vs unmasked derivative: {differ} of {} values differ, max {worst:.3e} of {range:.3e}",
                m.len()
            );
            assert!(worst <= 1e-5 * range, "{worst} of {range}");
        }
        let zeros = block_on(b.upload_image(&Volume::zeros(grid))).unwrap();
        let err = block_on(b.cc_value_derivative_pair_masked(&ga, &gb, &zeros, radius))
            .err()
            .expect("an empty mask is refused");
        assert!(err.to_string().contains(NO_VOXEL_IN_MASKS), "{err}");
    }
}

#[test]
fn zero_field_is_zero_after_pool_reuse() {
    let Some(b) = backend() else { return };
    let grid = Case::awkward([10, 9, 8], [1.0, 1.0, 1.0], false, 2)
        .grid()
        .unwrap();
    {
        let f = block_on(b.upload_field(&field(&grid, 3.0, 1))).unwrap();
        drop(f);
    }
    let z = block_on(b.zero_field(&grid)).unwrap();
    let down = block_on(b.download_field(&z)).unwrap();
    assert!(down.data().iter().all(|v| v == &[0.0, 0.0, 0.0]));
}

#[test]
fn zero_field_trims_the_pool_once_per_grid_until_new_data_arrives() {
    let Some(b) = backend() else { return };
    let cube = |n: usize| {
        fastwarp_reg::Grid::new([n; 3], [1.0; 3], [0.0; 3], fastwarp_reg::linalg::IDENTITY3)
            .unwrap()
    };
    let (large, small) = (cube(64), cube(16));
    let large_field_bytes = 64 * 64 * 64 * 12;
    let small_field_bytes = 16 * 16 * 16 * 12;
    let large_field = block_on(b.zero_field(&large)).unwrap();
    let _first = block_on(b.zero_field(&small)).unwrap();
    drop(block_on(b.copy_field(&large_field)).unwrap());
    let with_idle_copy = b.allocated_bytes();
    let _second = block_on(b.zero_field(&small)).unwrap();
    assert_eq!(b.allocated_bytes(), with_idle_copy + small_field_bytes);
    let _image = block_on(b.upload_image(&image(&small, 1, 0.1))).unwrap();
    let after_upload = b.allocated_bytes();
    let _third = block_on(b.zero_field(&small)).unwrap();
    assert_eq!(
        b.allocated_bytes(),
        after_upload - large_field_bytes + small_field_bytes
    );
    let resets: [(&str, &dyn Fn()); 3] = [
        ("resample_field", &|| {
            drop(block_on(b.resample_field(&large_field, &small)).unwrap())
        }),
        ("upload_field", &|| {
            drop(block_on(b.upload_field(&field(&small, 1.0, 2))).unwrap())
        }),
        ("release_idle", &|| {
            b.release_idle();
        }),
    ];
    for (label, reset) in resets {
        let _trimmed = block_on(b.zero_field(&small)).unwrap();
        reset();
        drop(block_on(b.copy_field(&large_field)).unwrap());
        let with_idle_copy = b.allocated_bytes();
        let _after_reset = block_on(b.zero_field(&small)).unwrap();
        assert!(
            b.allocated_bytes() + large_field_bytes <= with_idle_copy + small_field_bytes,
            "{label} did not make zero_field trim again"
        );
    }
}

#[test]
fn idle_pooled_buffers_are_released_on_request() {
    let Some(b) = backend() else { return };
    let grid = fastwarp_reg::Grid::new(
        [16, 16, 16],
        [1.0; 3],
        [0.0; 3],
        fastwarp_reg::linalg::IDENTITY3,
    )
    .unwrap();
    let field_bytes = 16 * 16 * 16 * 12;
    {
        let f = block_on(b.zero_field(&grid)).unwrap();
        let _g = block_on(b.copy_field(&f)).unwrap();
    }
    let before = b.allocated_bytes();
    let freed = b.release_idle();
    assert!(freed >= 2 * field_bytes, "{freed}");
    assert_eq!(b.allocated_bytes(), before - freed);
    let kept = block_on(b.zero_field(&grid)).unwrap();
    assert_eq!(b.release_idle(), 0);
    drop(kept);
    assert_eq!(b.release_idle(), field_bytes);
    assert_eq!(b.allocated_bytes(), 0);
}

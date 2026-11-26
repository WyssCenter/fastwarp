#![allow(clippy::unwrap_used, clippy::expect_used)]
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

mod common;

use fastwarp_gpu::parity::{image, Case};
use fastwarp_gpu::{GpuField, WgpuBackend};
use fastwarp_reg::backend::Backend;
use fastwarp_reg::cpu::CpuBackend;
use fastwarp_reg::field_ops::VectorField;
use fastwarp_reg::metric::cc::NO_VOXEL_IN_MASKS;
use fastwarp_reg::{Grid, Volume};
use pollster::block_on;

fn f32_image(grid: &Grid, seed: u64, contrast: bool) -> Volume<f64> {
    let base = image(grid, seed, 0.2);
    let data = base
        .data()
        .iter()
        .map(|v| {
            let w = if contrast { 2.0 - 1.5 * v * v } else { *v };
            f64::from(w as f32)
        })
        .collect();
    Volume::from_vec(*grid, data).unwrap()
}

fn masks(grid: &Grid) -> (Volume<f64>, Volume<f64>) {
    let [nx, ny, nz] = grid.size();
    let make = |f: &dyn Fn(usize, usize, usize) -> bool| {
        let data = (0..grid.num_voxels())
            .map(|idx| {
                let [i, j, k] = grid.ijk(idx);
                f64::from(u8::from(f(i, j, k)))
            })
            .collect();
        Volume::from_vec(*grid, data).unwrap()
    };
    let c = [nx as f64 / 2.0, ny as f64 / 2.0, nz as f64 / 2.0];
    let r = 0.42 * nx.min(ny).min(nz) as f64;
    let ball = make(&|i, j, k| {
        let d = (i as f64 - c[0]).powi(2) + (j as f64 - c[1]).powi(2) + (k as f64 - c[2]).powi(2);
        d.sqrt() < r
    });
    let half = make(&|i, _, _| i + 3 < nx);
    (ball, half)
}

fn download(b: &WgpuBackend, f: &GpuField) -> VectorField {
    block_on(b.download_field(f)).unwrap()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn flat(f: &VectorField) -> Vec<f64> {
    f.data().iter().flat_map(|v| v.iter().copied()).collect()
}

fn compare(label: &str, value: f64, want: f64, gpu: &VectorField, cpu: &VectorField) {
    assert!(
        (value - want).abs() <= 1e-6 * want.abs(),
        "{label}: value {value} vs CPU {want}"
    );
    let range = cpu
        .data()
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0f64, |a, x| a.max(x.abs()));
    assert!(range > 0.0, "{label}: zero CPU derivative");
    let n = cpu.data().len();
    let mut far = 0usize;
    let mut worst = 0.0f64;
    let mut zero_mismatch = 0usize;
    for (g, c) in gpu.data().iter().zip(cpu.data()) {
        if (*g == [0.0; 3]) != (*c == [0.0; 3]) {
            zero_mismatch += 1;
        }
        let d = (0..3).map(|k| (g[k] - c[k]).abs()).fold(0.0f64, f64::max);
        if d > 1e-3 * range {
            far += 1;
        } else {
            worst = worst.max(d);
        }
    }
    let mut rel: Vec<f64> = gpu
        .data()
        .iter()
        .zip(cpu.data())
        .map(|(g, c)| (0..3).map(|k| (g[k] - c[k]).abs()).fold(0.0f64, f64::max) / range)
        .collect();
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "{label}: relative derivative error quantiles 50% {:.2e} 90% {:.2e} 99% {:.2e} \
         99.9% {:.2e} max {:.2e}",
        rel[n / 2],
        rel[n * 9 / 10],
        rel[n * 99 / 100],
        rel[n * 999 / 1000],
        rel[n - 1]
    );
    eprintln!(
        "{label}: value {value:.9} vs {want:.9}; derivative: {far} of {n} voxels beyond 1e-3 of \
         {range:.3e}, others within {worst:.3e}; {zero_mismatch} zero/non-zero mismatches"
    );
    assert!(far * 200 <= n, "{label}: {far} of {n} voxels differ");
    assert!(worst <= 1e-4 * range, "{label}: {worst} of {range}");
    assert!(
        zero_mismatch * 200 <= n,
        "{label}: {zero_mismatch} zero mismatches"
    );
}

fn cases() -> Vec<Grid> {
    vec![
        Case::awkward([29, 23, 19], [1.2, 1.6, 2.0], true, 3)
            .grid()
            .unwrap(),
        Case::awkward([13, 11, 9], [1.0, 1.0, 1.0], false, 2)
            .grid()
            .unwrap(),
        Case::awkward([83, 71, 61], [0.9, 1.1, 1.4], true, 2)
            .grid()
            .unwrap(),
    ]
}

#[test]
fn the_mattes_pair_matches_the_cpu_in_both_directions() {
    let Some(b) = common::backend() else { return };
    let cpu = CpuBackend;
    for grid in cases() {
        let dims = grid.size();
        let (fa, fb) = (f32_image(&grid, 3, false), f32_image(&grid, 4, true));
        let (ga, gb) = (
            block_on(b.upload_image(&fa)).unwrap(),
            block_on(b.upload_image(&fb)).unwrap(),
        );
        for bins in [32usize, 12] {
            let ((v1, d1), (v2, d2)) =
                block_on(b.mi_value_derivative_pair(&ga, &gb, bins)).unwrap();
            let ((c1, e1), (c2, e2)) = cpu.mi_value_derivative_pair(&fa, &fb, bins).unwrap();
            compare(
                &format!("{dims:?} bins {bins} forward"),
                v1,
                c1,
                &download(&b, &d1),
                &e1,
            );
            compare(
                &format!("{dims:?} bins {bins} backward"),
                v2,
                c2,
                &download(&b, &d2),
                &e2,
            );
        }
        let (fm, mm) = masks(&grid);
        let (gfm, gmm) = (
            block_on(b.upload_image(&fm)).unwrap(),
            block_on(b.upload_image(&mm)).unwrap(),
        );
        let ((v1, d1), (v2, d2)) =
            block_on(b.mi_value_derivative_pair_masked(&ga, &gb, &gfm, &gmm, 32)).unwrap();
        let ((c1, e1), (c2, e2)) = cpu
            .mi_value_derivative_pair_masked(&fa, &fb, &fm, &mm, 32)
            .unwrap();
        let (g1, g2) = (download(&b, &d1), download(&b, &d2));
        for (idx, (a, c)) in g1.data().iter().zip(g2.data()).enumerate() {
            if fm.data()[idx] == 0.0 || mm.data()[idx] == 0.0 {
                assert_eq!(
                    (a, c),
                    (&[0.0; 3], &[0.0; 3]),
                    "voxel {idx} is outside a mask"
                );
            }
        }
        compare(&format!("{dims:?} masked forward"), v1, c1, &g1, &e1);
        compare(&format!("{dims:?} masked backward"), v2, c2, &g2, &e2);
    }
}

#[test]
fn the_mattes_pair_is_bit_identical_across_runs_and_with_norms() {
    let Some(b) = common::backend() else { return };
    let grid = cases()[0];
    let (fa, fb) = (f32_image(&grid, 5, false), f32_image(&grid, 6, true));
    let (fm, mm) = masks(&grid);
    let up = |v: &Volume<f64>| block_on(b.upload_image(v)).unwrap();
    let (ga, gb, gfm, gmm) = (up(&fa), up(&fb), up(&fm), up(&mm));
    for masked in [false, true] {
        let run = || {
            let before = b.stats().scalar_readbacks;
            let ((v1, d1), (v2, d2)) = if masked {
                block_on(b.mi_value_derivative_pair_masked(&ga, &gb, &gfm, &gmm, 32)).unwrap()
            } else {
                block_on(b.mi_value_derivative_pair(&ga, &gb, 32)).unwrap()
            };
            assert_eq!(b.stats().scalar_readbacks - before, 2);
            (v1, v2, flat(&download(&b, &d1)), flat(&download(&b, &d2)))
        };
        let first = run();
        let second = run();
        assert_eq!(bits(&[first.0, first.1]), bits(&[second.0, second.1]));
        assert_eq!(bits(&first.2), bits(&second.2));
        assert_eq!(bits(&first.3), bits(&second.3));
        let before = b.stats().scalar_readbacks;
        let (((w1, e1), (w2, e2)), [n1, n2]) = if masked {
            block_on(b.mi_value_derivative_pair_masked_with_norms(&ga, &gb, &gfm, &gmm, 32))
                .unwrap()
        } else {
            block_on(b.mi_value_derivative_pair_with_norms(&ga, &gb, 32)).unwrap()
        };
        assert_eq!(b.stats().scalar_readbacks - before, 3);
        assert_eq!(bits(&[w1, w2]), bits(&[first.0, first.1]));
        assert_eq!(bits(&flat(&download(&b, &e1))), bits(&first.2));
        assert_eq!(bits(&flat(&download(&b, &e2))), bits(&first.3));
        let m1 = block_on(b.l2_norm(&e1)).unwrap();
        let m2 = block_on(b.l2_norm(&e2)).unwrap();
        assert_eq!(bits(&[n1, n2]), bits(&[m1, m2]));
    }
}

#[test]
fn all_ones_masks_give_the_unmasked_pair_and_empty_or_wide_inputs_are_refused() {
    let Some(b) = common::backend() else { return };
    let grid = cases()[1];
    let n = grid.num_voxels();
    let (fa, fb) = (f32_image(&grid, 7, false), f32_image(&grid, 8, true));
    let up = |v: &Volume<f64>| block_on(b.upload_image(v)).unwrap();
    let (ga, gb) = (up(&fa), up(&fb));
    let ones = up(&Volume::from_vec(grid, vec![1.0; n]).unwrap());
    let zeros = up(&Volume::zeros(grid));
    let ((v1, d1), (v2, d2)) = block_on(b.mi_value_derivative_pair(&ga, &gb, 32)).unwrap();
    let ((w1, e1), (w2, e2)) =
        block_on(b.mi_value_derivative_pair_masked(&ga, &gb, &ones, &ones, 32)).unwrap();
    assert_eq!(bits(&[v1, v2]), bits(&[w1, w2]));
    assert_eq!(
        bits(&flat(&download(&b, &d1))),
        bits(&flat(&download(&b, &e1)))
    );
    assert_eq!(
        bits(&flat(&download(&b, &d2))),
        bits(&flat(&download(&b, &e2)))
    );
    let (fm, _) = masks(&grid);
    let disjoint = Volume::from_vec(grid, fm.data().iter().map(|v| 1.0 - v).collect()).unwrap();
    let (gfm, gdis) = (up(&fm), up(&disjoint));
    let err = block_on(b.mi_value_derivative_pair_masked(&ga, &gb, &gfm, &gdis, 32))
        .err()
        .expect("disjoint masks are refused");
    assert!(err.to_string().contains(NO_VOXEL_IN_MASKS), "{err}");
    let cpu_err = CpuBackend
        .mi_value_derivative_pair_masked(&fa, &fb, &fm, &disjoint, 32)
        .expect_err("the CPU refuses disjoint masks");
    assert!(cpu_err.to_string().contains(NO_VOXEL_IN_MASKS), "{cpu_err}");
    let err = block_on(b.mi_value_derivative_pair_masked(&ga, &gb, &zeros, &ones, 32))
        .err()
        .expect("an empty fixed mask is refused");
    assert!(err.to_string().contains("fixed mask"), "{err}");
    let err = block_on(b.mi_value_derivative_pair(&ga, &gb, 33))
        .err()
        .expect("more than 32 bins are refused on the GPU");
    assert!(err.to_string().contains("32"), "{err}");
}

#![allow(clippy::unwrap_used, clippy::expect_used)]
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

mod common;

use fastwarp_gpu::parity::{check_mi_dense, image, Case};
use fastwarp_gpu::{GpuField, WgpuBackend};
use fastwarp_reg::backend::Backend;
use fastwarp_reg::cpu::CpuBackend;
use fastwarp_reg::field_ops::VectorField;
use fastwarp_reg::linalg::IDENTITY3;
use fastwarp_reg::metric::cc::NO_VOXEL_IN_MASKS;
use fastwarp_reg::metric::mi::MiBinning;
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

fn compare(label: &str, value: f64, want: f64, gpu: &VectorField, reference: &VectorField) {
    compare_on(label, value, want, gpu, reference, false, 1e-4);
}

fn on_z_face(grid: &Grid, idx: usize) -> bool {
    let [_, _, k] = grid.ijk(idx);
    k == 0 || k + 1 == grid.size()[2]
}

fn compare_on(
    label: &str,
    value: f64,
    want: f64,
    gpu: &VectorField,
    reference: &VectorField,
    interior_only: bool,
    tolerance: f64,
) {
    let grid = *reference.grid();
    let keep = |idx: usize| !interior_only || !on_z_face(&grid, idx);
    assert!(
        (value - want).abs() <= 1e-6 * want.abs(),
        "{label}: value {value} vs reference {want}"
    );
    let range = reference
        .data()
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0f64, |a, x| a.max(x.abs()));
    assert!(range > 0.0, "{label}: zero reference derivative");
    let mut rel: Vec<f64> = gpu
        .data()
        .iter()
        .zip(reference.data())
        .enumerate()
        .filter(|(idx, _)| keep(*idx))
        .map(|(_, (g, c))| (0..3).map(|k| (g[k] - c[k]).abs()).fold(0.0f64, f64::max) / range)
        .collect();
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = rel.len();
    let zero_mismatch = gpu
        .data()
        .iter()
        .zip(reference.data())
        .enumerate()
        .filter(|(idx, (g, c))| keep(*idx) && ((**g == [0.0; 3]) != (**c == [0.0; 3])))
        .count();
    eprintln!(
        "{label}: value {value:.9} vs {want:.9}; derivative error / {range:.3e}: 50% {:.2e} \
         99% {:.2e} max {:.2e}; {zero_mismatch} zero/non-zero mismatches",
        rel[n / 2],
        rel[n * 99 / 100],
        rel[n - 1]
    );
    assert!(
        rel[n - 1] <= tolerance,
        "{label}: derivative error {} of the range",
        rel[n - 1]
    );
    assert_eq!(zero_mismatch, 0, "{label}: zero/non-zero mismatches");
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
            block_on(b.mi_value_derivative_pair_masked(&ga, &gb, Some(&gfm), Some(&gmm), 32))
                .unwrap();
        let ((c1, e1), (c2, e2)) = cpu
            .mi_value_derivative_pair_masked(&fa, &fb, Some(&fm), Some(&mm), 32)
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
        for (label, gpu_masks, cpu_masks) in [
            ("fixed mask only", (Some(&gfm), None), (Some(&fm), None)),
            ("moving mask only", (None, Some(&gmm)), (None, Some(&mm))),
        ] {
            let ((v1, d1), (v2, d2)) =
                block_on(b.mi_value_derivative_pair_masked(&ga, &gb, gpu_masks.0, gpu_masks.1, 32))
                    .unwrap();
            let ((c1, e1), (c2, e2)) = cpu
                .mi_value_derivative_pair_masked(&fa, &fb, cpu_masks.0, cpu_masks.1, 32)
                .unwrap();
            let open = cpu.mi_value_derivative_pair(&fa, &fb, 32).unwrap();
            assert_ne!(
                c1.to_bits(),
                open.0 .0.to_bits(),
                "{label}: the mask matters"
            );
            compare(
                &format!("{dims:?} {label} forward"),
                v1,
                c1,
                &download(&b, &d1),
                &e1,
            );
            compare(
                &format!("{dims:?} {label} backward"),
                v2,
                c2,
                &download(&b, &d2),
                &e2,
            );
        }
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
                block_on(b.mi_value_derivative_pair_masked(&ga, &gb, Some(&gfm), Some(&gmm), 32))
                    .unwrap()
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
            block_on(b.mi_value_derivative_pair_masked_with_norms(
                &ga,
                &gb,
                Some(&gfm),
                Some(&gmm),
                32,
            ))
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
        block_on(b.mi_value_derivative_pair_masked(&ga, &gb, Some(&ones), Some(&ones), 32))
            .unwrap();
    assert_eq!(bits(&[v1, v2]), bits(&[w1, w2]));
    assert_eq!(
        bits(&flat(&download(&b, &d1))),
        bits(&flat(&download(&b, &e1)))
    );
    assert_eq!(
        bits(&flat(&download(&b, &d2))),
        bits(&flat(&download(&b, &e2)))
    );
    for (fixed_mask, moving_mask) in [(Some(&ones), None), (None, Some(&ones)), (None, None)] {
        let ((u1, f1), (u2, f2)) =
            block_on(b.mi_value_derivative_pair_masked(&ga, &gb, fixed_mask, moving_mask, 32))
                .unwrap();
        assert_eq!(bits(&[v1, v2]), bits(&[u1, u2]));
        assert_eq!(
            bits(&flat(&download(&b, &d1))),
            bits(&flat(&download(&b, &f1)))
        );
        assert_eq!(
            bits(&flat(&download(&b, &d2))),
            bits(&flat(&download(&b, &f2)))
        );
    }
    let (fm, _) = masks(&grid);
    let disjoint = Volume::from_vec(grid, fm.data().iter().map(|v| 1.0 - v).collect()).unwrap();
    let (gfm, gdis) = (up(&fm), up(&disjoint));
    let err = block_on(b.mi_value_derivative_pair_masked(&ga, &gb, Some(&gfm), Some(&gdis), 32))
        .err()
        .expect("disjoint masks are refused");
    assert!(err.to_string().contains(NO_VOXEL_IN_MASKS), "{err}");
    let cpu_err = CpuBackend
        .mi_value_derivative_pair_masked(&fa, &fb, Some(&fm), Some(&disjoint), 32)
        .expect_err("the CPU refuses disjoint masks");
    assert!(cpu_err.to_string().contains(NO_VOXEL_IN_MASKS), "{cpu_err}");
    let err = block_on(b.mi_value_derivative_pair_masked(&ga, &gb, Some(&zeros), None, 32))
        .err()
        .expect("an empty fixed mask is refused");
    assert!(err.to_string().contains("fixed mask"), "{err}");
    let err = block_on(b.mi_value_derivative_pair(&ga, &gb, 33))
        .err()
        .expect("more than 32 bins are refused on the GPU");
    assert!(err.to_string().contains("32"), "{err}");
}

fn integer_image(grid: &Grid, seed: u64, contrast: bool) -> Volume<f64> {
    let base = image(grid, seed, 0.2);
    let n = grid.num_voxels();
    let data = base
        .data()
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            let w = if contrast { 1.0 - v * v } else { *v };
            match idx % 7 {
                _ if idx == 3 => 0.0,
                _ if idx == n - 1 => 1000.0,
                0 => 250.0,
                1 => 500.0,
                2 => 750.0,
                _ => (w.clamp(0.0, 1.0) * 1000.0).round(),
            }
        })
        .collect();
    Volume::from_vec(*grid, data).unwrap()
}

#[test]
fn integer_images_with_values_on_bin_edges_match_the_cpu() {
    let Some(b) = common::backend() else { return };
    let cpu = CpuBackend;
    let grid = Grid::new([29, 23, 19], [1.0, 2.0, 0.5], [0.0; 3], IDENTITY3).unwrap();
    let (fa, fb) = (
        integer_image(&grid, 9, false),
        integer_image(&grid, 10, true),
    );
    let (ga, gb) = (
        block_on(b.upload_image(&fa)).unwrap(),
        block_on(b.upload_image(&fb)).unwrap(),
    );
    for bins in [32usize, 12, 9] {
        let binning = MiBinning::dense(&fa, &fb, None, None, bins).unwrap();
        assert_eq!((binning.fixed_min, binning.fixed_max), (0.0, 1000.0));
        let on_edge = fa
            .data()
            .iter()
            .filter(|v| {
                binning.fixed_bin(**v) > binning.fixed_bin(f64::from((**v as f32).next_down()))
            })
            .count();
        eprintln!("integer bins {bins}: {on_edge} voxels on a fixed bin edge");
        assert!(on_edge > 0, "bins {bins}: no voxel on a fixed bin edge");
        let ((v1, d1), (v2, d2)) = block_on(b.mi_value_derivative_pair(&ga, &gb, bins)).unwrap();
        let ((c1, e1), (c2, e2)) = cpu.mi_value_derivative_pair(&fa, &fb, bins).unwrap();
        compare(
            &format!("integer bins {bins} forward"),
            v1,
            c1,
            &download(&b, &d1),
            &e1,
        );
        compare(
            &format!("integer bins {bins} backward"),
            v2,
            c2,
            &download(&b, &d2),
            &e2,
        );
    }
}

struct Fixture(std::collections::HashMap<String, Vec<f64>>);

impl Fixture {
    fn load(name: &str) -> Self {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../fastwarp-reg/tests/data/mattes")
            .join(format!("{name}.txt"));
        let text = std::fs::read_to_string(&path).unwrap();
        Self(
            text.lines()
                .filter(|l| !l.starts_with('#') && !l.is_empty())
                .map(|l| {
                    let (k, v) = l.split_once(':').unwrap();
                    let values = v.split_whitespace().map(|x| x.parse().unwrap()).collect();
                    (k.to_string(), values)
                })
                .collect(),
        )
    }

    fn get(&self, key: &str) -> &[f64] {
        &self.0[key]
    }

    fn grid(&self) -> Grid {
        let s = self.get("fixed_size");
        let sp = self.get("fixed_spacing");
        let o = self.get("fixed_origin");
        let d = self.get("fixed_direction");
        Grid::new(
            [s[0] as usize, s[1] as usize, s[2] as usize],
            [sp[0], sp[1], sp[2]],
            [o[0], o[1], o[2]],
            [[d[0], d[1], d[2]], [d[3], d[4], d[5]], [d[6], d[7], d[8]]],
        )
        .unwrap()
    }

    fn volume(&self, key: &str) -> Volume<f64> {
        Volume::from_vec(self.grid(), self.get(key).to_vec()).unwrap()
    }

    fn derivative(&self) -> VectorField {
        let data = self
            .get("derivative")
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        VectorField::from_vec(self.grid(), data).unwrap()
    }
}

#[test]
fn the_mattes_pair_matches_itk_on_the_virtual_grid_in_both_directions() {
    let Some(b) = common::backend() else { return };
    let up = |v: &Volume<f64>| block_on(b.upload_image(v)).unwrap();
    let plain = Fixture::load("virtual_b32");
    let (ga, gb) = (up(&plain.volume("fixed")), up(&plain.volume("moving")));
    let ((v1, d1), _) = block_on(b.mi_value_derivative_pair(&ga, &gb, 32)).unwrap();
    compare_on(
        "itk virtual_b32 (voxels off the z faces)",
        v1,
        plain.get("value")[0],
        &download(&b, &d1),
        &plain.derivative(),
        true,
        1e-3,
    );
    let masked = Fixture::load("virtual_b32_masked");
    let swapped = Fixture::load("virtual_b32_masked_swapped");
    let (gfm, gmm) = (
        up(&masked.volume("fixed_mask")),
        up(&masked.volume("moving_mask")),
    );
    let ((w1, e1), (w2, e2)) =
        block_on(b.mi_value_derivative_pair_masked(&ga, &gb, Some(&gfm), Some(&gmm), 32)).unwrap();
    compare_on(
        "itk virtual_b32_masked (voxels off the z faces)",
        w1,
        masked.get("value")[0],
        &download(&b, &e1),
        &masked.derivative(),
        true,
        1e-3,
    );
    compare_on(
        "itk virtual_b32_masked second direction (voxels off the z faces)",
        w2,
        swapped.get("value")[0],
        &download(&b, &e2),
        &swapped.derivative(),
        true,
        1e-3,
    );
}

#[test]
fn the_dense_mattes_known_answer_check_passes_on_the_self_test_and_an_oblique_case() {
    let Some(b) = common::backend() else { return };
    for case in [
        Case::selftest_mi(),
        Case::awkward([29, 23, 19], [1.2, 1.6, 2.0], true, 3),
    ] {
        let results = check_mi_dense(&b, &case).unwrap();
        assert_eq!(results.len(), 12);
        for r in &results {
            eprintln!("{}", r.line());
        }
        assert!(results.iter().all(|r| r.ok));
    }
}

#[test]
fn a_device_without_the_dense_mattes_kernels_keeps_cc() {
    let Some(mut b) = common::backend() else {
        return;
    };
    assert!(b.selftest_record().mi_dense_failure.is_none());
    let grid = cases()[1];
    let (fa, fb) = (f32_image(&grid, 3, false), f32_image(&grid, 4, true));
    let (ga, gb) = (
        block_on(b.upload_image(&fa)).unwrap(),
        block_on(b.upload_image(&fb)).unwrap(),
    );
    b.disable_dense_mi("forced for the test");
    let err = block_on(b.mi_value_derivative_pair(&ga, &gb, 32))
        .err()
        .expect("the dense Mattes kernels are off");
    assert!(err.to_string().contains("forced for the test"), "{err}");
    assert!(block_on(b.cc_value_derivative_pair(&ga, &gb, 2)).is_ok());
}

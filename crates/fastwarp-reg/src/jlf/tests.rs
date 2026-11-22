// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used)]

use super::patch::{search_offsets, zscore, Dims, Patch};
use super::*;

const SIZE: [usize; 3] = [12, 11, 10];

fn noise(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut h = (i as u32).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503);
            h ^= h >> 13;
            h = h.wrapping_mul(1_274_126_177);
            h ^= h >> 16;
            (h % 1000) as f32 / 10.0
        })
        .collect()
}

fn voxels() -> usize {
    SIZE.iter().product()
}

fn interior(v: usize, margin: isize) -> bool {
    let p = Dims::new(SIZE).ijk(v);
    (0..3).all(|d| p[d] >= margin && p[d] < SIZE[d] as isize - margin)
}

fn blocks() -> Vec<u32> {
    let d = Dims::new(SIZE);
    (0..voxels())
        .map(|v| {
            let p = d.ijk(v);
            u32::from(p[0] >= 6) + 2 * u32::from(p[2] >= 5)
        })
        .collect()
}

fn small() -> JlfOptions {
    JlfOptions {
        patch_radius: [1; 3],
        search: SearchRadius::Box([1; 3]),
        posteriors: true,
        voting_weights: true,
        ..JlfOptions::default()
    }
}

fn input(atlases: Vec<Vec<Vec<f32>>>, labels: Vec<Vec<u32>>) -> JlfInput {
    JlfInput {
        size: SIZE,
        spacing: [1.0, 1.0, 1.0],
        target: vec![noise(1, voxels())],
        atlases,
        labels,
        ..JlfInput::default()
    }
}

#[test]
fn search_offsets_are_ordered_by_distance_then_neighbourhood_index() {
    let o = search_offsets([1; 3], [1.0; 3]);
    assert_eq!(o.len(), 27);
    assert_eq!(o[0], [0, 0, 0]);
    assert_eq!(
        &o[1..7],
        &[
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    );
    let o = search_offsets([1; 3], [1.0, 1.0, 0.5]);
    assert_eq!(&o[1..3], &[[0, 0, -1], [0, 0, 1]]);
    assert_eq!(o[26], [1, 1, 1]);
}

#[test]
fn patches_outside_the_image_are_nan_and_constant_patches_blow_up() {
    let d = Dims::new(SIZE);
    let p = Patch::new([1; 3], &d);
    let image = noise(3, voxels());
    let corner = p.vectorize_list(std::slice::from_ref(&image), &d, [0, 0, 0], true);
    assert_eq!(corner.iter().filter(|v| v.is_nan()).count(), 27 - 8);
    assert!(corner.iter().filter(|v| v.is_finite()).count() == 8);
    let mut constant = vec![2.0f32; 5];
    zscore(&mut constant);
    assert!(constant.iter().all(|v| *v == f32::NEG_INFINITY));
    let mut zero = vec![0.0f32; 5];
    zscore(&mut zero);
    assert!(zero.iter().all(|v| v.is_nan()));
    let mut with_gap = vec![1.0f32, f32::NAN, 3.0];
    zscore(&mut with_gap);
    assert!((with_gap[0] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    assert!(with_gap[1].is_nan());
    assert!((with_gap[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn two_atlas_weights_have_the_closed_form() {
    let (a, b, c) = (3.0, 1.0, 2.0);
    let w = solve_weights(&[a, b, b, c], 2, false);
    let (w0, w1) = (c - b, a - b);
    assert!((w[0] - w0 / (w0 + w1)).abs() < 1e-15, "{w:?}");
    assert!((w[1] - w1 / (w0 + w1)).abs() < 1e-15, "{w:?}");
    let w = solve_weights(&[1.0, 2.0, 2.0, 5.0], 2, false);
    assert_eq!(w, vec![1.0, 0.0]);
    let nn = solve_weights(&[a, b, b, c], 2, true);
    assert!((nn[0] - w0 / (w0 + w1)).abs() < 1e-12, "{nn:?}");
    let nn = solve_weights(&[1.0, 2.0, 2.0, 5.0], 2, true);
    assert_eq!(nn, vec![1.0, 0.0]);
    let singular = solve_weights(&[1.0, 1.0, 1.0, 1.0], 2, false);
    assert!((singular[0] - 0.5).abs() < 1e-12 && (singular[1] - 0.5).abs() < 1e-12);
}

#[test]
fn nnls_meets_the_optimality_conditions() {
    let a = [4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
    let y = [1.0, -3.0, 2.0];
    let x = nonnegative_least_squares(&a, 3, 3, &y, 1e-10);
    let residual: Vec<f64> = (0..3)
        .map(|i| y[i] - (0..3).map(|j| a[i * 3 + j] * x[j]).sum::<f64>())
        .collect();
    for j in 0..3 {
        let g: f64 = (0..3).map(|i| a[i * 3 + j] * residual[i]).sum();
        assert!(x[j] >= 0.0);
        assert!(g <= 1e-9, "{j}: {g}");
        if x[j] > 0.0 {
            assert!(g.abs() < 1e-9, "{j}: {g}");
        }
    }
}

#[test]
fn atlases_equal_to_the_target_get_equal_weights_and_return_it() {
    let target = noise(1, voxels());
    let labels = blocks();
    let inp = input(vec![vec![target.clone()]; 3], vec![labels.clone(); 3]);
    let out = joint_fusion(&inp, &small(), None).unwrap();
    assert_eq!(out.label_set, vec![0, 1, 2, 3]);
    let got = out.labels.as_ref().unwrap();
    for (v, t) in target.iter().enumerate() {
        if interior(v, 3) {
            assert_eq!(got[v], labels[v]);
            assert!(
                (out.intensity[0][v] - t).abs() <= 1e-4 * t.abs().max(1.0),
                "{v}"
            );
        }
    }
    for w in &out.voting_weights {
        for v in w {
            assert!((v - 1.0 / 3.0).abs() < 1e-6, "{v}");
        }
    }
    for (label, p) in &out.posteriors {
        for (v, value) in p.iter().enumerate().filter(|(v, _)| interior(*v, 3)) {
            let want = if labels[v] == *label { 1.0 } else { 0.0 };
            assert!((value - want).abs() < 1e-6);
        }
    }
}

#[test]
fn the_closer_atlas_wins_and_the_result_does_not_depend_on_threads() {
    let n = voxels();
    let target = noise(1, n);
    let near: Vec<f32> = target
        .iter()
        .zip(noise(7, n))
        .map(|(t, e)| t + 0.05 * e)
        .collect();
    let far = noise(9, n);
    let labels = blocks();
    let other: Vec<u32> = labels.iter().map(|l| l + 4).collect();
    let inp = input(
        vec![vec![far.clone()], vec![near], vec![far]],
        vec![other.clone(), labels.clone(), other],
    );
    for metric in [PatchMetric::PearsonCorrelation, PatchMetric::MeanSquares] {
        for nonnegative in [false, true] {
            let opts = JlfOptions {
                metric,
                nonnegative,
                ..small()
            };
            let out = joint_fusion(&inp, &opts, None).unwrap();
            let got = out.labels.as_ref().unwrap();
            let inside: Vec<usize> = (0..n).filter(|v| interior(*v, 2)).collect();
            let agree = inside.iter().filter(|v| got[**v] == labels[**v]).count();
            assert!(
                agree * 20 > inside.len() * 19,
                "{metric:?} {nonnegative}: {agree} of {}",
                inside.len()
            );
            let one = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let serial = one.install(|| joint_fusion(&inp, &opts, None).unwrap());
            assert_eq!(serial, out);
        }
    }
}

#[test]
fn mask_exclusions_and_radius_images() {
    let n = voxels();
    let target = noise(1, n);
    let labels = blocks();
    let mut inp = input(vec![vec![target.clone()]; 2], vec![labels.clone(); 2]);
    let mask: Vec<u32> = (0..n).map(|v| u32::from(v % 3 != 0)).collect();
    inp.mask = Some(mask.clone());
    inp.exclusions = vec![(2, vec![0; n]), (2, vec![1; n])];
    let out = joint_fusion(&inp, &small(), None).unwrap();
    let got = out.labels.unwrap();
    for v in 0..n {
        if mask[v] == 0 {
            assert_eq!(got[v], 0);
            assert_eq!(out.intensity[0][v], 0.0);
        } else if labels[v] == 2 {
            assert_eq!(got[v], 0, "{v}");
        } else {
            assert_eq!(got[v], labels[v]);
        }
    }
    inp.mask = None;
    inp.exclusions.clear();
    let mut radii = vec![0u32; n];
    radii[Dims::new(SIZE).index([5, 5, 5])] = 1;
    let opts = JlfOptions {
        search: SearchRadius::Image(radii),
        ..small()
    };
    let out = joint_fusion(&inp, &opts, None).unwrap();
    let touched = out.intensity[0].iter().filter(|v| **v != 0.0).count();
    assert_eq!(touched, 27);
}

#[test]
fn one_target_modality_uses_the_first_atlas_modality_only() {
    let n = voxels();
    let target = noise(1, n);
    let labels = blocks();
    let make = |seed: u32| {
        input(
            vec![
                vec![target.clone(), noise(seed, n)],
                vec![noise(4, n), noise(seed + 1, n)],
            ],
            vec![labels.clone(), labels.iter().map(|l| l + 4).collect()],
        )
    };
    let a = joint_fusion(&make(20), &small(), None).unwrap();
    let b = joint_fusion(&make(30), &small(), None).unwrap();
    assert_eq!(a.intensity.len(), 2);
    assert_eq!(a.labels, b.labels);
    assert_eq!(a.voting_weights, b.voting_weights);
    assert_eq!(a.intensity[0], b.intensity[0]);
    assert_ne!(a.intensity[1], b.intensity[1]);
}

#[test]
fn a_non_finite_sum_resets_the_intensity() {
    let n = voxels();
    let target = noise(1, n);
    let mut bad = target.clone();
    bad[Dims::new(SIZE).index([5, 5, 5])] = f32::INFINITY;
    let inp = input(vec![vec![target.clone()], vec![bad]], Vec::new());
    let out = joint_fusion(&inp, &small(), None).unwrap();
    assert!(out.labels.is_none());
    assert!(out.intensity[0].iter().all(|v| v.is_finite()));
}

#[test]
fn bad_inputs_are_refused() {
    let n = voxels();
    let one = input(vec![vec![noise(2, n)]], Vec::new());
    assert!(joint_fusion(&one, &small(), None).is_err());
    let mut short = input(vec![vec![noise(2, n)]; 2], Vec::new());
    short.atlases[1][0].pop();
    assert!(joint_fusion(&short, &small(), None).is_err());
    let mut modalities = input(vec![vec![noise(2, n)]; 2], Vec::new());
    modalities.target.push(noise(3, n));
    assert!(joint_fusion(&modalities, &small(), None).is_err());
    let labels = input(vec![vec![noise(2, n)]; 2], vec![blocks()]);
    assert!(joint_fusion(&labels, &small(), None).is_err());
    let two = input(vec![vec![noise(2, n)]; 2], Vec::new());
    for search in [
        SearchRadius::Box([usize::MAX; 3]),
        SearchRadius::Box([MAX_SEARCH_RADIUS + 1, 0, 0]),
        SearchRadius::Image(vec![u32::MAX; n]),
        SearchRadius::Image(vec![MAX_SEARCH_RADIUS as u32 + 1; n]),
    ] {
        let opts = JlfOptions { search, ..small() };
        let e = joint_fusion(&two, &opts, None).unwrap_err().to_string();
        assert!(e.contains("search radius"), "{e}");
    }
    for patch_radius in [[usize::MAX, 0, 0], [1, MAX_PATCH_RADIUS + 1, 1], [127; 3]] {
        let opts = JlfOptions {
            patch_radius,
            ..small()
        };
        let e = joint_fusion(&two, &opts, None).unwrap_err().to_string();
        assert!(e.contains("patch radius"), "{e}");
    }
    let opts = JlfOptions {
        patch_radius: [MAX_PATCH_RADIUS, 0, 0],
        search: SearchRadius::Box([MAX_SEARCH_RADIUS, 0, 0]),
        ..small()
    };
    assert!(joint_fusion(&two, &opts, None).is_ok());
    assert_eq!(JlfOptions::default().alpha, f64::from(0.1f32));
    let hybrid = JlfOptions {
        device: JlfDevice::Hybrid,
        ..small()
    };
    let e = joint_fusion(&two, &hybrid, None).unwrap_err().to_string();
    assert!(e.contains("GPU"), "{e}");
}

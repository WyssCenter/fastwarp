// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::PathBuf;

use fastwarp_reg::cpu::CpuBackend;
use fastwarp_reg::linalg::{Mat3, Vec3};
use fastwarp_reg::metric::mi::{MattesMi, MiDenseInputs};
use fastwarp_reg::transform::DisplacementField;
use fastwarp_reg::{Grid, MaskVolume, TransformChain, TransformElem, Volume};

struct Fixture {
    fields: HashMap<String, Vec<f64>>,
}

impl Fixture {
    fn load(name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/mattes")
            .join(format!("{name}.txt"));
        let text = std::fs::read_to_string(&path).unwrap();
        let mut fields = HashMap::new();
        for line in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let (key, values) = line.split_once(':').unwrap();
            let values = values
                .split_whitespace()
                .map(|v| v.parse::<f64>().unwrap())
                .collect();
            fields.insert(key.to_string(), values);
        }
        Self { fields }
    }

    fn get(&self, key: &str) -> &[f64] {
        self.fields
            .get(key)
            .unwrap_or_else(|| panic!("{key} missing"))
    }

    fn has(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    fn scalar(&self, key: &str) -> f64 {
        self.get(key)[0]
    }

    fn grid(&self, role: &str) -> Grid {
        let size = self.get(&format!("{role}_size"));
        let v3 = |k: &str| -> Vec3 {
            let v = self.get(&format!("{role}_{k}"));
            [v[0], v[1], v[2]]
        };
        let d = self.get(&format!("{role}_direction"));
        let direction: Mat3 = [[d[0], d[1], d[2]], [d[3], d[4], d[5]], [d[6], d[7], d[8]]];
        Grid::new(
            [size[0] as usize, size[1] as usize, size[2] as usize],
            v3("spacing"),
            v3("origin"),
            direction,
        )
        .unwrap()
    }

    fn image(&self, role: &str) -> Volume<f64> {
        Volume::from_vec(self.grid(role), self.get(role).to_vec()).unwrap()
    }

    fn mask(&self, role: &str) -> Option<MaskVolume> {
        let key = format!("{role}_mask");
        self.has(&key)
            .then(|| MaskVolume::new(self.grid(role), self.get(&key).to_vec()).unwrap())
    }

    fn moving_transform(&self) -> TransformChain {
        let data = self.get("field").iter().map(|v| *v as f32).collect();
        let field = DisplacementField::new(self.grid("fixed"), data).unwrap();
        TransformChain::from_queue(vec![TransformElem::DisplacementField(field)])
    }

    fn derivative(&self) -> Vec<Vec3> {
        self.get("derivative")
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }
}

fn assert_matches(name: &str, value: f64, valid: Option<usize>, derivative: &[Vec3], fx: &Fixture) {
    let want_value = fx.scalar("value");
    let want = fx.derivative();
    if let Some(valid) = valid {
        assert_eq!(valid, fx.scalar("valid") as usize, "{name}: valid points");
    }
    assert!(
        (value - want_value).abs() <= 1e-12 * want_value.abs(),
        "{name}: value {value} vs ITK {want_value}"
    );
    assert_eq!(derivative.len(), want.len(), "{name}: derivative length");
    let range = want
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0f64, |a, x| a.max(x.abs()));
    assert!(range > 0.0, "{name}: ITK derivative is zero");
    let worst = derivative
        .iter()
        .zip(&want)
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
        .fold(0.0f64, f64::max);
    let zero_mismatch = derivative
        .iter()
        .zip(&want)
        .filter(|(a, b)| (**a == [0.0; 3]) != (**b == [0.0; 3]))
        .count();
    eprintln!(
        "{name}: value {value} (ITK {want_value}), derivative max diff {worst:.3e} of {range:.3e}"
    );
    assert_eq!(
        zero_mismatch, 0,
        "{name}: voxels with a zero derivative differ"
    );
    assert!(
        worst <= 1e-10 * range,
        "{name}: derivative {worst} of {range}"
    );
}

fn check_dense(name: &str) {
    let fx = Fixture::load(name);
    let fixed = fx.image("fixed");
    let moving = fx.image("moving");
    let (fixed_mask, moving_mask) = (fx.mask("fixed"), fx.mask("moving"));
    let identity = TransformChain::new();
    let moving_transform = fx.moving_transform();
    let e = MattesMi {
        bins: fx.scalar("bins") as usize,
    }
    .evaluate_dense(
        &MiDenseInputs {
            virtual_grid: fixed.grid(),
            fixed: &fixed,
            moving: &moving,
            fixed_transform: &identity,
            moving_transform: &moving_transform,
            fixed_mask: fixed_mask.as_ref(),
            moving_mask: moving_mask.as_ref(),
        },
        true,
    )
    .unwrap();
    assert_matches(name, e.value, Some(e.num_valid), &e.derivative, &fx);
}

#[test]
fn dense_mattes_matches_itk_with_a_displacement_field_on_oblique_grids() {
    check_dense("oblique_b32");
    check_dense("oblique_b12");
}

#[test]
fn dense_mattes_matches_itk_with_masks_on_oblique_grids() {
    check_dense("oblique_b32_masked");
}

#[test]
fn dense_mattes_matches_itk_on_the_virtual_grid() {
    check_dense("virtual_b32");
    check_dense("virtual_b32_masked");
}

fn binary(v: &[f64], grid: Grid) -> Volume<f64> {
    Volume::from_vec(grid, v.to_vec()).unwrap()
}

#[test]
fn the_cpu_backend_pair_matches_itk_and_swaps_images_and_masks() {
    let cpu = CpuBackend;
    let plain = Fixture::load("virtual_b32");
    let (fixed, moving) = (plain.image("fixed"), plain.image("moving"));
    let ((v1, d1), (v2, d2)) = cpu.mi_value_derivative_pair(&fixed, &moving, 32).unwrap();
    assert_matches("pair virtual_b32", v1, None, d1.data(), &plain);
    let masked = Fixture::load("virtual_b32_masked");
    let grid = *fixed.grid();
    let fm = binary(masked.get("fixed_mask"), grid);
    let mm = binary(masked.get("moving_mask"), grid);
    let ((w1, e1), (w2, e2)) = cpu
        .mi_value_derivative_pair_masked(&fixed, &moving, &fm, &mm, 32)
        .unwrap();
    assert_matches("pair virtual_b32_masked", w1, None, e1.data(), &masked);
    let identity = TransformChain::new();
    let swapped =
        |a: &Volume<f64>, b: &Volume<f64>, ma: Option<&MaskVolume>, mb: Option<&MaskVolume>| {
            MattesMi { bins: 32 }
                .evaluate_dense(
                    &MiDenseInputs {
                        virtual_grid: &grid,
                        fixed: a,
                        moving: b,
                        fixed_transform: &identity,
                        moving_transform: &identity,
                        fixed_mask: ma,
                        moving_mask: mb,
                    },
                    true,
                )
                .unwrap()
        };
    let s = swapped(&moving, &fixed, None, None);
    assert_eq!(v2.to_bits(), s.value.to_bits());
    assert_eq!(d2.data(), &s.derivative[..]);
    let (fmv, mmv) = (
        MaskVolume::from_volume(fm.clone()),
        MaskVolume::from_volume(mm.clone()),
    );
    let s = swapped(&moving, &fixed, Some(&mmv), Some(&fmv));
    assert_eq!(w2.to_bits(), s.value.to_bits());
    assert_eq!(e2.data(), &s.derivative[..]);
}

#![allow(clippy::unwrap_used, clippy::expect_used)]
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use fastwarp_gpu::parity::image;
use fastwarp_gpu::{BackendSelect, GpuError, WgpuBackend};
use fastwarp_reg::backend::Backend;
use fastwarp_reg::cpu::CpuBackend;
use fastwarp_reg::field_ops::VectorField;
use fastwarp_reg::linalg::IDENTITY3;
use fastwarp_reg::{Grid, Volume};
use pollster::block_on;

fn require_adapter() -> bool {
    fastwarp_gpu::device::env_var("FASTWARP_GPU_REQUIRE_ADAPTER")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn backend() -> Option<WgpuBackend> {
    match WgpuBackend::new(&BackendSelect::from_env().unwrap_or_default()) {
        Ok(b) => Some(b),
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

struct Look {
    background: f64,
    background_noise: f64,
    low: f64,
    high: f64,
}

fn phantom(grid: &Grid, seed: u64, centre_shift: f64, look: &Look) -> Volume<f64> {
    let [nx, ny, nz] = grid.size();
    let base = image(grid, seed, 0.2);
    let noise = image(grid, seed + 100, 1.0);
    let data = (0..grid.num_voxels())
        .map(|idx| {
            let [i, j, k] = grid.ijk(idx);
            let d = |p: usize, n: usize, s: f64| {
                (p as f64 - 0.5 * (n as f64 - 1.0) - s) / (0.34 * n as f64)
            };
            let r2 = d(i, nx, centre_shift).powi(2) + d(j, ny, 0.0).powi(2) + d(k, nz, 0.0).powi(2);
            let v = if r2 < 1.0 {
                look.low + (look.high - look.low) * base.data()[idx]
            } else {
                look.background + look.background_noise * (noise.data()[idx] - 0.5)
            };
            f64::from(v as f32)
        })
        .collect();
    Volume::from_vec(*grid, data).unwrap()
}

fn interior_diff(grid: &Grid, got: &VectorField, want: &VectorField) -> (f64, f64) {
    let [nx, ny, nz] = grid.size();
    let range = want
        .data()
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0f64, |a, x| a.max(x.abs()));
    let mut worst = 0.0f64;
    for (idx, (g, w)) in got.data().iter().zip(want.data()).enumerate() {
        let [x, y, z] = grid.ijk(idx);
        if x > 0 && y > 0 && z > 0 && x + 1 < nx && y + 1 < ny && z + 1 < nz {
            for k in 0..3 {
                worst = worst.max((g[k] - w[k]).abs());
            }
        }
    }
    (worst, range)
}

fn compare(b: &WgpuBackend, grid: &Grid, look: &Look, radius: usize) -> (f64, f64, f64, f64) {
    let f = phantom(grid, 3, 0.0, look);
    let m = phantom(grid, 4, 1.3, look);
    let (gf, gm) = (
        block_on(b.upload_image(&f)).unwrap(),
        block_on(b.upload_image(&m)).unwrap(),
    );
    let (gv, gd) = block_on(b.cc_value_derivative(&gf, &gm, radius)).unwrap();
    let gd = block_on(b.download_field(&gd)).unwrap();
    let (cv, cd) = CpuBackend.cc_value_derivative(&f, &m, radius).unwrap();
    let (worst, range) = interior_diff(grid, &gd, &cd);
    (gv, cv, worst, range)
}

fn assert_close(grid: &Grid, what: &str, got: (f64, &VectorField), want: (f64, &VectorField)) {
    let (worst, range) = interior_diff(grid, got.1, want.1);
    eprintln!(
        "{what}: value {} vs {}, derivative max diff {worst:.2e} of {range:.2e}",
        got.0, want.0
    );
    assert!(
        (got.0 - want.0).abs() < 1e-5,
        "{what}: value {} vs {}",
        got.0,
        want.0
    );
    assert!(range > 0.0, "{what}: zero derivative");
    assert!(
        worst < 2e-3 * range,
        "{what}: derivative {worst} of {range}"
    );
}

#[test]
fn images_with_an_intensity_offset_match_the_cpu_in_every_cc_path() {
    let Some(b) = backend() else { return };
    let grid = Grid::new([32; 3], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
    let down = |f| block_on(b.download_field(&f)).unwrap();
    for (background, moving_background, radius) in [(100.0, 30.0, 2usize), (1000.0, 300.0, 4usize)]
    {
        let look = |background: f64| Look {
            background,
            background_noise: 0.0,
            low: background + 0.05,
            high: background + 1.0,
        };
        let f = phantom(&grid, 3, 0.0, &look(background));
        let m = phantom(&grid, 4, 1.3, &look(moving_background));
        let (gf, gm) = (
            block_on(b.upload_image(&f)).unwrap(),
            block_on(b.upload_image(&m)).unwrap(),
        );
        let label = |what: &str| format!("background {background}, radius {radius}, {what}");

        let ((v1, d1), (v2, d2)) = block_on(b.cc_value_derivative_pair(&gf, &gm, radius)).unwrap();
        let ((c1, e1), (c2, e2)) = CpuBackend.cc_value_derivative_pair(&f, &m, radius).unwrap();
        assert_close(&grid, &label("pair forward"), (v1, &down(d1)), (c1, &e1));
        assert_close(&grid, &label("pair backward"), (v2, &down(d2)), (c2, &e2));

        let zero = block_on(b.zero_field(&grid)).unwrap();
        let warped = block_on(b.warp_through_field(&gm, &zero, None, None)).unwrap();
        let warped_pre = block_on(b.warp_through_field(&gm, &zero, Some(&zero), None)).unwrap();
        let (sv, sd) = block_on(b.cc_value_derivative(&gf, &gm, radius)).unwrap();
        let sd = down(sd);
        for (what, image) in [
            ("warped", &warped),
            ("warped with a prior field", &warped_pre),
        ] {
            let (wv, wd) = block_on(b.cc_value_derivative(&gf, image, radius)).unwrap();
            assert_eq!(wv.to_bits(), sv.to_bits(), "{}", label(what));
            assert_eq!(down(wd).data(), sd.data(), "{}", label(what));
        }

        let valid = Volume::from_vec(
            grid,
            (0..grid.num_voxels())
                .map(|idx| if grid.ijk(idx)[0] < 5 { 0.0 } else { 1.0 })
                .collect(),
        )
        .unwrap();
        let gv = block_on(b.upload_image(&valid)).unwrap();
        let ((v1, d1), (v2, d2)) =
            block_on(b.cc_value_derivative_pair_masked(&gf, &gm, &gv, radius)).unwrap();
        let ((c1, e1), (c2, e2)) = CpuBackend
            .cc_value_derivative_pair_masked(&f, &m, &valid, radius)
            .unwrap();
        assert_close(&grid, &label("masked forward"), (v1, &down(d1)), (c1, &e1));
        assert_close(&grid, &label("masked backward"), (v2, &down(d2)), (c2, &e2));
    }
}

#[test]
#[ignore = "prints the CPU-GPU difference of the CC metric over intensity offsets and scales"]
fn characterise_cc_precision_over_intensity_offsets() {
    let Some(b) = backend() else { return };
    eprintln!("using {}", b.name());
    let looks: Vec<(String, Look)> = vec![
        (
            "bg 0, 0..1".into(),
            Look {
                background: 0.0,
                background_noise: 0.0,
                low: 0.05,
                high: 1.0,
            },
        ),
        (
            "bg 1, 1..2".into(),
            Look {
                background: 1.0,
                background_noise: 0.0,
                low: 1.05,
                high: 2.0,
            },
        ),
        (
            "bg 100, 100..101".into(),
            Look {
                background: 100.0,
                background_noise: 0.0,
                low: 100.05,
                high: 101.0,
            },
        ),
        (
            "bg 1000, 1000..1001".into(),
            Look {
                background: 1000.0,
                background_noise: 0.0,
                low: 1000.05,
                high: 1001.0,
            },
        ),
        (
            "bg 0, 0..1000".into(),
            Look {
                background: 0.0,
                background_noise: 0.0,
                low: 50.0,
                high: 1000.0,
            },
        ),
        (
            "bg 1000, 1000..2000".into(),
            Look {
                background: 1000.0,
                background_noise: 0.0,
                low: 1050.0,
                high: 2000.0,
            },
        ),
        (
            "ct -1024, -100..3072".into(),
            Look {
                background: -1024.0,
                background_noise: 0.0,
                low: -100.0,
                high: 3072.0,
            },
        ),
        (
            "ct -1024, 0..100".into(),
            Look {
                background: -1024.0,
                background_noise: 0.0,
                low: 0.0,
                high: 100.0,
            },
        ),
        (
            "mri bg 50+-20, 800..3000".into(),
            Look {
                background: 50.0,
                background_noise: 40.0,
                low: 800.0,
                high: 3000.0,
            },
        ),
        (
            "mri bg 0, 2000..2500".into(),
            Look {
                background: 0.0,
                background_noise: 0.0,
                low: 2000.0,
                high: 2500.0,
            },
        ),
        (
            "mri bg 0, 2000..2050".into(),
            Look {
                background: 0.0,
                background_noise: 0.0,
                low: 2000.0,
                high: 2050.0,
            },
        ),
        (
            "bg 0, 100..101".into(),
            Look {
                background: 0.0,
                background_noise: 0.0,
                low: 100.05,
                high: 101.0,
            },
        ),
    ];
    for (n, radius) in [(32usize, 2usize), (32, 4), (64, 4)] {
        let grid = Grid::new([n; 3], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
        for (label, look) in &looks {
            let (gv, cv, worst, range) = compare(&b, &grid, look, radius);
            eprintln!(
                "n {n:3} r {radius} {label:28} gpu {gv:+.6} cpu {cv:+.6} dvalue {:.2e} deriv max diff/range {:.2e}",
                (gv - cv).abs(),
                worst / range
            );
        }
    }
}

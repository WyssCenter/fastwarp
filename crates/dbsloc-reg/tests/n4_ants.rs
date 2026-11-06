#![allow(clippy::unwrap_used, clippy::expect_used)]

//! CPU N4 against the bundled `N4BiasFieldCorrection`. Ignored by default: it runs the ANTs
//! binary (no data is checked in; the phantom is generated in the test).

use std::path::{Path, PathBuf};
use std::process::Command;

use dbsloc_reg::grid::Grid;
use dbsloc_reg::linalg::IDENTITY3;
use dbsloc_reg::n4::{run_n4, N4Params};
use dbsloc_reg::volume::Volume;

fn ants_exe() -> PathBuf {
    let sub = if cfg!(windows) {
        "win/N4BiasFieldCorrection.exe"
    } else if cfg!(target_os = "macos") {
        "mac/N4BiasFieldCorrection"
    } else {
        "linux/N4BiasFieldCorrection"
    };
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/ants")
        .join(sub);
    assert!(
        p.is_file(),
        "bundled N4BiasFieldCorrection missing: {}",
        p.display()
    );
    p
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dbsloc_reg_n4_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tissue(p: [f64; 3]) -> f64 {
    let r = ((p[0] / 30.0).powi(2) + (p[1] / 30.0).powi(2) + (p[2] / 22.0).powi(2)).sqrt();
    if r >= 1.0 {
        0.0
    } else if r >= 0.86 {
        0.35
    } else if r >= 0.76 {
        0.95
    } else if r >= 0.34 {
        0.65
    } else {
        0.45
    }
}

fn bias(p: [f64; 3]) -> f64 {
    (1.0 - 0.14 * (p[0] / 30.0).powi(2) - 0.12 * (p[1] / 30.0).powi(2) + 0.1 * (p[2] / 22.0))
        .clamp(0.7, 1.4)
}

fn cv(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    var.sqrt() / m
}

#[test]
#[ignore = "runs the bundled N4BiasFieldCorrection (CI runs it by name)"]
fn cpu_n4_agrees_with_bundled_ants() {
    let size = [40usize, 40, 30];
    let spacing = 2.0;
    let origin = [
        -(size[0] as f64 - 1.0) * spacing / 2.0,
        -(size[1] as f64 - 1.0) * spacing / 2.0,
        -(size[2] as f64 - 1.0) * spacing / 2.0,
    ];
    let g = Grid::new(size, [spacing; 3], origin, IDENTITY3).unwrap();
    let intensity: Vec<f32> = (0..g.num_voxels())
        .map(|lin| {
            let [i, j, k] = g.ijk(lin);
            let p = g.index_to_point([i as f64, j as f64, k as f64]);
            (tissue(p) * bias(p)) as f32
        })
        .collect();
    let input = Volume::from_vec(g, intensity).unwrap();

    let dir = tmp("parity");
    let in_path = dir.join("in.nii.gz");
    let out_path = dir.join("ants.nii.gz");
    input.save_nifti(&in_path).unwrap();
    let output = Command::new(ants_exe())
        .args(["-d", "3", "-i"])
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .env("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "4")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "N4 failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let ants = Volume::<f32>::load_nifti(&out_path).unwrap();

    let ours = run_n4(
        &input.to_f64(),
        &N4Params {
            shrink_factor: 4,
            ..N4Params::default()
        },
    )
    .unwrap();

    let mut ants_ratios = Vec::new();
    let mut our_ratios = Vec::new();
    for lin in 0..g.num_voxels() {
        let [i, j, k] = g.ijk(lin);
        let p = g.index_to_point([i as f64, j as f64, k as f64]);
        let t = tissue(p);
        if t <= 0.0 {
            continue;
        }
        ants_ratios.push(f64::from(ants.data()[lin]) / t);
        our_ratios.push(f64::from(ours.corrected.data()[lin]) / t);
    }
    let ants_vs_ours = {
        let mut sum = 0.0;
        let mut n = 0usize;
        for (a, o) in ants_ratios.iter().zip(&our_ratios) {
            sum += (a - o).abs();
            n += 1;
        }
        sum / n as f64
    };
    eprintln!(
        "within head: mean |ours/ants - 1| {ants_vs_ours:.6}, CV ants {:.4}, CV ours {:.4}",
        cv(&ants_ratios),
        cv(&our_ratios)
    );
    assert!(
        ants_vs_ours < 5e-3,
        "our N4 differs from ANTs by {ants_vs_ours} within the head"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

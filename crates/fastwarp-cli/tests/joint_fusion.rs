// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fastwarp_reg::linalg::IDENTITY3;
use fastwarp_reg::{Grid, Volume};

const NIFTI_UINT32: i16 = 768;
const NIFTI_FLOAT32: i16 = 16;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fastwarp_jlf_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fastwarp"))
        .arg("ants-joint-fusion")
        .args(args)
        .output()
        .unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn grid() -> Grid {
    Grid::new([10, 9, 8], [1.0, 1.0, 1.2], [-4.0, 2.0, 1.0], IDENTITY3).unwrap()
}

fn image(dir: &Path, name: &str, f: impl Fn([usize; 3]) -> f32) -> String {
    let g = grid();
    let data = (0..g.num_voxels()).map(|v| f(g.ijk(v))).collect();
    let path = dir.join(name);
    Volume::from_vec(g, data)
        .unwrap()
        .save_nifti(&path)
        .unwrap();
    path.to_str().unwrap().to_string()
}

fn wave(p: [usize; 3], shift: f32) -> f32 {
    let [x, y, z] = p.map(|v| v as f32);
    100.0 + 30.0 * (0.7 * x + shift).sin() * (0.5 * y).cos() + 5.0 * z + (x * y) % 7.0
}

fn label(p: [usize; 3], shift: usize) -> f32 {
    (1 + usize::from(p[0] + shift >= 5) + 2 * usize::from(p[2] >= 4)) as f32
}

fn datatype(path: &Path) -> i16 {
    nifti::NiftiHeader::from_file(path).unwrap().datatype
}

#[test]
fn joint_fusion_writes_the_files_ants_joint_fusion_writes() {
    let dir = tmp("files");
    let t = image(&dir, "t.nii.gz", |p| wave(p, 0.0));
    let mut args: Vec<String> = ["-d", "3", "-t", &t, "-p", "1", "-s", "1", "-v", "1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (i, shift) in [0.1f32, 0.4, 0.8].iter().enumerate() {
        args.push("-g".into());
        args.push(image(&dir, &format!("a{i}.nii.gz"), |p| wave(p, *shift)));
        args.push("-l".into());
        args.push(image(&dir, &format!("l{i}.nii"), |p| label(p, i)));
    }
    let out = dir.join("out");
    std::fs::create_dir_all(&out).unwrap();
    let o = |n: &str| out.join(n).to_str().unwrap().to_string();
    args.push("-o".into());
    args.push(format!(
        "[{},{},{},{}]",
        o("seg.nii.gz"),
        o("int%d.nii.gz"),
        o("post%02d.nii.gz"),
        o("w%d.nii")
    ));
    let r = run(&args);
    assert_eq!(
        r.status.code(),
        Some(0),
        "{}{}",
        text(&r.stdout),
        text(&r.stderr)
    );
    assert!(
        text(&r.stdout).contains("label set [1, 2, 3, 4]"),
        "{}",
        text(&r.stdout)
    );
    let files: BTreeSet<String> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let want: BTreeSet<String> = [
        "seg.nii.gz",
        "int1.nii.gz",
        "post01.nii.gz",
        "post02.nii.gz",
        "post03.nii.gz",
        "post04.nii.gz",
        "w1.nii",
        "w2.nii",
        "w3.nii",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(files, want);
    assert_eq!(datatype(&out.join("seg.nii.gz")), NIFTI_UINT32);
    assert_eq!(datatype(&out.join("post01.nii.gz")), NIFTI_FLOAT32);
    let seg = Volume::<f64>::load_nifti_f64(&out.join("seg.nii.gz")).unwrap();
    assert!(seg.grid().approx_eq(&grid(), 1e-6));
    assert!(seg.data().iter().all(|v| (1.0..=4.0).contains(v)));
    let weights: Vec<Volume<f64>> = (1..=3)
        .map(|i| Volume::<f64>::load_nifti_f64(&out.join(format!("w{i}.nii"))).unwrap())
        .collect();
    for v in 0..grid().num_voxels() {
        let sum: f64 = weights.iter().map(|w| w.data()[v]).sum();
        assert!((sum - 1.0).abs() < 1e-5, "{v}: {sum}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn large_label_ids_stay_exact_and_four_dimensional_targets_are_refused() {
    let dir = tmp("large");
    let t = image(&dir, "t.nii.gz", |p| wave(p, 0.0));
    let g = grid();
    let big = 600_000_001u32;
    let mut args: Vec<String> = ["-t", &t, "-p", "1", "-s", "1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (i, shift) in [0.1f32, 0.4].iter().enumerate() {
        args.push("-g".into());
        args.push(image(&dir, &format!("a{i}.nii.gz"), |p| wave(p, *shift)));
        let labels: Vec<u32> = (0..g.num_voxels())
            .map(|v| if g.ijk(v)[0] >= 5 { big + 2 } else { big })
            .collect();
        let path = dir.join(format!("l{i}.nii.gz"));
        fastwarp_cli::image::save_labels(&path, g, &labels).unwrap();
        args.push("-l".into());
        args.push(path.to_str().unwrap().to_string());
    }
    let seg = dir.join("seg.nii.gz");
    args.push("-o".into());
    args.push(seg.to_str().unwrap().to_string());
    let r = run(&args);
    assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
    assert_eq!(datatype(&seg), NIFTI_UINT32);
    let fused = Volume::<f64>::load_nifti_exact(&seg).unwrap();
    let values: Vec<u32> = fused.data().iter().map(|v| *v as u32).collect();
    let mut seen = values.clone();
    seen.sort_unstable();
    seen.dedup();
    assert!(
        values.contains(&big) && values.contains(&(big + 2)),
        "{seen:?}"
    );
    assert!(values.iter().all(|v| *v == big || *v == big + 2));
    let label_arg = args.iter().position(|a| a.ends_with("l1.nii.gz")).unwrap();
    for (bad, shown) in [(-1.0f32, "-1"), (2.5, "2.5"), (4294967296.0, "4294967296")] {
        let path = dir.join(format!("bad{shown}.nii"));
        let data = (0..g.num_voxels())
            .map(|v| if v == 7 { bad } else { 1.0 })
            .collect();
        Volume::from_vec(g, data)
            .unwrap()
            .save_nifti(&path)
            .unwrap();
        let mut a = args.clone();
        a[label_arg] = path.to_str().unwrap().to_string();
        let r = run(&a);
        assert_eq!(r.status.code(), Some(2), "{shown}: {}", text(&r.stderr));
        assert!(
            text(&r.stderr).contains(&format!("the value {shown} is not supported")),
            "{}",
            text(&r.stderr)
        );
    }
    let four = dir.join("four.nii.gz");
    fastwarp_niftiio::WriterOptions::new(&four)
        .write_nifti(&ndarray::Array4::<f32>::zeros((10, 9, 8, 2)))
        .unwrap();
    args[1] = four.to_str().unwrap().to_string();
    let r = run(&args);
    assert_eq!(r.status.code(), Some(2), "{}", text(&r.stderr));
    assert!(
        text(&r.stderr).contains("expected a 3D image"),
        "{}",
        text(&r.stderr)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn refusals_and_missing_files() {
    let dir = tmp("refusals");
    let t = image(&dir, "t.nii.gz", |p| wave(p, 0.0));
    let a = image(&dir, "a.nii.gz", |p| wave(p, 0.2));
    let missing = dir.join("missing.nii.gz").to_str().unwrap().to_string();
    let out = dir.join("i.nii.gz").to_str().unwrap().to_string();
    let base = |extra: &[&str], second: &str| {
        let mut v: Vec<String> = vec!["-t".into(), t.clone(), "-g".into(), a.clone()];
        v.extend([
            "-g".to_string(),
            second.to_string(),
            "-o".to_string(),
            out.clone(),
        ]);
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    let r = run(&base(&["-d", "2"], &a));
    assert_eq!(r.status.code(), Some(2), "{}", text(&r.stderr));
    assert!(text(&r.stderr).contains("ants-joint-fusion: -d 2 is not supported"));
    let r = run(&base(&[], &missing));
    assert_eq!(r.status.code(), Some(3), "{}", text(&r.stderr));
    assert!(!Path::new(&out).exists());
    let r = run(&base(&[], &a));
    assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
    assert_eq!(datatype(Path::new(&out)), NIFTI_FLOAT32);
    let r = run(&["--help".to_string()]);
    assert_eq!(r.status.code(), Some(0));
    assert!(text(&r.stdout).contains("antsJointFusion"));
    std::fs::remove_dir_all(&dir).ok();
}

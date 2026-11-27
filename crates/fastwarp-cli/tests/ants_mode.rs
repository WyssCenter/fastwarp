// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fastwarp_reg::linalg::IDENTITY3;
use fastwarp_reg::transform::itk_mat::write_mat_file;
use fastwarp_reg::transform::{
    read_transform_file, AffineTransform, TransformChain, TransformElem,
};
use fastwarp_reg::{Grid, Volume};

const NIFTI_INT16: i16 = 4;
const NIFTI_FLOAT32: i16 = 16;
const NIFTI_FLOAT64: i16 = 64;

fn exe() -> &'static str {
    env!("CARGO_BIN_EXE_fastwarp")
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fastwarp_ants_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[String]) -> Output {
    Command::new(exe()).args(args).output().unwrap()
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn argv(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

fn truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn gpu() -> bool {
    let o = run(&argv(&["gpu-info"]));
    match code(&o) {
        0 => true,
        4 => {
            assert!(
                !truthy("FASTWARP_GPU_REQUIRE_ADAPTER"),
                "no GPU adapter with FASTWARP_GPU_REQUIRE_ADAPTER set: {}",
                text(&o.stderr)
            );
            eprintln!("skip: no GPU adapter ({})", text(&o.stderr).trim());
            false
        }
        c => panic!("gpu-info exit {c}: {}", text(&o.stderr)),
    }
}

fn blob(g: &Grid, centre: [f64; 3], sigma: f64) -> Vec<f64> {
    (0..g.num_voxels())
        .map(|lin| {
            let [i, j, k] = g.ijk(lin);
            let (x, y, z) = (
                i as f64 - centre[0],
                j as f64 - centre[1],
                (k as f64 - centre[2]) * 1.1,
            );
            (-(x * x + y * y + z * z) / (2.0 * sigma * sigma)).exp()
        })
        .collect()
}

/// The 24³ pair of two Gaussian blobs at 2 mm the ANTs observations of the design used.
fn blob_pair(dir: &Path) -> (PathBuf, PathBuf) {
    let g = Grid::new([24; 3], [2.0; 3], [0.0; 3], IDENTITY3).unwrap();
    let pair = [
        ([12.0, 12.0, 12.0], 4.0, [8.0, 14.0, 12.0], 2.0),
        ([13.0, 11.5, 12.5], 4.3, [9.5, 13.0, 12.0], 2.2),
    ];
    let mut out = Vec::new();
    for (name, (c1, s1, c2, s2)) in ["f.nii.gz", "m.nii.gz"].iter().zip(pair) {
        let a = blob(&g, c1, s1);
        let b = blob(&g, c2, s2);
        let data: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x + 0.5 * y) as f32)
            .collect();
        let path = dir.join(name);
        Volume::from_vec(g, data)
            .unwrap()
            .save_nifti(&path)
            .unwrap();
        out.push(path);
    }
    (out[0].clone(), out[1].clone())
}

fn linear_stages(f: &Path, m: &Path) -> Vec<String> {
    let metric = format!("MI[{},{},1,32,Regular,0.25]", s(f), s(m));
    let mut v = Vec::new();
    for t in ["Rigid[0.1]", "Affine[0.1]"] {
        v.extend(argv(&[
            "-t",
            t,
            "-m",
            &metric,
            "-c",
            "[10x5,1e-6,10]",
            "-f",
            "2x1",
            "-s",
            "1x0vox",
        ]));
    }
    v
}

fn syn_stage(f: &Path, m: &Path) -> Vec<String> {
    argv(&[
        "-t",
        "SyN[0.1,3,0]",
        "-m",
        &format!("CC[{},{},1,2]", s(f), s(m)),
        "-c",
        "[5x3,1e-6,10]",
        "-f",
        "2x1",
        "-s",
        "1x0vox",
    ])
}

fn com(f: &Path, m: &Path) -> Vec<String> {
    argv(&["-r", &format!("[{},{},1]", s(f), s(m))])
}

fn files_in(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

fn datatype(path: &Path) -> i16 {
    nifti::NiftiHeader::from_file(path).unwrap().datatype
}

fn kinds(path: &Path) -> Vec<&'static str> {
    read_transform_file(path)
        .unwrap()
        .queue()
        .iter()
        .map(|t| match t {
            TransformElem::Affine(_) => "affine",
            TransformElem::DisplacementField(_) => "field",
        })
        .collect()
}

fn load(path: &Path) -> Volume<f64> {
    Volume::<f64>::load_nifti_f64(path).unwrap()
}

fn max_abs_diff(a: &Volume<f64>, b: &Volume<f64>) -> f64 {
    assert!(
        a.grid().approx_eq(b.grid(), 1e-4),
        "{:?} {:?}",
        a.grid(),
        b.grid()
    );
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn ncc(a: &Volume<f64>, b: &Volume<f64>) -> f64 {
    let n = a.data().len() as f64;
    let ma = a.data().iter().sum::<f64>() / n;
    let mb = b.data().iter().sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.data().iter().zip(b.data()) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

fn names(list: &[&str]) -> BTreeSet<String> {
    list.iter().map(|s| s.to_string()).collect()
}

fn ok(o: &Output, what: &str) {
    assert_eq!(code(o), 0, "{what}: {}{}", text(&o.stdout), text(&o.stderr));
}

fn write_affine_txt(path: &Path, matrix: [f64; 9], translation: [f64; 3]) {
    let p: Vec<String> = matrix
        .iter()
        .chain(translation.iter())
        .map(|v| v.to_string())
        .collect();
    std::fs::write(
        path,
        format!(
            "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\n\
             Parameters: {}\nFixedParameters: 1 2 3\n",
            p.join(" ")
        ),
    )
    .unwrap();
}

#[test]
fn refusals_exit_2_before_reading_or_writing_anything() {
    let d = tmp("refusals");
    let (f, m) = blob_pair(&d);
    let out = d.join("out");
    std::fs::create_dir_all(&out).unwrap();
    let prefix = s(&out.join("o_"));
    let cases: Vec<(Vec<String>, &str)> = vec![
        (
            [
                com(&f, &m),
                syn_stage(&f, &m),
                argv(&["-o", &prefix, "-u", "1"]),
            ]
            .concat(),
            "fastwarp: error: ants-registration: -u 1 is not supported: FastWarp has no \
             histogram matching.",
        ),
        (
            [
                argv(&["-m", &format!("MI[{},{},1,32,Regular,0.25]", s(&f), s(&m))]),
                argv(&[
                    "-t", "SyN[0.1]", "-c", "5", "-f", "1", "-s", "0", "-o", &prefix,
                ]),
            ]
            .concat(),
            "fastwarp: error: ants-registration: metric MI with Regular sampling in a SyN stage \
             is not supported: FastWarp's SyN metrics are dense; leave the sampling strategy out \
             or give None.",
        ),
        (
            [
                syn_stage(&f, &m),
                argv(&["-o", &s(&d.join("missing").join("o_"))]),
            ]
            .concat(),
            "fastwarp: error: ants-registration: output folder",
        ),
        (
            [
                syn_stage(&d.join("nope.nii.gz"), &m),
                argv(&["-o", &prefix]),
            ]
            .concat(),
            "fastwarp: error: ants-registration: fixed image",
        ),
        (
            [
                com(&f, &m),
                linear_stages(&f, &m),
                syn_stage(&f, &m),
                argv(&["-o", &prefix, "-x", &format!("[{},{}]", s(&f), s(&m))]),
            ]
            .concat(),
            "fastwarp: error: ants-registration: a moving-image mask on the Rigid stage \
             (--masks) is not supported: FastWarp applies a moving-image mask to SyN stages \
             only; give NULL as the moving mask of that stage.",
        ),
    ];
    for (i, (mut args, want)) in cases.into_iter().enumerate() {
        args.insert(0, "ants-registration".into());
        let o = run(&args);
        let err = text(&o.stderr);
        let status = code(&o);
        assert!(
            status == 2 || (i == 3 && status == 3),
            "{args:?}: {status} {err}"
        );
        assert!(err.starts_with(want), "{args:?}: {err}");
        assert_eq!(err.lines().count(), 1, "{err}");
        assert!(o.stdout.is_empty());
        assert!(files_in(&out).is_empty(), "{:?}", files_in(&out));
    }
    assert!(!d.join("missing").exists());
    let o = run(&argv(&[
        "ants-apply-transforms",
        "-i",
        &s(&m),
        "-r",
        &s(&f),
        "-o",
        &s(&out.join("w.nii.gz")),
        "-n",
        "GenericLabel",
    ]));
    assert_eq!(code(&o), 2);
    assert_eq!(
        text(&o.stderr).trim(),
        "fastwarp: error: ants-apply-transforms: interpolation GenericLabel is not supported: \
         use Linear, NearestNeighbor or LanczosWindowedSinc."
    );
    assert!(files_in(&out).is_empty());
}

#[test]
fn apply_transform_outputs_need_no_gpu() {
    let d = tmp("apply_outputs");
    let (f, m) = blob_pair(&d);
    let a = d.join("a.txt");
    write_affine_txt(
        &a,
        [1.02, 0.05, 0.0, -0.04, 0.98, 0.01, 0.0, 0.02, 1.01],
        [1.5, -2.0, 0.5],
    );
    let b = d.join("b.txt");
    write_affine_txt(
        &b,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, -1.0],
    );
    let ants = |extra: &[&str]| {
        let mut v = argv(&["ants-apply-transforms", "-d", "3"]);
        v.extend(argv(extra));
        run(&v)
    };
    let (sa, sb, sf, sm) = (s(&a), s(&b), s(&f), s(&m));
    let lin = d.join("l.mat");
    let o = ants(&[
        "-o",
        &format!("Linear[{},1]", s(&lin)),
        "-t",
        &sa,
        "-t",
        &sb,
    ]);
    ok(&o, "Linear");
    let chain = read_transform_file(&lin).unwrap();
    let TransformElem::Affine(l) = chain.queue()[0] else {
        panic!("not an affine")
    };
    let ta = read_transform_file(&a).unwrap();
    let tb = read_transform_file(&b).unwrap();
    for p in [[0.0, 0.0, 0.0], [10.0, -4.0, 30.0]] {
        let q = tb.apply(ta.apply(p));
        let back = l.apply(q);
        for k in 0..3 {
            assert!((back[k] - p[k]).abs() < 1e-9, "{back:?} {p:?}");
        }
    }
    let comp = d.join("c.h5");
    ok(
        &ants(&[
            "-o",
            &format!("CompositeTransform[{}]", s(&comp)),
            "-t",
            &sa,
            "-t",
            &format!("[{sb},1]"),
        ]),
        "CompositeTransform",
    );
    let c = read_transform_file(&comp).unwrap();
    assert_eq!(kinds(&comp), vec!["affine", "affine"]);
    let binv = AffineTransform::from_translation([-3.0, 0.0, 1.0]);
    let p = [5.0, 6.0, 7.0];
    let want = binv.apply(ta.apply(p));
    let got = c.apply(p);
    for k in 0..3 {
        assert!((got[k] - want[k]).abs() < 1e-9, "{got:?} {want:?}");
    }
    for (extra, out, dt) in [
        (vec![], "d64.nii.gz", NIFTI_FLOAT64),
        (vec!["--float", "1"], "d32.nii.gz", NIFTI_FLOAT32),
    ] {
        let path = d.join(out);
        let mut args = vec!["-r", &sf, "-o"];
        let target = format!("[{},1]", s(&path));
        args.push(&target);
        args.extend(["-t", &sa]);
        args.extend(extra);
        ok(&ants(&args), out);
        assert_eq!(datatype(&path), dt, "{out}");
        let field = read_transform_file(&path).unwrap();
        let q = [10.0, 12.0, 14.0];
        let (x, y) = (field.apply(q), ta.apply(q));
        for k in 0..3 {
            assert!((x[k] - y[k]).abs() < 1e-4, "{x:?} {y:?}");
        }
    }
    let h5 = d.join("d.h5");
    ok(
        &ants(&["-r", &sf, "-o", &format!("[{},1]", s(&h5)), "-t", &sa]),
        "h5 field",
    );
    assert_eq!(kinds(&h5), vec!["field"]);
    for (extra, out, dt) in [
        (vec![], "w64.nii.gz", NIFTI_FLOAT64),
        (vec!["--float"], "w32.nii.gz", NIFTI_FLOAT32),
        (vec!["-u", "short"], "w16.nii.gz", NIFTI_INT16),
    ] {
        let path = d.join(out);
        let ps = s(&path);
        let mut args = vec!["-i", &sm, "-r", &sf, "-o", &ps, "-t", &sb];
        args.extend(extra);
        ok(&ants(&args), out);
        assert_eq!(datatype(&path), dt, "{out}");
    }
    let w64 = load(&d.join("w64.nii.gz"));
    let w32 = load(&d.join("w32.nii.gz"));
    assert!(max_abs_diff(&w64, &w32) < 1e-6);
    let o = ants(&[
        "-o",
        &format!("Linear[{}]", s(&d.join("x.mat"))),
        "-t",
        &s(&d.join("d64.nii.gz")),
    ]);
    assert_eq!(code(&o), 2, "{}", text(&o.stderr));
    assert!(text(&o.stderr).contains("Linear[...] output is not supported"));
}

fn register(args: Vec<String>) -> Output {
    let mut v = argv(&["ants-registration"]);
    v.extend(args);
    run(&v)
}

#[test]
fn registration_writes_the_files_ants_writes() {
    if !gpu() {
        return;
    }
    let d = tmp("registration");
    let (f, m) = blob_pair(&d);
    let full = d.join("full");
    std::fs::create_dir_all(&full).unwrap();
    let o = register(
        [
            argv(&["-d", "3"]),
            com(&f, &m),
            linear_stages(&f, &m),
            syn_stage(&f, &m),
            argv(&[
                "-o",
                &format!(
                    "[{},{},{}]",
                    s(&full.join("o_")),
                    s(&full.join("W.nii.gz")),
                    s(&full.join("IW.nii.gz"))
                ),
                "-v",
                "1",
            ]),
        ]
        .concat(),
    );
    ok(&o, "full chain");
    assert!(text(&o.stdout).contains("syn level"), "{}", text(&o.stdout));
    assert_eq!(
        files_in(&full),
        names(&[
            "IW.nii.gz",
            "W.nii.gz",
            "o_0GenericAffine.mat",
            "o_1InverseWarp.nii.gz",
            "o_1Warp.nii.gz"
        ])
    );
    for name in [
        "W.nii.gz",
        "IW.nii.gz",
        "o_1Warp.nii.gz",
        "o_1InverseWarp.nii.gz",
    ] {
        assert_eq!(datatype(&full.join(name)), NIFTI_FLOAT64, "{name}");
    }
    let fixed = load(&f);
    let warped = load(&full.join("W.nii.gz"));
    assert!(warped.grid().approx_eq(fixed.grid(), 1e-6));
    let moving = load(&m);
    assert!(ncc(&fixed, &warped) > ncc(&fixed, &moving));
    let inverse = load(&full.join("IW.nii.gz"));
    assert!(inverse.grid().approx_eq(moving.grid(), 1e-6));
    assert!(ncc(&moving, &inverse) > ncc(&moving, &fixed));
    let again = d.join("again.nii.gz");
    let o = run(&argv(&[
        "ants-apply-transforms",
        "-d",
        "3",
        "-i",
        &s(&m),
        "-r",
        &s(&f),
        "-o",
        &s(&again),
        "-t",
        &s(&full.join("o_1Warp.nii.gz")),
        "-t",
        &s(&full.join("o_0GenericAffine.mat")),
    ]));
    ok(&o, "apply");
    assert!(max_abs_diff(&load(&again), &warped) < 1e-5);

    let syn_only = d.join("syn_only");
    std::fs::create_dir_all(&syn_only).unwrap();
    let o = register([syn_stage(&f, &m), argv(&["-o", &s(&syn_only.join("o_"))])].concat());
    ok(&o, "SyN only");
    assert_eq!(
        files_in(&syn_only),
        names(&["o_0InverseWarp.nii.gz", "o_0Warp.nii.gz"])
    );
    let comp = d.join("composite");
    std::fs::create_dir_all(&comp).unwrap();
    let o = register(
        [
            syn_stage(&f, &m),
            argv(&["-o", &s(&comp.join("o_")), "-a", "1", "--float", "1"]),
        ]
        .concat(),
    );
    ok(&o, "SyN only, composite");
    assert_eq!(
        files_in(&comp),
        names(&["o_Composite.h5", "o_InverseComposite.h5"])
    );
    assert_eq!(kinds(&comp.join("o_Composite.h5")), vec!["field"]);
    let lin = d.join("linear");
    std::fs::create_dir_all(&lin).unwrap();
    let o = register(
        [
            com(&f, &m),
            linear_stages(&f, &m),
            argv(&[
                "-o",
                &format!("[{},{}]", s(&lin.join("o")), s(&lin.join("W.nii"))),
                "--float",
                "-x",
                &format!("[{},NULL]", s(&f)),
            ]),
        ]
        .concat(),
    );
    ok(&o, "linear only");
    assert_eq!(files_in(&lin), names(&["W.nii", "o0GenericAffine.mat"]));
    assert_eq!(datatype(&lin.join("W.nii")), NIFTI_FLOAT32);
    let g = *fixed.grid();
    let box_mask: Vec<f32> = (0..g.num_voxels())
        .map(|i| {
            let [x, y, z] = g.ijk(i);
            if [x, y, z].iter().all(|c| (6..14).contains(c)) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mask = d.join("mask.nii.gz");
    Volume::from_vec(g, box_mask)
        .unwrap()
        .save_nifti(&mask)
        .unwrap();
    let mut params = Vec::new();
    for (name, x) in [("unmasked", "NULL".to_string()), ("masked", s(&mask))] {
        let dir = d.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let o = register(
            [
                com(&f, &m),
                linear_stages(&f, &m),
                argv(&["-o", &s(&dir.join("o_")), "-x", &format!("[{x},NULL]")]),
            ]
            .concat(),
        );
        ok(&o, name);
        let chain = read_transform_file(&dir.join("o_0GenericAffine.mat")).unwrap();
        let TransformElem::Affine(a) = chain.queue()[0] else {
            panic!("not an affine")
        };
        params.push(a.parameters());
    }
    assert_ne!(
        params[0], params[1],
        "the mask did not reach the linear stages"
    );
    let mut warps = Vec::new();
    for (name, x) in [
        ("syn_unmasked", "[NULL,NULL]".to_string()),
        ("syn_masked", format!("[{},{}]", s(&mask), s(&mask))),
        ("syn_moving_masked", format!("[NULL,{}]", s(&mask))),
    ] {
        let dir = d.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let o = register(
            [
                syn_stage(&f, &m),
                argv(&["-o", &s(&dir.join("o_")), "-x", &x]),
            ]
            .concat(),
        );
        ok(&o, name);
        assert_eq!(
            files_in(&dir),
            names(&["o_0InverseWarp.nii.gz", "o_0Warp.nii.gz"])
        );
        warps.push(std::fs::read(dir.join("o_0Warp.nii.gz")).unwrap());
    }
    assert_ne!(warps[0], warps[1], "the masks did not reach the SyN stage");
    assert_ne!(
        warps[0], warps[2],
        "the moving mask did not reach the SyN stage"
    );
    assert_ne!(warps[1], warps[2]);
}

fn ants_program(name: &str) -> PathBuf {
    let dir = std::env::var_os("FASTWARP_ANTS_DIR").expect(
        "FASTWARP_ANTS_DIR is not set: this ignored test compares FastWarp with ANTs 2.6.5 and \
         needs the directory holding antsRegistration and antsApplyTransforms",
    );
    let file = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let p = Path::new(&dir).join(file);
    assert!(
        p.is_file(),
        "FASTWARP_ANTS_DIR is set but {} is missing",
        p.display()
    );
    p
}

fn require_gpu() {
    let o = run(&argv(&["gpu-info"]));
    assert_eq!(
        code(&o),
        0,
        "this ignored test needs a GPU adapter: {}",
        text(&o.stderr)
    );
}

fn run_ants(program: &Path, args: &[String]) -> Output {
    Command::new(program)
        .args(args)
        .env("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
        .output()
        .unwrap()
}

fn forward_transforms(dir: &Path, files: &BTreeSet<String>) -> Vec<String> {
    let pick = |pred: &dyn Fn(&str) -> bool| -> Vec<String> {
        files
            .iter()
            .filter(|n| pred(n))
            .map(|n| s(&dir.join(n)))
            .collect()
    };
    let composite = pick(&|n| n.ends_with("_Composite.h5") && !n.contains("Inverse"));
    if !composite.is_empty() {
        return composite;
    }
    let mut t = pick(&|n| n.ends_with("Warp.nii.gz") && !n.contains("Inverse"));
    t.extend(pick(&|n| n.ends_with("GenericAffine.mat")));
    t
}

fn gauss(p: [f64; 3], c: [f64; 3], sigma: f64) -> f64 {
    let r2 = (0..3).map(|a| (p[a] - c[a]).powi(2)).sum::<f64>();
    (-r2 / (2.0 * sigma * sigma)).exp()
}

/// An asymmetric scene of three blobs, in millimetres.
fn scene(p: [f64; 3]) -> f64 {
    gauss(p, [23.0, 23.0, 23.0], 8.0)
        + 0.6 * gauss(p, [15.0, 29.0, 23.0], 4.0)
        + 0.5 * gauss(p, [31.0, 16.0, 29.0], 3.5)
}

/// Where a moving-image point shows the scene: a rotation of 8 degrees about z and 5 about x
/// around the centre, a shift of (4, -3, 2.5) mm and a smooth bend of up to 1.5 mm.
fn scene_point_of_moving(x: [f64; 3]) -> [f64; 3] {
    let c = [23.0, 23.0, 23.0];
    let (sz, cz) = 8f64.to_radians().sin_cos();
    let (sx, cx) = 5f64.to_radians().sin_cos();
    let v = [x[0] - c[0], x[1] - c[1], x[2] - c[2]];
    let v = [cz * v[0] - sz * v[1], sz * v[0] + cz * v[1], v[2]];
    let v = [v[0], cx * v[1] - sx * v[2], sx * v[1] + cx * v[2]];
    [
        v[0] + c[0] + 4.0 + 1.5 * (x[1] / 9.0).sin(),
        v[1] + c[1] - 3.0 + 1.2 * (x[0] / 11.0).cos(),
        v[2] + c[2] + 2.5 + (x[2] / 10.0).sin(),
    ]
}

/// A fixed image of the scene on `size` voxels of `spacing` mm and a moving image (24³, 2 mm)
/// that shows it moved, rotated and bent, so that a run that does nothing scores badly.
fn moved_pair(dir: &Path, size: [usize; 3], spacing: [f64; 3]) -> (PathBuf, PathBuf) {
    let fixed = Grid::new(size, spacing, [0.0; 3], IDENTITY3).unwrap();
    let moving = Grid::new([24; 3], [2.0; 3], [0.0; 3], IDENTITY3).unwrap();
    let mut out = Vec::new();
    for (name, g, moved) in [("f.nii.gz", fixed, false), ("m.nii.gz", moving, true)] {
        let data: Vec<f32> = (0..g.num_voxels())
            .map(|i| {
                let [a, b, c] = g.ijk(i);
                let p = g.index_to_point([a as f64, b as f64, c as f64]);
                let p = if moved { scene_point_of_moving(p) } else { p };
                scene(p) as f32
            })
            .collect();
        let path = dir.join(name);
        Volume::from_vec(g, data)
            .unwrap()
            .save_nifti(&path)
            .unwrap();
        out.push(path);
    }
    (out[0].clone(), out[1].clone())
}

fn chain_of(paths: &[PathBuf]) -> TransformChain {
    let mut c = TransformChain::new();
    for p in paths.iter().rev() {
        c.extend(read_transform_file(p).unwrap());
    }
    c
}

/// The forward map of a run (fixed to moving points) and its linear part alone.
fn maps(dir: &Path) -> (TransformChain, TransformChain) {
    let files = files_in(dir);
    let full: Vec<PathBuf> = forward_transforms(dir, &files)
        .into_iter()
        .map(PathBuf::from)
        .collect();
    let all = chain_of(&full);
    let linear = TransformChain::from_queue(
        all.queue()
            .iter()
            .filter(|t| matches!(t, TransformElem::Affine(_)))
            .cloned()
            .collect(),
    );
    (all, linear)
}

fn corners(g: &Grid) -> Vec<[f64; 3]> {
    let n = g.size();
    (0..8)
        .map(|c| {
            let idx: Vec<f64> = (0..3)
                .map(|a| {
                    if (c >> a) & 1 == 1 {
                        (n[a] - 1) as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            g.index_to_point([idx[0], idx[1], idx[2]])
        })
        .collect()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

fn max_dist(a: &TransformChain, b: &TransformChain, points: &[[f64; 3]]) -> f64 {
    points
        .iter()
        .map(|p| dist(a.apply(*p), b.apply(*p)))
        .fold(0.0, f64::max)
}

fn rms_dist(a: &TransformChain, b: &TransformChain, points: &[[f64; 3]]) -> f64 {
    let sum: f64 = points
        .iter()
        .map(|p| dist(a.apply(*p), b.apply(*p)).powi(2))
        .sum();
    (sum / points.len() as f64).sqrt()
}

/// Voxel centres of `fixed` inside the object (above a fifth of the maximum).
fn object_points(fixed: &Volume<f64>) -> Vec<[f64; 3]> {
    let g = fixed.grid();
    let top = fixed.data().iter().cloned().fold(f64::MIN, f64::max);
    (0..g.num_voxels())
        .filter(|i| fixed.data()[*i] > 0.2 * top)
        .map(|i| {
            let [a, b, c] = g.ijk(i);
            g.index_to_point([a as f64, b as f64, c as f64])
        })
        .collect()
}

/// `image` on the grid of `reference` through the identity, so images of two grids can be
/// compared voxel by voxel.
fn on_grid(image: &Path, reference: &Path, out: &Path) -> Volume<f64> {
    ok(
        &run(&argv(&[
            "ants-apply-transforms",
            "-i",
            &s(image),
            "-r",
            &s(reference),
            "-o",
            &s(out),
        ])),
        "identity resample",
    );
    load(out)
}

fn both_outputs(out: &Path) -> Vec<String> {
    vec![
        "-o".to_string(),
        format!(
            "[{},{},{}]",
            s(&out.join("o_")),
            s(&out.join("W.nii.gz")),
            s(&out.join("IW.nii.gz"))
        ),
    ]
}

fn stages(f: &Path, m: &Path, kinds: &[&str]) -> Vec<String> {
    let mut v = Vec::new();
    for k in kinds {
        if *k == "SyN" {
            v.extend(argv(&[
                "-t",
                "SyN[0.2,3,0]",
                "-m",
                &format!("CC[{},{},1,2]", s(f), s(m)),
                "-c",
                "[20x10,1e-6,10]",
                "-f",
                "2x1",
                "-s",
                "1x0vox",
            ]));
        } else {
            v.extend(argv(&[
                "-t",
                &format!("{k}[0.1]"),
                "-m",
                &format!("MI[{},{},1,32,Regular,0.5]", s(f), s(m)),
                "-c",
                "[100x50,1e-6,10]",
                "-f",
                "2x1",
                "-s",
                "1x0vox",
            ]));
        }
    }
    v
}

/// Runs `args` (with `out` the output folder) with antsRegistration into `<case>/ants` and
/// with FastWarp into `<case>/fastwarp`; checks that both write the same files, of the same
/// voxel types, and returns both folders.
fn run_both(
    reg: &Path,
    case: &Path,
    name: &str,
    args: &dyn Fn(&Path) -> Vec<String>,
) -> (PathBuf, PathBuf) {
    let theirs = case.join("ants");
    let mine = case.join("fastwarp");
    std::fs::create_dir_all(&theirs).unwrap();
    std::fs::create_dir_all(&mine).unwrap();
    let mut a = argv(&["-d", "3", "--random-seed", "1"]);
    a.extend(args(&theirs));
    ok(&run_ants(reg, &a), &format!("{name}: antsRegistration"));
    let mut b = argv(&["ants-registration", "-d", "3", "--random-seed", "1"]);
    b.extend(args(&mine));
    ok(&run(&b), &format!("{name}: fastwarp"));
    let (ft, fm) = (files_in(&theirs), files_in(&mine));
    assert_eq!(ft, fm, "{name}: file names");
    for n in &ft {
        if n.ends_with(".nii.gz") {
            let (x, y) = (
                nifti::NiftiHeader::from_file(theirs.join(n)).unwrap(),
                nifti::NiftiHeader::from_file(mine.join(n)).unwrap(),
            );
            assert_eq!(x.datatype, y.datatype, "{name}: {n} voxel type");
            assert_eq!(x.dim, y.dim, "{name}: {n} dimensions");
            for k in 1..6 {
                assert!(
                    (x.pixdim[k] - y.pixdim[k]).abs() < 1e-4,
                    "{name}: {n} pixdim {:?} {:?}",
                    x.pixdim,
                    y.pixdim
                );
            }
        }
        if n.ends_with(".h5") {
            assert_eq!(
                kinds(&theirs.join(n)),
                kinds(&mine.join(n)),
                "{name}: {n} transforms"
            );
        }
    }
    (theirs, mine)
}

/// Checks that FastWarp's warped images agree with ANTs' far better than the unregistered
/// images do, and that antsApplyTransforms reproduces FastWarp's warped image from its files.
fn check_warped(
    apply: &Path,
    name: &str,
    f: &Path,
    m: &Path,
    theirs: &Path,
    mine: &Path,
) -> [f64; 4] {
    let start_w = on_grid(m, f, &mine.join("m_on_f.nii.gz"));
    let start_iw = on_grid(f, m, &mine.join("f_on_m.nii.gz"));
    let (w_ants, iw_ants) = (
        load(&theirs.join("W.nii.gz")),
        load(&theirs.join("IW.nii.gz")),
    );
    let w = ncc(&load(&mine.join("W.nii.gz")), &w_ants);
    let iw = ncc(&load(&mine.join("IW.nii.gz")), &iw_ants);
    let (w0, iw0) = (ncc(&start_w, &w_ants), ncc(&start_iw, &iw_ants));
    eprintln!("{name}: NCC with ANTs W {w:.4} (unregistered {w0:.4}), IW {iw:.4} ({iw0:.4})");
    assert!(
        w > 0.99 && w - w0 > 0.05,
        "{name}: W {w} vs unregistered {w0}"
    );
    assert!(
        iw > 0.99 && iw - iw0 > 0.05,
        "{name}: IW {iw} vs unregistered {iw0}"
    );
    let check = mine.join("check.nii.gz");
    let mut a = argv(&[
        "-d",
        "3",
        "-i",
        &s(m),
        "-r",
        &s(f),
        "-o",
        &s(&check),
        "-n",
        "Linear",
    ]);
    for t in forward_transforms(mine, &files_in(mine)) {
        a.extend(["-t".to_string(), t]);
    }
    ok(
        &run_ants(apply, &a),
        &format!("{name}: antsApplyTransforms"),
    );
    let diff = max_abs_diff(&load(&check), &load(&mine.join("W.nii.gz")));
    assert!(
        diff < 1e-4,
        "{name}: ANTs reads FastWarp's transforms: {diff}"
    );
    [w, w0, iw, iw0]
}

#[test]
#[ignore = "runs antsRegistration and antsApplyTransforms from FASTWARP_ANTS_DIR and needs a GPU; \
            run with --ignored"]
fn the_output_layout_matches_ants_registration() {
    let reg = ants_program("antsRegistration");
    let apply = ants_program("antsApplyTransforms");
    require_gpu();
    let d = tmp("layout");
    let (f, m) = moved_pair(&d, [24; 3], [2.0; 3]);
    type Case = (&'static str, Vec<String>);
    let r = com(&f, &m);
    let lin = stages(&f, &m, &["Rigid", "Affine"]);
    let full = stages(&f, &m, &["Rigid", "Affine", "SyN"]);
    let syn = stages(&f, &m, &["SyN"]);
    let cases: Vec<Case> = vec![
        ("default", [r.clone(), full.clone()].concat()),
        (
            "composite",
            [r.clone(), full.clone(), argv(&["-a", "1", "-z", "1"])].concat(),
        ),
        (
            "float",
            [r.clone(), full.clone(), argv(&["--float", "1"])].concat(),
        ),
        ("linear", [r.clone(), lin.clone()].concat()),
        (
            "linear_composite",
            [r.clone(), lin, argv(&["-a", "1"])].concat(),
        ),
        (
            "identity",
            [argv(&["-r", "Identity"]), syn.clone()].concat(),
        ),
        ("syn_only", syn.clone()),
        ("syn_only_composite", [syn, argv(&["-a", "1"])].concat()),
    ];
    for (name, args) in &cases {
        let case = d.join(name);
        let (theirs, mine) = run_both(&reg, &case, name, &|o| {
            [args.clone(), both_outputs(o)].concat()
        });
        check_warped(&apply, name, &f, &m, &theirs, &mine);
    }
}

#[test]
#[ignore = "runs antsRegistration and antsApplyTransforms from FASTWARP_ANTS_DIR and needs a GPU; \
            run with --ignored"]
fn registrations_agree_with_ants() {
    let reg = ants_program("antsRegistration");
    let apply = ants_program("antsApplyTransforms");
    require_gpu();
    let d = tmp("agree");
    let iso = d.join("iso");
    let aniso = d.join("aniso");
    std::fs::create_dir_all(&iso).unwrap();
    std::fs::create_dir_all(&aniso).unwrap();
    let (f, m) = moved_pair(&iso, [24; 3], [2.0; 3]);
    let (fa, ma) = moved_pair(&aniso, [24, 24, 16], [2.0, 2.0, 3.0]);
    let start = AffineTransform::from_translation([-4.0, 3.0, -2.0]);
    let start_path = d.join("start.mat");
    let start_inv_path = d.join("start_inv.mat");
    write_mat_file(&start_path, &[TransformElem::Affine(start)]).unwrap();
    write_mat_file(
        &start_inv_path,
        &[TransformElem::Affine(start.inverse().unwrap())],
    )
    .unwrap();
    let start_chain = TransformChain::from_queue(vec![TransformElem::Affine(start)]);
    let cases: Vec<(&str, PathBuf, PathBuf, Vec<String>)> = vec![
        (
            "com",
            f.clone(),
            m.clone(),
            [com(&f, &m), stages(&f, &m, &["Rigid", "Affine", "SyN"])].concat(),
        ),
        (
            "file",
            f.clone(),
            m.clone(),
            [
                argv(&["-r", &s(&start_path)]),
                stages(&f, &m, &["Affine", "SyN"]),
            ]
            .concat(),
        ),
        (
            "inverted_file",
            f.clone(),
            m.clone(),
            [
                argv(&["-r", &format!("[{},1]", s(&start_inv_path))]),
                stages(&f, &m, &["Affine", "SyN"]),
            ]
            .concat(),
        ),
        (
            "anisotropic",
            fa.clone(),
            ma.clone(),
            [com(&fa, &ma), stages(&fa, &ma, &["Rigid", "Affine", "SyN"])].concat(),
        ),
    ];
    let identity = TransformChain::new();
    for (name, f, m, args) in &cases {
        let case = d.join(name);
        let (theirs, mine) = run_both(&reg, &case, name, &|o| {
            [args.clone(), both_outputs(o)].concat()
        });
        check_warped(&apply, name, f, m, &theirs, &mine);
        let fixed = load(f);
        let (ants_all, ants_linear) = maps(&theirs);
        let (mine_all, mine_linear) = maps(&mine);
        let box_corners = corners(fixed.grid());
        let linear_err = max_dist(&mine_linear, &ants_linear, &box_corners);
        let linear_noop = max_dist(&identity, &ants_linear, &box_corners).min(max_dist(
            &start_chain,
            &ants_linear,
            &box_corners,
        ));
        let inside = object_points(&fixed);
        let full_err = rms_dist(&mine_all, &ants_all, &inside);
        let warp_noop = rms_dist(&ants_linear, &ants_all, &inside);
        let full_noop = rms_dist(&identity, &ants_all, &inside);
        eprintln!(
            "{name}: affine corners max {linear_err:.3} mm (no-op {linear_noop:.3}); forward map \
             RMS in the object {full_err:.3} mm (ANTs affine alone {warp_noop:.3}, identity \
             {full_noop:.3}); {} points",
            inside.len()
        );
        let (corner_bound, map_ratio) = if *name == "anisotropic" {
            (2.0, 0.6)
        } else {
            (1.0, 0.25)
        };
        assert!(
            linear_err < corner_bound && linear_err < 0.2 * linear_noop,
            "{name}: affine corners {linear_err} mm, no-op {linear_noop}"
        );
        assert!(
            full_err < 0.5 && full_err < map_ratio * warp_noop,
            "{name}: forward map {full_err} mm, ANTs affine alone {warp_noop}"
        );
    }
}

/// The pair of [`moved_pair`] and its moving image in another contrast, `1.2 - 1.5 v^2`, which
/// only a metric without a linear intensity relation registers.
fn contrast_pair(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let (f, m) = moved_pair(dir, [24; 3], [2.0; 3]);
    let moving = load(&m);
    let other: Vec<f32> = moving
        .data()
        .iter()
        .map(|v| (1.2 - 1.5 * v * v) as f32)
        .collect();
    let path = dir.join("m_contrast.nii.gz");
    Volume::from_vec(*moving.grid(), other)
        .unwrap()
        .save_nifti(&path)
        .unwrap();
    (f, m, path)
}

#[test]
#[ignore = "runs antsRegistration and antsApplyTransforms from FASTWARP_ANTS_DIR and needs a GPU; \
            run with --ignored"]
fn syn_with_mattes_agrees_with_ants() {
    let reg = ants_program("antsRegistration");
    let apply = ants_program("antsApplyTransforms");
    require_gpu();
    let d = tmp("syn_mattes");
    let (f, same, m) = contrast_pair(&d);
    let mi_syn = |extra: &[&str]| {
        let mut v = argv(&[
            "-t",
            "SyN[0.2,3,0]",
            "-m",
            &format!("MI[{},{},1,32]", s(&f), s(&m)),
        ]);
        v.extend(argv(extra));
        v.extend(argv(&[
            "-c",
            "[20x10,1e-6,10]",
            "-f",
            "2x1",
            "-s",
            "1x0vox",
        ]));
        v
    };
    let mixed = mi_syn(&["-m", &format!("CC[{},{},1,2]", s(&f), s(&same))]);
    let cases: Vec<(&str, Vec<String>)> = vec![
        (
            "mattes",
            [
                com(&f, &m),
                stages(&f, &m, &["Rigid", "Affine"]),
                mi_syn(&[]),
            ]
            .concat(),
        ),
        ("mattes_only", [com(&f, &m), mi_syn(&[])].concat()),
        ("mattes_and_cc", [com(&f, &m), mixed].concat()),
    ];
    let fixed = load(&f);
    let inside = object_points(&fixed);
    let identity = TransformChain::new();
    for (name, args) in &cases {
        let case = d.join(name);
        let (theirs, mine) = run_both(&reg, &case, name, &|o| {
            [args.clone(), both_outputs(o)].concat()
        });
        check_warped(&apply, name, &f, &m, &theirs, &mine);
        let (ants_all, ants_linear) = maps(&theirs);
        let (mine_all, _) = maps(&mine);
        let full_err = rms_dist(&mine_all, &ants_all, &inside);
        let warp_noop = rms_dist(&ants_linear, &ants_all, &inside);
        let full_noop = rms_dist(&identity, &ants_all, &inside);
        eprintln!(
            "{name}: forward map RMS in the object {full_err:.3} mm (ANTs linear part alone \
             {warp_noop:.3}, identity {full_noop:.3}); {} points",
            inside.len()
        );
        assert!(
            full_err < 0.25 && full_err < 0.1 * full_noop,
            "{name}: forward map {full_err} mm, identity {full_noop}"
        );
    }
}

/// The pair of [`moved_pair`] with a cavity in the moving image (intensity 0 in a ball of 5 mm
/// on the second blob), a binary fixed mask of the object, a moving mask without a ball of 7 mm
/// at the cavity, and the cavity's centre. ANTs reads masks as `unsigned char`, so both are 0 or 1.
fn cavity_pair(dir: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf, [f64; 3]) {
    let (f, m) = moved_pair(dir, [24; 3], [2.0; 3]);
    let fixed = load(&f);
    let object: Vec<f32> = fixed
        .data()
        .iter()
        .map(|v| if *v > 0.05 { 1.0 } else { 0.0 })
        .collect();
    let fixed_mask = dir.join("f_mask.nii.gz");
    Volume::from_vec(*fixed.grid(), object)
        .unwrap()
        .save_nifti(&fixed_mask)
        .unwrap();
    let centre = [11.0, 32.0, 21.0];
    let moving = load(&m);
    let g = *moving.grid();
    let within = |idx: usize, r: f64| {
        let [i, j, k] = g.ijk(idx);
        dist(g.index_to_point([i as f64, j as f64, k as f64]), centre) < r
    };
    let lesioned: Vec<f32> = (0..g.num_voxels())
        .map(|idx| {
            if within(idx, 5.0) {
                0.0
            } else {
                moving.data()[idx] as f32
            }
        })
        .collect();
    let cavity = dir.join("m_cavity.nii.gz");
    Volume::from_vec(g, lesioned)
        .unwrap()
        .save_nifti(&cavity)
        .unwrap();
    let keep: Vec<f32> = (0..g.num_voxels())
        .map(|idx| if within(idx, 7.0) { 0.0 } else { 1.0 })
        .collect();
    let mask = dir.join("m_mask.nii.gz");
    Volume::from_vec(g, keep)
        .unwrap()
        .save_nifti(&mask)
        .unwrap();
    (f, cavity, fixed_mask, mask, centre)
}

#[test]
#[ignore = "runs antsRegistration and antsApplyTransforms from FASTWARP_ANTS_DIR and needs a GPU; \
            run with --ignored"]
fn syn_masks_agree_with_ants() {
    let reg = ants_program("antsRegistration");
    let apply = ants_program("antsApplyTransforms");
    require_gpu();
    let d = tmp("syn_masks");
    let (f, m, fixed_mask, mask, centre) = cavity_pair(&d);
    let chain = [com(&f, &m), stages(&f, &m, &["SyN"])].concat();
    let masks = argv(&["-x", &format!("[{},{}]", s(&fixed_mask), s(&mask))]);
    let mut runs = Vec::new();
    for (name, extra) in [("masked", masks), ("unmasked", Vec::new())] {
        let case = d.join(name);
        let (theirs, mine) = run_both(&reg, &case, name, &|o| {
            [chain.clone(), extra.clone(), both_outputs(o)].concat()
        });
        let w = ncc(
            &load(&mine.join("W.nii.gz")),
            &load(&theirs.join("W.nii.gz")),
        );
        let w0 = ncc(
            &on_grid(&m, &f, &mine.join("m_on_f.nii.gz")),
            &load(&theirs.join("W.nii.gz")),
        );
        eprintln!("{name}: NCC with ANTs W {w:.4} (unregistered {w0:.4})");
        assert!(
            w > 0.95 && w - w0 > 0.05,
            "{name}: W {w} vs unregistered {w0}"
        );
        let check = mine.join("check.nii.gz");
        let mut a = argv(&[
            "-d",
            "3",
            "-i",
            &s(&m),
            "-r",
            &s(&f),
            "-o",
            &s(&check),
            "-n",
            "Linear",
        ]);
        for t in forward_transforms(&mine, &files_in(&mine)) {
            a.extend(["-t".to_string(), t]);
        }
        ok(
            &run_ants(&apply, &a),
            &format!("{name}: antsApplyTransforms"),
        );
        let diff = max_abs_diff(&load(&check), &load(&mine.join("W.nii.gz")));
        assert!(
            diff < 1e-4,
            "{name}: ANTs reads FastWarp's transforms: {diff}"
        );
        runs.push((maps(&theirs), maps(&mine)));
    }
    let [((ants_masked, ants_linear), (mine_masked, _)), ((ants_unmasked, ants_unmasked_linear), (mine_unmasked, _))] =
        runs.try_into().ok().unwrap();
    let fixed = load(&f);
    let inside = object_points(&fixed);
    let near: Vec<[f64; 3]> = inside
        .iter()
        .copied()
        .filter(|p| dist(ants_masked.apply(*p), centre) < 8.0)
        .collect();
    let masked_err = rms_dist(&mine_masked, &ants_masked, &inside);
    let masked_warp = rms_dist(&ants_linear, &ants_masked, &inside);
    let unmasked_err = rms_dist(&mine_unmasked, &ants_unmasked, &inside);
    let unmasked_warp = rms_dist(&ants_unmasked_linear, &ants_unmasked, &inside);
    let near_err = rms_dist(&mine_masked, &ants_masked, &near);
    let near_unmasked = rms_dist(&mine_masked, &ants_unmasked, &near);
    let ants_effect = rms_dist(&ants_masked, &ants_unmasked, &near);
    let fastwarp_effect = rms_dist(&mine_masked, &mine_unmasked, &near);
    eprintln!(
        "forward map RMS in the object, FastWarp against ANTs: masked {masked_err:.2e} mm (ANTs' \
         warp {masked_warp:.3}), unmasked {unmasked_err:.2e} mm (ANTs' warp {unmasked_warp:.3}); \
         near the cavity FastWarp's masked run is {near_err:.2e} mm from ANTs' masked run and \
         {near_unmasked:.3} mm from its unmasked run; the masks move ANTs by {ants_effect:.3} mm \
         and FastWarp by {fastwarp_effect:.3} mm there; {} and {} points",
        inside.len(),
        near.len()
    );
    assert!(near.len() > 20, "{} points near the cavity", near.len());
    assert!(
        masked_err < 0.1 && masked_err < 0.1 * masked_warp,
        "masked forward map {masked_err} mm from ANTs', ANTs' warp {masked_warp} mm"
    );
    assert!(
        unmasked_err < 0.1,
        "unmasked forward map {unmasked_err} mm from ANTs'"
    );
    assert!(
        near_err < 0.1 * near_unmasked,
        "near the cavity: {near_err} mm from ANTs' masked run, {near_unmasked} from its unmasked one"
    );
    assert!(
        (fastwarp_effect - ants_effect).abs() < 0.1 * ants_effect,
        "near the cavity the masks move FastWarp by {fastwarp_effect} mm, ANTs by {ants_effect} mm"
    );
}

/// A binary mask on a grid of 1.7 mm whose origin is off the 2 mm grids of the images, so that
/// no level's voxel centre falls halfway between two mask voxels (where FastWarp's 32-bit
/// arithmetic may round otherwise than ANTs'): 1 within `radius` of `centre` when `inside`, else
/// 1 outside it.
fn off_grid_ball(path: &Path, centre: [f64; 3], radius: f64, inside: bool) {
    let g = Grid::new([29; 3], [1.7; 3], [-0.37, -0.41, -0.29], IDENTITY3).unwrap();
    let data: Vec<f32> = (0..g.num_voxels())
        .map(|idx| {
            let [i, j, k] = g.ijk(idx);
            let near = dist(g.index_to_point([i as f64, j as f64, k as f64]), centre) < radius;
            if near == inside {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Volume::from_vec(g, data).unwrap().save_nifti(path).unwrap();
}

#[test]
#[ignore = "runs antsRegistration from FASTWARP_ANTS_DIR and needs a GPU; run with --ignored"]
fn two_syn_stages_with_per_stage_masks_agree_with_ants() {
    let reg = ants_program("antsRegistration");
    require_gpu();
    let d = tmp("two_syn");
    let (f, m, _, _, cavity) = cavity_pair(&d);
    let refine_centre = [30.0, 17.0, 28.0];
    let refine = d.join("f_refine.nii.gz");
    off_grid_ball(&refine, refine_centre, 9.0, true);
    let lesion = d.join("m_lesion.nii.gz");
    off_grid_ball(&lesion, cavity, 7.0, false);
    let second = argv(&[
        "-t",
        "SyN[0.3,3,0]",
        "-m",
        &format!("CC[{},{},1,2]", s(&f), s(&m)),
        "-c",
        "[15,1e-6,10]",
        "-f",
        "1",
        "-s",
        "0vox",
    ]);
    let one_stage = [com(&f, &m), stages(&f, &m, &["SyN"])].concat();
    let two_stages = [one_stage.clone(), second].concat();
    let x = |fixed: Option<&Path>, moving: Option<&Path>| {
        let name = |p: Option<&Path>| p.map_or("NULL".to_string(), s);
        argv(&["-x", &format!("[{},{}]", name(fixed), name(moving))])
    };
    let cases = [
        ("one SyN stage", one_stage.clone(), Vec::new()),
        ("two SyN stages", two_stages.clone(), Vec::new()),
        (
            "two SyN stages, refine mask on the second",
            two_stages.clone(),
            [x(None, None), x(Some(&refine), None)].concat(),
        ),
        (
            "one SyN stage, lesion mask",
            one_stage,
            x(None, Some(&lesion)),
        ),
        (
            "two SyN stages, lesion mask on both, refine mask on the second",
            two_stages,
            [x(None, Some(&lesion)), x(Some(&refine), Some(&lesion))].concat(),
        ),
    ];
    let mut runs = Vec::new();
    for (name, chain, masks) in &cases {
        let case = d.join(name.replace([' ', ','], "_"));
        let (theirs, mine) = run_both(&reg, &case, name, &|o| {
            [chain.clone(), masks.clone(), both_outputs(o)].concat()
        });
        let w = ncc(
            &load(&mine.join("W.nii.gz")),
            &load(&theirs.join("W.nii.gz")),
        );
        eprintln!("{name}: NCC of the warped images with ANTs' {w:.5}");
        assert!(w > 0.99, "{name}: W {w}");
        runs.push((maps(&theirs), maps(&mine)));
    }
    let fixed = load(&f);
    let inside = object_points(&fixed);
    let near: Vec<[f64; 3]> = inside
        .iter()
        .copied()
        .filter(|p| dist(*p, refine_centre) < 9.0)
        .collect();
    let far: Vec<[f64; 3]> = inside
        .iter()
        .copied()
        .filter(|p| dist(*p, refine_centre) > 18.0)
        .collect();
    assert!(
        near.len() > 20 && far.len() > 20,
        "{} {}",
        near.len(),
        far.len()
    );
    let ((ants_plain, ants_plain_linear), (mine_plain, _)) = &runs[1];
    let plain_err = rms_dist(mine_plain, ants_plain, &inside);
    let plain_warp = rms_dist(ants_plain_linear, ants_plain, &inside);
    let plain_effect = rms_dist(ants_plain, &runs[0].0 .0, &inside);
    let plain_mine_effect = rms_dist(mine_plain, &runs[0].1 .0, &inside);
    eprintln!(
        "no masks: forward map RMS in the object, FastWarp against ANTs: two SyN stages \
         {plain_err:.2e} mm (ANTs' warp {plain_warp:.3} mm, the second stage moves it by \
         {plain_effect:.3} mm, FastWarp's by {plain_mine_effect:.3} mm)"
    );
    assert!(
        plain_err < 1e-3 && plain_err < 1e-2 * plain_effect,
        "no masks: two SyN stages {plain_err} mm from ANTs', second stage {plain_effect} mm"
    );
    for (one, two, label) in [(0, 2, "no lesion mask"), (3, 4, "lesion mask")] {
        let ((ants_one, _), (mine_one, _)) = &runs[one];
        let ((ants_two, ants_linear), (mine_two, _)) = &runs[two];
        let one_err = rms_dist(mine_one, ants_one, &inside);
        let two_err = rms_dist(mine_two, ants_two, &inside);
        let ants_warp = rms_dist(ants_linear, ants_two, &inside);
        let near_err = rms_dist(mine_two, ants_two, &near);
        let ants_effect = rms_dist(ants_two, ants_one, &near);
        let fastwarp_effect = rms_dist(mine_two, mine_one, &near);
        let ants_far = rms_dist(ants_two, ants_one, &far);
        let fastwarp_far = rms_dist(mine_two, mine_one, &far);
        eprintln!(
            "{label}: forward map RMS in the object, FastWarp against ANTs: one SyN stage \
             {one_err:.2e} mm, two SyN stages {two_err:.2e} mm (ANTs' warp {ants_warp:.3} mm), \
             inside the refine mask {near_err:.2e} mm; the second stage moves ANTs by \
             {ants_effect:.3} mm and FastWarp by {fastwarp_effect:.3} mm inside the refine mask, \
             {ants_far:.2e} and {fastwarp_far:.2e} mm far from it; {} points, {} inside, {} far",
            inside.len(),
            near.len(),
            far.len()
        );
        assert!(
            ants_effect > 0.05,
            "{label}: the second stage barely moves ANTs: {ants_effect} mm"
        );
        assert!(
            fastwarp_far < 0.2 * fastwarp_effect && ants_far < 0.2 * ants_effect,
            "{label}: far from the refine mask the second stage moves FastWarp by \
             {fastwarp_far} mm and ANTs by {ants_far} mm"
        );
        assert!(
            two_err < 0.1 * ants_warp && two_err < one_err + 0.05,
            "{label}: forward map {two_err} mm from ANTs' (one stage {one_err}), ANTs' warp \
             {ants_warp} mm"
        );
        assert!(
            (fastwarp_effect - ants_effect).abs() < 0.25 * ants_effect,
            "{label}: inside the refine mask the second stage moves FastWarp by \
             {fastwarp_effect} mm, ANTs by {ants_effect} mm"
        );
    }
}

#[test]
#[ignore = "runs antsApplyTransforms from FASTWARP_ANTS_DIR; run with --ignored"]
fn apply_outputs_match_ants_apply_transforms() {
    let apply = ants_program("antsApplyTransforms");
    let d = tmp("apply_ants");
    let (f, m) = blob_pair(&d);
    let a = d.join("a.txt");
    write_affine_txt(
        &a,
        [1.02, 0.05, 0.0, -0.04, 0.98, 0.01, 0.0, 0.02, 1.01],
        [1.5, -2.0, 0.5],
    );
    let field = d.join("field.nii.gz");
    let o = run(&argv(&[
        "compose",
        "-r",
        &s(&f),
        "-o",
        &s(&field),
        "-t",
        &s(&a),
    ]));
    ok(&o, "field");
    let b = d.join("b.txt");
    write_affine_txt(
        &b,
        [0.97, 0.0, 0.03, 0.02, 1.03, 0.0, -0.01, 0.0, 0.99],
        [-2.5, 1.0, 3.0],
    );
    let (sa, sb, sf, sm, sfield) = (s(&a), s(&b), s(&f), s(&m), s(&field));
    let cases: Vec<(&str, Vec<String>)> = vec![
        (
            "w.nii.gz",
            argv(&["-i", &sm, "-r", &sf, "-t", &sfield, "-t", &sa]),
        ),
        (
            "wf.nii.gz",
            argv(&["-i", &sm, "-r", &sf, "-t", &sa, "--float", "1"]),
        ),
        (
            "ws.nii.gz",
            argv(&[
                "-i",
                &sm,
                "-r",
                &sf,
                "-t",
                &sa,
                "-u",
                "short",
                "-n",
                "NearestNeighbor",
            ]),
        ),
        (
            "wl.nii.gz",
            argv(&[
                "-i",
                &sm,
                "-r",
                &sf,
                "-t",
                &format!("[{sa},1]"),
                "-n",
                "LanczosWindowedSinc",
            ]),
        ),
        (
            "[d.nii.gz,1]",
            argv(&["-r", &sf, "-t", &sfield, "-t", &sb, "-u", "short"]),
        ),
        (
            "DisplacementField[df.nii.gz]",
            argv(&["-r", &sf, "-t", &sa, "--float"]),
        ),
        ("[d.h5,1]", argv(&["-r", &sf, "-t", &sa])),
        (
            "CompositeTransform[c.h5]",
            argv(&["-t", &sfield, "-t", &format!("[{sb},1]")]),
        ),
        ("Linear[l.mat,1]", argv(&["-t", &sa, "-t", &sb])),
    ];
    let p = [11.0, 17.5, 20.0];
    for (target, args) in cases {
        let theirs = d.join("ants");
        let mine = d.join("fastwarp");
        std::fs::create_dir_all(&theirs).unwrap();
        std::fs::create_dir_all(&mine).unwrap();
        let out_arg = |dir: &Path| -> String {
            match target.find('[') {
                Some(open) => {
                    let inner = &target[open + 1..target.len() - 1];
                    let (file, rest) = inner.split_once(',').map_or((inner, ""), |(a, b)| (a, b));
                    let rest = if rest.is_empty() {
                        String::new()
                    } else {
                        format!(",{rest}")
                    };
                    format!("{}[{}{rest}]", &target[..open], s(&dir.join(file)))
                }
                None => s(&dir.join(target)),
            }
        };
        let mut a = argv(&["-d", "3", "-o", &out_arg(&theirs)]);
        a.extend(args.clone());
        ok(
            &run_ants(&apply, &a),
            &format!("{target}: antsApplyTransforms"),
        );
        let mut b = argv(&["ants-apply-transforms", "-d", "3", "-o", &out_arg(&mine)]);
        b.extend(args);
        ok(&run(&b), &format!("{target}: fastwarp"));
        let file = target
            .split(['[', ',', ']'])
            .find(|t| t.contains('.'))
            .unwrap();
        let (x, y) = (theirs.join(file), mine.join(file));
        if file.ends_with(".nii.gz") && !target.contains('[') {
            assert_eq!(datatype(&x), datatype(&y), "{target}: voxel type");
            let tol = if target.starts_with("wl") { 1e-4 } else { 1e-5 };
            let diff = max_abs_diff(&load(&x), &load(&y));
            assert!(diff < tol, "{target}: {diff}");
        } else {
            if file.ends_with(".nii.gz") {
                assert_eq!(datatype(&x), datatype(&y), "{target}: voxel type");
            }
            if file.ends_with(".h5") {
                assert_eq!(kinds(&x), kinds(&y), "{target}: transforms");
            }
            let (cx, cy) = (
                read_transform_file(&x).unwrap(),
                read_transform_file(&y).unwrap(),
            );
            let (qx, qy) = (cx.apply(p), cy.apply(p));
            for k in 0..3 {
                assert!((qx[k] - qy[k]).abs() < 1e-4, "{target}: {qx:?} {qy:?}");
            }
        }
        std::fs::remove_dir_all(&theirs).unwrap();
        std::fs::remove_dir_all(&mine).unwrap();
    }
    let g = Grid::new([10, 1, 1], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
    let values = vec![
        2.7f32, -2.7, 300.4, -300.6, 70000.6, -70000.2, 127.5, 128.5, 255.9, 0.49,
    ];
    let cast = d.join("cast.nii.gz");
    Volume::from_vec(g, values)
        .unwrap()
        .save_nifti(&cast)
        .unwrap();
    for t in ["char", "uchar", "short", "int"] {
        let (x, y) = (
            d.join(format!("a_{t}.nii.gz")),
            d.join(format!("f_{t}.nii.gz")),
        );
        let common = |out: &Path| {
            argv(&[
                "-d",
                "3",
                "-i",
                &s(&cast),
                "-r",
                &s(&cast),
                "-o",
                &s(out),
                "-u",
                t,
                "-n",
                "NearestNeighbor",
            ])
        };
        ok(&run_ants(&apply, &common(&x)), t);
        let mut b = argv(&["ants-apply-transforms"]);
        b.extend(common(&y));
        ok(&run(&b), t);
        assert_eq!(datatype(&x), datatype(&y), "{t}");
        assert_eq!(load(&x).data(), load(&y).data(), "{t}");
    }
}

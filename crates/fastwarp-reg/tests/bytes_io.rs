// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use fastwarp_reg::grid::Grid;
use fastwarp_reg::nifti_io::parse_nifti_grid;
use fastwarp_reg::transform::ants_io::{parse_transform, read_transform_file};
use fastwarp_reg::transform::field_nifti::{
    format_field_nifti, parse_field_nifti, read_field_nifti, write_field_nifti,
};
use fastwarp_reg::transform::itk_h5::format_h5;
use fastwarp_reg::transform::itk_mat::format_mat;
use fastwarp_reg::transform::{AffineTransform, DisplacementField, TransformChain, TransformElem};
use fastwarp_reg::Volume;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fastwarp_reg_bytes_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn grid() -> Grid {
    let c = 0.3f64.cos();
    let s = 0.3f64.sin();
    Grid::new(
        [6, 5, 4],
        [1.5, 2.0, 2.5],
        [-3.0, 4.0, -5.0],
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap()
}

fn volume() -> Volume<f32> {
    let g = grid();
    let data = (0..g.num_voxels()).map(|i| i as f32 * 0.5 - 7.0).collect();
    Volume::from_vec(g, data).unwrap()
}

#[test]
fn volume_bytes_equal_the_file_for_plain_and_gzip() {
    let v = volume();
    for (name, gzip) in [("v.nii", false), ("v.nii.gz", true)] {
        let path = tmp(name);
        v.save_nifti(&path).unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(v.to_nifti_bytes(gzip).unwrap(), on_disk, "{name}");
        let a = Volume::load_nifti(&path).unwrap();
        let b = Volume::from_nifti_bytes(on_disk.clone()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.data(), v.data());
        let c = Volume::from_nifti_bytes_f64(on_disk.clone()).unwrap();
        let d = Volume::load_nifti_f64(&path).unwrap();
        assert_eq!(c, d);
        assert_eq!(parse_nifti_grid(&on_disk).unwrap(), *a.grid());
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn field_bytes_equal_the_file_and_read_back() {
    let f = DisplacementField::from_fn(grid(), |p| [p[0] * 0.1, -p[1] * 0.2 + 1.0, p[2] * 0.05]);
    for (name, gzip) in [("f.nii", false), ("f.nii.gz", true)] {
        let path = tmp(name);
        write_field_nifti(&path, &f).unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(format_field_nifti(&f, gzip).unwrap(), on_disk, "{name}");
        let a = read_field_nifti(&path).unwrap();
        let b = parse_field_nifti(name, on_disk.clone()).unwrap();
        assert_eq!(a.data(), b.data());
        assert_eq!(b.data(), f.data());
        let chain = parse_transform(name, on_disk).unwrap();
        assert!(matches!(
            chain.queue().first(),
            Some(TransformElem::DisplacementField(_))
        ));
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn transform_bytes_read_like_the_files() {
    let affine = AffineTransform::new(
        [[1.02, 0.03, 0.0], [-0.03, 0.98, 0.0], [0.0, 0.01, 1.0]],
        [1.0, -2.0, 3.0],
        [0.5, 0.5, 0.5],
    );
    let chain = TransformChain::from_queue(vec![TransformElem::Affine(affine)]);
    for (name, bytes) in [
        ("t.mat", format_mat(chain.queue()).unwrap()),
        ("t.h5", format_h5(&chain).unwrap()),
    ] {
        let path = tmp(name);
        std::fs::write(&path, &bytes).unwrap();
        let a = read_transform_file(&path).unwrap();
        let b = parse_transform(name, bytes).unwrap();
        assert_eq!(a, b, "{name}");
        std::fs::remove_file(&path).ok();
    }
    assert!(parse_transform("t.xyz", Vec::new()).is_err());
}

#[test]
fn text_and_upper_case_names_are_read_like_the_files() {
    let text = "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\nParameters: 1 0 0 0 1 0 0 0 1 1 2 3\nFixedParameters: 0 0 0\n";
    for name in ["t.txt", "t.tfm", "T.TXT", "T.Tfm"] {
        let path = tmp(name);
        std::fs::write(&path, text).unwrap();
        let a = read_transform_file(&path).unwrap();
        let b = parse_transform(name, text.as_bytes().to_vec()).unwrap();
        assert_eq!(a, b, "{name}");
        std::fs::remove_file(&path).ok();
    }
    assert!(parse_transform("t.txt", vec![0xff, 0xfe]).is_err());

    let affine = AffineTransform::new(
        [[1.02, 0.03, 0.0], [-0.03, 0.98, 0.0], [0.0, 0.01, 1.0]],
        [1.0, -2.0, 3.0],
        [0.5, 0.5, 0.5],
    );
    let chain = TransformChain::from_queue(vec![TransformElem::Affine(affine)]);
    assert_eq!(
        parse_transform("T.MAT", format_mat(chain.queue()).unwrap()).unwrap(),
        chain
    );
    assert_eq!(
        parse_transform("T.H5", format_h5(&chain).unwrap()).unwrap(),
        chain
    );
    let f = DisplacementField::from_fn(grid(), |p| [p[0] * 0.1, 0.0, p[2] * 0.05]);
    let bytes = format_field_nifti(&f, true).unwrap();
    let chain = parse_transform("F.NII.GZ", bytes).unwrap();
    assert!(matches!(
        chain.queue().first(),
        Some(TransformElem::DisplacementField(_))
    ));
}

#[test]
fn nifti_errors_carry_the_given_name() {
    let err = parse_field_nifti("warp.nii.gz", vec![1, 2, 3]).unwrap_err();
    assert!(err.to_string().contains("warp.nii.gz"), "{err}");
    let err = parse_transform("moving.nii", vec![1, 2, 3]).unwrap_err();
    assert!(err.to_string().contains("moving.nii"), "{err}");
}

#[test]
fn large_volume_bytes_equal_the_file_above_16_mib() {
    let g = Grid::new(
        [170, 170, 170],
        [1.0; 3],
        [0.0; 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap();
    let data = (0..g.num_voxels())
        .map(|i| (i % 997) as f32 * 0.25)
        .collect();
    let v = Volume::from_vec(g, data).unwrap();
    for (name, gzip) in [("big.nii", false), ("big.nii.gz", true)] {
        let path = tmp(name);
        v.save_nifti(&path).unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        assert!(on_disk.len() > 16 << 20 || gzip);
        assert!(v.to_nifti_bytes(gzip).unwrap() == on_disk, "{name}");
        assert_eq!(Volume::from_nifti_bytes(on_disk).unwrap().data(), v.data());
        std::fs::remove_file(&path).ok();
    }
}

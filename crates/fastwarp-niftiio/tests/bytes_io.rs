// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used)]

use fastwarp_niftiio::{NiftiFile, WriterOptions};
use ndarray::Array3;

fn volume() -> Array3<f32> {
    Array3::from_shape_fn((9, 7, 5), |(i, j, k)| (i * 100 + j * 10 + k) as f32 * 0.25)
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("fastwarp_niftiio_bytes_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn check(name: &str, gzip: bool) {
    let data = volume();
    let path = tmp(name);
    WriterOptions::new(&path).write_nifti(&data).unwrap();
    let on_disk = std::fs::read(&path).unwrap();
    let in_memory = WriterOptions::in_memory(gzip).to_bytes(&data).unwrap();
    assert_eq!(in_memory, on_disk, "{name}");
    assert_eq!(in_memory.starts_with(&[0x1f, 0x8b]), gzip);

    let from_path = NiftiFile::open(&path).unwrap();
    let from_bytes = NiftiFile::from_bytes(on_disk).unwrap();
    assert_eq!(from_path.header(), from_bytes.header());
    assert_eq!(
        from_path.read_data(None).unwrap(),
        from_bytes.read_data(None).unwrap()
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn plain_bytes_equal_the_file() {
    check("plain.nii", false);
}

#[test]
fn gzip_bytes_equal_the_file() {
    check("gz.nii.gz", true);
}

#[test]
fn gzip_is_detected_by_magic_not_name() {
    let data = volume();
    let gz = WriterOptions::in_memory(true).to_bytes(&data).unwrap();
    let plain = WriterOptions::in_memory(false).to_bytes(&data).unwrap();
    let a = NiftiFile::from_bytes(gz).unwrap().read_data(None).unwrap();
    let b = NiftiFile::from_bytes(plain)
        .unwrap()
        .read_data(None)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn write_to_a_writer_matches_to_bytes() {
    let data = volume();
    for gzip in [false, true] {
        let opts = WriterOptions::in_memory(gzip);
        let out = opts.write_nifti_to(Vec::new(), &data).unwrap();
        assert_eq!(out, opts.to_bytes(&data).unwrap());
    }
}

#[test]
fn truncated_bytes_fail() {
    let data = volume();
    let mut b = WriterOptions::in_memory(false).to_bytes(&data).unwrap();
    b.truncate(400);
    assert!(NiftiFile::from_bytes(b).unwrap().read_data(None).is_err());
    assert!(NiftiFile::from_bytes(vec![1, 2, 3]).is_err());
}

#[test]
fn large_volume_bytes_equal_the_file_above_16_mib() {
    let data = Array3::from_shape_fn((170, 170, 170), |(i, j, k)| {
        ((i * 7 + j * 3 + k) % 251) as f32 * 0.5
    });
    for (name, gzip) in [("big.nii", false), ("big.nii.gz", true)] {
        let path = tmp(name);
        WriterOptions::new(&path).write_nifti(&data).unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        assert!(data.len() * 4 > 16 << 20);
        let in_memory = WriterOptions::in_memory(gzip).to_bytes(&data).unwrap();
        assert!(in_memory == on_disk, "{name}");
        std::fs::remove_file(&path).ok();
    }
}

fn hostile(gzip: bool) -> Vec<u8> {
    let data = Array3::<f32>::zeros((2, 2, 2));
    let mut b = WriterOptions::in_memory(false).to_bytes(&data).unwrap();
    b[42..44].copy_from_slice(&30000i16.to_le_bytes());
    b[44..46].copy_from_slice(&30000i16.to_le_bytes());
    b[46..48].copy_from_slice(&30000i16.to_le_bytes());
    b.truncate(352 + 32);
    if gzip {
        use std::io::Write;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(&b).unwrap();
        e.finish().unwrap()
    } else {
        b
    }
}

#[test]
fn hostile_header_fails_without_a_large_allocation() {
    for gzip in [false, true] {
        let bytes = hostile(gzip);
        assert!(bytes.len() < 1024);
        let file = NiftiFile::from_bytes(bytes).unwrap();
        assert!(file.voxel_count().unwrap() > 1 << 40);
        let start = std::time::Instant::now();
        let err = file.read_data(None).unwrap_err();
        assert!(
            matches!(
                err,
                fastwarp_niftiio::NiftiFileError::IncompatibleLength { .. }
            ),
            "{err:?}"
        );
        assert!(start.elapsed().as_secs() < 5);
    }
}

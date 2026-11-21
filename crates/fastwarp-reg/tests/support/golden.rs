// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, clippy::unwrap_used)]

use std::io::Read;
use std::path::{Path, PathBuf};

use fastwarp_core::sha256::{file_sha256_hex, sha256_hex};
use fastwarp_reg::transform::itk_h5::read_h5_file;
use fastwarp_reg::{Grid, TransformChain, TransformElem, Volume};

pub type Table = &'static [(&'static str, &'static str)];
pub type Tables = &'static [(&'static str, Table)];

pub fn platform_key() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "fastwarp_golden_{}_{serial}_{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn prefix(dir: &Scratch) -> PathBuf {
    dir.path().join("golden_")
}

fn push_f64s(out: &mut Vec<u8>, values: &[f64]) {
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

pub fn canonical_chain(chain: &TransformChain) -> Vec<u8> {
    let mut out = Vec::new();
    for elem in chain.queue() {
        match elem {
            TransformElem::Affine(a) => {
                out.push(1);
                for row in a.matrix {
                    push_f64s(&mut out, &row);
                }
                push_f64s(&mut out, &a.translation);
                push_f64s(&mut out, &a.center);
            }
            TransformElem::DisplacementField(f) => {
                out.push(2);
                let g = f.grid();
                for n in g.size() {
                    out.extend_from_slice(&(n as u64).to_le_bytes());
                }
                push_f64s(&mut out, &g.spacing());
                push_f64s(&mut out, &g.origin());
                for row in g.direction() {
                    push_f64s(&mut out, &row);
                }
                for v in f.data() {
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
    }
    out
}

fn gunzip(p: &Path) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(p).unwrap())
        .read_to_end(&mut out)
        .unwrap();
    out
}

pub fn hash_outputs(paths: &[PathBuf]) -> Vec<(String, String)> {
    let mut hashes = Vec::new();
    for p in paths {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if name.ends_with(".h5") {
            hashes.push((name.clone(), file_sha256_hex(p).unwrap()));
            hashes.push((
                format!("{name}#numeric"),
                sha256_hex(&canonical_chain(&read_h5_file(p).unwrap())),
            ));
        } else if name.ends_with(".nii.gz") {
            hashes.push((name, sha256_hex(&gunzip(p))));
        } else {
            hashes.push((name, file_sha256_hex(p).unwrap()));
        }
    }
    hashes
}

pub fn hash_f32s(label: &str, values: &[f32]) -> (String, String) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    (label.to_owned(), sha256_hex(&bytes))
}

fn const_name(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

pub fn check(case: &str, key: &str, got: &[(String, String)], tables: Tables) {
    for (i, (k, _)) in tables.iter().enumerate() {
        assert!(
            tables[..i].iter().all(|(other, _)| other != k),
            "duplicate golden key {k} in the {case} table"
        );
    }
    for (name, hex) in got {
        eprintln!("GOLDEN {name} {hex}");
    }
    let table = tables.iter().find(|(k, _)| *k == key).map(|(_, t)| *t);
    match table {
        Some(expected) => {
            let want: Vec<(String, String)> = expected
                .iter()
                .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
                .collect();
            assert_eq!(
                got,
                want.as_slice(),
                "hashes differ from the pins for {key}"
            );
        }
        None => {
            eprintln!("GOLDEN no pinned hashes for {key}; only determinism is checked");
            eprintln!("GOLDEN paste into the tables of this test:");
            let ident = const_name(&format!("{case}_{key}"));
            eprintln!("const {ident}: Table = &[");
            for (name, hex) in got {
                eprintln!("    (\"{name}\", \"{hex}\"),");
            }
            eprintln!("];");
            eprintln!("    (\"{key}\", {ident}),");
        }
    }
}

pub fn second_channel(v: &Volume<f64>) -> Volume<f64> {
    let data = v.data().iter().map(|x| (x * 0.8 + 0.1).sqrt()).collect();
    Volume::from_vec(*v.grid(), data).unwrap()
}

pub fn n4_phantom(size: [usize; 3], spacing: f64) -> Volume<f64> {
    let origin = [
        -(size[0] as f64 - 1.0) * spacing / 2.0,
        -(size[1] as f64 - 1.0) * spacing / 2.0,
        -(size[2] as f64 - 1.0) * spacing / 2.0,
    ];
    let g = Grid::new(
        size,
        [spacing; 3],
        origin,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap();
    let data: Vec<f64> = (0..g.num_voxels())
        .map(|lin| {
            let [i, j, k] = g.ijk(lin);
            let p = g.index_to_point([i as f64, j as f64, k as f64]);
            let r = ((p[0] / 22.0).powi(2) + (p[1] / 22.0).powi(2) + (p[2] / 16.0).powi(2)).sqrt();
            if r >= 1.0 {
                0.0
            } else {
                let edge = 0.5 * (1.0 + ((1.0 - r) / 0.05).tanh());
                edge * (0.5 + 0.3 * (1.0 - r) + 0.1 * (p[0] / 4.0).sin())
                    * (1.0 - 0.06 * (p[0] / 22.0).powi(2) + 0.05 * (p[2] / 16.0))
            }
        })
        .collect();
    Volume::from_vec(g, data).unwrap()
}

pub fn n4_params() -> fastwarp_reg::n4::N4Params {
    fastwarp_reg::n4::N4Params {
        shrink_factor: 2,
        iterations: vec![4; 4],
        ..fastwarp_reg::n4::N4Params::default()
    }
}

// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used)]

mod support {
    pub mod golden;
}

use fastwarp_reg::linear::{CpuLinearEngine, LinearRecipe};
use fastwarp_reg::n4::run_n4;
use fastwarp_reg::recipe::SynRecipe;
use fastwarp_reg::register::LinearPlan;
use fastwarp_reg::stages::{register_stages, write_stage_outputs, OutputLayout, StagePlan};
use fastwarp_reg::syn::SynOptions;
use fastwarp_reg::{CpuBackend, Grid, Volume};
use support::golden::{
    check, hash_f32s, hash_outputs, n4_params, n4_phantom, platform_key, prefix, second_channel,
    Scratch, Table, Tables,
};

const IDENTITY3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

const TABLES: Tables = &[("windows-x86_64", WINDOWS_X86_64)];

const TABLES_TWO_CHANNEL: Tables = &[("windows-x86_64", TWO_CHANNEL_WINDOWS_X86_64)];

const TABLES_N4: Tables = &[("windows-x86_64", N4_WINDOWS_X86_64)];

const WINDOWS_X86_64: Table = &[
    (
        "golden_Composite.h5",
        "559a70a6a3ce2c52437565b25ae88c380fcfbc5455bad38a40498a3c6f24656b",
    ),
    (
        "golden_Composite.h5#numeric",
        "369fe05ff28f8922f5b6909573b86f846b5b5ee4081cfba907f0db329bc9df6f",
    ),
    (
        "golden_InverseComposite.h5",
        "43d138355f6b9bbac3236fae2c256824707b70a06b6b9b7c74bb65f9bd39979e",
    ),
    (
        "golden_InverseComposite.h5#numeric",
        "3a7202c137e63004e5fc0e5c0cb9c0e10c23cb75fa798df7e2cddcbc3aae89b8",
    ),
    (
        "golden_0GenericAffine.mat",
        "f7710b7efb636b6b8f7038e5603f19c208df52ca4e7b2e905223f3325ebb6e0d",
    ),
    (
        "golden_1Warp.nii.gz",
        "54b9eeb9ae33fc116d8410da6ad5c8edc6bd393ffcf7f23fca9a2034e6762a4e",
    ),
    (
        "golden_1InverseWarp.nii.gz",
        "85451c5b301d8524971a2ab26affcc9efc98810c16b624d35edf927e6ef08573",
    ),
    (
        "golden__Warped.nii.gz",
        "64524c726becc84ce800bc85408e010552e9ae0ca6bc95613e1ef1f1c5fdc1f9",
    ),
];

const TWO_CHANNEL_WINDOWS_X86_64: Table = &[
    (
        "golden_Composite.h5",
        "59d8bb26a66def661ec9b0523a48a542c29b8c35d1d7a1933e47a81008490ec9",
    ),
    (
        "golden_Composite.h5#numeric",
        "5fe2ab7d4affd039e1f46018251a369b8c1aeadc9f6a9acbaf37a0cd17d970aa",
    ),
    (
        "golden_InverseComposite.h5",
        "4cfbd0dccc3980bcee2cf619845ee60f077ac3bc371b40dc6d9e26143ef72c41",
    ),
    (
        "golden_InverseComposite.h5#numeric",
        "20e4a519f7b9638d011d9ddcd50128fda6aeea1a3f3c04425c9a0ec28588f134",
    ),
    (
        "golden_0GenericAffine.mat",
        "f7710b7efb636b6b8f7038e5603f19c208df52ca4e7b2e905223f3325ebb6e0d",
    ),
    (
        "golden_1Warp.nii.gz",
        "726be59d1475b3cfb64957109207c9da613b6b4f40f96e239397cacffb3fa8a6",
    ),
    (
        "golden_1InverseWarp.nii.gz",
        "50c10529912d3ef61df12d150ebe4621d52f46621a30e197e672578052949ad7",
    ),
    (
        "golden__Warped.nii.gz",
        "7a312e932122fcf08c37417c58466e152d5f83271d9d661659c4a5cb163630c1",
    ),
];

const N4_WINDOWS_X86_64: Table = &[
    (
        "n4_corrected",
        "7acbc933060c1b53e933e52a2ba7a8793719fc253bbaeac96c78bb9f98cc0f16",
    ),
    (
        "n4_bias_field",
        "0f1132685717a425e8077f4e340a2e85525bbfd0bff801a7ae8b9caf126ea5f2",
    ),
];

fn blob_volume(n: usize, spacing: f64, shift: [f64; 3]) -> Volume<f64> {
    let g = Grid::new([n, n, n], [spacing; 3], [0.0; 3], IDENTITY3).unwrap();
    let c = (n as f64 - 1.0) * spacing / 2.0;
    let data = (0..g.num_voxels())
        .map(|lin| {
            let [i, j, k] = g.ijk(lin);
            let p = g.index_to_point([i as f64, j as f64, k as f64]);
            let r2 = (p[0] - c - shift[0]).powi(2)
                + (p[1] - c - shift[1]).powi(2)
                + (p[2] - c - shift[2]).powi(2);
            let lobe = (p[0] - c - shift[0] - 6.0).powi(2)
                + (p[1] - c - shift[1] + 3.0).powi(2)
                + (p[2] - c - shift[2]).powi(2);
            let noise = ((lin * 2654435761) % 1000) as f64 / 20000.0;
            (-r2 / (2.0 * (0.22 * n as f64 * spacing).powi(2))).exp()
                + 0.6 * (-lobe / 18.0).exp()
                + noise
        })
        .collect();
    Volume::from_vec(g, data).unwrap()
}

fn small(r: LinearRecipe) -> LinearRecipe {
    LinearRecipe {
        iterations: vec![12, 8],
        shrink_factors: vec![2, 1],
        smoothing_sigmas: vec![1.0, 0.0],
        ..r
    }
}

fn run(tag: &str, channels: usize) -> Vec<(String, String)> {
    let mut fixed = vec![blob_volume(24, 2.0, [0.0; 3])];
    let mut moving = vec![blob_volume(24, 2.0, [4.0, -2.0, 1.0])];
    if channels == 2 {
        fixed.push(second_channel(&fixed[0]));
        moving.push(second_channel(&moving[0]));
    }
    let linear = LinearPlan {
        rigid: small(LinearRecipe::cohort_rigid()),
        affine: small(LinearRecipe::cohort_affine()),
    };
    let syn = SynRecipe {
        iterations: vec![6, 4],
        shrink_factors: vec![2, 1],
        smoothing_sigmas: vec![1.0, 0.0],
        ..SynRecipe::cohort_channels(channels)
    };
    let out = register_stages(
        &CpuBackend,
        &CpuLinearEngine,
        &StagePlan::cohort(&linear, &syn),
        &fixed,
        &moving,
        Some(2.0),
        &SynOptions::default(),
    )
    .unwrap();
    let dir = Scratch::new(tag);
    let written = write_stage_outputs(&prefix(&dir), &out, OutputLayout::Both, true).unwrap();
    hash_outputs(&written)
}

fn n4_hashes() -> Vec<(String, String)> {
    let out = run_n4(&n4_phantom([20, 18, 16], 2.0), &n4_params()).unwrap();
    vec![
        hash_f32s("n4_corrected", out.corrected.data()),
        hash_f32s("n4_bias_field", out.bias_field.data()),
    ]
}

#[test]
fn cpu_registration_outputs_match_the_golden_hashes() {
    let first = run("a", 1);
    assert_eq!(first.len(), 8);
    assert_eq!(first, run("b", 1), "two CPU runs differ");
    check("REGISTRATION", &platform_key(), &first, TABLES);
}

#[test]
fn cpu_two_channel_registration_outputs_match_the_golden_hashes() {
    let first = run("c", 2);
    assert_eq!(first.len(), 8);
    assert_eq!(first, run("d", 2), "two CPU runs differ");
    check("TWO_CHANNEL", &platform_key(), &first, TABLES_TWO_CHANNEL);
}

#[test]
fn cpu_n4_outputs_match_the_golden_hashes() {
    let first = n4_hashes();
    assert_eq!(first, n4_hashes(), "two CPU N4 runs differ");
    check("N4", &platform_key(), &first, TABLES_N4);
}

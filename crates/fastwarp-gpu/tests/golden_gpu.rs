// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used)]

mod common;

#[path = "../../fastwarp-reg/tests/support/golden.rs"]
mod golden;

use common::*;
use fastwarp_gpu::{BackendSelect, GpuError, GpuN4};
use fastwarp_reg::apply::{apply_chain, Interpolation};
use fastwarp_reg::linear::LinearRecipe;
use fastwarp_reg::register::LinearPlan;
use fastwarp_reg::stages::{register_stages, write_stage_outputs, OutputLayout, StagePlan};
use fastwarp_reg::syn::SynOptions;
use fastwarp_reg::{AffineTransform, TransformChain, TransformElem};
use golden::{
    check, hash_f32s, hash_outputs, n4_params, n4_phantom, platform_key, prefix, second_channel,
    Scratch, Table, Tables,
};

const WINDOWS_ARC_140V_VULKAN: &str =
    "windows-x86_64 | Intel(R) Arc(TM) 140V GPU (16GB) | Vulkan | 101.8991";

const TABLES: Tables = &[(
    WINDOWS_ARC_140V_VULKAN,
    WINDOWS_ARC_140V_VULKAN_REGISTRATION,
)];

const TABLES_TWO_CHANNEL: Tables =
    &[(WINDOWS_ARC_140V_VULKAN, WINDOWS_ARC_140V_VULKAN_TWO_CHANNEL)];

const TABLES_N4: Tables = &[(WINDOWS_ARC_140V_VULKAN, WINDOWS_ARC_140V_VULKAN_N4)];

const WINDOWS_ARC_140V_VULKAN_REGISTRATION: Table = &[
    (
        "golden_Composite.h5",
        "04075a65774d709b102bd7f57e7b11f8624954a96d7c9c4ba61e8a721185608f",
    ),
    (
        "golden_Composite.h5#numeric",
        "42a053455054881b6d17e8ced438917f053902c898116d0e9cd7fd54820b70e2",
    ),
    (
        "golden_InverseComposite.h5",
        "cc990b89d2759326b9511f9631f59f264ec9569f8f0e961cdfcd7b0a26dd80da",
    ),
    (
        "golden_InverseComposite.h5#numeric",
        "53f4e0454326d2c19eb2bf4305db72e6695cc479885a4482a4acef24f6040fee",
    ),
    (
        "golden_0GenericAffine.mat",
        "64c59638343338fef691f29cff0407b84d06d2df4bb9636d1fcf516ac1480225",
    ),
    (
        "golden_1Warp.nii.gz",
        "3d9117aaa38aee54b1439cb1d00718933334ecfd50c707b97a2b2f2f4b29ab60",
    ),
    (
        "golden_1InverseWarp.nii.gz",
        "6c94decb54fa91529698aee97b1cc1856429c67709006e9cbd3e88c25a2b92d4",
    ),
    (
        "golden__Warped.nii.gz",
        "b8a31d7eaf45ffe306aece18a2c1605fafb85c57851b17a9b760357f583ee3fd",
    ),
];

const WINDOWS_ARC_140V_VULKAN_TWO_CHANNEL: Table = &[
    (
        "golden_Composite.h5",
        "f72c30c657e273dadf5ab8ff6d667dec039ead71fab6fad7ee1cdbb617bbbe22",
    ),
    (
        "golden_Composite.h5#numeric",
        "0acaed8db438e73c79b950b0161da7b5b8fad90234a5b2b4e3fafc1d5d4181ad",
    ),
    (
        "golden_InverseComposite.h5",
        "67c0e146692334428df253356973c2110d4a42c10e65eb4f19972aa5df6b2010",
    ),
    (
        "golden_InverseComposite.h5#numeric",
        "719ee510bde3cb8deb0a4bff9dd1983c96f293148fdeaaf8f781664a3a9c6825",
    ),
    (
        "golden_0GenericAffine.mat",
        "64c59638343338fef691f29cff0407b84d06d2df4bb9636d1fcf516ac1480225",
    ),
    (
        "golden_1Warp.nii.gz",
        "3d46f35e85fc3159177f16b8b44450307ca56e27d2dfcb058706ea9ff647aab3",
    ),
    (
        "golden_1InverseWarp.nii.gz",
        "9e7341e3e80ebc7d198e69f854266f92f22d85e2268c397e5ce86ca43db3cc41",
    ),
    (
        "golden__Warped.nii.gz",
        "d90c4447d4a5eb04f5c52cfe1ce09832f30fb7ef6a287ef4ee0d1952a8300347",
    ),
];

const WINDOWS_ARC_140V_VULKAN_N4: Table = &[
    (
        "n4_corrected",
        "7172a013b0b96297e71b7b0e683a5880a1e1ea6a3bbc26960204c050033999fc",
    ),
    (
        "n4_bias_field",
        "c57f66bd5e14272512b841c2c1d6b01446515a366dfca8e2ae61b1df8d625a9b",
    ),
];

fn short(r: LinearRecipe) -> LinearRecipe {
    LinearRecipe {
        iterations: vec![12, 8],
        shrink_factors: vec![2, 1],
        smoothing_sigmas: vec![1.0, 0.0],
        ..r
    }
}

type Hashes = Vec<(String, String)>;

fn adapter_key(name: &str, api: &str, driver_info: &str) -> String {
    format!("{} | {name} | {api} | {driver_info}", platform_key())
}

fn run(tag: &str, channels: usize) -> Option<(String, Hashes)> {
    let h = head([32; 3], 6.0, 3.0);
    let shift = TransformChain::from_queue(vec![TransformElem::Affine(
        AffineTransform::from_translation([6.0, -4.0, 3.0]),
    )]);
    let moving = apply_chain(&h.moving, &h.grid, &shift, Interpolation::Linear, 0.0).unwrap();
    let mut fixed = vec![h.fixed.clone()];
    let mut moving = vec![moving];
    if channels == 2 {
        fixed.push(second_channel(&fixed[0]));
        moving.push(second_channel(&moving[0]));
    }
    let linear = LinearPlan {
        rigid: short(LinearRecipe::cohort_rigid()),
        affine: short(LinearRecipe::cohort_affine()),
    };
    let mut syn = small_recipe(&[(2, 1.0, 4), (1, 0.0, 3)], 2);
    syn.channel_weights = vec![1.0 / channels as f64; channels];
    let b = backend()?;
    let info = &b.context().info;
    let key = adapter_key(&info.name, &info.backend, &info.driver_info);
    let out = register_stages(
        &b,
        &b,
        &StagePlan::cohort(&linear, &syn),
        &fixed,
        &moving,
        None,
        &SynOptions::default(),
    )
    .unwrap();
    drop(b);
    let dir = Scratch::new(tag);
    let written = write_stage_outputs(&prefix(&dir), &out, OutputLayout::Both, true).unwrap();
    Some((key, hash_outputs(&written)))
}

fn n4_run() -> Option<(String, Hashes)> {
    let g = match GpuN4::new(&BackendSelect::from_env().unwrap_or_default()) {
        Ok(g) => g,
        Err(GpuError::NoAdapter(why)) => {
            if require_adapter() {
                panic!("no GPU adapter with FASTWARP_GPU_REQUIRE_ADAPTER set: {why}");
            }
            eprintln!("skip: no adapter ({why})");
            return None;
        }
        Err(e) => panic!("device creation failed: {e}"),
    };
    let info = &g.context().info;
    let key = adapter_key(&info.name, &info.backend, &info.driver_info);
    let out = g
        .run_n4(&n4_phantom([20, 18, 16], 2.0), &n4_params())
        .unwrap();
    Some((
        key,
        vec![
            hash_f32s("n4_corrected", out.corrected.data()),
            hash_f32s("n4_bias_field", out.bias_field.data()),
        ],
    ))
}

fn registration_case(case: &str, channels: usize, tables: Tables) {
    let Some((key, first)) = run("a", channels) else {
        return;
    };
    assert_eq!(first.len(), 8);
    let Some((_, second)) = run("b", channels) else {
        return;
    };
    assert_eq!(first, second, "two GPU runs on the same adapter differ");
    check(case, &key, &first, tables);
}

#[test]
fn gpu_registration_outputs_are_deterministic_and_pinned_per_adapter() {
    registration_case("REGISTRATION", 1, TABLES);
}

#[test]
fn gpu_two_channel_registration_outputs_are_deterministic_and_pinned_per_adapter() {
    registration_case("TWO_CHANNEL", 2, TABLES_TWO_CHANNEL);
}

#[test]
fn gpu_n4_outputs_are_deterministic_and_pinned_per_adapter() {
    let Some((key, first)) = n4_run() else {
        return;
    };
    let Some((_, second)) = n4_run() else {
        return;
    };
    assert_eq!(first, second, "two GPU N4 runs on the same adapter differ");
    check("N4", &key, &first, TABLES_N4);
}

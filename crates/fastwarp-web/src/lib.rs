// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! FastWarp in a web browser: the default registration (centre of mass, rigid, affine, SyN) of
//! two NIfTI images held in memory, on WebGPU, with the ANTs-compatible output files returned
//! as one zip archive. The JavaScript entry points (`gpu_info`, `register`, `cancel`, `view_info`, `view_slice`,
//! `last_failure`, `debug_fault`) exist
//! on `wasm32-unknown-unknown` only; the rest also builds natively, for its tests.

pub mod diag;
pub mod options;
pub mod progress;
pub mod run;
pub mod viewer;
pub mod zip;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod web;

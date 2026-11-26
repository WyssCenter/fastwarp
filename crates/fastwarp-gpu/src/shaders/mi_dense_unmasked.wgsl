// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

@group(0) @binding(BINDING_UNIFORM) var<uniform> u: DenseMiU;

fn fixed_in(i: u32) -> bool {
    return true;
}

fn moving_in(i: u32) -> bool {
    return true;
}

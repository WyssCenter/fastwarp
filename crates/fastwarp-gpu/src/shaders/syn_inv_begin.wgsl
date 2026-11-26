// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

struct P {
    max_tol: f32,
    mean_tol: f32,
    pad0: u32,
    pad1: u32,
    groups: vec4<u32>,
}

struct St {
    max_err: f32,
    mean_err: f32,
    executed: u32,
    run: u32,
}

@group(0) @binding(0) var<storage, read_write> st: St;
@group(0) @binding(1) var<storage, read_write> args: array<u32, 6>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(1)
fn k_inv_begin() {
    if (st.run != 0u && !(st.max_err > p.max_tol && st.mean_err > p.mean_tol)) {
        st.run = 0u;
    }
    let on = st.run != 0u;
    args[0] = select(0u, p.groups.x, on);
    args[1] = select(0u, p.groups.y, on);
    args[2] = select(0u, p.groups.z, on);
    args[3] = select(0u, 1u, on);
    args[4] = select(0u, 1u, on);
    args[5] = select(0u, 1u, on);
}

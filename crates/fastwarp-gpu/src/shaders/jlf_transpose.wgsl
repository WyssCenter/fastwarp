// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

struct TransposeU {
  n: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: TransposeU;

@compute @workgroup_size(64)
fn k_jlf_transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x + gid.y * u.n.w;
  if (t >= u.n.z) {
    return;
  }
  let count = u.n.x;
  let width = u.n.y;
  let lane = t % 64u;
  let rest = t / 64u;
  let kk = rest % width;
  let c = (rest / width) * 64u + lane;
  var v = 0.0;
  if (c < count) {
    v = src[c * width + kk];
  }
  dst[t] = v;
}

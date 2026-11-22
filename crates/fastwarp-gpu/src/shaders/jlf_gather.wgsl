// SPDX-FileCopyrightText: 1999-2019 Insight Software Consortium
// SPDX-FileCopyrightText: 2020-present NumFOCUS
// SPDX-FileCopyrightText: ANTs developers
// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0
//
// Copyright NumFOCUS (Insight Software Consortium)
// Copyright 2026 Wyss Center for Bio and Neuroengineering
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions derived from ANTs `itkWeightedVotingFusionImageFilter.hxx` (the intensity fusion and the
// patch votes of `ThreadedGenerateDataForWeightedAveraging`: the reset of a non-finite intensity sum,
// the count of the first modality inside the mask, votes for labels of the label set only),
// gathered per voxel over the centres whose patch covers it, in the order the filter adds them.
// See `docs/THIRD_PARTY_PORTS.md`.

struct GatherU {
  dims: vec4<u32>,
  a: vec4<u32>,
  b: vec4<u32>,
  c: vec4<u32>,
  e: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> atlas: array<f32>;
@group(0) @binding(1) var<storage, read> slots_img: array<u32>;
@group(0) @binding(2) var<storage, read> cmap: array<u32>;
@group(0) @binding(3) var<storage, read> cw: array<f32>;
@group(0) @binding(4) var<storage, read> co: array<u32>;
@group(0) @binding(5) var<storage, read> maskbuf: array<u32>;
@group(0) @binding(6) var<storage, read> poffsets: array<vec4<i32>>;
@group(0) @binding(7) var<storage, read_write> out: array<u32>;
@group(0) @binding(8) var<uniform> u: GatherU;

const K: u32 = 16u;
const UNSET: u32 = 0xffffffffu;

fn finite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn inside(p: vec3<i32>) -> bool {
  return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < i32(u.dims.x) && p.y < i32(u.dims.y)
    && p.z < i32(u.dims.z);
}

fn unpack(o: u32) -> vec3<i32> {
  return vec3<i32>(i32(o & 0xffu) - 128, i32((o >> 8u) & 0xffu) - 128, i32((o >> 16u) & 0xffu) - 128);
}

fn centre_at(c: vec3<i32>) -> u32 {
  let lo = i32(u.a.z);
  if (c.z < lo || c.z >= lo + i32(u.c.y)) {
    return UNSET;
  }
  let nx = u.dims.x;
  return cmap[(u32(c.z - lo) * u.dims.y + u32(c.y)) * nx + u32(c.x)];
}

fn image_index(q: vec3<i32>) -> u32 {
  return (u32(q.z - i32(u.a.y)) * u.dims.y + u32(q.y)) * u.dims.x + u32(q.x);
}

@compute @workgroup_size(64)
fn k_jlf_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = u.e.x + gid.x + gid.y * u.b.z;
  if (t >= u.dims.w) {
    return;
  }
  let nx = u.dims.x;
  let ny = u.dims.y;
  let plane = nx * ny;
  let v = vec3<i32>(i32(t % nx), i32((t / nx) % ny), i32(u.a.x + t / plane));
  let n = u.a.w;
  let modalities = u.b.x;
  let p = u.b.y;
  let rsize = u.b.w;
  let apl = u.c.z;
  let base = t * rsize;
  var count = 0u;
  let inmask = maskbuf[t] != 0u;
  for (var mo = 0u; mo < modalities; mo = mo + 1u) {
    var acc = 0.0;
    if (inmask) {
      for (var jj = 0u; jj < p; jj = jj + 1u) {
        let c = v - poffsets[p - 1u - jj].xyz;
        if (!inside(c)) {
          continue;
        }
        let ci = centre_at(c);
        if (ci == UNSET) {
          continue;
        }
        var est = 0.0;
        var bad = false;
        for (var i = 0u; i < n; i = i + 1u) {
          let q = v + unpack(co[ci * n + i]);
          if (!inside(q)) {
            bad = true;
            break;
          }
          let a = atlas[(i * modalities + mo) * apl + image_index(q)];
          if (!finite(a)) {
            bad = true;
            break;
          }
          est = est + cw[ci * n + i] * a;
        }
        if (bad) {
          acc = 0.0;
        } else {
          acc = acc + est;
          if (!finite(acc)) {
            acc = 0.0;
          }
        }
        if (mo == 0u) {
          count = count + 1u;
        }
      }
    }
    out[base + 3u + 2u * K + mo] = bitcast<u32>(acc);
  }
  var lab: array<u32, 16>;
  var sw: array<f32, 16>;
  var vw: array<f32, 32>;
  var used = 0u;
  var overflow = 0u;
  var wsum = 0.0;
  if (u.c.x != 0u) {
    for (var jj = 0u; jj < p; jj = jj + 1u) {
      let c = v - poffsets[p - 1u - jj].xyz;
      if (!inside(c)) {
        continue;
      }
      let ci = centre_at(c);
      if (ci == UNSET) {
        continue;
      }
      for (var i = 0u; i < n; i = i + 1u) {
        let q = v + unpack(co[ci * n + i]);
        if (!inside(q)) {
          continue;
        }
        let sl = slots_img[i * apl + image_index(q)];
        if (sl == UNSET) {
          continue;
        }
        let w = cw[ci * n + i];
        var k = 0u;
        loop {
          if (k >= used) {
            break;
          }
          if (lab[k] == sl) {
            break;
          }
          k = k + 1u;
        }
        if (k < used) {
          sw[k] = sw[k] + w;
        } else if (used < K) {
          lab[used] = sl;
          sw[used] = w;
          used = used + 1u;
        } else {
          overflow = 1u;
        }
        wsum = wsum + w;
        if (i < 32u) {
          vw[i] = vw[i] + w;
        }
      }
    }
  }
  out[base] = used | (overflow << 31u);
  out[base + 1u] = count;
  out[base + 2u] = bitcast<u32>(wsum);
  for (var k = 0u; k < K; k = k + 1u) {
    out[base + 3u + 2u * k] = select(UNSET, lab[k], k < used);
    out[base + 4u + 2u * k] = bitcast<u32>(sw[k]);
  }
  if (u.c.w != 0u) {
    for (var i = 0u; i < n; i = i + 1u) {
      out[base + 3u + 2u * K + modalities + i] = bitcast<u32>(vw[i]);
    }
  }
}

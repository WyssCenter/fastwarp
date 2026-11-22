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
// Portions derived from ITKAdaptiveDenoising `itkNonLocalPatchBasedImageFilter.hxx`
// (`ComputeNeighborhoodPatchSimilarity`: PC and MSQ, voxels outside the image or with a non-finite
// target value skipped, fewer than 3 voxels give `FLT_MAX`) and ANTs
// `itkWeightedVotingFusionImageFilter.hxx` (the per-atlas search loop of
// `ThreadedGenerateDataForWeightedAveraging`). See `docs/THIRD_PARTY_PORTS.md`.

struct SearchU {
  dims: vec4<u32>,
  slab: vec4<u32>,
  pw: vec4<u32>,
  pad: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> atlas: array<f32>;
@group(0) @binding(1) var<storage, read> tpatch: array<f32>;
@group(0) @binding(2) var<storage, read> centres: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> offsets: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read> poffsets: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read_write> best_out: array<u32>;
@group(0) @binding(6) var<uniform> u: SearchU;

const FLT_MAX: f32 = 0x1.fffffep+127f;
const FLT_EPSILON: f32 = 0x1p-23f;

fn finite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn is_nan(v: f32) -> bool {
  let b = bitcast<u32>(v);
  return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

fn is_pos_inf(v: f32) -> bool {
  return bitcast<u32>(v) == 0x7f800000u;
}

fn inside(p: vec3<i32>) -> bool {
  return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < i32(u.dims.x) && p.y < i32(u.dims.y)
    && p.z < i32(u.dims.z);
}

@compute @workgroup_size(64)
fn k_jlf_search(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = u.pw.w + gid.x + gid.y * u.pw.z;
  if (t >= u.dims.w) {
    return;
  }
  let atlases = u.slab.z;
  let c = t / atlases;
  let a = t % atlases;
  let rec = centres[c];
  let nx = u.dims.x;
  let ny = u.dims.y;
  let plane = nx * ny;
  let at = vec3<i32>(i32(rec.x % nx), i32((rec.x / nx) % ny), i32(rec.x / plane));
  let p = u.pw.x;
  let used = u.slab.w;
  let zlo = i32(u.slab.x);
  let planes = u.slab.y;
  let ybase = c * p * used;
  var found = false;
  var best = 0.0;
  var best_j = 0u;
  for (var j = 0u; j < rec.z; j = j + 1u) {
    let s = at + offsets[rec.y + j].xyz;
    if (!inside(s)) {
      continue;
    }
    var sx = 0.0;
    var sxx = 0.0;
    var sxy = 0.0;
    var sdd = 0.0;
    var n = 0.0;
    var xbad = false;
    var count = ybase;
    for (var m = 0u; m < used; m = m + 1u) {
      let mbase = (a * used + m) * planes;
      for (var k = 0u; k < p; k = k + 1u) {
        let yv = tpatch[count];
        count = count + 1u;
        let q = s + poffsets[k].xyz;
        if (!inside(q) || !finite(yv)) {
          continue;
        }
        let x = atlas[(mbase + u32(q.z - zlo)) * plane + u32(q.y) * nx + u32(q.x)];
        n = n + 1.0;
        if (!finite(x)) {
          xbad = true;
          continue;
        }
        sx = sx + x;
        sxx = sxx + x * x;
        sxy = sxy + x * yv;
        let d = yv - x;
        sdd = sdd + d * d;
      }
    }
    var sim = FLT_MAX;
    var usable = true;
    if (n >= 3.0) {
      if (u.pw.y == 0u) {
        if (xbad) {
          sim = 0.0;
        } else {
          let variance = sxx - sx * sx / n;
          let s2 = sxy * sxy;
          let positive = !is_nan(sxy) && sxy > 0.0;
          var measure = 0.0;
          if (is_pos_inf(variance)) {
            if (!finite(s2)) {
              usable = false;
            }
          } else if (finite(variance) && variance > FLT_EPSILON) {
            if (is_nan(s2)) {
              usable = false;
            } else if (is_pos_inf(s2)) {
              if (positive) {
                measure = bitcast<f32>(0x7f800000u);
              } else {
                usable = false;
              }
            } else {
              measure = s2 / variance;
            }
          }
          if (positive) {
            sim = -measure;
          } else {
            sim = measure;
          }
        }
      } else if (xbad) {
        usable = false;
      } else {
        sim = sdd / n;
      }
    }
    usable = usable && !is_nan(sim) && !is_pos_inf(sim);
    if (usable && (!found || sim < best)) {
      found = true;
      best = sim;
      best_j = j;
    }
  }
  best_out[t] = best_j;
}

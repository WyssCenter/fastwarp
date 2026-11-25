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
  ext: vec4<i32>,
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

fn score(sx: f32, sxx: f32, sxy: f32, sdd: f32, n: f32, xbad: bool, ok: ptr<function, bool>) -> f32 {
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
  *ok = usable && !is_nan(sim) && !is_pos_inf(sim);
  return sim;
}

@compute @workgroup_size(64)
fn k_jlf_search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = u.pw.w + gid.x + gid.y * u.pw.z;
  let count = u.pad.x;
  let pcount = u.pad.y;
  let atlases = u.slab.z;
  let a = t / pcount;
  let c = t % pcount;
  let live = t < u.dims.w && c < count;
  let nx = u.dims.x;
  let ny = u.dims.y;
  let plane = nx * ny;
  let p = u.pw.x;
  let used = u.slab.w;
  let zlo = i32(u.slab.x);
  let planes = u.slab.y;
  var rec = vec4<u32>(0u);
  if (live) {
    rec = centres[c];
  }
  let at = vec3<i32>(i32(rec.x % nx), i32((rec.x / nx) % ny), i32(rec.x / plane));
  if (!live) {
    return;
  }
  let ybase = c * p * used;
  var found = false;
  var best = 0.0;
  var best_j = 0u;
  for (var j = 0u; j < rec.z; j = j + 4u) {
    let s0 = at + offsets[rec.y + min(j + 0u, rec.z - 1u)].xyz;
    let v0 = j + 0u < rec.z && inside(s0);
    var sx0 = 0.0;
    var sxx0 = 0.0;
    var sxy0 = 0.0;
    var sdd0 = 0.0;
    var n0 = 0.0;
    var xbad0 = false;
    let s1 = at + offsets[rec.y + min(j + 1u, rec.z - 1u)].xyz;
    let v1 = j + 1u < rec.z && inside(s1);
    var sx1 = 0.0;
    var sxx1 = 0.0;
    var sxy1 = 0.0;
    var sdd1 = 0.0;
    var n1 = 0.0;
    var xbad1 = false;
    let s2 = at + offsets[rec.y + min(j + 2u, rec.z - 1u)].xyz;
    let v2 = j + 2u < rec.z && inside(s2);
    var sx2 = 0.0;
    var sxx2 = 0.0;
    var sxy2 = 0.0;
    var sdd2 = 0.0;
    var n2 = 0.0;
    var xbad2 = false;
    let s3 = at + offsets[rec.y + min(j + 3u, rec.z - 1u)].xyz;
    let v3 = j + 3u < rec.z && inside(s3);
    var sx3 = 0.0;
    var sxx3 = 0.0;
    var sxy3 = 0.0;
    var sdd3 = 0.0;
    var n3 = 0.0;
    var xbad3 = false;
    if (v0 || v1 || v2 || v3) {
      var count_i = ybase;
      for (var m = 0u; m < used; m = m + 1u) {
        let mbase = (a * used + m) * planes;
        for (var k = 0u; k < p; k = k + 1u) {
          let yv = tpatch[count_i];
          count_i = count_i + 1u;
          if (!finite(yv)) {
            continue;
          }
          let po = poffsets[k].xyz;
          let q0 = s0 + po;
          if (v0 && inside(q0)) {
            let x = atlas[(mbase + u32(q0.z - zlo)) * plane + u32(q0.y) * nx + u32(q0.x)];
            n0 = n0 + 1.0;
            if (!finite(x)) {
              xbad0 = true;
            } else {
              sx0 = sx0 + x;
              sxx0 = sxx0 + x * x;
              sxy0 = sxy0 + x * yv;
              let d = yv - x;
              sdd0 = sdd0 + d * d;
            }
          }
          let q1 = s1 + po;
          if (v1 && inside(q1)) {
            let x = atlas[(mbase + u32(q1.z - zlo)) * plane + u32(q1.y) * nx + u32(q1.x)];
            n1 = n1 + 1.0;
            if (!finite(x)) {
              xbad1 = true;
            } else {
              sx1 = sx1 + x;
              sxx1 = sxx1 + x * x;
              sxy1 = sxy1 + x * yv;
              let d = yv - x;
              sdd1 = sdd1 + d * d;
            }
          }
          let q2 = s2 + po;
          if (v2 && inside(q2)) {
            let x = atlas[(mbase + u32(q2.z - zlo)) * plane + u32(q2.y) * nx + u32(q2.x)];
            n2 = n2 + 1.0;
            if (!finite(x)) {
              xbad2 = true;
            } else {
              sx2 = sx2 + x;
              sxx2 = sxx2 + x * x;
              sxy2 = sxy2 + x * yv;
              let d = yv - x;
              sdd2 = sdd2 + d * d;
            }
          }
          let q3 = s3 + po;
          if (v3 && inside(q3)) {
            let x = atlas[(mbase + u32(q3.z - zlo)) * plane + u32(q3.y) * nx + u32(q3.x)];
            n3 = n3 + 1.0;
            if (!finite(x)) {
              xbad3 = true;
            } else {
              sx3 = sx3 + x;
              sxx3 = sxx3 + x * x;
              sxy3 = sxy3 + x * yv;
              let d = yv - x;
              sdd3 = sdd3 + d * d;
            }
          }
        }
      }
    }
    if (v0) {
      var ok0 = false;
      let sim0 = score(sx0, sxx0, sxy0, sdd0, n0, xbad0, &ok0);
      if (ok0 && (!found || sim0 < best)) {
        found = true;
        best = sim0;
        best_j = j + 0u;
      }
    }
    if (v1) {
      var ok1 = false;
      let sim1 = score(sx1, sxx1, sxy1, sdd1, n1, xbad1, &ok1);
      if (ok1 && (!found || sim1 < best)) {
        found = true;
        best = sim1;
        best_j = j + 1u;
      }
    }
    if (v2) {
      var ok2 = false;
      let sim2 = score(sx2, sxx2, sxy2, sdd2, n2, xbad2, &ok2);
      if (ok2 && (!found || sim2 < best)) {
        found = true;
        best = sim2;
        best_j = j + 2u;
      }
    }
    if (v3) {
      var ok3 = false;
      let sim3 = score(sx3, sxx3, sxy3, sdd3, n3, xbad3, &ok3);
      if (ok3 && (!found || sim3 < best)) {
        found = true;
        best = sim3;
        best_j = j + 3u;
      }
    }
  }
  best_out[c * atlases + a] = best_j;
}

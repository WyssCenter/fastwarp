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
// Portions derived from ANTs `itkWeightedVotingFusionImageFilter.hxx` (the joint error matrix of
// `ThreadedGenerateDataForWeightedAveraging`: z-scored patch differences, `(P - 1)`, `beta`,
// non-finite entries to 0, `alpha` on the diagonal, the solve of `MxBar w = 1`, negative weights to
// 0, normalisation) and ITKAdaptiveDenoising `itkNonLocalPatchBasedImageFilter.hxx`
// (`VectorizeImagePatch`). See `docs/THIRD_PARTY_PORTS.md`.

struct WeighU {
  dims: vec4<u32>,
  slab: vec4<u32>,
  pw: vec4<u32>,
  par: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> atlas: array<f32>;
@group(0) @binding(1) var<storage, read> tnorm: array<f32>;
@group(0) @binding(2) var<storage, read> centres: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> offsets: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read> poffsets: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read> best: array<u32>;
@group(0) @binding(6) var<storage, read_write> wout: array<f32>;
@group(0) @binding(7) var<uniform> u: WeighU;

const MAXN: u32 = 32u;
const FLT_EPSILON: f32 = 0x1p-23f;

fn finite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn inside(p: vec3<i32>) -> bool {
  return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < i32(u.dims.x) && p.y < i32(u.dims.y)
    && p.z < i32(u.dims.z);
}

fn tri(i: u32, j: u32) -> u32 {
  return i * (i + 1u) / 2u + j;
}

@compute @workgroup_size(64)
fn k_jlf_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = u.pw.w + gid.x + gid.y * u.pw.z;
  if (c >= u.dims.w) {
    return;
  }
  let n = u.slab.z;
  let used = u.slab.w;
  let p = u.pw.x;
  let nx = u.dims.x;
  let plane = nx * u.dims.y;
  let zlo = i32(u.slab.x);
  let planes = u.slab.y;
  let rec = centres[c];
  let at = vec3<i32>(i32(rec.x % nx), i32((rec.x / nx) % u.dims.y), i32(rec.x / plane));
  var shift: array<vec3<i32>, 32>;
  for (var i = 0u; i < n; i = i + 1u) {
    shift[i] = at + offsets[rec.y + best[c * n + i]].xyz;
  }
  var m: array<f32, 528>;
  for (var e = 0u; e < n * (n + 1u) / 2u; e = e + 1u) {
    m[e] = 0.0;
  }
  var bad = 0u;
  var tbad = false;
  var mean: array<f32, 32>;
  var inv: array<f32, 32>;
  var d: array<f32, 32>;
  var fill = 0u;
  for (var mo = 0u; mo < used; mo = mo + 1u) {
    for (var i = 0u; i < n; i = i + 1u) {
      let base = (i * used + mo) * planes;
      var sum = 0.0;
      var sq = 0.0;
      var cnt = 0.0;
      for (var k = 0u; k < p; k = k + 1u) {
        let q = shift[i] + poffsets[k].xyz;
        if (!inside(q)) {
          continue;
        }
        let x = atlas[(base + u32(q.z - zlo)) * plane + u32(q.y) * nx + u32(q.x)];
        if (finite(x)) {
          sum = sum + x;
          sq = sq + x * x;
          cnt = cnt + 1.0;
        }
      }
      fill = fill & ~(1u << i);
      if (cnt < 2.0) {
        bad = bad | (1u << i);
        continue;
      }
      let mu = sum / cnt;
      let variance = (sq - cnt * (mu * mu)) / (cnt - 1.0);
      if (!finite(variance) || variance < 0.0) {
        bad = bad | (1u << i);
        continue;
      }
      let sd = sqrt(variance);
      if (sd < FLT_EPSILON) {
        if (sd == 0.0) {
          bad = bad | (1u << i);
          continue;
        }
        fill = fill | (1u << i);
      }
      mean[i] = mu;
      inv[i] = 1.0 / sd;
    }
    for (var k = 0u; k < p; k = k + 1u) {
      let t = tnorm[c * p * used + mo * p + k];
      if (!finite(t)) {
        tbad = true;
        continue;
      }
      for (var i = 0u; i < n; i = i + 1u) {
        d[i] = 0.0;
        if ((bad & (1u << i)) != 0u) {
          continue;
        }
        var z: f32;
        if ((fill & (1u << i)) != 0u) {
          z = (0.0 - mean[i]) * inv[i];
        } else {
          let q = shift[i] + poffsets[k].xyz;
          if (!inside(q)) {
            bad = bad | (1u << i);
            continue;
          }
          let base = (i * used + mo) * planes;
          let x = atlas[(base + u32(q.z - zlo)) * plane + u32(q.y) * nx + u32(q.x)];
          if (!finite(x)) {
            bad = bad | (1u << i);
            continue;
          }
          z = (x - mean[i]) * inv[i];
        }
        d[i] = abs(z - t);
      }
      for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j <= i; j = j + 1u) {
          m[tri(i, j)] = m[tri(i, j)] + d[i] * d[j];
        }
      }
    }
  }
  let denom = f32(p) - 1.0;
  let beta = u.par.y;
  let mode = u.pw.y;
  for (var i = 0u; i < n; i = i + 1u) {
    for (var j = 0u; j <= i; j = j + 1u) {
      var v = m[tri(i, j)];
      if (tbad || p < 2u || (bad & ((1u << i) | (1u << j))) != 0u) {
        v = 0.0;
      } else {
        v = v / denom;
        if (mode == 1u) {
          v = v * v;
        } else if (mode == 2u) {
          if (v > 0.0) {
            v = pow(v, beta);
          } else if (beta == 0.0) {
            v = 1.0;
          } else {
            v = 0.0;
          }
        }
        if (!finite(v)) {
          v = 0.0;
        }
      }
      if (i == j) {
        v = v + u.par.x;
      }
      m[tri(i, j)] = v;
    }
  }
  var anorm = 0.0;
  for (var j = 0u; j < n; j = j + 1u) {
    var col = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
      col = col + abs(m[select(tri(j, i), tri(i, j), i >= j)]);
    }
    anorm = max(anorm, col);
  }
  var flag = 0u;
  for (var j = 0u; j < n; j = j + 1u) {
    var s = m[tri(j, j)];
    for (var k = 0u; k < j; k = k + 1u) {
      s = s - m[tri(j, k)] * m[tri(j, k)];
    }
    if (!finite(s) || !(s > 0.0)) {
      flag = 1u;
      break;
    }
    let r = sqrt(s);
    m[tri(j, j)] = r;
    for (var i = j + 1u; i < n; i = i + 1u) {
      var a = m[tri(i, j)];
      for (var k = 0u; k < j; k = k + 1u) {
        a = a - m[tri(i, k)] * m[tri(j, k)];
      }
      m[tri(i, j)] = a / r;
    }
  }
  var w: array<f32, 32>;
  if (flag == 0u) {
    var inorm = 0.0;
    for (var e = 0u; e < n; e = e + 1u) {
      for (var i = 0u; i < n; i = i + 1u) {
        var s = select(0.0, 1.0, i == e);
        for (var k = 0u; k < i; k = k + 1u) {
          s = s - m[tri(i, k)] * w[k];
        }
        w[i] = s / m[tri(i, i)];
      }
      var col = 0.0;
      for (var ii = 0u; ii < n; ii = ii + 1u) {
        let i = n - 1u - ii;
        var s = w[i];
        for (var k = i + 1u; k < n; k = k + 1u) {
          s = s - m[tri(k, i)] * w[k];
        }
        w[i] = s / m[tri(i, i)];
        col = col + abs(w[i]);
      }
      inorm = max(inorm, col);
    }
    let cond = anorm * inorm;
    if (!finite(cond) || cond > u.par.z) {
      flag = 1u;
    }
  }
  if (flag == 0u) {
    for (var i = 0u; i < n; i = i + 1u) {
      var s = 1.0;
      for (var k = 0u; k < i; k = k + 1u) {
        s = s - m[tri(i, k)] * w[k];
      }
      w[i] = s / m[tri(i, i)];
    }
    for (var ii = 0u; ii < n; ii = ii + 1u) {
      let i = n - 1u - ii;
      var s = w[i];
      for (var k = i + 1u; k < n; k = k + 1u) {
        s = s - m[tri(k, i)] * w[k];
      }
      w[i] = s / m[tri(i, i)];
    }
    var total = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
      if (!finite(w[i])) {
        flag = 1u;
      }
      if (w[i] < 0.0) {
        w[i] = 0.0;
      }
      total = total + w[i];
    }
    if (!finite(total) || !(total > 0.0)) {
      flag = 1u;
    } else {
      let scale = 1.0 / total;
      for (var i = 0u; i < n; i = i + 1u) {
        w[i] = w[i] * scale;
      }
    }
  }
  let o = c * (n + 1u);
  for (var i = 0u; i < n; i = i + 1u) {
    wout[o + i] = select(0.0, w[i], flag == 0u);
  }
  wout[o + n] = bitcast<f32>(flag);
}

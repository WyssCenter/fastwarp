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
const TK: u32 = 32u;
const WGS: u32 = 64u;
const PER: u32 = 9u;
const FLT_EPSILON: f32 = 0x1p-23f;

var<workgroup> dsh: array<f32, 1024>;
var<workgroup> msh: array<f32, 528>;
var<workgroup> mean: array<f32, 32>;
var<workgroup> inv: array<f32, 32>;
var<workgroup> fill: array<u32, 32>;
var<workgroup> shift: array<vec4<i32>, 32>;
var<workgroup> bad: array<atomic<u32>, 32>;
var<workgroup> tbad: atomic<u32>;

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

fn sample(i: u32, mo: u32, q: vec3<i32>) -> f32 {
  let nx = u.dims.x;
  let plane = nx * u.dims.y;
  let base = (i * u.slab.w + mo) * u.slab.y;
  return atlas[(base + u32(q.z - i32(u.slab.x))) * plane + u32(q.y) * nx + u32(q.x)];
}

@compute @workgroup_size(64)
fn k_jlf_weights_tiled(
  @builtin(local_invocation_index) li: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let c = u.pw.w + wid.x + wid.y * u.pw.z;
  if (c >= u.dims.w) {
    return;
  }
  let n = u.slab.z;
  let used = u.slab.w;
  let p = u.pw.x;
  let nx = u.dims.x;
  let plane = nx * u.dims.y;
  let rec = centres[c];
  let at = vec3<i32>(i32(rec.x % nx), i32((rec.x / nx) % u.dims.y), i32(rec.x / plane));
  let pairs = n * (n + 1u) / 2u;
  if (li < n) {
    shift[li] = vec4<i32>(at + offsets[rec.y + best[c * n + li]].xyz, 0);
    atomicStore(&bad[li], 0u);
  }
  if (li == 0u) {
    atomicStore(&tbad, 0u);
  }
  var pi: array<u32, 9>;
  var pj: array<u32, 9>;
  var acc: array<f32, 9>;
  for (var r = 0u; r < PER; r = r + 1u) {
    let pair = li + r * WGS;
    var i = 0u;
    loop {
      if ((i + 1u) * (i + 2u) / 2u > pair) {
        break;
      }
      i = i + 1u;
    }
    pi[r] = i;
    pj[r] = pair - i * (i + 1u) / 2u;
    acc[r] = 0.0;
  }
  workgroupBarrier();
  for (var mo = 0u; mo < used; mo = mo + 1u) {
    if (li < n) {
      fill[li] = 0u;
      if (atomicLoad(&bad[li]) == 0u) {
        var sum = 0.0;
        var sq = 0.0;
        var cnt = 0.0;
        for (var k = 0u; k < p; k = k + 1u) {
          let q = shift[li].xyz + poffsets[k].xyz;
          if (!inside(q)) {
            continue;
          }
          let x = sample(li, mo, q);
          if (finite(x)) {
            sum = sum + x;
            sq = sq + x * x;
            cnt = cnt + 1.0;
          }
        }
        let mu = sum / cnt;
        let variance = (sq - cnt * (mu * mu)) / (cnt - 1.0);
        if (cnt < 2.0 || !finite(variance) || variance < 0.0) {
          atomicStore(&bad[li], 1u);
        } else {
          let sd = sqrt(variance);
          if (sd < FLT_EPSILON) {
            if (sd == 0.0) {
              atomicStore(&bad[li], 1u);
            }
            fill[li] = 1u;
          }
          mean[li] = mu;
          inv[li] = 1.0 / sd;
        }
      }
    }
    workgroupBarrier();
    for (var k0 = 0u; k0 < p; k0 = k0 + TK) {
      for (var e = li; e < n * TK; e = e + WGS) {
        let i = e / TK;
        let kk = e % TK;
        let k = k0 + kk;
        var dv = 0.0;
        if (k < p) {
          let t = tnorm[c * p * used + mo * p + k];
          if (!finite(t)) {
            atomicStore(&tbad, 1u);
          } else if (atomicLoad(&bad[i]) == 0u) {
            var z = 0.0;
            var ok = true;
            if (fill[i] != 0u) {
              z = (0.0 - mean[i]) * inv[i];
            } else {
              let q = shift[i].xyz + poffsets[k].xyz;
              if (!inside(q)) {
                ok = false;
              } else {
                let x = sample(i, mo, q);
                if (!finite(x)) {
                  ok = false;
                } else {
                  z = (x - mean[i]) * inv[i];
                }
              }
            }
            if (ok) {
              dv = abs(z - t);
            } else {
              atomicStore(&bad[i], 1u);
            }
          }
        }
        dsh[i * TK + kk] = dv;
      }
      workgroupBarrier();
      for (var r = 0u; r < PER; r = r + 1u) {
        if (li + r * WGS < pairs) {
          var s = acc[r];
          let bi = pi[r] * TK;
          let bj = pj[r] * TK;
          for (var kk = 0u; kk < TK; kk = kk + 1u) {
            s = s + dsh[bi + kk] * dsh[bj + kk];
          }
          acc[r] = s;
        }
      }
      workgroupBarrier();
    }
  }
  for (var r = 0u; r < PER; r = r + 1u) {
    if (li + r * WGS < pairs) {
      msh[li + r * WGS] = acc[r];
    }
  }
  workgroupBarrier();
  if (li != 0u) {
    return;
  }
  let tb = atomicLoad(&tbad) != 0u;
  let denom = f32(p) - 1.0;
  let beta = u.par.y;
  let mode = u.pw.y;
  for (var i = 0u; i < n; i = i + 1u) {
    for (var j = 0u; j <= i; j = j + 1u) {
      var v = msh[tri(i, j)];
      if (tb || p < 2u || atomicLoad(&bad[i]) != 0u || atomicLoad(&bad[j]) != 0u) {
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
      msh[tri(i, j)] = v;
    }
  }
  var anorm = 0.0;
  for (var j = 0u; j < n; j = j + 1u) {
    var col = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
      col = col + abs(msh[select(tri(j, i), tri(i, j), i >= j)]);
    }
    anorm = max(anorm, col);
  }
  var flag = 0u;
  for (var j = 0u; j < n; j = j + 1u) {
    var s = msh[tri(j, j)];
    for (var k = 0u; k < j; k = k + 1u) {
      s = s - msh[tri(j, k)] * msh[tri(j, k)];
    }
    if (!finite(s) || !(s > 0.0)) {
      flag = 1u;
      break;
    }
    let r = sqrt(s);
    msh[tri(j, j)] = r;
    for (var i = j + 1u; i < n; i = i + 1u) {
      var a = msh[tri(i, j)];
      for (var k = 0u; k < j; k = k + 1u) {
        a = a - msh[tri(i, k)] * msh[tri(j, k)];
      }
      msh[tri(i, j)] = a / r;
    }
  }
  var w: array<f32, 32>;
  if (flag == 0u) {
    var inorm = 0.0;
    for (var e = 0u; e < n; e = e + 1u) {
      for (var i = 0u; i < n; i = i + 1u) {
        var s = select(0.0, 1.0, i == e);
        for (var k = 0u; k < i; k = k + 1u) {
          s = s - msh[tri(i, k)] * w[k];
        }
        w[i] = s / msh[tri(i, i)];
      }
      var col = 0.0;
      for (var ii = 0u; ii < n; ii = ii + 1u) {
        let i = n - 1u - ii;
        var s = w[i];
        for (var k = i + 1u; k < n; k = k + 1u) {
          s = s - msh[tri(k, i)] * w[k];
        }
        w[i] = s / msh[tri(i, i)];
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
        s = s - msh[tri(i, k)] * w[k];
      }
      w[i] = s / msh[tri(i, i)];
    }
    for (var ii = 0u; ii < n; ii = ii + 1u) {
      let i = n - 1u - ii;
      var s = w[i];
      for (var k = i + 1u; k < n; k = k + 1u) {
        s = s - msh[tri(k, i)] * w[k];
      }
      w[i] = s / msh[tri(i, i)];
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

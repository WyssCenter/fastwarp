// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

struct BoxU {
  dims: vec4<u32>,
  slab: vec4<u32>,
  org: vec4<i32>,
  ext: vec4<u32>,
  pr: vec4<i32>,
  sr: vec4<i32>,
  jb: vec4<u32>,
  sh: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> atlas: array<f32>;
@group(0) @binding(1) var<storage, read> tgt: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> ecent: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> cstat: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> bx: array<f32>;
@group(0) @binding(6) var<storage, read_write> px: array<f32>;
@group(0) @binding(7) var<storage, read_write> pxy: array<f32>;
@group(0) @binding(8) var<storage, read_write> state: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read_write> best_out: array<u32>;
@group(0) @binding(10) var<uniform> u: BoxU;

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
  let zlo = i32(u.slab.x);
  return p.x >= 0 && p.y >= 0 && p.z >= zlo && p.x < i32(u.dims.x) && p.y < i32(u.dims.y)
    && p.z < zlo + i32(u.slab.y);
}

fn sidx(p: vec3<i32>) -> u32 {
  return (u32(p.z - i32(u.slab.x)) * u.dims.y + u32(p.y)) * u.dims.x + u32(p.x);
}

fn xat(p: vec3<i32>) -> f32 {
  let plane = u.dims.x * u.dims.y;
  return atlas[u.slab.z * u.slab.y * plane + sidx(p)] - u.sh.x;
}

fn yat(p: vec3<i32>) -> f32 {
  return tgt[sidx(p)] - u.sh.y;
}

fn ecentre(t: u32) -> vec3<i32> {
  let v = ecent[t].x;
  let nx = u.dims.x;
  let ny = u.dims.y;
  return vec3<i32>(i32(v % nx), i32((v / nx) % ny), i32(v / (nx * ny)));
}

@compute @workgroup_size(64)
fn k_jlf_box(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x + gid.y * u.jb.w;
  let mode = u.dims.w;
  let bw = u.ext.x;
  let bh = u.ext.y;
  let bd = u.ext.z;
  let ncent = u.ext.w;
  let pr = u.pr.xyz;
  let sr = u.sr.xyz;
  let n = u.sh.z;
  if (mode == 0u) {
    if (t >= ncent) {
      return;
    }
    let c = ecentre(t);
    var s = 0.0;
    var s2 = 0.0;
    for (var dz = -pr.z; dz <= pr.z; dz = dz + 1) {
      for (var dy = -pr.y; dy <= pr.y; dy = dy + 1) {
        for (var dx = -pr.x; dx <= pr.x; dx = dx + 1) {
          let y = yat(c + vec3<i32>(dx, dy, dz));
          s = s + y;
          s2 = s2 + y * y;
        }
      }
    }
    let mu = s / n;
    let sd = sqrt(max(s2 - s * s / n, 0.0) / (n - 1.0));
    cstat[t] = vec2<f32>(mu, sd);
    return;
  }
  if (mode == 1u) {
    let dw = bw + 2u * u32(sr.x);
    let dh = bh + 2u * u32(sr.y);
    let dd = bd + 2u * u32(sr.z);
    let size = dw * dh * dd;
    if (t >= size) {
      return;
    }
    let p = u.org.xyz - sr + vec3<i32>(i32(t % dw), i32((t / dw) % dh), i32(t / (dw * dh)));
    var s = 0.0;
    var s2 = 0.0;
    for (var dz = -pr.z; dz <= pr.z; dz = dz + 1) {
      for (var dy = -pr.y; dy <= pr.y; dy = dy + 1) {
        for (var dx = -pr.x; dx <= pr.x; dx = dx + 1) {
          let q = p + vec3<i32>(dx, dy, dz);
          if (inside(q)) {
            let x = xat(q);
            s = s + x;
            s2 = s2 + x * x;
          }
        }
      }
    }
    bx[t] = s;
    bx[size + t] = s2;
    return;
  }
  let dph = bh + 2u * u32(pr.y);
  let dpd = bd + 2u * u32(pr.z);
  let dp = bw * dph * dpd;
  let dq = bw * bh * dpd;
  let j0 = u.jb.x;
  let jn = u.jb.y;
  let lstart = u.slab.w;
  if (mode == 2u) {
    if (t >= jn * dp) {
      return;
    }
    let jj = t / dp;
    let w = t % dp;
    let p = u.org.xyz + vec3<i32>(i32(w % bw), i32((w / bw) % dph) - pr.y, i32(w / (bw * dph)) - pr.z);
    let s = offsets[lstart + j0 + jj].xyz;
    var acc = 0.0;
    for (var dx = -pr.x; dx <= pr.x; dx = dx + 1) {
      let q = p + vec3<i32>(dx, 0, 0);
      if (inside(q) && inside(q + s)) {
        acc = acc + xat(q + s) * yat(q);
      }
    }
    px[t] = acc;
    return;
  }
  if (mode == 3u) {
    if (t >= jn * dq) {
      return;
    }
    let jj = t / dq;
    let w = t % dq;
    let ix = w % bw;
    let iy = (w / bw) % bh;
    let iz = w / (bw * bh);
    var acc = 0.0;
    for (var dy = 0u; dy <= 2u * u32(pr.y); dy = dy + 1u) {
      acc = acc + px[jj * dp + (iz * dph + iy + dy) * bw + ix];
    }
    pxy[t] = acc;
    return;
  }
  if (t >= ncent) {
    return;
  }
  let c = ecentre(t);
  let l = vec3<u32>(c - u.org.xyz);
  let st = cstat[t];
  var cur = state[t];
  if (j0 == 0u) {
    cur = vec4<u32>(bitcast<u32>(FLT_MAX), 0u, 0u, 0u);
  }
  var best = bitcast<f32>(cur.x);
  var best_j = cur.y;
  var found = cur.z != 0u;
  let dw = bw + 2u * u32(sr.x);
  let dh = bh + 2u * u32(sr.y);
  let dd = bd + 2u * u32(sr.z);
  let size = dw * dh * dd;
  for (var jj = 0u; jj < jn; jj = jj + 1u) {
    var cc = 0.0;
    for (var dz = 0u; dz <= 2u * u32(pr.z); dz = dz + 1u) {
      cc = cc + pxy[jj * dq + ((l.z + dz) * bh + l.y) * bw + l.x];
    }
    let s = offsets[lstart + j0 + jj].xyz;
    let b = vec3<u32>(c + s - (u.org.xyz - sr));
    let bi = (b.z * dh + b.y) * dw + b.x;
    let sx = bx[bi];
    let sxx = bx[size + bi];
    let sxy = (cc - st.x * sx) / st.y;
    var sim = FLT_MAX;
    var usable = true;
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
    usable = usable && !is_nan(sim) && !is_pos_inf(sim);
    if (usable && (!found || sim < best)) {
      found = true;
      best = sim;
      best_j = j0 + jj;
    }
  }
  state[t] = vec4<u32>(bitcast<u32>(best), best_j, u32(found), 0u);
  if (j0 + jn == u.jb.z) {
    best_out[ecent[t].y * u32(u.org.w) + u.slab.z] = best_j;
  }
}

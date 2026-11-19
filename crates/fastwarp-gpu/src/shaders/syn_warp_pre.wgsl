// SPDX-FileCopyrightText: 1999-2019 Insight Software Consortium
// SPDX-FileCopyrightText: 2020-present NumFOCUS
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
// Portions derived from ITK `itkDisplacementFieldTransform.hxx` (`TransformPoint`: a point
// outside the field's buffer is returned unchanged), `itkVectorLinearInterpolateImageFunction.hxx`
// (`EvaluateAtContinuousIndex`), `itkNearestNeighborInterpolateImageFunction.h` and
// `itkImageFunction.hxx` (the nearest-neighbour branch: the buffer test, then the index rounded
// half up). See `docs/THIRD_PARTY_PORTS.md`.

struct Q {
    nx: u32,
    ny: u32,
    nz: u32,
    n: u32,
    pnx: u32,
    pny: u32,
    pnz: u32,
    pn: u32,
    snx: u32,
    sny: u32,
    snz: u32,
    sn: u32,
    flags: u32,
    nplanes: u32,
    pad0: u32,
    pad1: u32,
    mp0: vec4<f32>,
    mp1: vec4<f32>,
    mp2: vec4<f32>,
    lp0: vec4<f32>,
    lp1: vec4<f32>,
    lp2: vec4<f32>,
    ms0: vec4<f32>,
    ms1: vec4<f32>,
    ms2: vec4<f32>,
    ls0: vec4<f32>,
    ls1: vec4<f32>,
    ls2: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> pre: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> p: Q;

fn tri_pre(c: vec3<f32>, off: u32) -> f32 {
    let nx = p.pnx;
    let ny = p.pny;
    let nz = p.pnz;
    let cc = clamp(c, vec3<f32>(0.0), vec3<f32>(f32(nx - 1u), f32(ny - 1u), f32(nz - 1u)));
    let f = floor(cc);
    let t = cc - f;
    let x0 = u32(f.x);
    let y0 = u32(f.y);
    let z0 = u32(f.z);
    let x1 = min(x0 + 1u, nx - 1u);
    let y1 = min(y0 + 1u, ny - 1u);
    let z1 = min(z0 + 1u, nz - 1u);
    let sxy = nx * ny;
    let b00 = off + y0 * nx + z0 * sxy;
    let b10 = off + y1 * nx + z0 * sxy;
    let b01 = off + y0 * nx + z1 * sxy;
    let b11 = off + y1 * nx + z1 * sxy;
    let a00 = pre[b00 + x0];
    let a10 = pre[b10 + x0];
    let a01 = pre[b01 + x0];
    let a11 = pre[b11 + x0];
    let c00 = a00 + (pre[b00 + x1] - a00) * t.x;
    let c10 = a10 + (pre[b10 + x1] - a10) * t.x;
    let c01 = a01 + (pre[b01 + x1] - a01) * t.x;
    let c11 = a11 + (pre[b11 + x1] - a11) * t.x;
    let c0 = c00 + (c10 - c00) * t.y;
    let c1 = c01 + (c11 - c01) * t.y;
    return c0 + (c1 - c0) * t.z;
}

@compute @workgroup_size(8, 8, 4)
fn k_warp_pre(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.nx || id.y >= p.ny || id.z >= p.nz) {
        return;
    }
    let i = id.x + p.nx * (id.y + p.ny * id.z);
    let idx = vec3<f32>(id);
    let u = vec3<f32>(field[i], field[p.n + i], field[2u * p.n + i]);
    let cp = vec3<f32>(
        dot(p.mp0.xyz, idx) + p.mp0.w + dot(p.lp0.xyz, u),
        dot(p.mp1.xyz, idx) + p.mp1.w + dot(p.lp1.xyz, u),
        dot(p.mp2.xyz, idx) + p.mp2.w + dot(p.lp2.xyz, u),
    );
    var w = vec3<f32>(0.0);
    if (inside(cp, p.pnx, p.pny, p.pnz)) {
        w = vec3<f32>(tri_pre(cp, 0u), tri_pre(cp, p.pn), tri_pre(cp, 2u * p.pn));
    }
    let c = vec3<f32>(
        dot(p.ms0.xyz, cp) + p.ms0.w + dot(p.ls0.xyz, w),
        dot(p.ms1.xyz, cp) + p.ms1.w + dot(p.ls1.xyz, w),
        dot(p.ms2.xyz, cp) + p.ms2.w + dot(p.ls2.xyz, w),
    );
    let ok = inside(c, p.snx, p.sny, p.snz);
    if ((p.flags & 2u) != 0u) {
        var v = 0.0;
        if (ok) {
            let hi = vec3<f32>(f32(p.snx - 1u), f32(p.sny - 1u), f32(p.snz - 1u));
            let r = vec3<u32>(clamp(floor(c + vec3<f32>(0.5)), vec3<f32>(0.0), hi));
            if (src[r.x + p.snx * (r.y + p.sny * r.z)] != 0.0) {
                v = 1.0;
            }
        }
        if ((p.flags & 4u) != 0u) {
            v = v * dst[i];
        }
        dst[i] = v;
        return;
    }
    for (var k = 0u; k < p.nplanes; k = k + 1u) {
        var v = 0.0;
        if (ok) {
            v = tri(c, k * p.sn, p.snx, p.sny, p.snz);
        }
        dst[k * p.n + i] = v;
    }
}

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
// Portions derived from ITK `itkNearestNeighborInterpolateImageFunction.h` and
// `itkImageFunction.hxx` (the nearest-neighbour branch: the buffer test, then the index rounded
// half up). See `docs/THIRD_PARTY_PORTS.md`.

struct P {
    nx: u32,
    ny: u32,
    nz: u32,
    n: u32,
    snx: u32,
    sny: u32,
    snz: u32,
    sn: u32,
    flags: u32,
    nplanes: u32,
    pad0: u32,
    pad1: u32,
    mi0: vec4<f32>,
    mi1: vec4<f32>,
    mi2: vec4<f32>,
    l0: vec4<f32>,
    l1: vec4<f32>,
    l2: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(8, 8, 4)
fn k_warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.nx || id.y >= p.ny || id.z >= p.nz) {
        return;
    }
    let i = id.x + p.nx * (id.y + p.ny * id.z);
    let idx = vec3<f32>(id);
    var c = vec3<f32>(
        dot(p.mi0.xyz, idx) + p.mi0.w,
        dot(p.mi1.xyz, idx) + p.mi1.w,
        dot(p.mi2.xyz, idx) + p.mi2.w,
    );
    if ((p.flags & 1u) != 0u) {
        let u = vec3<f32>(field[i], field[p.n + i], field[2u * p.n + i]);
        c = c + vec3<f32>(dot(p.l0.xyz, u), dot(p.l1.xyz, u), dot(p.l2.xyz, u));
    }
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

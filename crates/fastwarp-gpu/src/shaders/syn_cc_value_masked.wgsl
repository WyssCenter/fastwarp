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
// Portions derived from ITK
// `itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.hxx`
// (`ThreadedExecution`: the local CC of each valid point and, per workgroup, the partial sums of
// the value and of the number of valid points, so the value is the mean over the valid points).
// See `docs/THIRD_PARTY_PORTS.md`.

struct P {
    nx: u32,
    ny: u32,
    nz: u32,
    n: u32,
    radius: u32,
    part_base: u32,
    pad1: u32,
    pad2: u32,
    o0: vec4<f32>,
    o1: vec4<f32>,
    o2: vec4<f32>,
    sp: vec4<f32>,
}

const EPS: f32 = 2.220446e-16;

@group(0) @binding(0) var<storage, read> s_ff: array<f32>;
@group(0) @binding(1) var<storage, read> s_mm: array<f32>;
@group(0) @binding(2) var<storage, read> s_fm: array<f32>;
@group(0) @binding(3) var<storage, read> valid: array<f32>;
@group(0) @binding(4) var<storage, read_write> part: array<f32>;
@group(0) @binding(5) var<storage, read_write> num_part: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

var<workgroup> sh: array<f32, 256>;
var<workgroup> shn: array<f32, 256>;

@compute @workgroup_size(8, 8, 4)
fn k_cc_value_masked(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) w: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
) {
    var val = 0.0;
    var num = 0.0;
    if (id.x < p.nx && id.y < p.ny && id.z < p.nz) {
        let i = id.x + p.nx * (id.y + p.ny * id.z);
        if (valid[i] != 0.0) {
            let sff = s_ff[i];
            let smm = s_mm[i];
            let sfm = s_fm[i];
            var cc = 1.0;
            let den = sff * smm;
            if (abs(den) > EPS) {
                cc = sfm * sfm / den;
            }
            val = -cc;
            num = 1.0;
        }
    }
    sh[li] = val;
    shn[li] = num;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) {
            sh[li] = sh[li] + sh[li + s];
            shn[li] = shn[li] + shn[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let g = w.x + nw.x * (w.y + nw.y * w.z);
        part[p.part_base + g] = sh[0];
        num_part[g] = shn[0];
    }
}

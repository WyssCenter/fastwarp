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
// (`ComputeInformationFromQueues`: the window means and centred second moments from the sums
// and the count of the valid points; a point whose centre is not valid gets no statistics),
// computed in place over the window-sum planes.
// See `docs/THIRD_PARTY_PORTS.md`.

struct P {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

const REL: f32 = 1e-6;

@group(0) @binding(0) var<storage, read_write> s_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_aa: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_bb: array<f32>;
@group(0) @binding(4) var<storage, read_write> s_ab: array<f32>;
@group(0) @binding(5) var<storage, read> s_n: array<f32>;
@group(0) @binding(6) var<storage, read> valid: array<f32>;
@group(0) @binding(7) var<uniform> p: P;

@compute @workgroup_size(256)
fn k_cc_stats_masked(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) w: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
) {
    let i = (w.y * nw.x + w.x) * 256u + li;
    if (i >= p.n) {
        return;
    }
    let cn = s_n[i];
    if (valid[i] == 0.0 || cn <= 0.0) {
        s_a[i] = 0.0;
        s_b[i] = 0.0;
        s_aa[i] = 0.0;
        s_bb[i] = 0.0;
        s_ab[i] = 0.0;
        return;
    }
    let si = s_a[i];
    let sj = s_b[i];
    let sii = s_aa[i];
    let sjj = s_bb[i];
    let sij = s_ab[i];
    var sff = sii - si * si / cn;
    var smm = sjj - sj * sj / cn;
    let sfm = sij - si * sj / cn;
    if (!(sff > REL * sii && smm > REL * sjj)) {
        sff = 0.0;
        smm = 0.0;
    }
    s_a[i] = si / cn;
    s_b[i] = sj / cn;
    s_aa[i] = sff;
    s_bb[i] = smm;
    s_ab[i] = sfm;
}

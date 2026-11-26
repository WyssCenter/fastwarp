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
// Portions derived from ITK `itkMattesMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader.hxx`
// (`ProcessPoint`: joint PDF and fixed marginal accumulation, here in `2^-32` fixed point with a
// 64-bit sum per workgroup over every voxel of the grid). See
// `docs/THIRD_PARTY_PORTS.md`.

const STRIDE: u32 = 1056u;
const SCALE: f32 = 4294967296.0;

@group(0) @binding(2) var<storage, read_write> partials: array<u32>;

var<workgroup> wg_lo: array<atomic<u32>, 1056>;
var<workgroup> wg_hi: array<atomic<u32>, 1056>;

fn add64(cell: u32, w: u32) {
    let old = atomicAdd(&wg_lo[cell], w);
    if (old + w < old) {
        atomicAdd(&wg_hi[cell], 1u);
    }
}

@compute @workgroup_size(256)
fn k_mi_dense_hist(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    for (var k = lid; k < STRIDE; k = k + 256u) {
        atomicStore(&wg_lo[k], 0u);
        atomicStore(&wg_hi[k], 0u);
    }
    workgroupBarrier();
    let bins = u.cfg.x;
    let base = wid.x * u.cfg.y;
    for (var j = 0u; j < u.cfg.y; j = j + 256u) {
        let i = base + j + lid;
        if (i < u.dims.w) {
            let q = dense_point(i);
            if (q.ok) {
                for (var k = -1; k <= 2; k = k + 1) {
                    let pm = q.mbin + k;
                    add64(q.fbin * bins + u32(pm), u32(b3(f32(pm) - q.term) * SCALE + 0.5));
                }
                add64(1024u + q.fbin, 1u);
            }
        }
    }
    workgroupBarrier();
    let out = 2u * wid.x * STRIDE;
    for (var k = lid; k < STRIDE; k = k + 256u) {
        partials[out + 2u * k] = atomicLoad(&wg_lo[k]);
        partials[out + 2u * k + 1u] = atomicLoad(&wg_hi[k]);
    }
}

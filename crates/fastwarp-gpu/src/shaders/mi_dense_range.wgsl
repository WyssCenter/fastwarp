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
// Portions derived from ITK `itkMattesMutualInformationImageToImageMetricv4.hxx` (`Initialize` for
// dense sampling: the minimum and maximum of each image, here per workgroup in `f32`). See
// `docs/THIRD_PARTY_PORTS.md`.

const BIG: f32 = 3.40282347e38;

@group(0) @binding(2) var<storage, read_write> range_parts: array<vec4<f32>>;

var<workgroup> wg_range: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn k_mi_dense_range(@builtin(local_invocation_id) lid3: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = lid3.x;
    var r = vec4<f32>(BIG, -BIG, BIG, -BIG);
    let base = wid.x * u.cfg.y;
    for (var j = 0u; j < u.cfg.y; j = j + 256u) {
        let i = base + j + lid;
        if (i < u.dims.w) {
            let f = fix[i];
            if (fixed_in(i) && finite(f)) {
                r.x = min(r.x, f);
                r.y = max(r.y, f);
            }
            let m = src[i];
            if (moving_in(i) && finite(m)) {
                r.z = min(r.z, m);
                r.w = max(r.w, m);
            }
        }
    }
    wg_range[lid] = r;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let a = wg_range[lid];
            let b = wg_range[lid + s];
            wg_range[lid] = vec4<f32>(min(a.x, b.x), max(a.y, b.y), min(a.z, b.z), max(a.w, b.w));
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        range_parts[wid.x] = wg_range[0];
    }
}

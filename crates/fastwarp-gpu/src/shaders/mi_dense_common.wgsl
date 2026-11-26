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
// (`ProcessPoint`: moving range test, moving Parzen window term and bin) and
// `itkMattesMutualInformationImageToImageMetricv4.hxx` (`ComputeSingleFixedImageParzenWindowIndex`, here as
// the count of the host's `f32` bin thresholds an intensity reaches). See
// `docs/THIRD_PARTY_PORTS.md`.

struct DenseMiU {
    dims: vec4<u32>,
    cfg: vec4<u32>,
    frng: vec4<f32>,
    mrng: vec4<f32>,
    o0: vec4<f32>,
    o1: vec4<f32>,
    o2: vec4<f32>,
    sp: vec4<f32>,
    fthr: array<vec4<f32>, 7>,
}

struct DensePoint {
    ok: bool,
    fbin: u32,
    term: f32,
    mbin: i32,
}

@group(0) @binding(0) var<storage, read> fix: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;

fn finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn fixed_bin(v: f32) -> u32 {
    var b = 2u;
    for (var k = 0u; k < 7u; k = k + 1u) {
        let c = select(vec4<u32>(0u), vec4<u32>(1u), vec4<f32>(v) >= u.fthr[k]);
        b = b + c.x + c.y + c.z + c.w;
    }
    return b;
}

fn dense_point(i: u32) -> DensePoint {
    var q: DensePoint;
    q.ok = false;
    q.fbin = 0u;
    q.term = 0.0;
    q.mbin = 0;
    if (!fixed_in(i) || !moving_in(i)) {
        return q;
    }
    let fv = fix[i];
    let mv = src[i];
    if (!finite(fv) || !finite(mv) || mv < u.mrng.x || mv > u.mrng.y) {
        return q;
    }
    let top = i32(u.cfg.x) - 3;
    let term = mv / u.mrng.z - u.mrng.w;
    q.ok = true;
    q.fbin = fixed_bin(fv);
    q.term = term;
    q.mbin = clamp(i32(term), 2, top);
    return q;
}

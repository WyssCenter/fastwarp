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
// (`ProcessPoint` and `ComputePDFDerivativesLocalSupportTransform` with the identity Jacobian of a
// displacement field transform), `itkMattesMutualInformationImageToImageMetricv4.hxx` (the
// local-support branch of `ComputeResults`: `m_PRatioArray` applied per point) and
// `itkCentralDifferenceImageFunction.hxx` (the moving-image gradient). See
// `docs/THIRD_PARTY_PORTS.md`.

@group(0) @binding(2) var<storage, read> ratio: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

fn probe_gradient(c: vec3<f32>, o: vec3<f32>, spacing: f32) -> f32 {
    let cp = c + o;
    let cm = c - o;
    if (inside(cp, u.dims.x, u.dims.y, u.dims.z) && inside(cm, u.dims.x, u.dims.y, u.dims.z)) {
        return (tri(cp, 0u, u.dims.x, u.dims.y, u.dims.z) - tri(cm, 0u, u.dims.x, u.dims.y, u.dims.z)) / spacing;
    }
    return 0.0;
}

@compute @workgroup_size(8, 8, 4)
fn k_mi_dense_deriv(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u.dims.x && id.y < u.dims.y && id.z < u.dims.z) {
        let i = id.x + u.dims.x * (id.y + u.dims.y * id.z);
        let q = dense_point(i);
        var d = vec3<f32>(0.0);
        if (q.ok) {
            var s = 0.0;
            for (var k = -1; k <= 2; k = k + 1) {
                let pm = q.mbin + k;
                s = s + db3(f32(pm) - q.term) * ratio[q.fbin * u.cfg.x + u32(pm)];
            }
            let c = vec3<f32>(id);
            let g = vec3<f32>(
                probe_gradient(c, u.o0.xyz, u.sp.x),
                probe_gradient(c, u.o1.xyz, u.sp.y),
                probe_gradient(c, u.o2.xyz, u.sp.z),
            );
            d = s * g;
        }
        dst[i] = d.x;
        dst[u.dims.w + i] = d.y;
        dst[2u * u.dims.w + i] = d.z;
    }
}

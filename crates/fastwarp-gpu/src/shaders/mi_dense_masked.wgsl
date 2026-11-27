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
// Portions derived from ITK `itkImageToImageMetricv4.hxx` (`TransformAndEvaluateFixedPoint`,
// `TransformAndEvaluateMovingPoint`: a point outside the fixed or the moving mask is skipped) and
// `itkMattesMutualInformationImageToImageMetricv4.hxx` (`Initialize`: each intensity range over the
// voxels inside its own mask). A mask flagged absent in `u.cfg.z` has every voxel inside. See
// `docs/THIRD_PARTY_PORTS.md`.

@group(0) @binding(BINDING_FIXED_MASK) var<storage, read> fixed_mask: array<f32>;
@group(0) @binding(BINDING_MOVING_MASK) var<storage, read> moving_mask: array<f32>;
@group(0) @binding(BINDING_UNIFORM) var<uniform> u: DenseMiU;

fn fixed_in(i: u32) -> bool {
    return (u.cfg.z & 1u) != 0u || fixed_mask[i] != 0.0;
}

fn moving_in(i: u32) -> bool {
    return (u.cfg.z & 2u) != 0u || moving_mask[i] != 0.0;
}

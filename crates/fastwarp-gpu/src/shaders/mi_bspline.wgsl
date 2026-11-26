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
// Portions derived from ITK `itkBSplineKernelFunction.h` and
// `itkBSplineDerivativeKernelFunction.h` (order 3). See `docs/THIRD_PARTY_PORTS.md`.

fn b3(u: f32) -> f32 {
    let a = abs(u);
    if (a < 1.0) {
        let s = a * a;
        return (4.0 - 6.0 * s + 3.0 * s * a) / 6.0;
    }
    if (a < 2.0) {
        let s = a * a;
        return (8.0 - 12.0 * a + 6.0 * s - s * a) / 6.0;
    }
    return 0.0;
}

fn db3(u: f32) -> f32 {
    if (u >= 0.0 && u < 1.0) {
        return -2.0 * u + 1.5 * u * u;
    }
    if (u > -1.0 && u < 0.0) {
        return -2.0 * u - 1.5 * u * u;
    }
    if (u >= 1.0 && u < 2.0) {
        return -2.0 + 2.0 * u - 0.5 * u * u;
    }
    if (u > -2.0 && u <= -1.0) {
        return 2.0 + 2.0 * u + 0.5 * u * u;
    }
    return 0.0;
}

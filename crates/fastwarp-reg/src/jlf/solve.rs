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
// Portions derived from ANTs `itkWeightedVotingFusionImageFilter.hxx` (the weight solve of
// `ThreadedGenerateDataForWeightedAveraging`: `vnl_cholesky` when its condition estimate exceeds
// `sqrteps`, else `vnl_svd::solve`, negative weights to 0; `NonNegativeLeastSquares`, Lawson-Hanson
// with `vnl_svd::pinverse`) and ITK `itkMath.h` (`FloatAlmostEqual`). The Cholesky factor follows
// LINPACK `dpofa` / `dposl`; the reciprocal condition number is the exact 1-norm value instead of
// the `dpoco` estimate; the SVD is a one-sided Jacobi SVD instead of LINPACK `dsvdc`; the inner
// NNLS loop is bounded. See `docs/THIRD_PARTY_PORTS.md`.

//! The atlas weights: the solve of `MxBar w = 1` and the non-negative least squares.

const SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

pub(crate) fn float_almost_equal(a: f64, b: f64) -> bool {
    if (a - b).abs() <= 0.1 * f64::EPSILON {
        return true;
    }
    if (a < 0.0) != (b < 0.0) {
        return false;
    }
    let ulps = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
    ulps <= 4
}

fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut r = a.to_vec();
    for j in 0..n {
        let mut s = 0.0f64;
        for k in 0..j {
            let mut dot = 0.0f64;
            for i in 0..k {
                dot += r[i * n + k] * r[i * n + j];
            }
            let t = (r[k * n + j] - dot) / r[k * n + k];
            r[k * n + j] = t;
            s += t * t;
        }
        s = r[j * n + j] - s;
        if s <= 0.0 || !s.is_finite() {
            return None;
        }
        r[j * n + j] = s.sqrt();
    }
    Some(r)
}

fn cholesky_solve(r: &[f64], n: usize, b: &mut [f64]) {
    for k in 0..n {
        let mut t = 0.0f64;
        for i in 0..k {
            t += r[i * n + k] * b[i];
        }
        b[k] = (b[k] - t) / r[k * n + k];
    }
    for k in (0..n).rev() {
        b[k] /= r[k * n + k];
        let t = -b[k];
        for i in 0..k {
            b[i] += t * r[i * n + k];
        }
    }
}

fn reciprocal_condition(a: &[f64], r: &[f64], n: usize) -> f64 {
    let column_sum = |col: &dyn Fn(usize) -> f64| (0..n).map(|i| col(i).abs()).sum::<f64>();
    let mut norm = 0.0f64;
    let mut inverse_norm = 0.0f64;
    for j in 0..n {
        norm = norm.max(column_sum(&|i| a[i * n + j]));
        let mut e = vec![0.0f64; n];
        e[j] = 1.0;
        cholesky_solve(r, n, &mut e);
        inverse_norm = inverse_norm.max(column_sum(&|i| e[i]));
    }
    let rcond = 1.0 / (norm * inverse_norm);
    if rcond.is_finite() {
        rcond
    } else {
        0.0
    }
}

fn svd_solve(a: &[f64], rows: usize, cols: usize, b: &[f64]) -> Vec<f64> {
    let mut u: Vec<Vec<f64>> = (0..cols)
        .map(|j| (0..rows).map(|i| a[i * cols + j]).collect())
        .collect();
    let mut v: Vec<Vec<f64>> = (0..cols)
        .map(|j| (0..cols).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _ in 0..100 {
        let mut rotated = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let alpha: f64 = u[p].iter().map(|x| x * x).sum();
                let beta: f64 = u[q].iter().map(|x| x * x).sum();
                let gamma: f64 = u[p].iter().zip(&u[q]).map(|(x, y)| x * y).sum();
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for m in [&mut u, &mut v] {
                    let (lo, hi) = m.split_at_mut(q);
                    for (x, y) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
                        let (xp, yq) = (*x, *y);
                        *x = c * xp - s * yq;
                        *y = s * xp + c * yq;
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut x = vec![0.0f64; cols];
    for j in 0..cols {
        let sigma2: f64 = u[j].iter().map(|x| x * x).sum();
        if sigma2 == 0.0 || !sigma2.is_finite() {
            continue;
        }
        let projection: f64 = u[j].iter().zip(b).map(|(x, y)| x * y).sum();
        let factor = projection / sigma2;
        for (xi, vi) in x.iter_mut().zip(&v[j]) {
            *xi += factor * vi;
        }
    }
    x
}

/// Lawson-Hanson non-negative least squares as ANTs' `NonNegativeLeastSquares` runs it: `a` is
/// `rows` x `cols`, row-major; at most `3 cols` outer iterations.
pub fn nonnegative_least_squares(
    a: &[f64],
    rows: usize,
    cols: usize,
    y: &[f64],
    tolerance: f64,
) -> Vec<f64> {
    let n = cols;
    let max_iterations = 3 * n;
    let mut p = vec![false; n];
    let mut x = vec![0.0f64; n];
    let mut s = vec![0.0f64; n];
    let gradient = |x: &[f64]| -> Vec<f64> {
        let residual: Vec<f64> = (0..rows)
            .map(|i| y[i] - (0..n).map(|j| a[i * cols + j] * x[j]).sum::<f64>())
            .collect();
        (0..n)
            .map(|j| {
                (0..rows)
                    .map(|i| a[i * cols + j] * residual[i])
                    .sum::<f64>()
            })
            .collect()
    };
    let pick = |w: &[f64], p: &[bool]| -> (Option<usize>, f64) {
        let mut index = None;
        let mut max = f64::MIN;
        for j in 0..n {
            if !p[j] && max < w[j] {
                index = Some(j);
                max = w[j];
            }
        }
        (index, max)
    };
    let passive_solve = |p: &[bool]| -> Vec<f64> {
        let columns: Vec<usize> = (0..n).filter(|&j| p[j]).collect();
        let k = columns.len();
        let mut sub = vec![0.0f64; rows * k];
        for i in 0..rows {
            for (c, &j) in columns.iter().enumerate() {
                sub[i * k + c] = a[i * cols + j];
            }
        }
        svd_solve(&sub, rows, k, y)
    };
    let spread = |p: &[bool], sp: &[f64], s: &mut [f64]| {
        let mut next = sp.iter();
        for j in 0..n {
            s[j] = if p[j] {
                *next.next().unwrap_or(&0.0)
            } else {
                0.0
            };
        }
    };
    let (mut index, mut w_max) = pick(&gradient(&x), &p);
    let mut iterations = 0;
    while p.iter().any(|v| !v) && w_max > tolerance && {
        let go = iterations < max_iterations;
        iterations += 1;
        go
    } {
        let Some(j) = index else {
            break;
        };
        p[j] = true;
        let mut sp = passive_solve(&p);
        spread(&p, &sp, &mut s);
        let mut inner = 0;
        while !sp.is_empty()
            && sp.iter().copied().fold(f64::INFINITY, f64::min) <= tolerance
            && inner < 10 * n + 10
        {
            inner += 1;
            let mut alpha = f64::MAX;
            for i in 0..n {
                if p[i] && s[i] <= tolerance {
                    let value = x[i] / (x[i] - s[i]);
                    if value < alpha {
                        alpha = value;
                    }
                }
            }
            for i in 0..n {
                x[i] += alpha * (s[i] - x[i]);
            }
            for i in 0..n {
                if p[i] && x[i].abs() < tolerance {
                    p[i] = false;
                }
            }
            if !p.iter().any(|v| *v) {
                break;
            }
            sp = passive_solve(&p);
            spread(&p, &sp, &mut s);
        }
        x.copy_from_slice(&s);
        (index, w_max) = pick(&gradient(&x), &p);
    }
    x
}

/// The normalised atlas weights for the regularised joint error matrix `mxbar` (`n` x `n`,
/// row-major): non-negative least squares of `mxbar w = 1` with `nonnegative`, else the Cholesky
/// solve (an SVD solve when the reciprocal condition number is at most `sqrt(eps)`) with negative
/// weights set to 0; then divided by their sum.
pub fn solve_weights(mxbar: &[f64], n: usize, nonnegative: bool) -> Vec<f64> {
    let ones = vec![1.0f64; n];
    let mut w = if nonnegative {
        nonnegative_least_squares(mxbar, n, n, &ones, 1e-6)
    } else {
        let mut w = match cholesky(mxbar, n) {
            Some(r) if reciprocal_condition(mxbar, &r, n) > SQRT_EPS => {
                let mut b = ones.clone();
                cholesky_solve(&r, n, &mut b);
                b
            }
            _ => svd_solve(mxbar, n, n, &ones),
        };
        for v in &mut w {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        w
    };
    let mut sum = 0.0f64;
    for v in &w {
        sum += v;
    }
    let scale = 1.0 / sum;
    for v in &mut w {
        *v *= scale;
    }
    w
}

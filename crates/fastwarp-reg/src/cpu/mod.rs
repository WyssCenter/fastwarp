// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! The `f64` CPU reference backend.
//!
//! Every kernel is data-parallel over voxels with rayon; reductions sum in a fixed chunk order
//! (`field_ops::ordered_sum`), so two runs produce bit-identical results whatever the number
//! of threads.

use rayon::prelude::*;

use crate::backend::{Backend, InversionPair, MetricPair};
use crate::error::{invalid, Result};
use crate::field_ops::{
    compose_displacement_fields, gaussian_smooth_displacement_field, invert_displacement_field,
    resample_field, InvertParams, InvertStats, VectorField,
};
use crate::grid::Grid;
use crate::interp::linear_at_point;
use crate::mask::{nearest_voxel, MaskVolume};
use crate::metric::cc::{CcInputs, CcMetric};
use crate::metric::mi::{MattesMi, MiDenseInputs};
use crate::transform::{AffineTransform, TransformChain};
use crate::volume::Volume;

/// The reference backend: images are `Volume<f64>`, fields are [`VectorField`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    #[doc(hidden)]
    pub fn upload_image(&self, volume: &Volume<f64>) -> Result<Volume<f64>> {
        Ok(volume.clone())
    }

    pub(crate) fn upload_image_owned(&self, volume: Volume<f64>) -> Result<Volume<f64>> {
        Ok(volume)
    }

    #[doc(hidden)]
    pub fn zero_field(&self, grid: &Grid) -> Result<VectorField> {
        Ok(VectorField::zeros(*grid))
    }

    #[doc(hidden)]
    pub fn upload_field(&self, field: &VectorField) -> Result<VectorField> {
        Ok(field.clone())
    }

    pub(crate) fn download_field(&self, field: &VectorField) -> Result<VectorField> {
        Ok(field.clone())
    }

    #[doc(hidden)]
    pub fn copy_field(&self, field: &VectorField) -> Result<VectorField> {
        Ok(field.clone())
    }

    #[doc(hidden)]
    pub fn warp_through_field(
        &self,
        source: &Volume<f64>,
        field: &VectorField,
        pre: Option<&VectorField>,
        affine: Option<&AffineTransform>,
    ) -> Result<Volume<f64>> {
        let grid = *field.grid();
        let data: Vec<f64> = field
            .data()
            .par_iter()
            .enumerate()
            .map(|(idx, u)| {
                let [i, j, k] = grid.ijk(idx);
                let x = grid.index_to_point([i as f64, j as f64, k as f64]);
                let mut p = [x[0] + u[0], x[1] + u[1], x[2] + u[2]];
                if let Some(d) = pre.and_then(|w| w.interpolate(p)) {
                    p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
                }
                if let Some(a) = affine {
                    p = a.apply(p);
                }
                linear_at_point(source, p).unwrap_or(0.0)
            })
            .collect();
        Volume::from_vec(grid, data)
    }

    #[doc(hidden)]
    pub fn warp_mask_through_field(
        &self,
        mask: &Volume<f64>,
        field: &VectorField,
        pre: Option<&VectorField>,
        affine: Option<&AffineTransform>,
        product: Option<Volume<f64>>,
    ) -> Result<Volume<f64>> {
        let grid = *field.grid();
        if let Some(p) = &product {
            if p.grid() != &grid {
                return Err(invalid(
                    "warp mask",
                    "the product image is not on the grid of the field",
                ));
            }
        }
        let mut data: Vec<f64> = field
            .data()
            .par_iter()
            .enumerate()
            .map(|(idx, u)| {
                let [i, j, k] = grid.ijk(idx);
                let x = grid.index_to_point([i as f64, j as f64, k as f64]);
                let mut p = [x[0] + u[0], x[1] + u[1], x[2] + u[2]];
                if let Some(d) = pre.and_then(|w| w.interpolate(p)) {
                    p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
                }
                if let Some(a) = affine {
                    p = a.apply(p);
                }
                match nearest_voxel(mask.grid(), p) {
                    Some(v) if mask.data()[v] != 0.0 => 1.0,
                    _ => 0.0,
                }
            })
            .collect();
        if let Some(p) = product {
            for (d, q) in data.iter_mut().zip(p.data()) {
                *d *= q;
            }
        }
        Volume::from_vec(grid, data)
    }

    #[doc(hidden)]
    pub fn cc_value_derivative_pair_masked(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        valid: &Volume<f64>,
        radius: usize,
    ) -> Result<MetricPair<VectorField>> {
        let identity = TransformChain::new();
        let grid = *fixed.grid();
        let metric = CcMetric::new(radius);
        let mut out = Vec::with_capacity(2);
        for (a, b) in [(fixed, moving), (moving, fixed)] {
            let e = metric.evaluate_masked(
                &CcInputs {
                    virtual_grid: &grid,
                    fixed: a,
                    moving: b,
                    fixed_transform: &identity,
                    moving_transform: &identity,
                },
                Some(valid),
                true,
            )?;
            out.push((e.value, VectorField::from_vec(grid, e.derivative)?));
        }
        let second = out.pop().ok_or_else(|| invalid("cc metric", "no result"))?;
        let first = out.pop().ok_or_else(|| invalid("cc metric", "no result"))?;
        Ok((first, second))
    }

    #[doc(hidden)]
    pub fn cc_value_derivative(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        radius: usize,
    ) -> Result<(f64, VectorField)> {
        let identity = TransformChain::new();
        let grid = *fixed.grid();
        let evaluation = CcMetric::new(radius).evaluate(
            &CcInputs {
                virtual_grid: &grid,
                fixed,
                moving,
                fixed_transform: &identity,
                moving_transform: &identity,
            },
            true,
        )?;
        Ok((
            evaluation.value,
            VectorField::from_vec(grid, evaluation.derivative)?,
        ))
    }

    #[doc(hidden)]
    pub fn mi_value_derivative_pair(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        bins: usize,
    ) -> Result<MetricPair<VectorField>> {
        mi_pair(fixed, moving, None, bins)
    }

    #[doc(hidden)]
    pub fn mi_value_derivative_pair_masked(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        fixed_mask: &Volume<f64>,
        moving_mask: &Volume<f64>,
        bins: usize,
    ) -> Result<MetricPair<VectorField>> {
        for (what, mask) in [("fixed", fixed_mask), ("moving", moving_mask)] {
            if mask.grid() != fixed.grid() {
                return Err(invalid(
                    "mattes",
                    format!("the {what} mask is not on the grid of the images"),
                ));
            }
        }
        let masks = (
            MaskVolume::from_volume(fixed_mask.clone()),
            MaskVolume::from_volume(moving_mask.clone()),
        );
        mi_pair(fixed, moving, Some((&masks.0, &masks.1)), bins)
    }

    pub(crate) fn mi_value_derivative_pair_with_norms(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        bins: usize,
    ) -> Result<(MetricPair<VectorField>, [f64; 2])> {
        let pair = self.mi_value_derivative_pair(fixed, moving, bins)?;
        let norms = [self.l2_norm(&pair.0 .1)?, self.l2_norm(&pair.1 .1)?];
        Ok((pair, norms))
    }

    pub(crate) fn mi_value_derivative_pair_masked_with_norms(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        fixed_mask: &Volume<f64>,
        moving_mask: &Volume<f64>,
        bins: usize,
    ) -> Result<(MetricPair<VectorField>, [f64; 2])> {
        let pair =
            self.mi_value_derivative_pair_masked(fixed, moving, fixed_mask, moving_mask, bins)?;
        let norms = [self.l2_norm(&pair.0 .1)?, self.l2_norm(&pair.1 .1)?];
        Ok((pair, norms))
    }

    #[doc(hidden)]
    pub fn smooth_field(&self, field: VectorField, variance: f64) -> Result<VectorField> {
        if variance <= 0.0 {
            return Ok(field);
        }
        Ok(gaussian_smooth_displacement_field(&field, variance))
    }

    #[doc(hidden)]
    pub fn max_scaled_norm(&self, field: &VectorField) -> Result<f64> {
        Ok(field.max_scaled_norm())
    }

    #[doc(hidden)]
    pub fn l2_norm(&self, field: &VectorField) -> Result<f64> {
        Ok(field.l2_norm())
    }

    #[doc(hidden)]
    pub fn scale(&self, field: &mut VectorField, s: f64) -> Result<()> {
        field.scale(s);
        Ok(())
    }

    #[doc(hidden)]
    pub fn axpy(&self, acc: &mut VectorField, a: f64, x: &VectorField) -> Result<()> {
        acc.axpy(a, x)
    }

    #[doc(hidden)]
    pub fn compose(
        &self,
        displacement: &VectorField,
        warping: &VectorField,
    ) -> Result<VectorField> {
        compose_displacement_fields(displacement, warping)
    }

    #[doc(hidden)]
    pub fn invert(
        &self,
        field: &VectorField,
        initial: VectorField,
        params: &InvertParams,
    ) -> Result<(VectorField, InvertStats)> {
        invert_displacement_field(field, Some(&initial), params)
    }

    #[doc(hidden)]
    pub fn resample_field(&self, field: &VectorField, grid: &Grid) -> Result<VectorField> {
        resample_field(field, grid)
    }

    #[doc(hidden)]
    pub fn cc_value_derivative_pair(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        radius: usize,
    ) -> Result<MetricPair<VectorField>> {
        let forward = self.cc_value_derivative(fixed, moving, radius)?;
        let backward = self.cc_value_derivative(moving, fixed, radius)?;
        Ok((forward, backward))
    }

    pub(crate) fn cc_value_derivative_pair_masked_with_norms(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        valid: &Volume<f64>,
        radius: usize,
    ) -> Result<(MetricPair<VectorField>, [f64; 2])> {
        let pair = self.cc_value_derivative_pair_masked(fixed, moving, valid, radius)?;
        let norms = [self.l2_norm(&pair.0 .1)?, self.l2_norm(&pair.1 .1)?];
        Ok((pair, norms))
    }

    pub(crate) fn cc_value_derivative_pair_with_norms(
        &self,
        fixed: &Volume<f64>,
        moving: &Volume<f64>,
        radius: usize,
    ) -> Result<(MetricPair<VectorField>, [f64; 2])> {
        let pair = self.cc_value_derivative_pair(fixed, moving, radius)?;
        let norms = [self.l2_norm(&pair.0 .1)?, self.l2_norm(&pair.1 .1)?];
        Ok((pair, norms))
    }

    pub(crate) fn max_scaled_norm_pair(
        &self,
        a: &VectorField,
        b: &VectorField,
    ) -> Result<(f64, f64)> {
        Ok((self.max_scaled_norm(a)?, self.max_scaled_norm(b)?))
    }

    pub(crate) fn invert_pair(
        &self,
        field: VectorField,
        inverse: VectorField,
        params: &InvertParams,
    ) -> Result<InversionPair<VectorField>> {
        let inverse = self.invert(&field, inverse, params)?;
        let field = self.invert(&inverse.0, field, params)?;
        Ok((inverse, field))
    }
}

fn mi_pair(
    fixed: &Volume<f64>,
    moving: &Volume<f64>,
    masks: Option<(&MaskVolume, &MaskVolume)>,
    bins: usize,
) -> Result<MetricPair<VectorField>> {
    let grid = *fixed.grid();
    if moving.grid() != &grid {
        return Err(invalid("mattes", "the images are not on one grid"));
    }
    let identity = TransformChain::new();
    let metric = MattesMi { bins };
    let run = |a: &Volume<f64>, b: &Volume<f64>, ma, mb| -> Result<(f64, VectorField)> {
        let e = metric.evaluate_dense(
            &MiDenseInputs {
                virtual_grid: &grid,
                fixed: a,
                moving: b,
                fixed_transform: &identity,
                moving_transform: &identity,
                fixed_mask: ma,
                moving_mask: mb,
            },
            true,
        )?;
        Ok((e.value, VectorField::from_vec(grid, e.derivative)?))
    };
    let (fm, mm) = masks.map_or((None, None), |(f, m)| (Some(f), Some(m)));
    let first = run(fixed, moving, fm, mm)?;
    let second = run(moving, fixed, mm, fm)?;
    Ok((first, second))
}

impl Backend for CpuBackend {
    type Image = Volume<f64>;
    type Field = VectorField;

    fn name(&self) -> String {
        "cpu-f64".to_string()
    }

    async fn upload_image(&self, volume: &Volume<f64>) -> Result<Self::Image> {
        CpuBackend::upload_image(self, volume)
    }

    async fn upload_image_owned(&self, volume: Volume<f64>) -> Result<Self::Image> {
        CpuBackend::upload_image_owned(self, volume)
    }

    async fn zero_field(&self, grid: &Grid) -> Result<Self::Field> {
        CpuBackend::zero_field(self, grid)
    }

    async fn upload_field(&self, field: &VectorField) -> Result<Self::Field> {
        CpuBackend::upload_field(self, field)
    }

    async fn download_field(&self, field: &Self::Field) -> Result<VectorField> {
        CpuBackend::download_field(self, field)
    }

    fn field_grid(&self, field: &Self::Field) -> Grid {
        *field.grid()
    }

    async fn copy_field(&self, field: &Self::Field) -> Result<Self::Field> {
        CpuBackend::copy_field(self, field)
    }

    async fn warp_through_field(
        &self,
        source: &Self::Image,
        field: &Self::Field,
        pre: Option<&Self::Field>,
        affine: Option<&AffineTransform>,
    ) -> Result<Self::Image> {
        CpuBackend::warp_through_field(self, source, field, pre, affine)
    }

    async fn warp_mask_through_field(
        &self,
        mask: &Self::Image,
        field: &Self::Field,
        pre: Option<&Self::Field>,
        affine: Option<&AffineTransform>,
        product: Option<Self::Image>,
    ) -> Result<Self::Image> {
        CpuBackend::warp_mask_through_field(self, mask, field, pre, affine, product)
    }

    async fn cc_value_derivative_pair_masked(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        valid: &Self::Image,
        radius: usize,
    ) -> Result<MetricPair<Self::Field>> {
        CpuBackend::cc_value_derivative_pair_masked(self, fixed, moving, valid, radius)
    }

    async fn cc_value_derivative(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<(f64, Self::Field)> {
        CpuBackend::cc_value_derivative(self, fixed, moving, radius)
    }

    async fn cc_value_derivative_pair(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<MetricPair<Self::Field>> {
        CpuBackend::cc_value_derivative_pair(self, fixed, moving, radius)
    }

    async fn cc_value_derivative_pair_masked_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        valid: &Self::Image,
        radius: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        CpuBackend::cc_value_derivative_pair_masked_with_norms(self, fixed, moving, valid, radius)
    }

    async fn cc_value_derivative_pair_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        CpuBackend::cc_value_derivative_pair_with_norms(self, fixed, moving, radius)
    }

    async fn mi_value_derivative_pair(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        bins: usize,
    ) -> Result<MetricPair<Self::Field>> {
        CpuBackend::mi_value_derivative_pair(self, fixed, moving, bins)
    }

    async fn mi_value_derivative_pair_masked(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        fixed_mask: &Self::Image,
        moving_mask: &Self::Image,
        bins: usize,
    ) -> Result<MetricPair<Self::Field>> {
        CpuBackend::mi_value_derivative_pair_masked(
            self,
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            bins,
        )
    }

    async fn mi_value_derivative_pair_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        bins: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        CpuBackend::mi_value_derivative_pair_with_norms(self, fixed, moving, bins)
    }

    async fn mi_value_derivative_pair_masked_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        fixed_mask: &Self::Image,
        moving_mask: &Self::Image,
        bins: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        CpuBackend::mi_value_derivative_pair_masked_with_norms(
            self,
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            bins,
        )
    }

    async fn smooth_field(&self, field: Self::Field, variance: f64) -> Result<Self::Field> {
        CpuBackend::smooth_field(self, field, variance)
    }

    async fn max_scaled_norm(&self, field: &Self::Field) -> Result<f64> {
        CpuBackend::max_scaled_norm(self, field)
    }

    async fn max_scaled_norm_pair(&self, a: &Self::Field, b: &Self::Field) -> Result<(f64, f64)> {
        CpuBackend::max_scaled_norm_pair(self, a, b)
    }

    async fn l2_norm(&self, field: &Self::Field) -> Result<f64> {
        CpuBackend::l2_norm(self, field)
    }

    async fn scale(&self, field: &mut Self::Field, s: f64) -> Result<()> {
        CpuBackend::scale(self, field, s)
    }

    async fn axpy(&self, acc: &mut Self::Field, a: f64, x: &Self::Field) -> Result<()> {
        CpuBackend::axpy(self, acc, a, x)
    }

    async fn compose(
        &self,
        displacement: &Self::Field,
        warping: &Self::Field,
    ) -> Result<Self::Field> {
        CpuBackend::compose(self, displacement, warping)
    }

    async fn invert(
        &self,
        field: &Self::Field,
        initial: Self::Field,
        params: &InvertParams,
    ) -> Result<(Self::Field, InvertStats)> {
        CpuBackend::invert(self, field, initial, params)
    }

    async fn invert_pair(
        &self,
        field: Self::Field,
        inverse: Self::Field,
        params: &InvertParams,
    ) -> Result<InversionPair<Self::Field>> {
        CpuBackend::invert_pair(self, field, inverse, params)
    }

    async fn resample_field(&self, field: &Self::Field, grid: &Grid) -> Result<Self::Field> {
        CpuBackend::resample_field(self, field, grid)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::linalg::IDENTITY3;

    fn grid() -> Grid {
        Grid::new([8, 7, 6], [1.5, 1.5, 2.0], [-3.0, 2.0, 1.0], IDENTITY3).unwrap()
    }

    fn ramp(g: Grid) -> Volume<f64> {
        let data = (0..g.num_voxels())
            .map(|idx| {
                let [i, j, k] = g.ijk(idx);
                2.0 * i as f64 + 5.0 * j as f64 - k as f64
            })
            .collect();
        Volume::from_vec(g, data).unwrap()
    }

    #[test]
    fn warp_through_a_zero_field_is_the_identity_and_translates_with_an_affine() {
        let b = CpuBackend;
        let g = grid();
        let v = ramp(g);
        let zero = b.zero_field(&g).unwrap();
        let same = b.warp_through_field(&v, &zero, None, None).unwrap();
        for (a, c) in same.data().iter().zip(v.data()) {
            assert!((a - c).abs() < 1e-12);
        }
        let shift = AffineTransform::from_translation([3.0, 0.0, 0.0]);
        let moved = b.warp_through_field(&v, &zero, None, Some(&shift)).unwrap();
        let at = |i, j, k| moved.data()[g.linear_index(i, j, k)];
        assert!((at(2, 3, 2) - (2.0 * 4.0 + 15.0 - 2.0)).abs() < 1e-12);
        assert_eq!(at(7, 3, 2), 0.0);
    }

    #[test]
    fn warp_applies_the_field_before_the_affine() {
        let b = CpuBackend;
        let g = grid();
        let v = ramp(g);
        let field = VectorField::from_fn(g, |_| [3.0, 0.0, 0.0]);
        let scale = AffineTransform::from_offset(
            [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let f = b.upload_field(&field).unwrap();
        let out = b.warp_through_field(&v, &f, None, Some(&scale)).unwrap();
        let x = g.index_to_point([1.0, 2.0, 2.0]);
        let q = [0.5 * (x[0] + 3.0), x[1], x[2]];
        let got = out.data()[g.linear_index(1, 2, 2)];
        match linear_at_point(&v, q) {
            Some(w) => assert!((got - w).abs() < 1e-12),
            None => assert_eq!(got, 0.0),
        }
        let swapped = linear_at_point(&v, [0.5 * x[0] + 3.0, x[1], x[2]]);
        assert_ne!(swapped, linear_at_point(&v, q));
    }

    #[test]
    fn warp_applies_an_earlier_field_between_the_field_and_the_affine() {
        let b = CpuBackend;
        let g = grid();
        let v = ramp(g);
        let field = VectorField::from_fn(g, |_| [1.5, 0.0, 0.0]);
        let pre_grid = Grid::new([4, 7, 6], [1.5, 1.5, 2.0], [-3.0, 2.0, 1.0], IDENTITY3).unwrap();
        let pre = VectorField::from_fn(pre_grid, |_| [0.0, 1.5, 0.0]);
        let scale = AffineTransform::from_offset(
            [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let out = b
            .warp_through_field(&v, &field, Some(&pre), Some(&scale))
            .unwrap();
        let ones = Volume::from_vec(g, vec![1.0; g.num_voxels()]).unwrap();
        let mask = b
            .warp_mask_through_field(&ones, &field, Some(&pre), Some(&scale), None)
            .unwrap();
        for (i, in_pre) in [(0usize, true), (6, false)] {
            let x = g.index_to_point([i as f64, 2.0, 2.0]);
            let q = [x[0] + 1.5, x[1], x[2]];
            assert_eq!(
                pre_grid.contains_continuous_index(pre_grid.point_to_index(q)),
                in_pre
            );
            let r = if in_pre { [q[0], q[1] + 1.5, q[2]] } else { q };
            let s = [0.5 * r[0], r[1], r[2]];
            let want = linear_at_point(&v, s).unwrap_or(0.0);
            assert!((out.data()[g.linear_index(i, 2, 2)] - want).abs() < 1e-12);
            let inside = nearest_voxel(&g, s).is_some();
            assert_eq!(
                mask.data()[g.linear_index(i, 2, 2)],
                f64::from(u8::from(inside))
            );
        }
    }

    #[test]
    fn cc_of_an_image_with_itself_is_minus_one_with_zero_derivative() {
        let b = CpuBackend;
        let g = Grid::new([9, 9, 9], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
        let data = (0..g.num_voxels())
            .map(|i| (((i * 2_654_435_761usize) % 997) as f64) / 997.0)
            .collect();
        let v = Volume::from_vec(g, data).unwrap();
        let (value, d) = b.cc_value_derivative(&v, &v, 2).unwrap();
        assert!((value + 1.0).abs() < 1e-12);
        assert!(d.data().iter().all(|c| c.iter().all(|x| x.abs() < 1e-9)));
    }

    #[test]
    fn the_default_cc_pair_is_two_calls_and_smoothing_by_value_keeps_a_zero_variance_field() {
        let b = CpuBackend;
        let g = Grid::new([9, 8, 7], [1.0; 3], [0.0; 3], IDENTITY3).unwrap();
        let image = |seed: usize| {
            let data = (0..g.num_voxels())
                .map(|i| (((i + seed) * 2_654_435_761usize) % 997) as f64 / 997.0)
                .collect();
            Volume::from_vec(g, data).unwrap()
        };
        let (a, c) = (image(1), image(5));
        let ((v1, d1), (v2, d2)) = b.cc_value_derivative_pair(&a, &c, 2).unwrap();
        let (w1, e1) = b.cc_value_derivative(&a, &c, 2).unwrap();
        let (w2, e2) = b.cc_value_derivative(&c, &a, 2).unwrap();
        assert_eq!((v1.to_bits(), v2.to_bits()), (w1.to_bits(), w2.to_bits()));
        assert_eq!(d1.data(), e1.data());
        assert_eq!(d2.data(), e2.data());
        let kept = b.smooth_field(d1.clone(), 0.0).unwrap();
        assert_eq!(kept.data(), d1.data());
        let init = VectorField::zeros(g);
        let (inv, _) = b.invert(&d1, init, &InvertParams::SYN).unwrap();
        assert_eq!(inv.grid(), &g);
    }

    #[test]
    fn name_and_basic_reductions() {
        let b = CpuBackend;
        assert_eq!(b.name(), "cpu-f64");
        let g = grid();
        let mut f = b
            .upload_field(&VectorField::from_fn(g, |_| [3.0, 0.0, 4.0]))
            .unwrap();
        assert!((b.l2_norm(&f).unwrap() - 5.0 * (g.num_voxels() as f64).sqrt()).abs() < 1e-9);
        b.scale(&mut f, 2.0).unwrap();
        let h = b.copy_field(&f).unwrap();
        b.axpy(&mut f, -1.0, &h).unwrap();
        assert_eq!(b.max_scaled_norm(&f).unwrap(), 0.0);
        assert_eq!(b.field_grid(&f), g);
        assert_eq!(b.download_field(&h).unwrap().data()[0], [6.0, 0.0, 8.0]);
    }
}

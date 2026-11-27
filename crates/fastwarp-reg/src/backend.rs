// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! The compute backend a registration loop runs on.
//!
//! The loop in `syn` is written once against [`Backend`]; the CPU implementation in `cpu` is the
//! `f64` reference, a GPU implementation provides the same operations on device buffers. Every
//! operation is a whole-volume kernel; scalars cross the boundary as `f64`.

use crate::error::Result;
use crate::field_ops::{InvertParams, InvertStats, VectorField};
use crate::grid::Grid;
use crate::transform::AffineTransform;
use crate::volume::Volume;

/// A metric value with its derivative field.
pub type MetricEvaluation<F> = (f64, F);

/// The two directions of one metric evaluation: `(fixed, moving)` then `(moving, fixed)`.
pub type MetricPair<F> = (MetricEvaluation<F>, MetricEvaluation<F>);

/// An inverted field with the statistics of its inversion.
pub type Inversion<F> = (F, InvertStats);

/// The two inversions of [`Backend::invert_pair`] in call order: the new inverse, then the new
/// field.
pub type InversionPair<F> = (Inversion<F>, Inversion<F>);

/// Operations the symmetric normalisation loop needs.
///
/// Scalar images ([`Backend::Image`]) and displacement fields ([`Backend::Field`]) are opaque
/// buffers owned by the backend. Fields always live on the grid they were created with.
///
/// Every operation on images and fields is `async`, so that a backend can wait for its device
/// without blocking (WebGPU in a browser). The futures are not required to be `Send`: the loops
/// drive them on the thread that calls them, and on a native backend every future is ready when
/// it is first polled.
#[allow(async_fn_in_trait)]
pub trait Backend: Sync {
    /// A scalar volume buffer.
    type Image: Send + Sync;
    /// A displacement-field buffer (three components per voxel).
    type Field: Send + Sync;

    /// A short name for records and logs.
    fn name(&self) -> String;

    /// Frees buffers the backend keeps for reuse but nothing holds, before a phase with other
    /// buffer sizes (the SyN stage after linear stages). No-op by default.
    fn release_idle_buffers(&self) {}

    /// Moves a volume onto the backend.
    async fn upload_image(&self, volume: &Volume<f64>) -> Result<Self::Image>;

    async fn upload_image_owned(&self, volume: Volume<f64>) -> Result<Self::Image> {
        self.upload_image(&volume).await
    }

    /// A zero displacement field on `grid`.
    async fn zero_field(&self, grid: &Grid) -> Result<Self::Field>;

    /// Moves a field onto the backend.
    async fn upload_field(&self, field: &VectorField) -> Result<Self::Field>;

    /// Brings a field back to the host.
    async fn download_field(&self, field: &Self::Field) -> Result<VectorField>;

    /// The grid of a field.
    fn field_grid(&self, field: &Self::Field) -> Grid;

    /// A copy of a field.
    async fn copy_field(&self, field: &Self::Field) -> Result<Self::Field>;

    /// Resamples `source` onto the grid of `field`: output voxel `x` takes the linearly
    /// interpolated value of `source` at `affine(q + pre(q))` with `q = x + field(x)` (the field
    /// first, then the optional earlier field `pre`, then the optional affine), and zero where
    /// that point lies outside `source`'s buffer. `pre` lives on its own grid: `pre(q)` is
    /// interpolated linearly there and is zero where `q` lies outside its buffer, as ITK's
    /// `DisplacementFieldTransform` maps a point. The image is interpolated once.
    async fn warp_through_field(
        &self,
        source: &Self::Image,
        field: &Self::Field,
        pre: Option<&Self::Field>,
        affine: Option<&AffineTransform>,
    ) -> Result<Self::Image>;

    /// Resamples a mask (an image of zeros and ones) onto the grid of `field` through the same
    /// map as [`Backend::warp_through_field`], with nearest-neighbour interpolation: output voxel
    /// `x` is 1 where the mask voxel nearest to `affine(q + pre(q))`, `q = x + field(x)`
    /// (rounding half up, as ITK's `NearestNeighborInterpolateImageFunction`) is not zero, and 0
    /// where it is zero or the point lies outside the mask's buffer. With `product` (an image on the grid of
    /// `field`), the result is multiplied into `product` voxel by voxel and returned in its
    /// buffer.
    async fn warp_mask_through_field(
        &self,
        mask: &Self::Image,
        field: &Self::Field,
        pre: Option<&Self::Field>,
        affine: Option<&AffineTransform>,
        product: Option<Self::Image>,
    ) -> Result<Self::Image>;

    /// ANTs neighbourhood cross-correlation of two images on the same grid: the metric value
    /// (`-mean(localCC)`) and the dense image-space derivative with respect to the moving image
    /// (`coefficient * grad(moving)` per voxel, not divided by the voxel count).
    async fn cc_value_derivative(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<(f64, Self::Field)>;

    /// Both directions of [`Backend::cc_value_derivative`] on one image pair: first
    /// `(fixed, moving)`, then `(moving, fixed)`. The default evaluates them one after the other;
    /// a backend that can share the symmetric window sums overrides it with bit-identical results.
    async fn cc_value_derivative_pair(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<MetricPair<Self::Field>> {
        let forward = self.cc_value_derivative(fixed, moving, radius).await?;
        let backward = self.cc_value_derivative(moving, fixed, radius).await?;
        Ok((forward, backward))
    }

    /// [`Backend::cc_value_derivative_pair`] restricted to the voxels where `valid` (an image of
    /// zeros and ones on the grid of the two images: the fixed and the moving mask resolved on
    /// it) is not zero, as ITK's metric treats points outside its masks: the window sums and
    /// their voxel count take only valid voxels, a voxel that is not valid has no value and a
    /// zero derivative, and each value is the mean over the valid voxels. Fails when no voxel is
    /// valid.
    async fn cc_value_derivative_pair_masked(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        valid: &Self::Image,
        radius: usize,
    ) -> Result<MetricPair<Self::Field>>;

    /// [`Backend::cc_value_derivative_pair_masked`] with the [`Backend::l2_norm`] of both
    /// derivatives, in the same order. The default calls them one after the other; a backend
    /// that can return all the scalars at once overrides it with bit-identical results.
    async fn cc_value_derivative_pair_masked_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        valid: &Self::Image,
        radius: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        let pair = self
            .cc_value_derivative_pair_masked(fixed, moving, valid, radius)
            .await?;
        let norms = [
            self.l2_norm(&pair.0 .1).await?,
            self.l2_norm(&pair.1 .1).await?,
        ];
        Ok((pair, norms))
    }

    /// [`Backend::cc_value_derivative_pair`] with the [`Backend::l2_norm`] of both derivatives,
    /// in the same order. The default calls them one after the other; a backend that can return
    /// all the scalars at once overrides it with bit-identical results.
    async fn cc_value_derivative_pair_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        let pair = self.cc_value_derivative_pair(fixed, moving, radius).await?;
        let norms = [
            self.l2_norm(&pair.0 .1).await?,
            self.l2_norm(&pair.1 .1).await?,
        ];
        Ok((pair, norms))
    }

    /// Mattes mutual information (`MI[fixed,moving,weight,bins]` with dense sampling) of two
    /// images on the same grid, in both directions: first `(fixed, moving)`, then
    /// `(moving, fixed)`. Each value is `-MI`; each derivative is ITK's local-support derivative of
    /// a displacement field transform, `sum_k B3'(m_k - term) * ratio[f, m_k] * grad(moving)` per
    /// voxel. Every call measures the intensity ranges of the histogram anew from the two images,
    /// as ITK's metric `Initialize` does before each evaluation.
    async fn mi_value_derivative_pair(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        bins: usize,
    ) -> Result<MetricPair<Self::Field>>;

    /// [`Backend::mi_value_derivative_pair`] with image masks resolved on the grid of the two
    /// images (zeros and ones; `None` for a mask that was not given, every voxel inside it): a
    /// voxel contributes only where both masks are not zero, the intensity range of each image is
    /// measured over the voxels inside its own mask, and a voxel that does not contribute has a
    /// zero derivative. The second direction swaps the masks with the images. Fails when no voxel
    /// is inside both masks.
    async fn mi_value_derivative_pair_masked(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        fixed_mask: Option<&Self::Image>,
        moving_mask: Option<&Self::Image>,
        bins: usize,
    ) -> Result<MetricPair<Self::Field>>;

    /// [`Backend::mi_value_derivative_pair`] with the [`Backend::l2_norm`] of both derivatives,
    /// in the same order. The default calls them one after the other; a backend that can return
    /// all the scalars at once overrides it with bit-identical results.
    async fn mi_value_derivative_pair_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        bins: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        let pair = self.mi_value_derivative_pair(fixed, moving, bins).await?;
        let norms = [
            self.l2_norm(&pair.0 .1).await?,
            self.l2_norm(&pair.1 .1).await?,
        ];
        Ok((pair, norms))
    }

    /// [`Backend::mi_value_derivative_pair_masked`] with the [`Backend::l2_norm`] of both
    /// derivatives, in the same order. The default calls them one after the other; a backend that
    /// can return all the scalars at once overrides it with bit-identical results.
    async fn mi_value_derivative_pair_masked_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        fixed_mask: Option<&Self::Image>,
        moving_mask: Option<&Self::Image>,
        bins: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        let pair = self
            .mi_value_derivative_pair_masked(fixed, moving, fixed_mask, moving_mask, bins)
            .await?;
        let norms = [
            self.l2_norm(&pair.0 .1).await?,
            self.l2_norm(&pair.1 .1).await?,
        ];
        Ok((pair, norms))
    }

    /// SyN's Gaussian smoothing of a field (`variance` in squared voxels of the field's grid,
    /// faces pinned to zero). Takes the field so that `variance <= 0` returns it unchanged
    /// without a copy.
    async fn smooth_field(&self, field: Self::Field, variance: f64) -> Result<Self::Field>;

    /// Largest `|u / spacing|` over the field.
    async fn max_scaled_norm(&self, field: &Self::Field) -> Result<f64>;

    /// [`Backend::max_scaled_norm`] of two fields, `(a, b)`. The default calls it twice; a
    /// backend that can return both values at once overrides it with bit-identical results.
    async fn max_scaled_norm_pair(&self, a: &Self::Field, b: &Self::Field) -> Result<(f64, f64)> {
        Ok((
            self.max_scaled_norm(a).await?,
            self.max_scaled_norm(b).await?,
        ))
    }

    /// Euclidean norm of the whole field as one vector.
    async fn l2_norm(&self, field: &Self::Field) -> Result<f64>;

    /// `field *= s`.
    async fn scale(&self, field: &mut Self::Field, s: f64) -> Result<()>;

    /// `acc += a * x`.
    async fn axpy(&self, acc: &mut Self::Field, a: f64, x: &Self::Field) -> Result<()>;

    /// `out(x) = warping(x) + displacement(x + warping(x))`, zero displacement outside the
    /// buffer (ITK `ComposeDisplacementFieldsImageFilter`).
    async fn compose(
        &self,
        displacement: &Self::Field,
        warping: &Self::Field,
    ) -> Result<Self::Field>;

    /// ITK's fixed-point inverse of `field` starting from `initial`; `initial`'s buffer is
    /// updated in place and returned as the inverse.
    async fn invert(
        &self,
        field: &Self::Field,
        initial: Self::Field,
        params: &InvertParams,
    ) -> Result<(Self::Field, InvertStats)>;

    /// The two inversions with which SyN re-estimates a field and its inverse:
    /// `inverse = invert(field, initial = inverse)`, then `field = invert(inverse, initial =
    /// field)`. The default calls [`Backend::invert`] twice; a backend that can return both
    /// statistics at once overrides it with bit-identical results.
    async fn invert_pair(
        &self,
        field: Self::Field,
        inverse: Self::Field,
        params: &InvertParams,
    ) -> Result<InversionPair<Self::Field>> {
        let inverse = self.invert(&field, inverse, params).await?;
        let field = self.invert(&inverse.0, field, params).await?;
        Ok((inverse, field))
    }

    /// The field resampled onto `grid` (linear, zero outside the old buffer).
    async fn resample_field(&self, field: &Self::Field, grid: &Grid) -> Result<Self::Field>;
}

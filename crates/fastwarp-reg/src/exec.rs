// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! Running the `async` registration code on a native backend without an executor.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use crate::error::{invalid, Result};

/// The output of `future`, which must be ready when it is first polled (every future of a native
/// backend is); a future that is not ready is an error.
pub(crate) fn block_ready<T>(future: impl Future<Output = Result<T>>) -> Result<T> {
    let mut future = pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(output) => output,
        Poll::Pending => Err(invalid(
            "backend",
            "the backend's operation did not complete when first polled; the synchronous entry points \
             need a backend whose futures are ready at once (use the async forms: run_syn_async, \
             run_syn_from_async, run_syn_cached_async, register_async, register_full_async or \
             stages::register_stages_async)",
        )),
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::future::Future;
    use std::pin::{pin, Pin};
    use std::task::{Context, Poll, Waker};

    use crate::backend::{Backend, MetricPair};
    use crate::cpu::CpuBackend;
    use crate::error::Result;
    use crate::field_ops::{InvertParams, InvertStats, VectorField};
    use crate::grid::Grid;
    use crate::linear::{
        CpuLevel, CpuLinearEngine, LinearEngine, LinearStageResult, PreparedSamples,
    };
    use crate::optimizer::LinearMetric;
    use crate::transform::{AffineTransform, LinearMovingTransform};
    use crate::volume::Volume;

    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                return Poll::Ready(());
            }
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    fn yield_once() -> YieldOnce {
        YieldOnce(false)
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct YieldingCpu;

    impl Backend for YieldingCpu {
        type Image = Volume<f64>;
        type Field = VectorField;

        fn name(&self) -> String {
            "cpu-f64 (yielding)".to_string()
        }

        fn field_grid(&self, field: &Self::Field) -> Grid {
            *field.grid()
        }

        async fn upload_image(&self, volume: &Volume<f64>) -> Result<Self::Image> {
            yield_once().await;
            Backend::upload_image(&CpuBackend, volume).await
        }

        async fn zero_field(&self, grid: &Grid) -> Result<Self::Field> {
            yield_once().await;
            Backend::zero_field(&CpuBackend, grid).await
        }

        async fn upload_field(&self, field: &VectorField) -> Result<Self::Field> {
            yield_once().await;
            Backend::upload_field(&CpuBackend, field).await
        }

        async fn download_field(&self, field: &Self::Field) -> Result<VectorField> {
            yield_once().await;
            Backend::download_field(&CpuBackend, field).await
        }

        async fn download_displacement_field(
            &self,
            field: &Self::Field,
        ) -> Result<crate::transform::DisplacementField> {
            yield_once().await;
            Backend::download_displacement_field(&CpuBackend, field).await
        }

        async fn copy_field(&self, field: &Self::Field) -> Result<Self::Field> {
            yield_once().await;
            Backend::copy_field(&CpuBackend, field).await
        }

        async fn warp_through_field(
            &self,
            source: &Self::Image,
            field: &Self::Field,
            pre: Option<&Self::Field>,
            affine: Option<&AffineTransform>,
        ) -> Result<Self::Image> {
            yield_once().await;
            Backend::warp_through_field(&CpuBackend, source, field, pre, affine).await
        }

        async fn warp_mask_through_field(
            &self,
            mask: &Self::Image,
            field: &Self::Field,
            pre: Option<&Self::Field>,
            affine: Option<&AffineTransform>,
            product: Option<Self::Image>,
        ) -> Result<Self::Image> {
            yield_once().await;
            Backend::warp_mask_through_field(&CpuBackend, mask, field, pre, affine, product).await
        }

        async fn cc_value_derivative(
            &self,
            fixed: &Self::Image,
            moving: &Self::Image,
            radius: usize,
        ) -> Result<(f64, Self::Field)> {
            yield_once().await;
            Backend::cc_value_derivative(&CpuBackend, fixed, moving, radius).await
        }

        async fn cc_value_derivative_pair_masked(
            &self,
            fixed: &Self::Image,
            moving: &Self::Image,
            valid: &Self::Image,
            radius: usize,
        ) -> Result<MetricPair<Self::Field>> {
            yield_once().await;
            Backend::cc_value_derivative_pair_masked(&CpuBackend, fixed, moving, valid, radius)
                .await
        }

        async fn mi_value_derivative_pair(
            &self,
            fixed: &Self::Image,
            moving: &Self::Image,
            bins: usize,
        ) -> Result<MetricPair<Self::Field>> {
            yield_once().await;
            Backend::mi_value_derivative_pair(&CpuBackend, fixed, moving, bins).await
        }

        async fn mi_value_derivative_pair_masked(
            &self,
            fixed: &Self::Image,
            moving: &Self::Image,
            fixed_mask: Option<&Self::Image>,
            moving_mask: Option<&Self::Image>,
            bins: usize,
        ) -> Result<MetricPair<Self::Field>> {
            yield_once().await;
            Backend::mi_value_derivative_pair_masked(
                &CpuBackend,
                fixed,
                moving,
                fixed_mask,
                moving_mask,
                bins,
            )
            .await
        }

        async fn smooth_field(&self, field: Self::Field, variance: f64) -> Result<Self::Field> {
            yield_once().await;
            Backend::smooth_field(&CpuBackend, field, variance).await
        }

        async fn max_scaled_norm(&self, field: &Self::Field) -> Result<f64> {
            yield_once().await;
            Backend::max_scaled_norm(&CpuBackend, field).await
        }

        async fn l2_norm(&self, field: &Self::Field) -> Result<f64> {
            yield_once().await;
            Backend::l2_norm(&CpuBackend, field).await
        }

        async fn scale(&self, field: &mut Self::Field, s: f64) -> Result<()> {
            yield_once().await;
            Backend::scale(&CpuBackend, field, s).await
        }

        async fn axpy(&self, acc: &mut Self::Field, a: f64, x: &Self::Field) -> Result<()> {
            yield_once().await;
            Backend::axpy(&CpuBackend, acc, a, x).await
        }

        async fn compose(
            &self,
            displacement: &Self::Field,
            warping: &Self::Field,
        ) -> Result<Self::Field> {
            yield_once().await;
            Backend::compose(&CpuBackend, displacement, warping).await
        }

        async fn invert(
            &self,
            field: &Self::Field,
            initial: Self::Field,
            params: &InvertParams,
        ) -> Result<(Self::Field, InvertStats)> {
            yield_once().await;
            Backend::invert(&CpuBackend, field, initial, params).await
        }

        async fn resample_field(&self, field: &Self::Field, grid: &Grid) -> Result<Self::Field> {
            yield_once().await;
            Backend::resample_field(&CpuBackend, field, grid).await
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct YieldingLinear;

    pub(crate) struct YieldingLevel<'a>(CpuLevel<'a>);

    impl LinearMetric for YieldingLevel<'_> {
        async fn value(&self, t: &LinearMovingTransform) -> Result<f64> {
            yield_once().await;
            self.0.value(t).await
        }

        async fn value_and_derivative(&self, t: &LinearMovingTransform) -> Result<(f64, Vec<f64>)> {
            yield_once().await;
            self.0.value_and_derivative(t).await
        }
    }

    impl LinearEngine for YieldingLinear {
        type Level<'a> = YieldingLevel<'a>;

        fn name(&self) -> String {
            "cpu-f64 (yielding)".to_string()
        }

        async fn prepare_level<'a>(
            &'a self,
            fixed: Volume<f64>,
            moving: Volume<f64>,
            samples: PreparedSamples,
        ) -> Result<YieldingLevel<'a>> {
            yield_once().await;
            let level =
                LinearEngine::prepare_level(&CpuLinearEngine, fixed, moving, samples).await?;
            Ok(YieldingLevel(level))
        }

        async fn prepare_level_borrowed<'a>(
            &'a self,
            fixed: &'a Volume<f64>,
            moving: &'a Volume<f64>,
            samples: PreparedSamples,
        ) -> Result<YieldingLevel<'a>> {
            yield_once().await;
            let level =
                LinearEngine::prepare_level_borrowed(&CpuLinearEngine, fixed, moving, samples)
                    .await?;
            Ok(YieldingLevel(level))
        }
    }

    pub(crate) fn linear_bits(stage: &LinearStageResult) -> String {
        let levels: Vec<String> = stage
            .levels
            .iter()
            .map(|l| {
                format!(
                    "{} {:?} {:?} {} {:?} {:?} {:?}",
                    l.level,
                    l.shrink_factors,
                    l.grid_size,
                    l.samples,
                    l.cg,
                    l.parameters,
                    l.skipped
                )
            })
            .collect();
        format!("{:?} {:?} {levels:?}", stage.transform, stage.affine)
    }

    pub(crate) fn drive<T>(future: impl Future<Output = T>) -> (T, usize) {
        let mut future = pin!(future);
        let mut cx = Context::from_waker(Waker::noop());
        for pending in 0..1usize << 24 {
            if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                return (output, pending);
            }
        }
        panic!("the future did not complete within the poll bound");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::error::RegError;

    #[test]
    fn a_ready_future_gives_its_output_and_a_pending_one_an_error() {
        assert_eq!(block_ready(async { Ok(7) }).unwrap(), 7);
        let e = block_ready(async {
            std::future::pending::<()>().await;
            Ok(())
        })
        .unwrap_err();
        assert!(
            matches!(
                e,
                RegError::Invalid {
                    what: "backend",
                    ..
                }
            ),
            "{e}"
        );
    }
}

// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! Progress of a registration without hooks in the engine: [`Observed`] passes every call on
//! to the backend and the linear engine it wraps, unchanged, and counts the linear levels it
//! prepares, the metric derivatives it evaluates (one per optimiser iteration) and the SyN
//! metric evaluations (one per SyN iteration and channel).

use std::sync::Mutex;

use fastwarp_core::Stopwatch;
use fastwarp_reg::backend::{InversionPair, MetricPair};
use fastwarp_reg::field_ops::{InvertParams, InvertStats, VectorField};
use fastwarp_reg::linear::{LinearEngine, LinearKind, PreparedSamples};
use fastwarp_reg::optimizer::LinearMetric;
use fastwarp_reg::stages::StagePlan;
use fastwarp_reg::transform::{AffineTransform, LinearMovingTransform};
use fastwarp_reg::{Backend, Grid, Result, Volume};
use serde_json::{json, Value};

const REPORT_INTERVAL_S: f64 = 0.25;

#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    pub phase: &'static str,
    pub level: usize,
    pub levels: usize,
    pub iteration: usize,
    pub iterations: usize,
    pub grid: Option<[usize; 3]>,
}

impl Progress {
    pub fn phase(phase: &'static str) -> Self {
        Self {
            phase,
            level: 0,
            levels: 0,
            iteration: 0,
            iterations: 0,
            grid: None,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "phase": self.phase,
            "level": self.level,
            "levels": self.levels,
            "iteration": self.iteration,
            "iterations": self.iterations,
            "grid": self.grid,
        })
    }
}

#[derive(Default)]
struct State {
    linear_stage: usize,
    linear_level: usize,
    linear_iteration: usize,
    syn_levels: usize,
    syn_grid: Option<[usize; 3]>,
    syn_calls: usize,
    last: Option<Stopwatch>,
}

struct LinearStage {
    name: &'static str,
    parameters: usize,
    iterations: Vec<usize>,
}

pub struct Tracker<'a> {
    linear: Vec<LinearStage>,
    syn: Vec<usize>,
    channels: usize,
    state: Mutex<State>,
    report: &'a (dyn Fn(&Progress) + Sync),
}

impl<'a> Tracker<'a> {
    pub fn new(plan: &StagePlan, report: &'a (dyn Fn(&Progress) + Sync)) -> Self {
        let linear = plan
            .linear
            .iter()
            .map(|stage| {
                let (name, parameters) = match stage.kind {
                    LinearKind::Rigid => ("rigid", 6),
                    LinearKind::Affine => ("affine", 12),
                };
                let iterations = (0..stage.shrink_factors.len())
                    .map(|l| stage.iterations.get(l).copied().unwrap_or(0) as usize)
                    .collect();
                LinearStage {
                    name,
                    parameters,
                    iterations,
                }
            })
            .collect();
        let syn = plan
            .syn
            .iter()
            .flat_map(|s| {
                (0..s.shrink_factors.len())
                    .map(|l| s.iterations.get(l).copied().unwrap_or(0) as usize)
            })
            .collect();
        let channels = plan
            .syn
            .first()
            .map(|s| s.channel_weights.len().max(1))
            .unwrap_or(1);
        Self {
            linear,
            syn,
            channels,
            state: Mutex::new(State::default()),
            report,
        }
    }

    pub fn phase(&self, phase: &'static str) {
        self.emit(&Progress::phase(phase), true);
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn emit(&self, p: &Progress, force: bool) {
        {
            let mut s = self.state();
            let due = s
                .last
                .as_ref()
                .is_none_or(|t| t.elapsed().as_secs_f64() >= REPORT_INTERVAL_S);
            if !(force || due) {
                return;
            }
            s.last = Some(Stopwatch::start());
        }
        (self.report)(p);
    }

    fn linear_progress(&self, s: &State) -> Progress {
        match self.linear.get(s.linear_stage) {
            Some(stage) => Progress {
                phase: stage.name,
                level: s.linear_level,
                levels: stage.iterations.len(),
                iteration: s.linear_iteration,
                iterations: stage
                    .iterations
                    .get(s.linear_level.saturating_sub(1))
                    .copied()
                    .unwrap_or(0),
                grid: None,
            },
            None => Progress::phase("linear"),
        }
    }

    fn linear_level(&self) {
        let p = {
            let mut s = self.state();
            s.linear_level += 1;
            let levels = self
                .linear
                .get(s.linear_stage)
                .map_or(0, |stage| stage.iterations.len());
            if s.linear_level > levels && s.linear_stage + 1 < self.linear.len() {
                s.linear_stage += 1;
                s.linear_level = 1;
            }
            s.linear_iteration = 0;
            self.linear_progress(&s)
        };
        self.emit(&p, true);
    }

    fn linear_iteration(&self, parameters: usize) {
        let (p, moved) = {
            let mut s = self.state();
            let current = self
                .linear
                .get(s.linear_stage)
                .map(|stage| stage.parameters);
            let mut moved = false;
            if current.is_some_and(|c| c != parameters) {
                let next = (s.linear_stage + 1..self.linear.len())
                    .find(|&i| self.linear[i].parameters == parameters);
                if let Some(i) = next {
                    s.linear_stage = i;
                    s.linear_level = 1;
                    moved = true;
                }
            }
            s.linear_iteration += 1;
            (self.linear_progress(&s), moved)
        };
        self.emit(&p, moved);
    }

    fn syn_evaluation(&self, grid: [usize; 3]) {
        let (p, new_level) = {
            let mut s = self.state();
            let new_level = s.syn_grid != Some(grid) || s.syn_levels == 0;
            if new_level {
                s.syn_levels += 1;
                s.syn_grid = Some(grid);
                s.syn_calls = 0;
            }
            s.syn_calls += 1;
            let level = s.syn_levels;
            let p = Progress {
                phase: "syn",
                level,
                levels: self.syn.len(),
                iteration: s.syn_calls.div_ceil(self.channels),
                iterations: self.syn.get(level - 1).copied().unwrap_or(0),
                grid: Some(grid),
            };
            (p, new_level)
        };
        self.emit(&p, new_level);
    }
}

/// A backend and a linear engine that report their progress to a [`Tracker`].
pub struct Observed<'a, B, E> {
    backend: &'a B,
    engine: &'a E,
    tracker: &'a Tracker<'a>,
}

impl<'a, B, E> Observed<'a, B, E> {
    pub fn new(backend: &'a B, engine: &'a E, tracker: &'a Tracker<'a>) -> Self {
        Self {
            backend,
            engine,
            tracker,
        }
    }
}

pub struct ObservedLevel<'a, L> {
    inner: L,
    tracker: &'a Tracker<'a>,
}

impl<L: LinearMetric> LinearMetric for ObservedLevel<'_, L> {
    async fn value(&self, t: &LinearMovingTransform) -> Result<f64> {
        self.inner.value(t).await
    }

    async fn value_and_derivative(&self, t: &LinearMovingTransform) -> Result<(f64, Vec<f64>)> {
        self.tracker
            .linear_iteration(t.optimised.number_of_parameters());
        self.inner.value_and_derivative(t).await
    }
}

impl<B: Sync, E: LinearEngine> LinearEngine for Observed<'_, B, E> {
    type Level<'b>
        = ObservedLevel<'b, E::Level<'b>>
    where
        Self: 'b;

    fn name(&self) -> String {
        self.engine.name()
    }

    async fn prepare_level<'b>(
        &'b self,
        fixed: Volume<f64>,
        moving: Volume<f64>,
        samples: PreparedSamples,
    ) -> Result<Self::Level<'b>> {
        self.tracker.linear_level();
        let inner = self.engine.prepare_level(fixed, moving, samples).await?;
        Ok(ObservedLevel {
            inner,
            tracker: self.tracker,
        })
    }

    async fn prepare_level_borrowed<'b>(
        &'b self,
        fixed: &'b Volume<f64>,
        moving: &'b Volume<f64>,
        samples: PreparedSamples,
    ) -> Result<Self::Level<'b>> {
        self.tracker.linear_level();
        let inner = self
            .engine
            .prepare_level_borrowed(fixed, moving, samples)
            .await?;
        Ok(ObservedLevel {
            inner,
            tracker: self.tracker,
        })
    }
}

impl<B: Backend, E: Sync> Observed<'_, B, E> {
    fn seen(&self, out: Result<MetricPair<B::Field>>) -> Result<MetricPair<B::Field>> {
        if let Ok(pair) = &out {
            self.tracker
                .syn_evaluation(self.backend.field_grid(&pair.0 .1).size());
        }
        out
    }
}

impl<B: Backend, E: Sync> Backend for Observed<'_, B, E> {
    type Image = B::Image;
    type Field = B::Field;

    fn name(&self) -> String {
        self.backend.name()
    }

    fn release_idle_buffers(&self) {
        self.backend.release_idle_buffers();
    }

    async fn upload_image(&self, volume: &Volume<f64>) -> Result<Self::Image> {
        self.backend.upload_image(volume).await
    }

    async fn upload_image_owned(&self, volume: Volume<f64>) -> Result<Self::Image> {
        self.backend.upload_image_owned(volume).await
    }

    async fn zero_field(&self, grid: &Grid) -> Result<Self::Field> {
        self.backend.zero_field(grid).await
    }

    async fn upload_field(&self, field: &VectorField) -> Result<Self::Field> {
        self.backend.upload_field(field).await
    }

    async fn download_field(&self, field: &Self::Field) -> Result<VectorField> {
        self.backend.download_field(field).await
    }

    fn field_grid(&self, field: &Self::Field) -> Grid {
        self.backend.field_grid(field)
    }

    async fn copy_field(&self, field: &Self::Field) -> Result<Self::Field> {
        self.backend.copy_field(field).await
    }

    async fn warp_through_field(
        &self,
        source: &Self::Image,
        field: &Self::Field,
        pre: Option<&Self::Field>,
        affine: Option<&AffineTransform>,
    ) -> Result<Self::Image> {
        self.backend
            .warp_through_field(source, field, pre, affine)
            .await
    }

    async fn warp_mask_through_field(
        &self,
        mask: &Self::Image,
        field: &Self::Field,
        pre: Option<&Self::Field>,
        affine: Option<&AffineTransform>,
        product: Option<Self::Image>,
    ) -> Result<Self::Image> {
        self.backend
            .warp_mask_through_field(mask, field, pre, affine, product)
            .await
    }

    async fn cc_value_derivative(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<(f64, Self::Field)> {
        self.backend
            .cc_value_derivative(fixed, moving, radius)
            .await
    }

    async fn cc_value_derivative_pair(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<MetricPair<Self::Field>> {
        let out = self
            .backend
            .cc_value_derivative_pair(fixed, moving, radius)
            .await;
        self.seen(out)
    }

    async fn cc_value_derivative_pair_masked(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        valid: &Self::Image,
        radius: usize,
    ) -> Result<MetricPair<Self::Field>> {
        let out = self
            .backend
            .cc_value_derivative_pair_masked(fixed, moving, valid, radius)
            .await;
        self.seen(out)
    }

    async fn cc_value_derivative_pair_masked_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        valid: &Self::Image,
        radius: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        let (pair, norms) = self
            .backend
            .cc_value_derivative_pair_masked_with_norms(fixed, moving, valid, radius)
            .await?;
        Ok((self.seen(Ok(pair))?, norms))
    }

    async fn cc_value_derivative_pair_with_norms(
        &self,
        fixed: &Self::Image,
        moving: &Self::Image,
        radius: usize,
    ) -> Result<(MetricPair<Self::Field>, [f64; 2])> {
        let (pair, norms) = self
            .backend
            .cc_value_derivative_pair_with_norms(fixed, moving, radius)
            .await?;
        Ok((self.seen(Ok(pair))?, norms))
    }

    async fn smooth_field(&self, field: Self::Field, variance: f64) -> Result<Self::Field> {
        self.backend.smooth_field(field, variance).await
    }

    async fn max_scaled_norm(&self, field: &Self::Field) -> Result<f64> {
        self.backend.max_scaled_norm(field).await
    }

    async fn max_scaled_norm_pair(&self, a: &Self::Field, b: &Self::Field) -> Result<(f64, f64)> {
        self.backend.max_scaled_norm_pair(a, b).await
    }

    async fn l2_norm(&self, field: &Self::Field) -> Result<f64> {
        self.backend.l2_norm(field).await
    }

    async fn scale(&self, field: &mut Self::Field, s: f64) -> Result<()> {
        self.backend.scale(field, s).await
    }

    async fn axpy(&self, acc: &mut Self::Field, a: f64, x: &Self::Field) -> Result<()> {
        self.backend.axpy(acc, a, x).await
    }

    async fn compose(
        &self,
        displacement: &Self::Field,
        warping: &Self::Field,
    ) -> Result<Self::Field> {
        self.backend.compose(displacement, warping).await
    }

    async fn invert(
        &self,
        field: &Self::Field,
        initial: Self::Field,
        params: &InvertParams,
    ) -> Result<(Self::Field, InvertStats)> {
        self.backend.invert(field, initial, params).await
    }

    async fn invert_pair(
        &self,
        field: Self::Field,
        inverse: Self::Field,
        params: &InvertParams,
    ) -> Result<InversionPair<Self::Field>> {
        self.backend.invert_pair(field, inverse, params).await
    }

    async fn resample_field(&self, field: &Self::Field, grid: &Grid) -> Result<Self::Field> {
        self.backend.resample_field(field, grid).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_skipped_level_does_not_shift_the_later_stages() {
        let seen: Mutex<Vec<Progress>> = Mutex::new(Vec::new());
        let report = |p: &Progress| seen.lock().unwrap().push(p.clone());
        let plan = crate::run::default_plan();
        let t = Tracker::new(&plan, &report);
        for _ in 0..3 {
            t.linear_level();
            t.linear_iteration(6);
        }
        t.linear_level();
        t.linear_iteration(12);
        t.linear_level();
        t.linear_iteration(12);
        let now = t.linear_progress(&t.state());
        assert_eq!(
            (
                now.phase,
                now.level,
                now.levels,
                now.iteration,
                now.iterations
            ),
            ("affine", 2, 4, 1, 500)
        );
        drop(t);
        let seen = seen.into_inner().unwrap();
        assert!(seen
            .iter()
            .any(|p| p.phase == "affine" && p.level == 1 && p.iteration == 1));
        assert_eq!(seen[0].phase, "rigid");
        assert_eq!((seen[0].level, seen[0].iterations), (1, 1000));
    }
}

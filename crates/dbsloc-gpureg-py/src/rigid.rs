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
// The rigid fit is dbsloc-reg's `run_rigid_registration_from_cancellable` (itself after ANTs'
// `--initial-moving-transform <start> --transform Rigid[...]`: the start in the moving initial
// composite, a fresh Euler3DTransform optimised on top, both collapsed into one affine), with its
// smoothing cache kept across calls instead of made per call. See `THIRD_PARTY_PORTS.md`.

//! Several rigid fits of one image pair on one device: the images are converted, winsorized and
//! uploaded once, and their smoothed copies are reused by every fit that asks for the same
//! sigma -- the shape of a multi-start, where five coarse fits and two full ones register the
//! same CT to the same anchor.

use std::sync::atomic::AtomicBool;

use dbsloc_gpu::WgpuBackend;
use dbsloc_reg::linear::{run_linear_stage_cached, with_center, LinearRecipe, SmoothCache};
use dbsloc_reg::mask::MaskVolume;
use dbsloc_reg::transform::{AffineTransform, ParametricTransform};
use dbsloc_reg::{Result, Volume};

/// The prepared pair and everything reused between fits.
pub struct Session {
    pub gpu: WgpuBackend,
    pub fixed: Volume<f64>,
    pub moving: Volume<f64>,
    pub mask: Option<MaskVolume>,
    pub cache: SmoothCache,
}

/// One fit's result: the collapsed affine (start then rigid) and the final MI value.
pub struct Fit {
    pub total: AffineTransform,
    pub metric: Option<f64>,
    pub iterations: Vec<u32>,
}

impl Session {
    /// One `Rigid` stage from `initial`, masked by the session's mask when `use_mask`.
    pub fn fit(
        &mut self,
        recipe: &LinearRecipe,
        initial: AffineTransform,
        use_mask: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<Fit> {
        let mut recipe = recipe.clone();
        recipe.mask = if use_mask { self.mask.clone() } else { None };
        let rigid = run_linear_stage_cached(
            &self.gpu,
            &recipe,
            &self.fixed,
            &self.moving,
            initial,
            ParametricTransform::euler3d([0.0; 6], initial.center),
            &mut self.cache,
            cancel,
        )?;
        let collapsed = AffineTransform::compose(&initial, &rigid.affine);
        Ok(Fit {
            total: with_center(&collapsed, rigid.affine.center),
            metric: rigid.final_metric(),
            iterations: rigid.levels.iter().map(|l| l.cg.iterations).collect(),
        })
    }
}

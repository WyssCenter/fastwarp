// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! 8-bit copies of the images of a finished run on the fixed image's grid, for the result viewer
//! of the page: the fixed image, the registered moving image, the moving image with the start of
//! the chain only (centre of mass) and an edge map of the fixed image. The copies are reordered to
//! a fixed display order (x to the patient's right, y to anterior, z to superior) so that the
//! page needs no knowledge of the image's orientation.

use fastwarp_reg::{Grid, Volume};
use serde_json::{json, Value};

const BINS: usize = 4096;
const HEAD_FRACTION: f64 = 0.1;
const WINDOW_LOW: f64 = 0.01;
const WINDOW_HIGH: f64 = 0.995;
const EDGE_REFERENCE_QUANTILE: f64 = 0.99;
const EDGE_START: f32 = 0.2;
const EDGE_SLICES: usize = 24;
const EDGE_INSIDE: u8 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Fixed,
    Warped,
    Before,
    Edges,
}

impl Layer {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Layer::Fixed),
            1 => Some(Layer::Warped),
            2 => Some(Layer::Before),
            3 => Some(Layer::Edges),
            _ => None,
        }
    }
}

/// Which voxel axis runs along each display axis, and whether it runs backwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orientation {
    pub axis_of: [usize; 3],
    pub flip: [bool; 3],
    pub oriented: bool,
}

impl Orientation {
    /// The nearest display order of a grid (LPS direction columns). An oblique or degenerate
    /// grid whose axes do not map to distinct display axes is shown in voxel order and is not
    /// `oriented`.
    pub fn of(grid: &Grid) -> Self {
        const LPS_TO_DISPLAY: [f64; 3] = [-1.0, -1.0, 1.0];
        let d = grid.direction();
        let mut axis_of = [usize::MAX; 3];
        let mut flip = [false; 3];
        let columns: [[f64; 3]; 3] = std::array::from_fn(|a| [d[0][a], d[1][a], d[2][a]]);
        for (a, column) in columns.iter().enumerate() {
            let mut w = 0;
            for r in 1..3 {
                if column[r].abs() > column[w].abs() {
                    w = r;
                }
            }
            if axis_of[w] != usize::MAX {
                return Self {
                    axis_of: [0, 1, 2],
                    flip: [false; 3],
                    oriented: false,
                };
            }
            axis_of[w] = a;
            flip[w] = column[w] * LPS_TO_DISPLAY[w] < 0.0;
        }
        Self {
            axis_of,
            flip,
            oriented: true,
        }
    }

    pub fn dims(&self, size: [usize; 3]) -> [usize; 3] {
        [
            size[self.axis_of[0]],
            size[self.axis_of[1]],
            size[self.axis_of[2]],
        ]
    }

    pub fn spacing(&self, spacing: [f64; 3]) -> [f64; 3] {
        [
            spacing[self.axis_of[0]],
            spacing[self.axis_of[1]],
            spacing[self.axis_of[2]],
        ]
    }

    /// Reorders a volume of voxel order and size `size` into display order.
    pub fn reorder(&self, data: &[u8], size: [usize; 3]) -> Vec<u8> {
        let dims = self.dims(size);
        let voxel_stride = [1, size[0], size[0] * size[1]];
        let mut start = 0usize;
        let mut step = [0isize; 3];
        for w in 0..3 {
            let stride = voxel_stride[self.axis_of[w]];
            if self.flip[w] {
                start += (dims[w] - 1) * stride;
                step[w] = -(stride as isize);
            } else {
                step[w] = stride as isize;
            }
        }
        let mut out = Vec::with_capacity(data.len());
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                let row = start as isize + step[2] * z as isize + step[1] * y as isize;
                for x in 0..dims[0] {
                    out.push(data[(row + step[0] * x as isize) as usize]);
                }
            }
        }
        out
    }
}

fn histogram(values: impl Iterator<Item = f64>, lo: f64, hi: f64) -> (Vec<u32>, u64) {
    let mut hist = vec![0u32; BINS];
    let mut total = 0u64;
    let scale = BINS as f64 / (hi - lo);
    for v in values {
        let bin = (((v - lo) * scale) as isize).clamp(0, BINS as isize - 1) as usize;
        hist[bin] += 1;
        total += 1;
    }
    (hist, total)
}

fn quantile(hist: &[u32], total: u64, lo: f64, hi: f64, q: f64) -> f64 {
    let target = (q * total as f64).ceil() as u64;
    let mut seen = 0u64;
    for (i, n) in hist.iter().enumerate() {
        seen += u64::from(*n);
        if seen >= target.max(1) {
            return lo + (hi - lo) * (i + 1) as f64 / hist.len() as f64;
        }
    }
    hi
}

/// The intensity window of an image: the 1st and 99.5th percentiles of the voxels inside the
/// head, which are those above a tenth of the way from the minimum to the 99.9th percentile of
/// all finite voxels.
pub fn head_window(data: &[f64]) -> (f64, f64) {
    let finite = || data.iter().copied().filter(|v| v.is_finite());
    let (min, max) = finite().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    let spread = max - min;
    if spread.is_nan() || spread <= 0.0 {
        let base = if min.is_finite() { min } else { 0.0 };
        return (base, base + 1.0);
    }
    let (hist, total) = histogram(finite(), min, max);
    let top = quantile(&hist, total, min, max, 0.999);
    let floor = min + HEAD_FRACTION * (top - min);
    let inside = || finite().filter(|v| *v >= floor);
    let (hist, total) = histogram(inside(), floor, top.max(floor + f64::EPSILON));
    if total == 0 {
        return (min, max);
    }
    let hi_edge = top.max(floor + f64::EPSILON);
    let lo = quantile(&hist, total, floor, hi_edge, WINDOW_LOW);
    let hi = quantile(&hist, total, floor, hi_edge, WINDOW_HIGH);
    if hi > lo {
        (lo, hi)
    } else {
        (lo, lo + 1.0)
    }
}

pub fn quantize(data: &[f64], window: (f64, f64)) -> Vec<u8> {
    let scale = 255.0 / (window.1 - window.0);
    data.iter()
        .map(|v| {
            if v.is_finite() {
                ((v - window.0) * scale).round().clamp(0.0, 255.0) as u8
            } else {
                0
            }
        })
        .collect()
}

/// Gradient magnitude (per millimetre, central differences) and the direction sector of the
/// gradient of a slice: 0 along the columns, 1 along the rows, 2 and 3 the two diagonals.
fn gradient(image: &[u8], width: usize, height: usize, step: [f64; 2]) -> (Vec<f32>, Vec<u8>) {
    let at = |x: usize, y: usize| f32::from(image[x + width * y]);
    let inv = [(0.5 / step[0]) as f32, (0.5 / step[1]) as f32];
    let mut magnitude = vec![0f32; image.len()];
    let mut sector = vec![0u8; image.len()];
    for y in 0..height {
        let (y0, y1) = (y.saturating_sub(1), (y + 1).min(height - 1));
        for x in 0..width {
            let (x0, x1) = (x.saturating_sub(1), (x + 1).min(width - 1));
            let gx = (at(x1, y) - at(x0, y)) * inv[0];
            let gy = (at(x, y1) - at(x, y0)) * inv[1];
            let i = x + width * y;
            magnitude[i] = (gx * gx + gy * gy).sqrt();
            let (ax, ay) = (gx.abs(), gy.abs());
            sector[i] = if ay < 0.4142 * ax {
                0
            } else if ay > 2.4142 * ax {
                1
            } else if gx * gy > 0.0 {
                2
            } else {
                3
            };
        }
    }
    (magnitude, sector)
}

/// The edges of one slice of the fixed image: the 2D gradient magnitude in the plane of the
/// slice, thinned to its ridge (non-maximum suppression along the gradient), zero below a
/// fifth of `reference` and rising to 255 at twice that.
pub fn edge_slice(
    image: &[u8],
    width: usize,
    height: usize,
    step: [f64; 2],
    reference: f32,
) -> Vec<u8> {
    let (magnitude, sector) = gradient(image, width, height, step);
    let low = (EDGE_START * reference).max(f32::EPSILON);
    let mut out = vec![0u8; image.len()];
    for y in 0..height {
        for x in 0..width {
            let i = x + width * y;
            let m = magnitude[i];
            if m <= low {
                continue;
            }
            let (dx, dy): (isize, isize) = match sector[i] {
                0 => (1, 0),
                1 => (0, 1),
                2 => (1, 1),
                _ => (1, -1),
            };
            let neighbour = |sign: isize| {
                let (nx, ny) = (x as isize + sign * dx, y as isize + sign * dy);
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    0.0
                } else {
                    magnitude[nx as usize + width * ny as usize]
                }
            };
            if m >= neighbour(1) && m >= neighbour(-1) {
                out[i] = (((m - low) / low).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

fn plane(
    volume: &[u8],
    dims: [usize; 3],
    axis: usize,
    index: usize,
) -> Option<(usize, usize, Vec<u8>)> {
    let [nx, ny, nz] = dims;
    let (width, height) = match axis {
        0 => (ny, nz),
        1 => (nx, nz),
        2 => (nx, ny),
        _ => return None,
    };
    if index >= dims[axis] {
        return None;
    }
    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        match axis {
            0 => {
                let z = nz - 1 - row;
                data.extend((0..ny).map(|y| volume[index + nx * (y + ny * z)]));
            }
            1 => {
                let z = nz - 1 - row;
                let start = nx * (index + ny * z);
                data.extend_from_slice(&volume[start..start + nx]);
            }
            _ => {
                let y = ny - 1 - row;
                let start = nx * (y + ny * index);
                data.extend_from_slice(&volume[start..start + nx]);
            }
        }
    }
    Some((width, height, data))
}

fn plane_steps(spacing: [f64; 3], axis: usize) -> [f64; 2] {
    match axis {
        0 => [spacing[1], spacing[2]],
        1 => [spacing[0], spacing[2]],
        _ => [spacing[0], spacing[1]],
    }
}

/// The 99th percentile of the in-plane gradient magnitude inside the head over slices spread
/// through the volume in all three directions: the scale of the edge maps.
fn edge_reference(fixed: &[u8], dims: [usize; 3], spacing: [f64; 3]) -> f32 {
    let mut values: Vec<f32> = Vec::new();
    for axis in 0..3 {
        let stride = (dims[axis] / EDGE_SLICES).max(1);
        for index in (0..dims[axis]).step_by(stride) {
            let Some((width, height, image)) = plane(fixed, dims, axis, index) else {
                continue;
            };
            let (magnitude, _) = gradient(&image, width, height, plane_steps(spacing, axis));
            values.extend(
                image
                    .iter()
                    .zip(&magnitude)
                    .filter(|(v, _)| **v >= EDGE_INSIDE)
                    .map(|(_, m)| *m),
            );
        }
    }
    if values.is_empty() {
        return 0.0;
    }
    let k = ((values.len() - 1) as f64 * EDGE_REFERENCE_QUANTILE) as usize;
    let (_, kth, _) = values.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *kth
}

/// One slice of a layer: `width` x `height` bytes, row by row from the top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// The 8-bit volumes of one run in display order. It starts from the fixed image
/// ([`Viewer::new`]) so that the fixed volume can be freed early, and takes the registered image
/// and the one before the registration as they are made ([`Viewer::set_warped`],
/// [`Viewer::set_before`]).
#[derive(Debug)]
pub struct Viewer {
    orientation: Orientation,
    size: [usize; 3],
    dims: [usize; 3],
    spacing: [f64; 3],
    fixed: Vec<u8>,
    warped: Vec<u8>,
    before: Option<Vec<u8>>,
    edge_reference: f32,
    windows: [(f64, f64); 2],
}

impl Viewer {
    pub fn new(fixed: &Volume<f64>) -> Self {
        let grid = fixed.grid();
        let size = grid.size();
        let orientation = Orientation::of(grid);
        let window = head_window(fixed.data());
        let bytes = orientation.reorder(&quantize(fixed.data(), window), size);
        Self::from_parts(
            orientation,
            size,
            orientation.spacing(grid.spacing()),
            bytes,
            window,
        )
    }

    fn from_parts(
        orientation: Orientation,
        size: [usize; 3],
        spacing: [f64; 3],
        fixed: Vec<u8>,
        window: (f64, f64),
    ) -> Self {
        let dims = orientation.dims(size);
        let edge_reference = edge_reference(&fixed, dims, spacing);
        Self {
            orientation,
            size,
            dims,
            spacing,
            fixed,
            warped: Vec::new(),
            before: None,
            edge_reference,
            windows: [window, (0.0, 1.0)],
        }
    }

    /// Adds the registered image (on the fixed grid) with its own window.
    pub fn set_warped(&mut self, warped: &Volume<f64>) {
        let window = head_window(warped.data());
        self.windows[1] = window;
        self.warped = self
            .orientation
            .reorder(&quantize(warped.data(), window), self.size);
    }

    /// The window of the registered image, which the image before the registration shares.
    pub fn warped_window(&self) -> (f64, f64) {
        self.windows[1]
    }

    /// Adds the image before the registration as bytes in the fixed grid's voxel order, already
    /// quantized with [`Viewer::warped_window`].
    pub fn set_before_bytes(&mut self, voxel_order: &[u8]) {
        self.before = Some(self.orientation.reorder(voxel_order, self.size));
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn has_before(&self) -> bool {
        self.before.is_some()
    }

    /// Sizes, voxel spacing in millimetres (display order), whether the orientation labels are
    /// valid, the layers available and the intensity windows used.
    pub fn info(&self) -> Value {
        json!({
            "dims": self.dims,
            "spacing": self.spacing,
            "oriented": self.orientation.oriented,
            "before": self.before.is_some(),
            "windows": {
                "fixed": [self.windows[0].0, self.windows[0].1],
                "warped": [self.windows[1].0, self.windows[1].1],
            },
        })
    }

    fn layer(&self, layer: Layer) -> Option<&[u8]> {
        match layer {
            Layer::Fixed | Layer::Edges => Some(&self.fixed),
            Layer::Warped if self.warped.is_empty() => None,
            Layer::Warped => Some(&self.warped),
            Layer::Before => self.before.as_deref(),
        }
    }

    /// One slice: `axis` 0 is sagittal (columns run to anterior, rows to inferior), 1 coronal
    /// (columns to the patient's right, rows to inferior) and 2 axial (columns to the patient's
    /// right, rows to posterior). `None` for an axis or index out of range or a missing layer.
    pub fn slice(&self, axis: usize, index: usize, layer: Layer) -> Option<Slice> {
        let volume = self.layer(layer)?;
        let (width, height, mut data) = plane(volume, self.dims, axis, index)?;
        if layer == Layer::Edges {
            data = edge_slice(
                &data,
                width,
                height,
                plane_steps(self.spacing, axis),
                self.edge_reference,
            );
        }
        Some(Slice {
            width,
            height,
            data,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use fastwarp_reg::grid::lps_to_ras;
    use fastwarp_reg::linalg::{Mat3, IDENTITY3};

    fn grid(size: [usize; 3], spacing: [f64; 3], direction: Mat3) -> Grid {
        Grid::new(size, spacing, [10.0, -20.0, 30.0], direction).unwrap()
    }

    fn ramp(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    fn viewer(dims: [usize; 3], spacing: [f64; 3], before: bool) -> Viewer {
        let n = dims[0] * dims[1] * dims[2];
        let mut v = Viewer::from_parts(
            Orientation {
                axis_of: [0, 1, 2],
                flip: [false; 3],
                oriented: true,
            },
            dims,
            spacing,
            ramp(n),
            (0.0, 1.0),
        );
        v.warped = ramp(n).iter().map(|x| x.wrapping_add(1)).collect();
        v.before = before.then(|| ramp(n).iter().map(|x| x.wrapping_add(2)).collect());
        v
    }

    #[test]
    fn display_order_follows_the_direction_of_the_grid() {
        let lps = Orientation::of(&grid([4, 5, 6], [1.0; 3], IDENTITY3));
        assert_eq!(lps.axis_of, [0, 1, 2]);
        assert_eq!(lps.flip, [true, true, false]);
        let ras = Orientation::of(&grid(
            [4, 5, 6],
            [1.0; 3],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ));
        assert_eq!(ras.flip, [false, false, false]);
        let permuted = Orientation::of(&grid(
            [4, 5, 6],
            [1.0; 3],
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ));
        assert_eq!(permuted.axis_of, [2, 0, 1]);
        assert_eq!(permuted.flip, [true, false, false]);
        assert_eq!(permuted.dims([4, 5, 6]), [6, 4, 5]);
        assert_eq!(permuted.spacing([1.0, 2.0, 3.0]), [3.0, 1.0, 2.0]);
        let degenerate = Orientation::of(&grid(
            [4, 5, 6],
            [1.0; 3],
            [[1.0, 0.9, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
        ));
        assert_eq!(degenerate.axis_of, [0, 1, 2]);
        assert!(!degenerate.oriented);
        assert!(lps.oriented);
    }

    #[test]
    fn reordering_puts_every_voxel_at_its_position_in_ras() {
        let directions: [Mat3; 3] = [
            IDENTITY3,
            [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ];
        for direction in directions {
            let size = [3, 4, 5];
            let g = grid(size, [1.0, 2.0, 3.0], direction);
            let o = Orientation::of(&g);
            let dims = o.dims(size);
            let spacing = o.spacing(g.spacing());
            let src: Vec<u8> = (0..60).map(|i| i as u8).collect();
            let out = o.reorder(&src, size);
            let mut low = [f64::INFINITY; 3];
            for lin in 0..60 {
                let [i, j, k] = g.ijk(lin);
                let ras = lps_to_ras(g.index_to_point([i as f64, j as f64, k as f64]));
                for w in 0..3 {
                    low[w] = low[w].min(ras[w]);
                }
            }
            for (lin, &value) in src.iter().enumerate() {
                let [i, j, k] = g.ijk(lin);
                let ras = lps_to_ras(g.index_to_point([i as f64, j as f64, k as f64]));
                let d: Vec<usize> = (0..3)
                    .map(|w| ((ras[w] - low[w]) / spacing[w]).round() as usize)
                    .collect();
                assert_eq!(
                    out[d[0] + dims[0] * (d[1] + dims[1] * d[2])],
                    value,
                    "{direction:?} voxel {lin}"
                );
            }
        }
    }

    #[test]
    fn slices_have_the_axes_sizes_and_orientation_of_the_views() {
        let v = viewer([3, 4, 5], [1.0, 2.0, 3.0], true);
        let at = |x: usize, y: usize, z: usize| (x + 3 * (y + 4 * z)) as u8;
        let sag = v.slice(0, 1, Layer::Fixed).unwrap();
        assert_eq!((sag.width, sag.height), (4, 5));
        assert_eq!(sag.data[0], at(1, 0, 4));
        assert_eq!(sag.data[3], at(1, 3, 4));
        assert_eq!(sag.data[4 * 4], at(1, 0, 0));
        let cor = v.slice(1, 2, Layer::Fixed).unwrap();
        assert_eq!((cor.width, cor.height), (3, 5));
        assert_eq!(cor.data[0], at(0, 2, 4));
        assert_eq!(cor.data[2], at(2, 2, 4));
        assert_eq!(cor.data[3 * 4 + 1], at(1, 2, 0));
        let ax = v.slice(2, 4, Layer::Fixed).unwrap();
        assert_eq!((ax.width, ax.height), (3, 4));
        assert_eq!(ax.data[0], at(0, 3, 4));
        assert_eq!(ax.data[3 * 3 + 2], at(2, 0, 4));
        assert_eq!(ax.data.len(), 12);
    }

    #[test]
    fn layers_are_separate_and_a_missing_layer_or_bound_gives_none() {
        let v = viewer([3, 4, 5], [1.0; 3], true);
        let fixed = v.slice(2, 0, Layer::Fixed).unwrap();
        let warped = v.slice(2, 0, Layer::Warped).unwrap();
        let before = v.slice(2, 0, Layer::Before).unwrap();
        assert_eq!(warped.data[0], fixed.data[0].wrapping_add(1));
        assert_eq!(before.data[0], fixed.data[0].wrapping_add(2));
        assert_eq!(v.slice(2, 0, Layer::Edges).unwrap().data.len(), 12);
        assert!(v.slice(0, 3, Layer::Fixed).is_none());
        assert!(v.slice(1, 4, Layer::Fixed).is_none());
        assert!(v.slice(2, 5, Layer::Fixed).is_none());
        assert!(v.slice(3, 0, Layer::Fixed).is_none());
        let without = viewer([3, 4, 5], [1.0; 3], false);
        assert!(!without.has_before());
        assert!(without.slice(2, 0, Layer::Before).is_none());
        assert_eq!(Layer::from_code(2), Some(Layer::Before));
        assert_eq!(Layer::from_code(4), None);
    }

    #[test]
    fn the_info_reports_dimensions_and_spacing() {
        let v = viewer([3, 4, 5], [1.0, 2.0, 3.0], true);
        let info = v.info();
        assert_eq!(info["dims"], json!([3, 4, 5]));
        assert_eq!(info["spacing"], json!([1.0, 2.0, 3.0]));
        assert_eq!(info["before"], true);
        assert_eq!(info["oriented"], true);
    }

    #[test]
    fn the_window_ignores_the_background_and_outliers() {
        let mut data = vec![0.0; 6000];
        data.extend((0..4000).map(|i| 100.0 + (i % 100) as f64));
        data.push(100000.0);
        data.push(f64::NAN);
        let (lo, hi) = head_window(&data);
        assert!((99.0..110.0).contains(&lo), "{lo}");
        assert!((190.0..260.0).contains(&hi), "{hi}");
        let flat = head_window(&[5.0; 10]);
        assert_eq!(flat, (5.0, 6.0));
        assert_eq!(head_window(&[]), (0.0, 1.0));
    }

    #[test]
    fn quantizing_maps_the_window_to_the_byte_range() {
        let out = quantize(&[-5.0, 10.0, 15.0, 20.0, 99.0, f64::NAN], (10.0, 20.0));
        assert_eq!(out, [0, 0, 128, 255, 255, 0]);
    }

    #[test]
    fn edges_are_thin_lines_on_the_boundary_of_a_square() {
        let (w, h) = (40, 40);
        let mut image = vec![0u8; w * h];
        for y in 10..30 {
            for x in 10..30 {
                image[x + w * y] = 200;
            }
        }
        let edges = edge_slice(&image, w, h, [1.0, 1.0], 100.0);
        let at = |x: usize, y: usize| edges[x + w * y];
        assert!(at(9, 20) > 200 || at(10, 20) > 200);
        assert!(at(29, 20) > 200 || at(30, 20) > 200);
        assert_eq!(at(20, 20), 0);
        assert_eq!(at(2, 2), 0);
        assert_eq!(at(20, 2), 0);
        let on_row = (0..w).filter(|x| at(*x, 20) > 0).count();
        assert!((2..=4).contains(&on_row), "{on_row}");
        let flat = edge_slice(&[200u8; 64], 8, 8, [1.0, 1.0], 100.0);
        assert_eq!(flat, vec![0u8; 64]);
    }

    #[test]
    fn the_edge_layer_follows_the_slice_and_is_empty_in_a_flat_volume() {
        let dims = [12, 12, 12];
        let mut v = viewer(dims, [1.0; 3], false);
        v.fixed = vec![200u8; 12 * 12 * 12];
        v.edge_reference = 100.0;
        for axis in 0..3 {
            let s = v.slice(axis, 6, Layer::Edges).unwrap();
            assert!(s.data.iter().all(|b| *b == 0));
        }
        for z in 3..9 {
            for y in 3..9 {
                for x in 0..12 {
                    v.fixed[x + 12 * (y + 12 * z)] = if (3..9).contains(&x) { 200 } else { 0 };
                }
            }
        }
        let s = v.slice(2, 5, Layer::Edges).unwrap();
        assert!(s.data.iter().any(|b| *b > 200));
        assert_eq!(s.data[6 * 12 + 6], 0);
    }

    #[test]
    fn a_viewer_built_from_volumes_takes_the_later_images_in_turn() {
        let g = Grid::new([6, 5, 4], [1.0, 2.0, 3.0], [0.0; 3], IDENTITY3).unwrap();
        let vol = |scale: f64| {
            Volume::from_vec(
                g,
                (0..120).map(|i| 50.0 + scale * (i % 40) as f64).collect(),
            )
            .unwrap()
        };
        let mut v = Viewer::new(&vol(1.0));
        assert_eq!(v.dims(), [6, 5, 4]);
        assert!(v.slice(2, 0, Layer::Warped).is_none());
        v.set_warped(&vol(2.0));
        assert!(!v.has_before());
        let w = v.warped_window();
        v.set_before_bytes(&quantize(vol(3.0).data(), w));
        assert!(v.has_before());
        assert!(v.slice(2, 0, Layer::Warped).is_some());
        assert_eq!(v.info()["oriented"], true);
    }
}

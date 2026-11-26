# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0
"""Writes the ITK golden fixtures of the dense Mattes mutual information tests.

ITK's `MattesMutualInformationImageToImageMetricv4` evaluates small volumes with a
`DisplacementFieldTransform` as the moving transform (dense sampling, central-difference
gradients, as `antsRegistration` sets it up), once without and once with image masks. Each case
is written as a text file of `key: values` lines (images x-fastest, vectors interleaved) to
`crates/fastwarp-reg/tests/data/mattes/`. Run from the repository root:

    uv run --project scripts/itk_fixtures --locked python -I scripts/itk_fixtures/mattes_fixtures.py
"""

import math
import sys
from pathlib import Path

import itk
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "crates" / "fastwarp-reg" / "tests" / "data" / "mattes"
IMAGE = itk.Image[itk.D, 3]
MASK = itk.Image[itk.UC, 3]
VECTOR = itk.Vector[itk.D, 3]
FIELD = itk.Image[VECTOR, 3]


def rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    r = np.eye(3)
    i, j = [a for a in range(3) if a != axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


class Grid:
    def __init__(self, size, spacing, origin, direction):
        self.size = tuple(size)
        self.spacing = np.array(spacing, dtype=float)
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)

    def points(self):
        k, j, i = np.meshgrid(*(np.arange(n) for n in self.size[::-1]), indexing="ij")
        index = np.stack([i, j, k], axis=-1).astype(float)
        return self.origin + (index * self.spacing) @ self.direction.T

    def place(self, image):
        image.SetSpacing(self.spacing.tolist())
        image.SetOrigin(self.origin.tolist())
        image.SetDirection(itk.matrix_from_array(self.direction))
        return image


def quantise(values, step):
    return np.round(np.asarray(values) / step) * step


def anatomy(p):
    blobs = [
        ((0.4, -0.6, 0.3), 2.2, 1.0),
        ((-1.8, 1.4, -0.5), 1.4, 0.8),
        ((1.9, 1.2, 1.1), 1.1, 0.6),
    ]
    v = np.zeros(p.shape[:-1])
    for centre, sigma, amplitude in blobs:
        d2 = np.sum((p - np.array(centre)) ** 2, axis=-1)
        v += amplitude * np.exp(-d2 / (2.0 * sigma * sigma))
    return v + 0.05 * p[..., 0]


def fixed_values(grid):
    return quantise(anatomy(grid.points()), 2.0**-12)


def moving_values(grid):
    q = grid.points() - np.array([0.35, -0.25, 0.2])
    return quantise(2.0 - 1.7 * np.clip(anatomy(q), 0.0, None) ** 1.5, 2.0**-12)


def displacement(grid, amplitude):
    p = grid.points()
    u = np.stack(
        [
            amplitude * np.sin(0.7 * p[..., 1] + 0.3),
            0.8 * amplitude * np.cos(0.6 * p[..., 2] - 0.2),
            0.6 * amplitude * np.sin(0.5 * p[..., 0] + 0.9),
        ],
        axis=-1,
    )
    return u.astype(np.float32).astype(np.float64)


def ball(grid, centre, radius):
    d = np.linalg.norm(grid.points() - np.array(centre), axis=-1)
    return (d < radius).astype(np.uint8)


def half_space(grid, axis, below):
    return (grid.points()[..., axis] < below).astype(np.uint8)


def to_image(values, grid, kind):
    image = itk.image_from_array(np.ascontiguousarray(values), ttype=kind)
    return grid.place(image)


def to_field(values, grid):
    field = FIELD.New()
    field.SetRegions(list(grid.size))
    field.Allocate()
    for k in range(grid.size[2]):
        for j in range(grid.size[1]):
            for i in range(grid.size[0]):
                field.SetPixel([i, j, k], VECTOR([float(c) for c in values[k, j, i]]))
    return grid.place(field)


def to_mask(values, grid):
    mask = itk.ImageMaskSpatialObject[3].New()
    mask.SetImage(to_image(values, grid, MASK))
    mask.Update()
    return mask


def evaluate(case):
    metric = itk.MattesMutualInformationImageToImageMetricv4[IMAGE, IMAGE].New()
    metric.SetMaximumNumberOfWorkUnits(1)
    metric.SetFixedImage(to_image(case["fixed"], case["fixed_grid"], IMAGE))
    metric.SetMovingImage(to_image(case["moving"], case["moving_grid"], IMAGE))
    metric.SetNumberOfHistogramBins(case["bins"])
    metric.SetUseFixedImageGradientFilter(False)
    metric.SetUseMovingImageGradientFilter(False)
    if case.get("fixed_mask") is not None:
        metric.SetFixedImageMask(to_mask(case["fixed_mask"], case["fixed_grid"]))
    if case.get("moving_mask") is not None:
        metric.SetMovingImageMask(to_mask(case["moving_mask"], case["moving_grid"]))
    transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    transform.SetDisplacementField(to_field(case["field"], case["fixed_grid"]))
    metric.SetMovingTransform(transform)
    metric.SetVirtualDomainFromImage(to_image(case["fixed"], case["fixed_grid"], IMAGE))
    metric.Initialize()
    derivative = itk.Array[itk.D]()
    metric.GetDerivative(derivative)
    values = np.array([derivative.GetElement(i) for i in range(derivative.GetSize())])
    return metric.GetValue(), metric.GetNumberOfValidPoints(), values


def numbers(values):
    return " ".join(repr(float(v)) for v in np.asarray(values, dtype=float).ravel())


def write(name, case):
    value, valid, derivative = evaluate(case)
    lines = [
        "# Written by scripts/itk_fixtures/mattes_fixtures.py with ITK "
        + itk.Version.GetITKVersion()
        + " (MattesMutualInformationImageToImageMetricv4, DisplacementFieldTransform).",
        f"bins: {case['bins']}",
    ]
    for role in ("fixed", "moving"):
        grid = case[f"{role}_grid"]
        lines += [
            f"{role}_size: " + " ".join(str(n) for n in grid.size),
            f"{role}_spacing: " + numbers(grid.spacing),
            f"{role}_origin: " + numbers(grid.origin),
            f"{role}_direction: " + numbers(grid.direction),
            f"{role}: " + numbers(case[role]),
        ]
        if case.get(f"{role}_mask") is not None:
            lines.append(
                f"{role}_mask: " + " ".join(str(int(v)) for v in case[f"{role}_mask"].ravel())
            )
    lines += [
        "field: " + numbers(case["field"]),
        f"value: {value!r}",
        f"valid: {valid}",
        "derivative: " + numbers(derivative),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"{name}: value {value!r}, {valid} valid points", file=sys.stderr)


def main():
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)
    fixed_grid = Grid((8, 7, 6), (1.0, 1.25, 0.9), (-3.5, -4.0, -2.2), rotation(2, 10.0))
    moving_grid = Grid((9, 8, 7), (1.1, 1.1, 1.0), (-4.3, -4.1, -3.0), rotation(0, -8.0))
    oblique = {
        "fixed_grid": fixed_grid,
        "moving_grid": moving_grid,
        "fixed": fixed_values(fixed_grid),
        "moving": moving_values(moving_grid),
        "field": displacement(fixed_grid, 0.6),
    }
    masks = {
        "fixed_mask": ball(fixed_grid, (0.3, -0.2, 0.1), 3.4),
        "moving_mask": half_space(moving_grid, 0, 2.2),
    }
    write("oblique_b32", {**oblique, "bins": 32})
    write("oblique_b12", {**oblique, "bins": 12})
    write("oblique_b32_masked", {**oblique, **masks, "bins": 32})
    same = {
        "fixed_grid": fixed_grid,
        "moving_grid": fixed_grid,
        "fixed": fixed_values(fixed_grid),
        "moving": moving_values(fixed_grid),
        "field": np.zeros(fixed_grid.size[::-1] + (3,)),
    }
    write("virtual_b32", {**same, "bins": 32})
    write(
        "virtual_b32_masked",
        {
            **same,
            "fixed_mask": ball(fixed_grid, (0.3, -0.2, 0.1), 3.4),
            "moving_mask": half_space(fixed_grid, 1, 1.9),
            "bins": 32,
        },
    )


if __name__ == "__main__":
    main()

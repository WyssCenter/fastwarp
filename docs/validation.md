# Performance and validation

How fast FastWarp is on measured hardware, how its accuracy is measured against ANTs, and how
the code is tested.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`stopwatch` Performance
:link: performance
:link-type: doc

Run times of the registration, the linear fits and the resampling, with the data and hardware
they were measured on.
:::

:::{grid-item-card} {fas}`bullseye` Accuracy benchmark
:link: benchmarks
:link-type: doc

FastWarp and ANTs on public brain MRI with manual labels and landmarks. Anyone can download the
data and run it.
:::

:::{grid-item-card} {fas}`vial` Testing
:link: testing
:link-type: doc

The Rust and Python tests, the comparisons with the ANTs programs, and what CI runs.
:::
::::

```{toctree}
:hidden:

performance
benchmarks
testing
```

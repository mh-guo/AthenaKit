# AthenaKit Documentation

**AthenaKit** is a Python toolkit for analyzing and visualizing simulation data from [AthenaK](https://github.com/IAS-Astrophysics/athenak) — a performance-portable astrophysics code built on Kokkos.

## Key Features

- **Performance-portable** — runs on both CPU and CUDA GPU via CuPy
- **MPI-aware** — distributes analysis across many ranks
- **Rich data model** — lazy evaluation of 50+ derived variables from raw dumps
- **Flexible I/O** — reads `.bin`, `.athdf`, `.h5`, and `.pkl` formats
- **Built-in analysis** — histograms, profiles, slices, and interpolation
- **Visualization** — slices, phase diagrams, and radial profiles via Matplotlib

## Quick Navigation

::::{grid} 2
:::{grid-item-card} Getting Started
:link: getting_started/index
:link-type: doc

Install AthenaKit and run your first analysis in minutes.
:::
:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Full documentation of all classes and functions.
:::
:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Worked examples for common astrophysical problems.
:::
:::{grid-item-card} Source on GitHub
:link: https://github.com/mh-guo/AthenaKit
:link-type: url

Browse the source code, open issues, or contribute.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

getting_started/index
examples/index
```

```{toctree}
:maxdepth: 3
:hidden:
:caption: API Reference

api/index
```

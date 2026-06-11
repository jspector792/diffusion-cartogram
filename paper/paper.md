---
title: 'diffusion-cartogram: A Python package for diffusion deformations in 2 and 3 dimensions'
tags:
  - Python
  - computational geometry
  - 3D shape deformation
  - cartography
  - scientific visualization
  - adaptive mesh refinement
authors:
  - name: Jonah Spector
    orcid: 0000-0002-5128-9129
    affiliation: 1
  - name: Albert-László Barabási
    orcid: 0000-0002-4028-3522
    affiliation: "1, 2, 3"
affiliations:
  - name: Network Science Institute, Northeastern University, USA
    index: 1
  - name: Department of Medicine, Brigham and Women's Hospital, Harvard Medical School, USA
    index: 2
  - name: Department of Network and Data Science, Central European University
    index: 3
date: 11 March 2026
bibliography: paper.bib
---

# Summary

`diffusion-cartogram` is a Python package for computing density-equalizing reference maps in both
two and three dimensions. Given a spatial domain and a user-defined scalar density
distribution, the package deforms the domain so that regions with higher density expand
and regions with lower density contract, producing a smooth, orientation-preserving map
that equalizes density across the domain.

In two dimensions, `diffusion-cartogram` implements a diffusion-based density-equalizing map for
planar domains, enabling the construction of statistical cartograms from point clouds
derived from shapefiles, GeoPandas GeoDataFrames, or plain CSV/XYZ files. In three
dimensions, it implements the Volumetric Density-Equalizing Reference Map (VDERM)
algorithm of @choi2021volumetric, which extends this framework to fully volumetric
domains in $\mathbb{R}^3$ using the reference map technique [@kamrin2012]. Both modes
are built on the same core architecture: a regular Cartesian grid, iterative backward
Euler diffusion of the density field, and reference map tracking to recover the final
deformation without per-iteration interpolation.

The package exposes both modes through a unified high-level API, includes built-in
visualization utilities for animating and inspecting deformations, and supports export
to XYZ, STL, and VTK formats for use in downstream tools such as ParaView. Example
Jupyter notebooks cover a dependency-free 2D workflow (CSV inputs only), a world
population cartogram, and 3D volumetric deformation examples.

# Statement of Need

Density-equalizing maps are widely used in cartography, data visualization, and medical
imaging, yet accessible Python implementations have been lacking for both planar and
volumetric domains.

For two-dimensional cartograms, despite the foundational @gastner2004 algorithm being
over two decades old, no general-purpose Python implementation of diffusion-based
contiguous cartograms existed prior to `diffusion-cartogram`. Existing tools are either implemented
in other languages, limited to specific input formats, or not openly maintained. This
gap is particularly notable given Python's dominance in data science and geospatial
analysis workflows.

For three-dimensional volumetric deformations, the VDERM algorithm [@choi2021volumetric]
was previously available only as a C++ prototype. No open Python implementation
existed for data-driven 3D shape deformation via density equalization.

`diffusion-cartogram` fills both gaps in a single, pip-installable package. It is designed for
researchers and practitioners in computational mathematics, cartography, geospatial data
science, medical imaging, and computational geometry who need density-driven spatial
deformations without requiring expertise in numerical PDE methods. A dependency-free
mode (using only NumPy and SciPy) ensures accessibility in restricted computational
environments.

# State of the Field

Density-equalizing maps originate from the work of @gastner2004, who introduced a
diffusion-based algorithm for constructing contiguous cartograms—maps in which regions
are rescaled in proportion to a statistical quantity such as population. This method was
later extended to simply-connected open surfaces [@choi2018dem] and, most recently, to
genus-0 closed surfaces [@lyu2024sdem].

The extension to volumetric domains was introduced by @choi2021volumetric via the
reference map technique [@kamrin2012], which tracks deformation on a fixed Eulerian
grid and requires interpolation only once, at the end of the iteration. This is a
significant computational advantage over naively extending the original @gastner2004
approach to 3D. To our knowledge, `diffusion-cartogram` is the first openly available Python
implementation of both the 2D diffusion-based cartogram and the 3D VDERM method.

Related open-source tools for spatial deformation include general-purpose mesh
processing libraries such as PyMeshLab [@pymeshlab2021] and PyVista [@sullivan2019],
as well as geometry processing packages like libigl [@libigl2018]. Geospatial
libraries such as GeoPandas [@geopandas2020] provide rich support for working with
geographic data but do not perform density-driven deformations. None of these tools
provide the density-equalizing deformation functionality of `diffusion-cartogram`.

# Software Design

`diffusion-cartogram` is organized into parallel 2D and 3D modules (`core_2d` / `core` and
`visualization_2d` / `visualization`) that share a common design pattern across
three stages: grid initialization, density diffusion, and surface interpolation.

**Grid initialization.** A regular Cartesian grid is constructed over the bounding box
of the input geometry, with automatic sizing based on a target point count and optional
padding. The `VDERMGrid` class (2D and 3D variants) stores the reference map field and
density field on this grid. The density field is defined by a user-supplied callable
$\rho(\mathbf{x})$, providing full flexibility in specifying the deformation.

**Density diffusion and reference map update.** The core iteration alternates between
solving the diffusion equation for the density field using a backward Euler scheme with
a pre-factored sparse matrix, and updating the reference map using the resulting
velocity field. No-flux boundary conditions are enforced via ghost nodes; free boundary
conditions are available by embedding the domain in a uniform-density region. The
iteration proceeds until a convergence criterion on the maximum per-step displacement
is met, or a maximum iteration count is reached.

**Surface interpolation.** After convergence, `interpolate_to_surface` maps input
points to their deformed positions using fast regular grid interpolation
(`scipy.interpolate`). Performing this interpolation once at the end—rather than at
every iteration—is the key efficiency of the reference map approach [@choi2021volumetric].

**Input and output.** The 2D module accepts point clouds from shapefiles, GeoPandas
GeoDataFrames, or plain CSV/XYZ files; a dependency-free mode (NumPy and SciPy only)
supports CSV inputs without requiring GeoPandas or Shapely. The 3D module supports
point cloud and mesh inputs via optional PyMeshLab integration. Both modules export to
XYZ format; the 3D module additionally supports STL and VTK export. Deformed meshes
can be reconstructed from point clouds via Poisson surface reconstruction.

# Research Impact Statement

`diffusion-cartogram` makes density-equalizing map methods accessible to the broader scientific
Python community for the first time in both two and three dimensions. The 2D module
enables reproducible cartogram creation within standard geospatial Python workflows,
while the 3D module opens the VDERM algorithm to researchers in medical imaging,
computational geometry, and data visualization. Anticipated applications include
population and demographic cartograms, data-driven adaptive remeshing of 3D domains,
volumetric medical data visualization, and time-dependent shape morphing. Worked
examples—including a world population cartogram and a 3D volumetric deformation
tutorial—are provided as Jupyter notebooks to support immediate use in research
pipelines.

# AI Usage Disclosure

The software implementation of `diffusion-cartogram` was developed with the assistance of
Claude (Anthropic), a large language model, used as a programming aid for code
generation, debugging, and documentation. This paper was drafted with the assistance
of Claude. All algorithmic decisions, design choices, validation, and scientific
content are the responsibility of the authors. The original VDERM algorithm is the
work of @choi2021volumetric and is unmodified in its mathematical substance by this
implementation.

# Acknowledgements

The authors thank Gary P. T. Choi and Chris H. Rycroft for the original development of
the VDERM algorithm and for approving this Python implementation. Work by J.S. was supported in part by the National Institute of Health grant number T32 HL007427

# References
---
title: 'pyVDERM: A Python Package for Volumetric Density-Equalizing Reference Maps'
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
  - name: Gary P. T. Choi
    orcid: 0000-0001-5407-9111
    affiliation: 2
  - name: Albert László Barabási
    orcid: 0000-0002-4028-3522
    affiliation: "1, 3, 4"
affiliations:
  - name: Network Science Institute, Northeastern University, USA
    index: 1
  - name: Department of Mathematics, The Chinese University of Hong Kong, Hong Kong
    index: 2
  - name: Department of Medicine, Brigham and Women's Hospital, Harvard Medical School, USA
    index: 3
  - name: Department of Network and Data Science, Central European University
    index: 4
date: 11 March 2026
bibliography: paper.bib
---

# Summary

`pyVDERM` is a Python package implementing the Volumetric Density-Equalizing Reference
Map (VDERM) algorithm [@choi2021volumetric], a method for continuously deforming
three-dimensional objects according to a prescribed scalar density distribution. Given
a volumetric domain and a user-defined density function, the algorithm diffuses the
density field and tracks the resulting deformation using a reference map, producing a
smooth, orientation-preserving map that enlarges regions of high density and shrinks
regions of low density. The method extends the classical diffusion-based cartogram
technique of @gastner2004 from two-dimensional maps to fully three-dimensional domains.

`pyVDERM` exposes this algorithm through a high-level Python API centered on the
`VDERMGrid` class, which discretizes the computation onto a regular Cartesian grid.
The package handles point cloud and mesh inputs (via optional PyMeshLab integration),
provides flexible export to XYZ, STL, and VTK formats for use in tools such as
ParaView, and includes built-in visualization utilities for producing animations of the
deformation process. A suite of Jupyter notebook examples covers the full workflow from
mesh loading and density field design through to export and visualization.

# Statement of Need

Density-equalizing maps have proven valuable across data visualization, medical
imaging, and computational geometry, yet accessible software implementations have
historically been limited to two-dimensional domains. The original VDERM method
[@choi2021volumetric] was demonstrated using a MATLAB prototype; no open,
general-purpose implementation for 3D density-equalizing deformations existed in Python
prior to `pyVDERM`.

Python has become the dominant language for scientific computing and data analysis,
with a large ecosystem of numerical, visualization, and mesh processing libraries.
Researchers wishing to apply 3D density-equalizing deformations to their data—whether
for volumetric cartogram creation, adaptive remeshing, or shape morphing—currently have
no suitable Python tool available. They must either implement the algorithm from scratch
or use the original MATLAB code, which limits accessibility, reproducibility, and
integration with modern scientific Python workflows.

`pyVDERM` addresses this gap by providing a well-documented, tested, and pip-installable
Python implementation suitable for use in research pipelines. It is designed for
researchers and practitioners in computational mathematics, medical imaging, data
visualization, and computational geometry who need to perform data-driven 3D shape
deformations without requiring specialized expertise in numerical PDE methods.

# State of the Field

Density-equalizing maps originate from the work of @gastner2004, who introduced a
diffusion-based algorithm for creating 2D cartograms—maps in which regions are rescaled
in proportion to a statistical quantity such as population. This method has since been
widely adopted for 2D data visualization [@dorling2008] and extended to surface
domains, including open surfaces [@choi2018dem] and genus-0 closed surfaces
[@lyu2024sdem].

The extension to volumetric (3D) domains was introduced by @choi2021volumetric via the
reference map technique [@kamrin2012], which tracks the deformation of a material on a
fixed Eulerian grid, avoiding the need to interpolate a velocity field at every
iteration. This is a significant computational advantage over a naive 3D extension of
the original [@gastner2004] method. To our knowledge, `pyVDERM` is the first
openly available Python implementation of this volumetric approach.

Related open-source tools for 3D shape deformation include general-purpose mesh
processing libraries such as PyMeshLab [@pymeshlab2021] and PyVista [@sullivan2019],
as well as geometry processing packages like libigl [@libigl2018]. However, none of
these provide density-driven volumetric deformation in the manner of the VDERM method.

# Software Design

`pyVDERM` is structured around three core stages: grid initialization, density
diffusion, and surface interpolation.

A regular Cartesian grid is constructed over the bounding box of the input geometry
using `make_initial_grid`, with automatic sizing based on a user-specified target point
count and optional padding. The `VDERMGrid` class stores the reference map field and
density field on this grid. The density field is set via a user-supplied callable
$\rho(x, y, z)$, giving full flexibility in defining the deformation.

The VDERM iteration is performed by `run_VDERM`, which alternates between (i) solving
the diffusion equation for the density field using a backward Euler scheme with a
pre-factored sparse matrix for efficiency, and (ii) updating the reference map field
using the resulting velocity. No-flux boundary conditions are enforced via ghost nodes,
and free boundary conditions are optionally available by embedding the domain in a
uniform-density "sea". The iteration proceeds until a convergence criterion on the
maximum deformation per step ($\varepsilon$) is met, or a maximum iteration count is
reached.

Finally, `interpolate_to_surface` applies the computed displacement field to an input
point cloud using fast regular grid interpolation (via `scipy.interpolate`), mapping
surface points to their deformed positions without requiring per-iteration interpolation.
This design—performing interpolation only once at the end—follows the key efficiency
insight of the reference map approach [@choi2021volumetric].

The package includes progress tracking via `tqdm`, intermediate state export for
checkpointing, and mesh reconstruction from deformed point clouds via PyMeshLab's
Poisson surface reconstruction.

# Research Impact Statement

`pyVDERM` makes the VDERM algorithm accessible to the broader scientific Python
community for the first time. Anticipated research applications include 3D statistical
cartogram creation for visualizing volumetric demographic or medical data, data-driven
adaptive remeshing of complex 3D domains, and time-dependent shape morphing between
3D objects. By providing a pip-installable package with worked examples and
comprehensive documentation, `pyVDERM` lowers the barrier to applying these techniques
in reproducible research workflows.

# AI Usage Disclosure

The software implementation of `pyVDERM` was developed with the assistance of
Claude (Anthropic), a large language model, used as a programming aid for code
generation, debugging, and documentation. This paper was drafted with the assistance
of Claude. All algorithmic decisions, design choices, validation, and scientific
content are the responsibility of the authors. The original VDERM algorithm is the
work of @choi2021volumetric and is unmodified in its mathematical substance by this
implementation.

# Acknowledgements

The authors thank Gary P. T. Choi and Chris H. Rycroft for the original development of
the VDERM algorithm and for approving this Python implementation. [ADDITIONAL
ACKNOWLEDGEMENTS / FUNDING HERE]

# References
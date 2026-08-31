# Changelog

## [0.2.2] - 2026-08-31

### New: Additional reconstruction options

- **Added** `load_mesh_topology(mesh_path)`: loads a mesh's own vertices,
  faces, and per-vertex normals with **no resampling**. This is the new
  default entry point for the deformation pipeline (replaces `create_pcd` as
  the "get me surface points" call for most users; `create_pcd` still exists
  unchanged for anyone who wants a lighter/resampled point cloud, or as the
  Poisson fallback's input).
- **Modified** `export_mesh_file(filename, deformed_pcd, ...)`:
  - New parameters: `method='none'` (new default), `original_faces=None`.
  - `method='none'`: requires `original_faces`; if given, saves
    `(deformed_pcd, original_faces)` directly via
    `pymeshlab.Mesh(vertex_matrix=..., face_matrix=...)` -- no reconstruction. 
  - If `method='none'` but `original_faces` is `None`: emits a
    `RuntimeWarning` and **falls back to `method='poisson'`** (the prior,
    only, behavior) automatically.
  - `method='poisson'`: unchanged prior behavior (screened Poisson
    reconstruction with estimated normals), reachable explicitly or via the
    automatic fallback above.
  - Added `method='ball_pivoting'` and `method='alpha_shape'` alongside `'none'`/  `'poisson'`. New parameters, all with defaults matching pymeshlab's own except `alpha_filtering`. `ball_radius=0.0` (0% = auto-estimated), `ball_clustering=20.0`, `ball_creasethr=90.0`, `ball_deletefaces=False`, `alpha=1.0`, `alpha_filtering='Alpha Shape'` (pymeshlab's own filter default is `'Alpha Complex'`, deliberately overridden -- `'Alpha Complex'` retains interior simplicial-complex faces, not just the outer boundary).
  - Backward compatible: existing callers that only ever passed
    `(filename, deformed_pcd)` keep working exactly as before, just now with
    a printed warning nudging them toward the no-reconstruction path.
  

## [0.2.1] - 2026-07-31

### New: Post hoc animation 
- **`animate_surface_posthoc`** - Interpolate the final displacement field and densities to create an animation for display purposes. Faster than running with tracking but not guaranteed to be accurate at intermediate steps
- **`animate_map_posthoc`** - Same function implemented for 2D maps

## [0.2.0] - 2026-04-29

### New: 2D VDERM Pipeline

- **`VDERMGrid2D`** — 2-D Lagrangian-Eulerian grid; same physics as `VDERMGrid` (diffusion + gradient advection) with one dimension removed. Compatible with the existing `run_VDERM()` function.
- **`run_VDERM_2d_with_tracking()`** — tracking run with grid and map-point exports (CSV format: `x y v_x v_y rho` for grid, `x y rho` for map points).
- **`make_initial_grid_2d()`** / **`compute_grid_dimensions_2d()`** / **`print_grid_info_2d()`** — 2-D grid utilities mirroring the 3-D equivalents.
- **`interpolate_to_map_2d()`** / **`interpolate_densities_2d()`** / **`interpolate_velocities_2d()`** — 2-D interpolation from grid to arbitrary point sets.

### New: Geographic I/O (`pip install diffusion-cartogram[2D]`)

- **`read_geojson(filepath)`** — extract 2-D point array from GeoJSON (polygon boundaries, point features, line features).
- **`read_shapefile(filepath)`** — same for Shapefiles.
- **`read_geotiff(filepath, band=1)`** — read a raster band plus coordinate arrays.
- **`density_from_geotiff(grid_2d, filepath)`** — sample a GeoTIFF raster onto a `VDERMGrid2D` density field with bilinear interpolation; handles north-up rasters and nodata automatically.
- **`write_csv_2d()`** / **`read_csv_2d()`** — simple 2-column / 3-column CSV I/O for 2-D point sets.

### New: 2D Visualization

- **`plot_map_2d()`** — scatter plot of 2-D points with optional density colouring.
- **`plot_density_field_2d()`** — heatmap of the `VDERMGrid2D` density field.
- **`plot_map_before_after()`** — side-by-side comparison of original and deformed map.
- **`animate_map_deformation_2d()`** — GIF / MP4 animation from `vderm_map/` CSV exports.
- **`animate_grid_deformation_2d()`** — GIF / MP4 animation from `vderm_grid/` CSV exports.
- **`plot_density_evolution_2d()`** — density statistics over iterations.

### Packaging

- Version bumped to **0.2.0**.
- New optional dependency group **`[2D]`**: geopandas, rasterio, shapely.
- New optional dependency group **`[3D]`**: pymeshlab.
- New optional dependency group **`[all]`**: `[2D]` + `[3D]`.
- `pyproject.toml` now points `readme` at `README_pypi.md` (no embedded GIF) for PyPI; `README.md` remains the full GitHub README.

## [0.1.0] - 2026-02-09

### Initial Release

- Core VDERM algorithm implementation
- Flexible XYZ file I/O
- Optional mesh support via PyMeshLab
- Visualization and animation tools
- Tracking with intermediate exports
- ParaView export capabilities
- Comprehensive example notebooks

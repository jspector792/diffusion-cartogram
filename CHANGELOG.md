# Changelog

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

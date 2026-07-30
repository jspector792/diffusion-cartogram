"""
2D VDERM core: VDERMGrid2D, grid utilities, I/O, and interpolation.

Mirrors the 3D core.py API but operates on a 2D (L × M) grid.  The
density diffusion and node advection physics are identical — just one
spatial dimension removed — so run_VDERM() from core.py works unchanged
with VDERMGrid2D objects.

For geographic workflows, read_geojson / read_shapefile extract 2-D
point arrays and density_from_geotiff samples a raster onto the grid.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None


def _require_geopandas(fname):
    if not HAS_GEOPANDAS:
        raise ImportError(
            f"{fname} requires geopandas for geographic file I/O.\n"
            "Install with: pip install diffusion-cartogram[2D]"
        )


def _require_rasterio(fname):
    if not HAS_RASTERIO:
        raise ImportError(
            f"{fname} requires rasterio for GeoTIFF support.\n"
            "Install with: pip install diffusion-cartogram[2D]"
        )


# ─── Grid class ──────────────────────────────────────────────────────────────

class VDERMGrid2D:
    """
    2-D Lagrangian-Eulerian grid for VDERM deformation.

    Grid nodes are both computational points (Eulerian) and material
    points (Lagrangian), mirroring VDERMGrid but for a planar (L × M) domain.

    Compatible with run_VDERM() from core.py — all required methods are
    present.  Use run_VDERM_2d_with_tracking() for intermediate exports.

    Parameters
    ----------
    shape : tuple (L, M)
        Grid dimensions along x and y axes.
    h : float
        Uniform grid spacing.
    min_bounds : array-like, shape (2,)
        Lower-left corner [x_min, y_min].
    """

    def __init__(self, shape, h, min_bounds):
        self.L, self.M = shape
        self.h = float(h)
        self.min_bounds = np.asarray(min_bounds, dtype=float)

        # Eulerian density field
        self.rho = np.ones((self.L, self.M))

        # Lagrangian node positions and velocities — shape (L*M, 2)
        self.positions = self._initialize_positions()
        self.velocities = np.zeros_like(self.positions)
        self.initial_positions = self.positions.copy()

        self.epsilon = None

    # ── initialisation ───────────────────────────────────────────────────────

    def _initialize_positions(self):
        i_idx = np.arange(self.L)
        j_idx = np.arange(self.M)
        ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
        return np.stack(
            [self.min_bounds[0] + self.h * ii,
             self.min_bounds[1] + self.h * jj],
            axis=-1
        ).reshape(-1, 2)

    # ── index helpers ─────────────────────────────────────────────────────────

    def _index_to_flat(self, i, j):
        return i * self.M + j

    def _flat_to_index(self, flat_idx):
        return flat_idx // self.M, flat_idx % self.M

    # ── density ───────────────────────────────────────────────────────────────

    def set_density(self, density_func):
        """
        Set density field from a callable or array.

        Parameters
        ----------
        density_func : callable or ndarray
            If callable: ``density_func(x, y) → scalar`` evaluated at each
            grid node (scalar x, y arguments, same signature as the 3-D
            ``density_func(x, y, z)`` pattern in VDERMGrid).
            If ndarray: shape must match (L, M).
        """
        if callable(density_func):
            xx = self.min_bounds[0] + np.arange(self.L) * self.h
            yy = self.min_bounds[1] + np.arange(self.M) * self.h
            for i, x in enumerate(xx):
                for j, y in enumerate(yy):
                    self.rho[i, j] = density_func(x, y)
        else:
            self.rho = np.asarray(density_func, dtype=float)

    def update_density(self, dt):
        """Diffuse density field one step with the 2-D heat equation."""
        rho = self.rho
        # Pad with edge values → Neumann (zero-flux) boundary conditions
        rho_pad = np.pad(rho, pad_width=1, mode='edge')
        laplacian = (
            rho_pad[2:,  1:-1] + rho_pad[:-2, 1:-1] +
            rho_pad[1:-1, 2:] + rho_pad[1:-1, :-2] -
            4.0 * rho
        )
        rho_new = rho + (dt / self.h ** 2) * laplacian
        self.epsilon = np.linalg.norm(rho_new - rho) / np.mean(rho)
        self.rho = rho_new

    # ── velocities & positions ────────────────────────────────────────────────

    def update_velocities(self):
        """Compute nodal velocities from the density gradient."""
        rho = self.rho
        rho_pad = np.pad(rho, pad_width=1, mode='edge')
        # Centred differences → ∇ρ
        grad_x = rho_pad[2:, 1:-1] - rho_pad[:-2, 1:-1]   # shape (L, M)
        grad_y = rho_pad[1:-1, 2:] - rho_pad[1:-1, :-2]   # shape (L, M)
        # v = -∇ρ / (2h · ρ)
        vx = -grad_x / (2.0 * self.h * rho)
        vy = -grad_y / (2.0 * self.h * rho)
        self.velocities[:, 0] = vx.ravel()
        self.velocities[:, 1] = vy.ravel()

    def update_positions(self, dt):
        """Advect grid nodes: positions += dt * velocities."""
        self.positions += dt * self.velocities

    def get_displacement_field(self):
        """Return per-node displacement vectors (shape: L*M × 2)."""
        return self.positions - self.initial_positions

    def compute_timestep(self):
        """
        CFL- and diffusion-stable timestep.

        2-D diffusion stability limit: dt ≤ h² / 4   (vs h² / 6 in 3-D).
        """
        max_speed = np.max(np.abs(self.velocities).sum(axis=1))
        dt_advection = (2.0 * self.h / (3.0 * max_speed)
                        if max_speed > 1e-10 else np.inf)
        dt_diffusion = self.h ** 2 / 4.0
        dt = min(dt_advection, dt_diffusion) * 0.9
        return min(dt, 0.01)


# ─── Grid utilities ──────────────────────────────────────────────────────────

def compute_grid_dimensions_2d(box_dims, max_points=4096):
    """
    Compute 2-D grid dimensions (L, M) and spacing h for a given box size.

    Parameters
    ----------
    box_dims : array-like, shape (2,)
        Desired box dimensions [x_size, y_size].
    max_points : int, default=4096
        Target total grid points.

    Returns
    -------
    shape : tuple (L, M)
    h : float
    """
    box_dims = np.asarray(box_dims, dtype=float)
    aspect = box_dims / box_dims.min()
    area_factor = np.prod(aspect)
    base = (max_points / area_factor) ** 0.5
    L, M = np.maximum(np.round(aspect * base).astype(int), 3)
    h = box_dims[0] / (L - 1)
    M = max(3, int(np.round(box_dims[1] / h)) + 1)
    return (L, M), h


def make_initial_grid_2d(pts_2d, max_points=4096, padding=(2, 2)):
    """
    Generate grid parameters automatically sized to fit a 2-D point set.

    Mirrors make_initial_grid() from core.py for 2-D inputs.

    Parameters
    ----------
    pts_2d : ndarray, shape (n_points, 2)
        Input point cloud [x, y].
    max_points : int, default=4096
        Target grid points (≈ 64² by default).
    padding : tuple (x_ratio, y_ratio), default=(2, 2)
        Grid extent as a multiple of the object bounding box.

    Returns
    -------
    grid_params : dict
        Keys: 'shape', 'h', 'min_bounds', 'max_bounds',
        'object_bounds', 'padding', 'actual_points'.
    """
    padding = np.asarray(padding, dtype=float)
    obj_min = pts_2d.min(axis=0)
    obj_max = pts_2d.max(axis=0)
    obj_size = obj_max - obj_min
    obj_center = (obj_min + obj_max) / 2.0

    box_dims = padding * obj_size
    shape, h = compute_grid_dimensions_2d(box_dims, max_points)
    L, M = shape

    grid_half = h * np.array([L - 1, M - 1]) / 2.0
    grid_min = obj_center - grid_half
    grid_max = obj_center + grid_half

    return {
        'shape': (L, M),
        'h': h,
        'min_bounds': grid_min,
        'max_bounds': grid_max,
        'object_bounds': {
            'min': obj_min,
            'max': obj_max,
            'size': obj_size,
            'center': obj_center,
        },
        'padding': tuple(padding),
        'actual_points': L * M,
    }


def print_grid_info_2d(grid_params):
    """Print a summary of a 2-D grid parameter dict."""
    L, M = grid_params['shape']
    h = grid_params['h']
    ob = grid_params['object_bounds']
    gsize = grid_params['max_bounds'] - grid_params['min_bounds']
    osize = ob['size']

    print("=" * 60)
    print("2D GRID INFORMATION")
    print("=" * 60)
    print(f"\nGrid dimensions : {L} × {M} = {grid_params['actual_points']:,} points")
    print(f"Grid spacing (h): {h:.6f}")
    print(f"\nGrid size  : [{gsize[0]:.4f}, {gsize[1]:.4f}]")
    print(f"Object size: [{osize[0]:.4f}, {osize[1]:.4f}]")
    print(f"Ratio      : [{gsize[0]/osize[0]:.2f}x, {gsize[1]/osize[1]:.2f}x]")
    print(f"\nGrid bounds:")
    print(f"  x: [{grid_params['min_bounds'][0]:.4f}, {grid_params['max_bounds'][0]:.4f}]")
    print(f"  y: [{grid_params['min_bounds'][1]:.4f}, {grid_params['max_bounds'][1]:.4f}]")
    print(f"\nObject bounds:")
    print(f"  x: [{ob['min'][0]:.4f}, {ob['max'][0]:.4f}]")
    print(f"  y: [{ob['min'][1]:.4f}, {ob['max'][1]:.4f}]")

    margins_min = ob['min'] - grid_params['min_bounds']
    margins_max = grid_params['max_bounds'] - ob['max']
    print(f"\nObject margins:")
    print(f"  x: min={margins_min[0]:.4f}, max={margins_max[0]:.4f}")
    print(f"  y: min={margins_min[1]:.4f}, max={margins_max[1]:.4f}")
    print("=" * 60)


# ─── Interpolation ────────────────────────────────────────────────────────────

def interpolate_densities_2d(map_points, grid):
    """
    Interpolate density values from a VDERMGrid2D to a 2-D point set.

    Parameters
    ----------
    map_points : ndarray, shape (n, 2)
    grid : VDERMGrid2D

    Returns
    -------
    densities : ndarray, shape (n,)
    """
    x = grid.min_bounds[0] + np.arange(grid.L) * grid.h
    y = grid.min_bounds[1] + np.arange(grid.M) * grid.h
    interp = RegularGridInterpolator(
        (x, y), grid.rho, bounds_error=False, fill_value=0.0
    )
    return interp(map_points)


def interpolate_velocities_2d(map_points, grid_params, velocity_field):
    """
    Interpolate the 2-D velocity field to arbitrary map points.

    Parameters
    ----------
    map_points : ndarray, shape (n, 2)
    grid_params : dict
        From make_initial_grid_2d; needs 'shape', 'h', 'min_bounds'.
    velocity_field : ndarray, shape (L*M, 2)

    Returns
    -------
    velocities : ndarray, shape (n, 2)
    """
    L, M = grid_params['shape']
    h = grid_params['h']
    mb = grid_params['min_bounds']
    x = mb[0] + np.arange(L) * h
    y = mb[1] + np.arange(M) * h
    vg = velocity_field.reshape(L, M, 2)
    iu = RegularGridInterpolator((x, y), vg[:, :, 0], bounds_error=False, fill_value=0.0)
    iv = RegularGridInterpolator((x, y), vg[:, :, 1], bounds_error=False, fill_value=0.0)
    return np.column_stack([iu(map_points), iv(map_points)])


def interpolate_to_map_2d(map_points, grid_params, displacement_field):
    """
    Apply grid displacements to 2-D map points and return deformed positions.

    Analogous to interpolate_to_surface() in core.py.

    Parameters
    ----------
    map_points : ndarray, shape (n, 2)
        Original point positions.
    grid_params : dict
    displacement_field : ndarray, shape (L*M, 2)
        From VDERMGrid2D.get_displacement_field().

    Returns
    -------
    deformed_points : ndarray, shape (n, 2)
    """
    L, M = grid_params['shape']
    h = grid_params['h']
    mb = grid_params['min_bounds']
    x = mb[0] + np.arange(L) * h
    y = mb[1] + np.arange(M) * h
    dg = displacement_field.reshape(L, M, 2)
    iu = RegularGridInterpolator((x, y), dg[:, :, 0], bounds_error=False, fill_value=0.0)
    iv = RegularGridInterpolator((x, y), dg[:, :, 1], bounds_error=False, fill_value=0.0)
    return map_points + np.column_stack([iu(map_points), iv(map_points)])


# ─── 2-D I/O ─────────────────────────────────────────────────────────────────

def write_csv_2d(filepath, positions, densities=None):
    """
    Write 2-D point positions (and optionally densities) to a
    space-delimited file.

    Column formats:
    - 2 columns: ``x y``
    - 3 columns: ``x y rho``

    Parameters
    ----------
    filepath : str
    positions : ndarray, shape (n, 2)
    densities : ndarray, shape (n,), optional
    """
    if densities is None:
        data = positions
    else:
        dens = densities.reshape(-1, 1) if densities.ndim == 1 else densities
        data = np.hstack([positions, dens])
    np.savetxt(filepath, data, fmt='%.6e', delimiter=' ')


def read_csv_2d(filepath):
    """
    Read 2-D point data from a space-delimited file.

    Accepts 2-column (x y) or 3-column (x y rho) files.

    Returns
    -------
    positions : ndarray, shape (n, 2)
    densities : ndarray, shape (n,) or None
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_cols = data.shape[1]
    if n_cols == 2:
        return data[:, :2], None
    elif n_cols == 3:
        return data[:, :2], data[:, 2]
    else:
        raise ValueError(
            f"Unrecognised 2-D CSV format: {n_cols} columns. "
            "Expected 2 (x y) or 3 (x y rho)."
        )


# ─── Geographic I/O ──────────────────────────────────────────────────────────

def _extract_from_geometry(geom):
    """Recursively extract (x, y) coordinate pairs from a Shapely geometry."""
    gtype = geom.geom_type
    coords = []
    if gtype == 'Point':
        coords.append([geom.x, geom.y])
    elif gtype == 'MultiPoint':
        for pt in geom.geoms:
            coords.append([pt.x, pt.y])
    elif gtype == 'LineString':
        coords.extend([[c[0], c[1]] for c in geom.coords])
    elif gtype == 'MultiLineString':
        for ln in geom.geoms:
            coords.extend([[c[0], c[1]] for c in ln.coords])
    elif gtype == 'Polygon':
        coords.extend([[c[0], c[1]] for c in geom.exterior.coords])
    elif gtype == 'MultiPolygon':
        for poly in geom.geoms:
            coords.extend([[c[0], c[1]] for c in poly.exterior.coords])
    elif gtype == 'GeometryCollection':
        for g in geom.geoms:
            coords.extend(_extract_from_geometry(g))
    return coords


def _points_from_geodataframe(gdf):
    """Return (ndarray (n,2), crs_str) from a GeoDataFrame."""
    crs = str(gdf.crs) if gdf.crs is not None else None
    all_coords = []
    for geom in gdf.geometry:
        if geom is not None:
            all_coords.extend(_extract_from_geometry(geom))
    if not all_coords:
        raise ValueError("No coordinates found in file.")
    return np.array(all_coords), crs


def read_geojson(filepath):
    """
    Extract a 2-D point array from a GeoJSON file.

    For Polygon / MultiPolygon features the exterior ring vertices are
    extracted.  Point and LineString features are also supported.

    Requires geopandas (``pip install diffusion-cartogram[2D]``).

    Parameters
    ----------
    filepath : str
        Path to a .geojson or .json file.

    Returns
    -------
    points : ndarray, shape (n, 2)
        [x, y] coordinates in the file's native CRS.
    crs : str or None
        Coordinate reference system string.

    Examples
    --------
    >>> pts, crs = vd.read_geojson('countries.geojson')
    >>> grid_params = vd.make_initial_grid_2d(pts)
    """
    _require_geopandas('read_geojson')
    gdf = gpd.read_file(filepath)
    return _points_from_geodataframe(gdf)


def read_shapefile(filepath):
    """
    Extract a 2-D point array from a Shapefile (.shp).

    Requires geopandas (``pip install diffusion-cartogram[2D]``).

    If the .shx index file is missing, it is reconstructed automatically
    via GDAL's ``SHAPE_RESTORE_SHX`` option (geometry-only recovery).
    Note that without the companion .dbf file, attribute columns will
    not be available, but geometry extraction still works.

    Parameters
    ----------
    filepath : str
        Path to .shp file.

    Returns
    -------
    points : ndarray, shape (n, 2)
    crs : str or None

    Examples
    --------
    >>> pts, crs = vd.read_shapefile('states.shp')
    >>> grid_params = vd.make_initial_grid_2d(pts)
    """
    _require_geopandas('read_shapefile')
    import os
    try:
        gdf = gpd.read_file(filepath)
    except Exception:
        # Missing .shx — try with GDAL's auto-recovery option
        old = os.environ.get('SHAPE_RESTORE_SHX')
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        try:
            gdf = gpd.read_file(filepath)
        finally:
            if old is None:
                os.environ.pop('SHAPE_RESTORE_SHX', None)
            else:
                os.environ['SHAPE_RESTORE_SHX'] = old
    return _points_from_geodataframe(gdf)


def read_geotiff(filepath, band=1):
    """
    Read a GeoTIFF raster band and return data plus coordinate arrays.

    Requires rasterio (``pip install diffusion-cartogram[2D]``).

    Parameters
    ----------
    filepath : str
    band : int, default=1

    Returns
    -------
    data : ndarray, shape (H, W)
        Raster values; NaN where nodata.
    x_coords : ndarray, shape (W,)
        Pixel-centre x coordinates (ascending).
    y_coords : ndarray, shape (H,)
        Pixel-centre y coordinates (may be descending for north-up).
    crs : str or None

    Examples
    --------
    >>> data, xs, ys, crs = vd.read_geotiff('population.tif')
    """
    _require_rasterio('read_geotiff')
    with rasterio.open(filepath) as src:
        data = src.read(band).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        H, W = data.shape
        x_coords = np.array([src.xy(0, c)[0] for c in range(W)])
        y_coords = np.array([src.xy(r, 0)[1] for r in range(H)])
        crs = str(src.crs) if src.crs is not None else None
    return data, x_coords, y_coords, crs


def density_from_geotiff(grid_2d, filepath, band=1, nodata_fill=1.0,
                         normalize=True):
    """
    Sample a GeoTIFF raster onto a VDERMGrid2D density field.

    The raster is bilinearly interpolated to every grid node.  NaN /
    nodata pixels are replaced by ``nodata_fill``.  The grid and the
    raster must share the same coordinate reference system.

    Parameters
    ----------
    grid_2d : VDERMGrid2D
    filepath : str
    band : int, default=1
    nodata_fill : float, default=1.0
        Value substituted for missing data.
    normalize : bool, default=True
        If True, rescale so that mean(ρ) = 1.

    Returns
    -------
    grid_2d : VDERMGrid2D
        The same object with its ``rho`` field updated.

    Examples
    --------
    >>> grid = vd.VDERMGrid2D(shape, h, min_bounds)
    >>> vd.density_from_geotiff(grid, 'population.tif')
    >>> result = vd.run_VDERM(grid)
    """
    data, x_coords, y_coords, _ = read_geotiff(filepath, band=band)
    data = np.where(np.isnan(data), nodata_fill, data)

    # RegularGridInterpolator requires strictly ascending axes
    if y_coords[0] > y_coords[-1]:
        y_sorted = y_coords[::-1]
        data_sorted = data[::-1, :]
    else:
        y_sorted = y_coords
        data_sorted = data

    # Transpose: data is (H, W) = (y, x); interpolator wants (x, y)
    interp = RegularGridInterpolator(
        (x_coords, y_sorted),
        data_sorted.T,
        bounds_error=False,
        fill_value=nodata_fill,
    )

    density = interp(grid_2d.positions).reshape(grid_2d.L, grid_2d.M)

    if normalize:
        mean_d = np.mean(density)
        if mean_d > 0:
            density = density / mean_d

    grid_2d.rho = density
    return grid_2d


# ─── Tracking run ─────────────────────────────────────────────────────────────

def run_VDERM_2d_with_tracking(
        grid, map_points,
        n_max=100, max_eps=0.01, dt=None,
        export_grid=False, export_grid_frequency=10,
        export_map=False, export_map_frequency=10,
        base_folder='vderm_2d_exports',
        grid_folder='vderm_grid',
        map_folder='vderm_map'):
    """
    Run 2-D VDERM deformation with optional intermediate state exports.

    Extends run_VDERM() (which works unchanged with VDERMGrid2D) by
    supporting export of:

    - **Grid states**: node positions, velocities, and densities
      (5-column CSV: ``x y v_x v_y rho``).
    - **Map point states**: deformed map points with interpolated
      densities (3-column CSV: ``x y rho``).

    Parameters
    ----------
    grid : VDERMGrid2D
        Grid with density field already set via ``grid.set_density()``.
    map_points : ndarray, shape (n, 2)
        Point set whose deformation will be tracked (e.g. polygon
        boundary extracted from a GeoJSON file).
    n_max : int, default=100
        Maximum iterations.
    max_eps : float, default=0.01
        Convergence threshold (relative L2 norm of density change).
    dt : float, optional
        Manual timestep.  Auto-computed from CFL / diffusion limits if None.
    export_grid : bool, default=False
        Export grid state CSVs.
    export_grid_frequency : int, default=10
        Export every N iterations.
    export_map : bool, default=False
        Export deformed map point CSVs.
    export_map_frequency : int, default=10
        Export every N iterations.
    base_folder : str, default='vderm_2d_exports'
        Root export directory.
    grid_folder : str, default='vderm_grid'
        Subfolder for grid exports.
    map_folder : str, default='vderm_map'
        Subfolder for map exports.

    Returns
    -------
    grid : VDERMGrid2D
        Final deformed grid.

    Examples
    --------
    >>> pts, _ = vd.read_geojson('countries.geojson')
    >>> gp = vd.make_initial_grid_2d(pts, max_points=16384)
    >>> grid = vd.VDERMGrid2D(gp['shape'], gp['h'], gp['min_bounds'])
    >>> vd.density_from_geotiff(grid, 'population.tif')
    >>> final = vd.run_VDERM_2d_with_tracking(
    ...     grid, pts,
    ...     export_grid=True, export_map=True
    ... )
    >>> deformed = vd.interpolate_to_map_2d(
    ...     pts, gp, final.get_displacement_field()
    ... )
    """
    any_exports = export_grid or export_map

    if any_exports:
        os.makedirs(base_folder, exist_ok=True)
        if export_grid:
            os.makedirs(os.path.join(base_folder, grid_folder), exist_ok=True)
        if export_map:
            os.makedirs(os.path.join(base_folder, map_folder), exist_ok=True)

    grid.update_velocities()

    if dt is None:
        dt = grid.compute_timestep()
        if dt < 0.005:
            print(f"  Warning: Very small timestep ({dt:.6f})")
            print(f"    This may indicate strong density gradients.")

    grid.epsilon = None

    # ── initial state export (iteration 0) ───────────────────────────────
    if any_exports:
        params = {'shape': (grid.L, grid.M), 'h': grid.h,
                  'min_bounds': grid.min_bounds}

        if export_grid:
            densities = grid.rho.ravel()
            vels = grid.velocities
            data = np.hstack([grid.positions, vels,
                              densities.reshape(-1, 1)])
            np.savetxt(
                os.path.join(base_folder, grid_folder,
                             'grid_iteration_0000.csv'),
                data, fmt='%.6e', delimiter=' '
            )

        if export_map:
            map_dens = interpolate_densities_2d(map_points, grid)
            write_csv_2d(
                os.path.join(base_folder, map_folder,
                             'map_iteration_0000.csv'),
                map_points, map_dens
            )

    pbar = tqdm(range(n_max), desc='Deforming (2D)')

    for iteration in pbar:
        grid.update_density(dt)
        if iteration > 0:
            grid.update_velocities()
        grid.update_positions(dt)

        # ── instability check ─────────────────────────────────────────────
        if grid.epsilon is not None:
            if grid.epsilon > 1e6 or grid.epsilon < -1e-6 or np.isnan(grid.epsilon):
                pbar.close()
                print(f"\nINSTABILITY at iteration {iteration}!")
                print(f"   Epsilon: {grid.epsilon:.3e}")
                print(f"   Current dt: {dt:.6f}")
                print(f"\n   Solution: set a smaller timestep manually:")
                print(f"   run_VDERM_2d_with_tracking(grid, ..., dt={dt/10:.6f})")
                raise RuntimeError(
                    "Numerical instability detected. "
                    "Please manually set a smaller timestep and rerun"
                )

        if grid.epsilon is not None:
            pbar.set_postfix({'eps': f'{grid.epsilon:.3e}',
                              'target': f'{max_eps:.3e}'})

        # ── periodic exports ──────────────────────────────────────────────
        if any_exports:
            params = {'shape': (grid.L, grid.M), 'h': grid.h,
                      'min_bounds': grid.min_bounds}

            if export_grid and (iteration + 1) % export_grid_frequency == 0:
                densities = grid.rho.ravel()
                data = np.hstack([grid.positions, grid.velocities,
                                  densities.reshape(-1, 1)])
                np.savetxt(
                    os.path.join(base_folder, grid_folder,
                                 f'grid_iteration_{iteration+1:04d}.csv'),
                    data, fmt='%.6e', delimiter=' '
                )

            if export_map and (iteration + 1) % export_map_frequency == 0:
                disp = grid.get_displacement_field()
                cur_map = interpolate_to_map_2d(map_points, params, disp)
                cur_dens = interpolate_densities_2d(map_points, grid)
                write_csv_2d(
                    os.path.join(base_folder, map_folder,
                                 f'map_iteration_{iteration+1:04d}.csv'),
                    cur_map, cur_dens
                )

        # ── convergence ───────────────────────────────────────────────────
        if grid.epsilon is not None and grid.epsilon <= max_eps:
            pbar.set_description('Converged')
            pbar.close()
            print(f'\nConverged at iteration {iteration + 1}')

            if any_exports:
                params = {'shape': (grid.L, grid.M), 'h': grid.h,
                          'min_bounds': grid.min_bounds}
                disp = grid.get_displacement_field()

                if export_grid:
                    densities = grid.rho.ravel()
                    data = np.hstack([grid.positions, grid.velocities,
                                      densities.reshape(-1, 1)])
                    np.savetxt(
                        os.path.join(base_folder, grid_folder,
                                     f'grid_final_iteration_{iteration+1:04d}.csv'),
                        data, fmt='%.6e', delimiter=' '
                    )

                if export_map:
                    final_map = interpolate_to_map_2d(map_points, params, disp)
                    final_dens = interpolate_densities_2d(map_points, grid)
                    write_csv_2d(
                        os.path.join(base_folder, map_folder,
                                     f'map_final_iteration_{iteration+1:04d}.csv'),
                        final_map, final_dens
                    )
            break

    if any_exports:
        print(f"\nExports saved to: {base_folder}/")
        if export_grid:
            print(f"  Grid states (x y v_x v_y rho): {grid_folder}/")
        if export_map:
            print(f"  Map point states (x y rho): {map_folder}/")

    return grid


def _interpolate_field_to_points_2d(points, grid_params, field):
    """Interpolate a scalar field defined on the (L, M) grid to arbitrary points."""
    L, M = grid_params['shape']
    h = grid_params['h']
    mb = grid_params['min_bounds']

    x = mb[0] + np.arange(L) * h
    y = mb[1] + np.arange(M) * h

    interp = RegularGridInterpolator((x, y), field, bounds_error=False, fill_value=0.0)
    return interp(points)


def animate_map_posthoc(grid, map_points, n_frames=30,
                         output_folder='vderm_2d_posthoc_exports',
                         initial_densities=None, tau=0.3,
                         map_folder='vderm_map'):
    """
    Generate an approximate 2-D map animation from a completed VDERM run
    without re-running the (expensive) vector-field interpolation at
    every intermediate step.

    Mirrors animate_surface_posthoc() from core.py — only the initial and
    final map states are ever interpolated from the grid's displacement
    field. Intermediate frames are produced by linearly interpolating
    point positions (and, if provided, densities) between those two
    states along an eased timescale that starts fast and decelerates
    (``1 - exp(-t/tau)``), approximating the true VDERM motion (fast
    initial advection, slowing as the density field equalizes).

    This is a display tool, not a physically accurate reconstruction of
    intermediate states — see the warning printed when this function
    runs, and use ``run_VDERM_2d_with_tracking`` if intermediate accuracy
    matters.

    Parameters
    ----------
    grid : VDERMGrid2D
        A grid that has already been deformed (e.g. via run_VDERM or
        run_VDERM_2d_with_tracking). Its current positions are treated
        as the final state.
    map_points : ndarray, shape (n_points, 2)
        Original (undeformed) map point set.
    n_frames : int, default=30
        Number of frames to export, including the initial and final
        frames. Must be >= 2.
    output_folder : str, default='vderm_2d_posthoc_exports'
        Base directory for exports.
    initial_densities : callable or ndarray, optional
        Density field assigned to the grid before deformation, in the
        same form accepted by ``VDERMGrid2D.set_density``: either a
        callable ``density_func(x, y) -> density`` or an array of shape
        (L, M). If given, per-point densities are eased from this
        initial field to the grid's current (final) density field and
        each frame is colored accordingly. If omitted, frames are
        exported without a density column.
    tau : float, default=0.3
        Time constant of the easing curve ``1 - exp(-t/tau)``, evaluated
        over a normalized t in [0, 1] and renormalized so the curve runs
        exactly from 0 to 1. Smaller values front-load more of the motion
        into the earliest frames.
    map_folder : str, default='vderm_map'
        Subfolder (under output_folder) that exports are written to.
        Matches the default subfolder expected by
        ``animate_map_deformation_2d``, so the output of this function
        can be fed directly into it.

    Returns
    -------
    frame_paths : list of str
        Paths to the exported .csv files, in frame order.

    Notes
    -----
    Exported files use the same CSV format ('x y' or 'x y rho') and the
    same 'map_iteration_NNNN.csv' / 'map_final_iteration_NNNN.csv' naming
    convention as run_VDERM_2d_with_tracking's map exports, so they can be
    visualized with animate_map_deformation_2d() without any extra
    arguments.

    Examples
    --------
    >>> final_grid = vd.run_VDERM(grid, n_max=500)
    >>> vd.animate_map_posthoc(final_grid, pts, n_frames=40,
    ...                        output_folder='my_cartogram_posthoc',
    ...                        initial_densities=grid.rho.copy())
    >>> vd.animate_map_deformation_2d('my_cartogram_posthoc')
    """
    print("=" * 70)
    print("POST-HOC ANIMATION NOTICE")
    print("=" * 70)
    print("This post-hoc animation tool is meant for display purposes only and")
    print("may not be faithful to the real deformation during intermediate")
    print("steps. If you are interested in viewing the process of the real")
    print("deformation, use the run_VDERM_2d_with_tracking function, which is")
    print("slower, but is guaranteed to be accurate at each frame.")
    print("=" * 70)

    if n_frames < 2:
        raise ValueError("n_frames must be >= 2 (need at least an initial and final frame)")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    out_dir = os.path.join(output_folder, map_folder)
    os.makedirs(out_dir, exist_ok=True)

    params = {'shape': (grid.L, grid.M), 'h': grid.h, 'min_bounds': grid.min_bounds}
    displacement_field = grid.get_displacement_field()

    initial_map = map_points
    final_map = interpolate_to_map_2d(map_points, params, displacement_field)

    # Eased timescale: fast at first, decelerating; renormalized to hit exactly 0 and 1
    t = np.linspace(0.0, 1.0, n_frames)
    ease = (1.0 - np.exp(-t / tau))
    ease = ease / ease[-1]

    have_densities = initial_densities is not None
    if have_densities:
        final_map_densities = interpolate_densities_2d(map_points, grid)

        if callable(initial_densities):
            xs = grid.min_bounds[0] + np.arange(grid.L) * grid.h
            ys = grid.min_bounds[1] + np.arange(grid.M) * grid.h
            initial_rho = np.zeros_like(grid.rho)
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    initial_rho[i, j] = initial_densities(x, y)
        else:
            initial_rho = np.asarray(initial_densities, dtype=float)
            if initial_rho.shape != grid.rho.shape:
                raise ValueError(
                    f"initial_densities array shape {initial_rho.shape} must match "
                    f"grid shape {grid.rho.shape}"
                )
        initial_map_densities = _interpolate_field_to_points_2d(map_points, params, initial_rho)

    frame_paths = []
    for frame_idx, alpha in enumerate(ease):
        positions_i = initial_map + alpha * (final_map - initial_map)

        if have_densities:
            densities_i = initial_map_densities + alpha * (final_map_densities - initial_map_densities)
        else:
            densities_i = None

        if frame_idx == n_frames - 1:
            filename = f'map_final_iteration_{frame_idx:04d}.csv'
        else:
            filename = f'map_iteration_{frame_idx:04d}.csv'

        filepath = os.path.join(out_dir, filename)
        write_csv_2d(filepath, positions_i, densities_i)
        frame_paths.append(filepath)

    print(f"\n{n_frames} post-hoc animation frames saved to: {out_dir}/")

    return frame_paths

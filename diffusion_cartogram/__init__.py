"""
diffusion-cartogram: Volumetric Density-Equalizing Reference Map

Python implementation of the VDERM algorithm for 3-D shape deformation
(Choi & Rycroft 2020) and — new in v2.0 — 2-D density-equalizing
cartogram deformation from GeoJSON / Shapefile inputs.

3-D Quick Start
---------------
>>> import diffusion_cartogram as vd
>>>
>>> surface_pts, normals = vd.create_pcd('mesh.stl', n_pts=25000)
>>> gp = vd.make_initial_grid(surface_pts, max_points=32768)
>>> grid = vd.VDERMGrid(gp['shape'], gp['h'], gp['min_bounds'])
>>> grid.set_density(my_density_function)
>>> result = vd.run_VDERM(grid, n_max=100, max_eps=0.02)
>>> deformed = vd.interpolate_to_surface(surface_pts, gp,
...                                      result.get_displacement_field())

2-D Quick Start
---------------
>>> pts, crs = vd.read_geojson('countries.geojson')
>>> gp = vd.make_initial_grid_2d(pts, max_points=16384)
>>> grid = vd.VDERMGrid2D(gp['shape'], gp['h'], gp['min_bounds'])
>>> vd.density_from_geotiff(grid, 'population.tif')
>>> result = vd.run_VDERM(grid)          # run_VDERM works for both 2D and 3D
>>> deformed = vd.interpolate_to_map_2d(pts, gp,
...                                     result.get_displacement_field())
>>> vd.plot_map_2d(deformed, title='Population Cartogram')
"""

__version__ = '0.2.1'

# Core VDERM classes and algorithms
from .core import (
    VDERMGrid,
    run_VDERM,
    run_VDERM_with_tracking,
    animate_surface_posthoc,
)

# I/O functions
from .core import (
    write_xyz,
    read_xyz,
    create_pcd,
    load_mesh_topology,
    export_mesh_file,
    export_mesh_vtk,
)

# Grid utilities
from .core import (
    compute_grid_dimensions,
    make_initial_grid,
    print_grid_info,
)

# interpolation and remeshing utilities
from .core import (
    HAS_PYMESHLAB,
    interpolate_densities,
    interpolate_to_surface,
    interpolate_velocities,
)

# Visualization functions (optional - only if matplotlib available)
try:
    from .visualization import (
        animate_grid_deformation,
        animate_surface_deformation,
        create_side_by_side_animation,
        plot_density_evolution,
        export_all_to_paraview,
        export_meshes_to_paraview,
        export_surface_to_paraview,
        export_grid_to_paraview,
        plot_pcd,
        interactive_pcd_plot,
    )
    _has_visualization = True
except ImportError:
    _has_visualization = False

__all__ = [
    # Core classes and algorithms
    'VDERMGrid',
    'run_VDERM',
    'run_VDERM_with_tracking',
    'animate_surface_posthoc',

    # I/O
    'write_xyz',
    'read_xyz',
    'create_pcd',
    'load_mesh_topology',
    'export_mesh_file',
    'export_mesh_vtk',
    
    # Grid utilities
    'compute_grid_dimensions',
    'make_initial_grid',
    'print_grid_info',
    
    # mesh utilities
    'HAS_PYMESHLAB',
    'interpolate_densities',
    'interpolate_to_surface',
    'interpolate_velocities',
]

# Add visualization functions to __all__ if available
if _has_visualization:
    __all__.extend([
        'animate_grid_deformation',
        'animate_surface_deformation',
        'create_side_by_side_animation',
        'plot_density_evolution',
        'export_grid_to_paraview',
        'export_surface_to_paraview',
        'export_meshes_to_paraview',
        'export_all_to_paraview',
        'plot_pcd',
        'interactive_pcd_plot',
    ])

# ── 2-D API ──────────────────────────────────────────────────────────────────

# 2-D core classes and algorithms
from .core_2d import (
    VDERMGrid2D,
    run_VDERM_2d_with_tracking,
    animate_map_posthoc,
)

# 2-D grid utilities
from .core_2d import (
    compute_grid_dimensions_2d,
    make_initial_grid_2d,
    print_grid_info_2d,
)

# 2-D interpolation
from .core_2d import (
    interpolate_densities_2d,
    interpolate_velocities_2d,
    interpolate_to_map_2d,
)

# 2-D I/O (always available)
from .core_2d import (
    write_csv_2d,
    read_csv_2d,
    HAS_GEOPANDAS,
    HAS_RASTERIO,
    read_geojson,
    read_shapefile,
    read_geotiff,
    density_from_geotiff,
)

# 2-D visualization (optional)
try:
    from .visualization_2d import (
        plot_map_2d,
        plot_density_field_2d,
        plot_map_before_after,
        animate_map_deformation_2d,
        animate_grid_deformation_2d,
        plot_density_evolution_2d,
    )
    _has_visualization_2d = True
except ImportError:
    _has_visualization_2d = False

__all__ += [
    # 2-D classes and algorithms
    'VDERMGrid2D',
    'run_VDERM_2d_with_tracking',
    'animate_map_posthoc',
# 2-D grid utilities
    'compute_grid_dimensions_2d',
    'make_initial_grid_2d',
    'print_grid_info_2d',
    # 2-D interpolation
    'interpolate_densities_2d',
    'interpolate_velocities_2d',
    'interpolate_to_map_2d',
    # 2-D I/O
    'write_csv_2d',
    'read_csv_2d',
    'HAS_GEOPANDAS',
    'HAS_RASTERIO',
    'read_geojson',
    'read_shapefile',
    'read_geotiff',
    'density_from_geotiff',
]

if _has_visualization_2d:
    __all__ += [
        'plot_map_2d',
        'plot_density_field_2d',
        'plot_map_before_after',
        'animate_map_deformation_2d',
        'animate_grid_deformation_2d',
        'plot_density_evolution_2d',
    ]
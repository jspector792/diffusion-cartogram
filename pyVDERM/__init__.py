"""
VDERM: Volumetric Density-Equalizing Reference Map

A Python implementation of the VDERM algorithm for 3D shape deformation
based on the method by Choi & Rycroft (2020).

Basic Usage
-----------
>>> import vderm
>>> 
>>> # Load mesh and create surface point cloud
>>> surface_pts, normals = vderm.create_pcd('mesh.stl', n_pts=25000)
>>> 
>>> # Create computational grid
>>> grid, params = vderm.make_initial_grid(surface_pts, max_points=32768)
>>> 
>>> # Set up VDERM grid and density field
>>> vderm_grid = vderm.VDERMGrid(params['shape'], params['h'], params['min_bounds'])
>>> vderm_grid.set_density(my_density_function)
>>> 
>>> # Run deformation
>>> final_grid = vderm.run_VDERM(vderm_grid, n_max=100, max_eps=0.02)
>>> 
>>> # Interpolate to surface and export
>>> final_surface = final_grid.interpolate_to_points(surface_pts)
>>> vderm.export_mesh('output.stl', final_surface, normals)
"""

__version__ = '0.1.5'

# Core VDERM classes and algorithms
from .core import (
    VDERMGrid,
    run_VDERM,
    run_VDERM_with_tracking,
)

# I/O functions
from .core import (
    write_xyz,
    read_xyz,
    create_pcd,
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
    
    # I/O
    'write_xyz',
    'read_xyz',
    'write_xyz_underlying',
    'read_xyz_underlying',
    'create_pcd',
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
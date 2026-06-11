import numpy as np
from scipy import interpolate
from tqdm import tqdm
import os
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
try:
    import pymeshlab as ml
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False
    ml = None
    
def _require_pymeshlab(function_name):
    """Raise error if pymeshlab not available."""
    if not HAS_PYMESHLAB:
        raise ImportError(
            f"{function_name} requires pymeshlab for mesh operations.\n"
            f"Install with: pip install pymeshlab\n"
            f"Or install with 3-D mesh support: pip install diffusion-cartogram[3D]"
        )

def create_pcd(mesh_path, n_pts=25_000, sampling_method='poisson'):
    """
    Parameters
    ----------
    mesh_path : str
        mesh file path
    n_pts : int
        number of points to sample
    sampling_method : str, 'poisson' or 'uniform'
        pymeshlab sampling method
    
    Returns
    -------
    outs : ndarray, shape (n_points, 3)
        Point positions [x, y, z]
    normals : ndarray, shape (n_points, 3)
        Normal vectors [n_x, n_y, n_z]

    """
    _require_pymeshlab('create_pcd')
    
    ms = ml.MeshSet()
    ms.load_new_mesh(mesh_path)
    
    if sampling_method == 'poisson':
        ms.generate_sampling_poisson_disk(samplenum=n_pts)
    else:  # uniform
        ms.generate_sampling_montecarlo(samplenum=n_pts)
    
    current_mesh = ms.current_mesh()
    out = current_mesh.vertex_matrix()
    norms = current_mesh.vertex_normal_matrix()
    
    return out, norms
    
            
def write_xyz(filepath, positions, normals=None, densities=None):
    """
    Write grid positions (and optionally normal vectors and densities) to space-delimited .xyz file.
    
    Parameters
    ----------
    filepath : str
        Output file path
    positions : ndarray, shape (n_points, 3)
        Point positions [x, y, z]
    normals : ndarray, shape (n_points, 3), optional
        Normal vectors [n_x, n_y, n_z]
    densities : ndarray, shape (n_points,), optional
        Density values at each grid node. If None, only positions are written.
    
    Examples
    --------
    >>> positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    >>> densities = np.array([1.0, 1.5, 2.0])
    >>> write_xyz('grid.xyz', positions, normals, densities)
    
    Output file format:
    x y z n_x n_y n_z rho
    0.0 0.0 0.0 0.0 0.0 1.0 1.0
    1.0 0.0 0.0 0.0 1.0 0.0 1.5
    0.0 1.0 0.0 1.0 0.0 0.0 2.0
    """
    #  Validate inputs
    n_points = len(positions)
    
    if normals is not None and len(normals) != n_points:
        raise ValueError(
            f"Normals length ({len(normals)}) must match positions ({n_points})"
        )
    
    if densities is not None and len(densities) != n_points:
        raise ValueError(
            f"Densities length ({len(densities)}) must match positions ({n_points})"
        )
    
    # Build data array based on what's provided
    data_columns = [positions]
    
    if normals is not None:
        data_columns.append(normals)
    
    if densities is not None:
        # Reshape densities to column vector if needed
        if densities.ndim == 1:
            densities = densities.reshape(-1, 1)
        data_columns.append(densities)
    
    # Stack all columns
    data = np.hstack(data_columns)
    
    # Save with numpy
    np.savetxt(filepath, data, fmt='%.6e', delimiter=' ')


def read_xyz(filepath):
    """
    Read point data from space-delimited .xyz file.
    
    Automatically detects file format based on number of columns:
    - 3 columns: x y z (positions only)
    - 4 columns: x y z rho (positions + densities)
    - 6 columns: x y z n_x n_y n_z (positions + normals)
    - 7 columns: x y z n_x n_y n_z rho (positions + normals + densities)
    
    Parameters
    ----------
    filepath : str
        Input file path
    
    Returns
    -------
    positions : ndarray, shape (n_points, 3)
        Point positions [x, y, z]
    normals : ndarray, shape (n_points, 3) or None
        Normal vectors [n_x, n_y, n_z], or None if not in file
    densities : ndarray, shape (n_points,) or None
        Density values, or None if not in file
    
    Raises
    ------
    ValueError
        If file doesn't have a recognized format (3, 4, 6, or 7 columns)
    
    Examples
    --------
    >>> # File with positions only
    >>> positions, normals, densities = read_xyz('points.xyz')
    >>> # normals is None, densities is None
    
    >>> # File with positions + normals
    >>> positions, normals, densities = read_xyz('surface.xyz')
    >>> # normals is array, densities is None
    
    >>> # File with positions + densities
    >>> positions, normals, densities = read_xyz('grid.xyz')
    >>> # normals is None, densities is array
    
    >>> # File with everything
    >>> positions, normals, densities = read_xyz('complete.xyz')
    >>> # Both normals and densities are arrays
    """
    # Load data
    data = np.loadtxt(filepath)
    
    # Handle single point case
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_cols = data.shape[1]
    
    # Extract data based on number of columns
    if n_cols == 3:
        # x y z
        positions = data[:, 0:3]
        normals = None
        densities = None
        
    elif n_cols == 4:
        # x y z rho
        positions = data[:, 0:3]
        normals = None
        densities = data[:, 3]
        
    elif n_cols == 6:
        # x y z n_x n_y n_z
        positions = data[:, 0:3]
        normals = data[:, 3:6]
        densities = None
        
    elif n_cols == 7:
        # x y z n_x n_y n_z rho
        positions = data[:, 0:3]
        normals = data[:, 3:6]
        densities = data[:, 6]
        
    else:
        raise ValueError(
            f"Unrecognized file format: {n_cols} columns.\n"
            f"Expected 3 (xyz), 4 (xyz+rho), 6 (xyz+normals), or 7 (xyz+normals+rho).\n"
            f"File: {filepath}"
        )
    
    return positions, normals, densities
                
    
def compute_grid_dimensions(box_dims, max_points=32768):
    """
    Compute grid dimensions (L, M, N) and spacing (h) for a given box size.
    
    Creates a grid with approximately max_points that maintains uniform
    spacing and respects the box aspect ratio.
    
    Parameters
    ----------
    box_dims : array-like, shape (3,)
        Desired box dimensions [x_size, y_size, z_size]
    max_points : int, default=32768
        Target number of grid points
    
    Returns
    -------
    shape : tuple (L, M, N)
        Grid dimensions
    h : float
        Grid spacing (uniform in all directions)
    
    Examples
    --------
    >>> # For a 2×3×4 box with ~10k points
    >>> shape, h = compute_grid_dimensions([2, 3, 4], max_points=10000)
    >>> print(f"Grid: {shape}, spacing: {h:.4f}")
    """
    # Compute aspect ratios
    aspect_ratios = box_dims / box_dims.min()
    
    # Find grid dimensions that give approximately max_points
    volume_factor = np.prod(aspect_ratios)
    base_dim = (max_points / volume_factor) ** (1/3)
    
    # Compute dimensions based on aspect ratios
    dimensions = aspect_ratios * base_dim
    
    # Round to integers (at least 3 points per dimension)
    L, M, N = np.maximum(np.round(dimensions).astype(int), 3)
    
    # Compute uniform grid spacing
    h = box_dims[0] / (L - 1)
    
    # Adjust other dimensions to maintain uniform spacing
    M = max(3, int(np.round(box_dims[1] / h)) + 1)
    N = max(3, int(np.round(box_dims[2] / h)) + 1)
    
    return (L, M, N), h
    
def make_initial_grid(pcd, max_points=32768, padding=(2, 2, 2)):
    """
    Generate the parameters for a computational grid automatically sized to fit a point cloud object.
    
    The grid will:
    - Have the same aspect ratio as the object's bounding box
    - Be padded by specified ratios in each dimension
    - Have approximately max_points total grid points
    - Be centered on the object
    
    Parameters
    ----------
    pcd : ndarray, shape (n_points, 3)
        Point cloud defining the object (from create_pcd)
    max_points : int, default=32768
        Target maximum number of grid points. The function will create
        a grid with approximately this many points while maintaining
        the aspect ratio. Default is 32^3 = 32768.
    padding : tuple of floats (x_ratio, y_ratio, z_ratio), default=(2, 2, 2)
        Padding ratios for each axis. Grid size along each axis will be
        ratio * object_size for that axis. Default (2, 2, 2) means grid
        is 2x object size in each dimension (object uses central 1/3).
        Examples:
        - (2, 2, 2): Uniform 2x padding (default)
        - (3, 3, 3): More accomodating 3x padding 
        - (5, 3, 3): More padding in X direction
    
    Returns
    -------
    grid_params : dict
        Dictionary containing:
        - 'shape': tuple (L, M, N) - grid dimensions
        - 'h': float - grid spacing
        - 'min_bounds': ndarray - lower corner [x_min, y_min, z_min]
        - 'max_bounds': ndarray - upper corner [x_max, y_max, z_max]
        - 'object_bounds': dict with 'min' and 'max' - original object bounds
        - 'padding': tuple - the padding ratios used
    
    Examples
    --------
    >>> # Default 2x padding
    >>> pcd, normals = create_pcd('mesh.stl', n_pts=25000)
    >>> grid, params = make_initial_grid(pcd, max_points=32768)
    
    >>> # Tighter fit with 1.5x padding
    >>> grid, params = make_initial_grid(pcd, max_points=32768, padding=(1.5, 1.5, 1.5))
    
    >>> # Asymmetric padding (more space in Z direction)
    >>> grid, params = make_initial_grid(pcd, max_points=32768, padding=(3, 3, 5))
    """
    
    # Convert padding to array
    padding = np.array(padding)
    
    # Get object bounding box
    obj_min = pcd.min(axis=0)
    obj_max = pcd.max(axis=0)
    obj_size = obj_max - obj_min
    obj_center = (obj_min + obj_max) / 2
    
    # Compute box size
    padding = np.array(padding)
    box_dims = padding * obj_size
    
    # Use helper to get grid dimensions
    shape, h = compute_grid_dimensions(box_dims, max_points)
    
    # Round to integers (at least 3 points per dimension for derivatives)
    L, M, N = shape
    
    # Compute grid bounds (centered on object)
    grid_half_size = h * np.array([L-1, M-1, N-1]) / 2
    grid_min = obj_center - grid_half_size
    grid_max = obj_center + grid_half_size
    
    # Package parameters
    grid_params = {
        'shape': (L, M, N),
        'h': h,
        'min_bounds': grid_min,
        'max_bounds': grid_max,
        'object_bounds': {
            'min': obj_min,
            'max': obj_max,
            'size': obj_size,
            'center': obj_center
        },
        'padding': tuple(padding),
        'actual_points': L * M * N
    }
    
    return grid_params


def print_grid_info(grid_params):
    """
    Print information about the generated grid.
    
    Parameters
    ----------
    grid_params : dict
        Dictionary returned by make_initial_grid
    """
    L, M, N = grid_params['shape']
    h = grid_params['h']
    obj_bounds = grid_params['object_bounds']
    
    print("=" * 70)
    print("GRID INFORMATION")
    print("=" * 70)
    print(f"\nGrid dimensions: {L} × {M} × {N} = {grid_params['actual_points']:,} points")
    print(f"Grid spacing (h): {h:.6f}")
    
    grid_size = grid_params['max_bounds'] - grid_params['min_bounds']
    obj_size = obj_bounds['size']
    
    print(f"\nGrid size:   [{grid_size[0]:.4f}, {grid_size[1]:.4f}, {grid_size[2]:.4f}]")
    print(f"Object size: [{obj_size[0]:.4f}, {obj_size[1]:.4f}, {obj_size[2]:.4f}]")
    print(f"Ratio:       [{grid_size[0]/obj_size[0]:.2f}x, "
          f"{grid_size[1]/obj_size[1]:.2f}x, {grid_size[2]/obj_size[2]:.2f}x]")
    
    print(f"\nGrid bounds:")
    print(f"  x: [{grid_params['min_bounds'][0]:.4f}, {grid_params['max_bounds'][0]:.4f}]")
    print(f"  y: [{grid_params['min_bounds'][1]:.4f}, {grid_params['max_bounds'][1]:.4f}]")
    print(f"  z: [{grid_params['min_bounds'][2]:.4f}, {grid_params['max_bounds'][2]:.4f}]")
    
    print(f"\nObject bounds:")
    print(f"  x: [{obj_bounds['min'][0]:.4f}, {obj_bounds['max'][0]:.4f}]")
    print(f"  y: [{obj_bounds['min'][1]:.4f}, {obj_bounds['max'][1]:.4f}]")
    print(f"  z: [{obj_bounds['min'][2]:.4f}, {obj_bounds['max'][2]:.4f}]")
    
    # Check margins
    margins_min = obj_bounds['min'] - grid_params['min_bounds']
    margins_max = grid_params['max_bounds'] - obj_bounds['max']
    
    print(f"\nObject margins (distance from grid boundary):")
    print(f"  x: min={margins_min[0]:.4f}, max={margins_max[0]:.4f}")
    print(f"  y: min={margins_min[1]:.4f}, max={margins_max[1]:.4f}")
    print(f"  z: min={margins_min[2]:.4f}, max={margins_max[2]:.4f}")
    
    # Verify padding relationship
    print(f"\nVerification for padding ratios:")
    actual_ratios = grid_size / obj_size
    print(f"  x: {actual_ratios[0]:.3f}x")
    print(f"  y: {actual_ratios[1]:.3f}x")
    print(f"  z: {actual_ratios[2]:.3f}x")
    
    print("=" * 70)
    
class VDERMGrid:
    """
    Combined Lagrangian-Eulerian grid for VDERM deformation.
    Grid nodes are both computational points (Eulerian) and material points (Lagrangian).
    """
    
    def __init__(self, shape, h, min_bounds):
        """
        Parameters
        ----------
        shape : tuple (L, M, N)
            Grid dimensions
        h : float
            Grid spacing
        min_bounds : array-like [x_min, y_min, z_min]
            Lower corner of grid
        """
        self.L, self.M, self.N = shape
        self.h = h
        self.min_bounds = np.array(min_bounds)
        
        # Density field (Eulerian)
        self.rho = np.ones(shape)
        
        # Node positions (Lagrangian) - flattened for easier iteration
        self.positions = self._initialize_positions()
        self.velocities = np.zeros_like(self.positions)
        
        # Store initial positions for computing displacement field
        self.initial_positions = self.positions.copy()
    
    def _initialize_positions(self):
        """Create initial grid node positions"""
        positions = []
        for i in range(self.L):
            for j in range(self.M):
                for k in range(self.N):
                    pos = self.min_bounds + self.h * np.array([i, j, k])
                    positions.append(pos)
        return np.array(positions)
    
    def _index_to_flat(self, i, j, k):
        """Convert 3D grid index to flat array index"""
        return i * (self.M * self.N) + j * self.N + k
    
    def _flat_to_index(self, flat_idx):
        """Convert flat array index to 3D grid index"""
        i = flat_idx // (self.M * self.N)
        remainder = flat_idx % (self.M * self.N)
        j = remainder // self.N
        k = remainder % self.N
        return i, j, k
    
    def set_density(self, density_func):
        """
        Set density field using a function or array.
        
        Parameters
        ----------
        density_func : callable or array
            If callable: density_func(x, y, z) -> density
            If array: shape must match (L, M, N)
        """
        if callable(density_func):
            for idx, pos in enumerate(self.positions):
                i, j, k = self._flat_to_index(idx)
                self.rho[i, j, k] = density_func(*pos)
        else:
            self.rho = np.array(density_func)
    
    def update_density(self, dt):
        """Diffuse density field using heat equation"""
        rho_new = np.zeros_like(self.rho)
        
        for i in range(self.L):
            for j in range(self.M):
                for k in range(self.N):
                    # Get neighbor indices with boundary conditions
                    i_p, i_m = self._get_neighbors(i, self.L)
                    j_p, j_m = self._get_neighbors(j, self.M)
                    k_p, k_m = self._get_neighbors(k, self.N)
                    
                    # Laplacian
                    laplacian = (
                        self.rho[i_p, j, k] + self.rho[i_m, j, k] +
                        self.rho[i, j_p, k] + self.rho[i, j_m, k] +
                        self.rho[i, j, k_p] + self.rho[i, j, k_m] -
                        6 * self.rho[i, j, k]
                    )
                    
                    rho_new[i, j, k] = self.rho[i, j, k] + (dt / self.h**2) * laplacian
        
        # Convergence metric
        self.epsilon = np.linalg.norm(rho_new - self.rho) / np.mean(self.rho)
        self.rho = rho_new
    
    def _get_neighbors(self, idx, max_idx):
        """Get neighbor indices with boundary conditions"""
        if idx == max_idx - 1:
            idx_plus = idx
            idx_minus = idx - 1
        elif idx == 0:
            idx_plus = idx + 1
            idx_minus = idx
        else:
            idx_plus = idx + 1
            idx_minus = idx - 1
        return idx_plus, idx_minus
    
    def update_velocities(self):
        """Compute velocity for each node from density gradient"""
        for idx in range(len(self.positions)):
            i, j, k = self._flat_to_index(idx)
            
            # Get neighbors
            i_p, i_m = self._get_neighbors(i, self.L)
            j_p, j_m = self._get_neighbors(j, self.M)
            k_p, k_m = self._get_neighbors(k, self.N)
            
            # Density gradient (centered difference)
            grad_rho = np.array([
                self.rho[i_p, j, k] - self.rho[i_m, j, k],
                self.rho[i, j_p, k] - self.rho[i, j_m, k],
                self.rho[i, j, k_p] - self.rho[i, j, k_m]
            ])
            
            # Velocity from gradient
            rho_at_node = self.rho[i, j, k]
            self.velocities[idx] = -grad_rho / (2 * self.h * rho_at_node)
    
    def update_positions(self, dt):
        """Move grid nodes based on velocities"""
        self.positions += dt * self.velocities
    
    def get_displacement_field(self):
        """
        Compute displacement vectors for each grid node.
        
        Returns
        -------
        displacements : ndarray, shape (n_nodes, 3)
            Displacement vectors: final_position - initial_position
        """
        return self.positions - self.initial_positions
    
    def compute_timestep(self):
        """Compute stable timestep based on velocities"""
        # Compute both stability limits
        max_speed = np.max(np.abs(self.velocities).sum(axis=1))
        
        if max_speed > 1e-10:
            dt_advection = 2 * self.h / (3 * max_speed)
        else:
            dt_advection = np.inf
        
        # Diffusion stability (most restrictive for fine grids)
        dt_diffusion = self.h**2 / 6
        
        # Take minimum with safety factor of 0.9
        dt = min(dt_advection, dt_diffusion) * 0.9
        dt = min(dt, 0.01)  # Cap at 0.01
        return min(dt, 0.01)

def run_VDERM(grid, n_max=100, max_eps=0.01, dt=None):
    """
    Run VDERM deformation algorithm on a computational grid.
    
    Iteratively deforms the grid by diffusing the density field and advecting
    grid nodes according to the resulting velocity field. Continues until either
    convergence (epsilon ≤ max_eps) or maximum iterations reached.
    
    Parameters
    ----------
    grid : VDERMGrid
        Grid object with density field already set via grid.set_density()
    n_max : int, default=100
        Maximum number of iterations to perform
    max_eps : float, default=0.01
        Convergence threshold. Algorithm stops when the relative change in
        density field (epsilon) falls below this value.
    dt : float, optional
        Manual timestep override. If None, timestep is computed automatically
        based on CFL and diffusion stability conditions.
        
        **Note:** For density fields with strong, sustained gradients (e.g., 
        linear gradients), automatic timestep selection may be 
        insufficient and numerical instability can occur. Symptoms include 
        epsilon becoming larger over time, or negative.
        
        If instability occurs, manually set a smaller timestep:
        - Typical values: dt=0.001 to 0.01
        - Strong gradients: dt=0.0001 to 0.001
    
    Returns
    -------
    grid : VDERMGrid
        The deformed grid object. Grid positions have been updated to reflect
        the deformation. Original positions are preserved in grid.initial_positions.
    
    Raises
    ------
    RuntimeError
        If numerical instability is detected (epsilon > 1e10 or epsilon < 0)
    
    Notes
    -----
    The algorithm alternates between:
    1. Diffusing the density field (heat equation)
    2. Computing velocities from density gradients
    3. Advecting grid nodes according to velocities
    
    Convergence is measured by epsilon, the relative L2 norm of density change:
        epsilon = ||ρ^(n+1) - ρ^n|| / mean(ρ^n)
    
    Examples
    --------
    >>> # Basic usage with automatic timestep
    >>> grid = vderm.VDERMGrid(shape=(32, 32, 32), h=0.1, min_bounds=[0, 0, 0])
    >>> grid.set_density(my_density_function)
    >>> deformed_grid = vderm.run_VDERM(grid, n_max=100, max_eps=0.02)
    >>> print(f"Converged with epsilon={deformed_grid.epsilon:.3e}")
    
    >>> # Manual timestep for strong gradients
    >>> grid.set_density(lambda x, y, z: 1.0 + 5.0 * x)  # Strong linear gradient
    >>> deformed_grid = vderm.run_VDERM(grid, dt=0.001)  # Smaller dt for stability
    
    >>> # Relaxed convergence for faster results
    >>> deformed_grid = vderm.run_VDERM(grid, n_max=50, max_eps=0.05)
    
    See Also
    --------
    run_VDERM_with_tracking : Extended version with export capabilities
    VDERMGrid.set_density : Set the density field before running
    """
    
    # Initial velocities
    grid.update_velocities()
    
    # Compute timestep if not provided
    if dt is None:
        dt = grid.compute_timestep()
        
        # Warn if timestep is very small
        if dt < 0.005:
            print(f"  ⚠ Warning: Very small timestep ({dt:.6f})")
            print(f"    This may indicate strong density gradients.")
            print(f"    Expect longer computation time.")
    
    grid.epsilon = None
    
    pbar = tqdm(range(n_max), desc='Deforming')
    
    for iteration in pbar:
        grid.update_density(dt)
        
        if iteration > 0:
            grid.update_velocities()
        
        grid.update_positions(dt)
        
        # Early instability detection (check every iteration)
        if grid.epsilon is not None:
            if grid.epsilon > 1e6 or grid.epsilon < -1e-6 or np.isnan(grid.epsilon):
                pbar.close()
                print(f"\n❌ INSTABILITY at iteration {iteration}!")
                print(f"   Epsilon: {grid.epsilon:.3e}")
                print(f"   Current dt: {dt:.6f}")
                print(f"   Grid spacing h: {grid.h:.6f}")
                print(f"\n   Solution: Manually set smaller timestep:")
                print(f"   vderm.run_VDERM(grid, dt={dt/10:.6f})")
                raise RuntimeError("Numerical instability detected. Please manually set a smaller timestep and rerun")
        
        if grid.epsilon is not None:
            pbar.set_postfix({'ε': f'{grid.epsilon:.3e}', 'target': f'{max_eps:.3e}'})
        
        if grid.epsilon is not None and grid.epsilon <= max_eps:
            pbar.set_description('Converged')
            pbar.close()
            print(f'\nConverged at iteration {iteration}')
            break
    
    return grid
    

def run_VDERM_with_tracking(grid, surface_points,
                            n_max=100, max_eps=0.01, dt=None,
                            export_grid=False, export_grid_frequency=10,
                            export_surface=False, export_surface_frequency=10,
                            export_mesh=False, export_mesh_frequency=20,
                            mesh_depth=8,
                            mesh_format='stl',
                            base_folder='vderm_exports',
                            grid_folder='vderm_grid',
                            surface_folder='vderm_surface',
                            mesh_folder='vderm_mesh'):
    """
    Run VDERM deformation with optional tracking of grid, surface, and mesh states.
    
    This function extends run_VDERM() by allowing intermediate exports of:
    - Grid states (positions + densities)
    - Interpolated surface point clouds (positions + normals + densities)
    - Reconstructed meshes (STL or VTK format)
    
    Each export type has independent frequency control and output folders.
    All exports include complete data.
    
    Parameters
    ----------
    grid : VDERMGrid
        Grid object with initial density field set
    surface_points : ndarray, shape (n_points, 3)
        Original surface point cloud positions
    n_max : int, default=100
        Maximum VDERM iterations
    max_eps : float, default=0.01
        Convergence threshold for epsilon
    dt : float, optional
        timestep can be manually assigned if the auto assigned timestep is not sufficient
    export_grid : bool, default=False
        If True, export grid positions and densities
    export_grid_frequency : int, default=10
        Export grid every N iterations
    
    export_surface : bool, default=False
        If True, export interpolated surface point clouds (with normals and densities)
    export_surface_frequency : int, default=10
        Export surface every N iterations
    
    export_mesh : bool, default=False
        If True, export reconstructed meshes (slowest option)
    export_mesh_frequency : int, default=20
        Export mesh every N iterations
    
    mesh_depth : int, default=8
        Poisson reconstruction depth for mesh exports
    
    mesh_format : str, default='stl'
        Mesh export format: 'stl' or 'vtk'
        - 'stl': Standard triangle mesh format (geometry only)
        - 'vtk': VTK format with vertex normals and density data (for ParaView)
        Both formats also save accompanying .xyz file with complete vertex data.
    
    base_folder : str, default='vderm_exports'
        Base directory for all exports
    grid_folder : str, default='vderm_grid'
        Subfolder name for grid exports
    surface_folder : str, default='vderm_surface'
        Subfolder name for surface exports
    mesh_folder : str, default='vderm_mesh'
        Subfolder name for mesh exports
    
    Returns
    -------
    grid : VDERMGrid
        Final deformed grid
    
    Notes
    -----
    Export file formats:
    - Grid: .xyz with "x y z rho" (4 columns)
    - Surface: .xyz with "x y z n_x n_y n_z rho" (7 columns)
    - Mesh: .stl or .vtk file + .xyz with "x y z n_x n_y n_z rho" (7 columns)
    
    Performance considerations:
    - Grid exports: Fast (~0.1s for 32k points)
    - Surface exports: Medium (~0.5-2s depending on grid size)
    - Mesh exports: Slow (~5-30s depending on complexity)
    
    Examples
    --------
    >>> # Standard STL meshes
    >>> final_grid, final_surface = run_VDERM_with_tracking(
    ...     grid, surface_pts, normals,
    ...     export_mesh=True,
    ...     mesh_format='stl'
    ... )
    
    >>> # VTK meshes for ParaView with density coloring
    >>> final_grid, final_surface = run_VDERM_with_tracking(
    ...     grid, surface_pts, normals,
    ...     export_mesh=True,
    ...     mesh_format='vtk'
    ... )
    """
    
    # Check for pymeshlab if mesh export requested
    if export_mesh and not HAS_PYMESHLAB:
        raise ImportError(
            "Mesh export requires pymeshlab.\n"
            "Install with: pip install pymeshlab\n"
            "Or install with 3-D mesh support: pip install diffusion-cartogram[3D]"
        )
    
    # Validate mesh format
    if mesh_format not in ['stl', 'vtk']:
        raise ValueError(f"mesh_format must be 'stl' or 'vtk', got '{mesh_format}'")
    
    # Create export directories if any exports are enabled
    any_exports = export_grid or export_surface or export_mesh
    
    if any_exports:
        os.makedirs(base_folder, exist_ok=True)
        
        if export_grid:
            grid_path = os.path.join(base_folder, grid_folder)
            os.makedirs(grid_path, exist_ok=True)
        
        if export_surface:
            surface_path = os.path.join(base_folder, surface_folder)
            os.makedirs(surface_path, exist_ok=True)
        
        if export_mesh:
            mesh_path = os.path.join(base_folder, mesh_folder)
            os.makedirs(mesh_path, exist_ok=True)
    
    # Initial velocities and timestep
    grid.update_velocities()
    # Compute timestep if not provided
    if dt is None:
        dt = grid.compute_timestep()
        
        # Warn if timestep is very small
        if dt < 0.005:
            print(f"  ⚠ Warning: Very small timestep ({dt:.6f})")
            print(f"    This may indicate strong density gradients.")
            print(f"    Expect longer computation time.")
    grid.epsilon = None
    
    # Export initial states (iteration 0)
    if any_exports:
        if export_grid:
            # Grid: positions + densities
            densities = grid.rho.ravel()
            velocities = grid.velocities
            filepath = os.path.join(base_folder, grid_folder, 'grid_iteration_0000.xyz')
            write_xyz(filepath, grid.positions, normals=velocities, densities=densities)
        
        if export_surface:
            # Surface: positions + normals + densities
            initial_surface_densities = interpolate_densities(surface_points, grid)
            params = {'shape': (grid.L, grid.M, grid.N),'h': grid.h,'min_bounds': grid.min_bounds}
            initial_surface_velocities = interpolate_velocities(surface_points, params, grid.velocities)
            filepath = os.path.join(base_folder, surface_folder, 'surface_iteration_0000.xyz')
            write_xyz(filepath, surface_points, 
                     normals=initial_surface_velocities, 
                     densities=initial_surface_densities)
        
        if export_mesh:
            # Mesh: save in requested format + xyz
            initial_mesh_densities = interpolate_densities(surface_points, grid)
            
            mesh_filepath = os.path.join(base_folder, mesh_folder, 
                                        f'mesh_iteration_0000.{mesh_format}')
            
            if mesh_format == 'stl':
                export_mesh_file(mesh_filepath, surface_points, depth=mesh_depth)
            else:  # vtk
                export_mesh_vtk(mesh_filepath, surface_points, initial_mesh_densities, depth=mesh_depth)
    
    # Main iteration loop with progress bar
    pbar = tqdm(range(n_max), desc='Deforming with tracking')
    
    for iteration in pbar:
        # Diffuse density
        grid.update_density(dt)
        
        # Update velocities from new density field
        if iteration > 0:
            grid.update_velocities()
        
        # Move grid nodes
        grid.update_positions(dt)
        
        # Handle exports
        should_export = (iteration + 1) % min(
            export_grid_frequency if export_grid else float('inf'),
            export_surface_frequency if export_surface else float('inf'),
            export_mesh_frequency if export_mesh else float('inf')
        ) == 0
        
        if any_exports and should_export:
            
            # Check if we need to interpolate surface
            need_interpolation = (
                (export_surface and (iteration + 1) % export_surface_frequency == 0) or
                (export_mesh and (iteration + 1) % export_mesh_frequency == 0)
            )
            
            if need_interpolation:
                # Interpolate surface positions
                params = {
                    'shape': (grid.L, grid.M, grid.N),
                    'h': grid.h,
                    'min_bounds': grid.min_bounds
                }
                displacement_field = grid.get_displacement_field()
                current_surface = interpolate_to_surface(surface_points, params, displacement_field)
                
                # Interpolate densities to surface points
                current_surface_densities = interpolate_densities(surface_points, grid)
                current_surface_velocities = interpolate_velocities(surface_points, params, grid.velocities)
            
            # Export grid if requested
            if export_grid and (iteration + 1) % export_grid_frequency == 0:
                densities = grid.rho.ravel()
                velocities = grid.velocities
                filepath = os.path.join(base_folder, grid_folder, 
                                       f'grid_iteration_{iteration+1:04d}.xyz')
                write_xyz(filepath, grid.positions, normals=velocities, densities=densities)
            
            # Export surface if requested
            if export_surface and (iteration + 1) % export_surface_frequency == 0:
                filepath = os.path.join(base_folder, surface_folder,
                                       f'surface_iteration_{iteration+1:04d}.xyz')
                write_xyz(filepath, current_surface, 
                         normals=current_surface_velocities, 
                         densities=current_surface_densities)
            
            # Export mesh if requested
            if export_mesh and (iteration + 1) % export_mesh_frequency == 0:
                mesh_filepath = os.path.join(base_folder, mesh_folder,
                                            f'mesh_iteration_{iteration+1:04d}.{mesh_format}')
                
                if mesh_format == 'stl':
                    export_mesh_file(mesh_filepath, current_surface, depth=mesh_depth)
                else:  # vtk
                    export_mesh_vtk(mesh_filepath, current_surface, current_surface_densities, depth=mesh_depth)
                
        
        # Early instability detection (check every iteration)
        if grid.epsilon is not None:
            if grid.epsilon > 1e6 or grid.epsilon < -1e-6 or np.isnan(grid.epsilon):
                pbar.close()
                print(f"\n❌ INSTABILITY at iteration {iteration}!")
                print(f"   Epsilon: {grid.epsilon:.3e}")
                print(f"   Current dt: {dt:.6f}")
                print(f"   Grid spacing h: {grid.h:.6f}")
                print(f"\n   Solution: Manually set smaller timestep:")
                print(f"   vderm.run_VDERM(grid, dt={dt/10:.6f})")
                raise RuntimeError("Numerical instability detected. Please manually set a smaller timestep and rerun")
        
        # Update progress bar
        if grid.epsilon is not None:
            pbar.set_postfix({'ε': f'{grid.epsilon:.3e}', 'target': f'{max_eps:.3e}'})
        
        # Check convergence
        if grid.epsilon is not None and grid.epsilon <= max_eps:
            pbar.set_description('Converged')
            pbar.close()
            print(f'\nConverged at iteration {iteration + 1}')
            
            # Export final states
            if any_exports:
                # Interpolate final surface and densities
                params = {
                    'shape': (grid.L, grid.M, grid.N),
                    'h': grid.h,
                    'min_bounds': grid.min_bounds
                }
                displacement_field = grid.get_displacement_field()
                
                if export_grid:
                    densities = grid.rho.ravel()
                    velocities = grid.velocities
                    filepath = os.path.join(base_folder, grid_folder,
                                           f'grid_final_iteration_{iteration+1:04d}.xyz')
                    write_xyz(filepath, grid.positions, normals=velocities, densities=densities)
                
                if export_surface or export_mesh:

                    final_surface = interpolate_to_surface(
                        surface_points, params, displacement_field
                    )
                    final_surface_densities = interpolate_densities(surface_points, grid)
                    final_surface_velocities = interpolate_velocities(surface_points, params, grid.velocities)
                
                if export_surface:
                    filepath = os.path.join(base_folder, surface_folder,
                                           f'surface_final_iteration_{iteration+1:04d}.xyz')
                    write_xyz(filepath, final_surface,
                             normals=final_surface_velocities,
                             densities=final_surface_densities)
                
                if export_mesh:
                    mesh_filepath = os.path.join(base_folder, mesh_folder,
                                                f'mesh_final_iteration_{iteration+1:04d}.{mesh_format}')
                    
                    if mesh_format == 'stl':
                        export_mesh_file(mesh_filepath, final_surface, depth=mesh_depth)
                    else:  # vtk
                        export_mesh_vtk(mesh_filepath, final_surface, final_surface_densities, depth=mesh_depth)
            
            break
    
    # Final interpolation for return value
    params = {
        'shape': (grid.L, grid.M, grid.N),
        'h': grid.h,
        'min_bounds': grid.min_bounds
    }
    displacement_field = grid.get_displacement_field()
    
    # Print export summary
    if any_exports:
        print(f"\nExports saved to: {base_folder}/")
        if export_grid:
            print(f"  - Grid states (x y z rho): {grid_folder}/")
        if export_surface:
            print(f"  - Surface point clouds (x y z n_x n_y n_z rho): {surface_folder}/")
        if export_mesh:
            print(f"  - Meshes (.{mesh_format} + .xyz with x y z n_x n_y n_z rho): {mesh_folder}/")
    
    return grid

    
def interpolate_densities(surface_points, grid):
    """
    Interpolate density values from grid to surface points.
    
    Parameters
    ----------
    surface_points : ndarray, shape (n_points, 3)
        Surface point positions (in initial/reference coordinates)
    grid : VDERMGrid
        VDERM grid object with density field
    
    Returns
    -------
    surface_densities : ndarray, shape (n_points,)
        Interpolated density values at each surface point
    
    Notes
    -----
    Surface points outside the grid bounds will have density set to 0.
    """
    
    # Create coordinate arrays for each axis
    x = grid.min_bounds[0] + np.arange(grid.L) * grid.h
    y = grid.min_bounds[1] + np.arange(grid.M) * grid.h
    z = grid.min_bounds[2] + np.arange(grid.N) * grid.h
    
    # Create interpolator for density field
    # density field is already in (L, M, N) shape
    interp_rho = RegularGridInterpolator(
        (x, y, z), 
        grid.rho,
        bounds_error=False, 
        fill_value=0  # Points outside grid get density 0
    )
    
    # Interpolate densities
    surface_densities = interp_rho(surface_points)
    
    return surface_densities
    
def interpolate_velocities(surface_points, grid_params, velocity_field):
    """
    Interpolate velocities from the regular grid to the surface
    
    Parameters
    ----------
    surface_points : ndarray, shape (n_points, 3)
        Surface point cloud positions
    grid_params : dict
        Dictionary with 'shape', 'h', 'min_bounds'
    velocity_field : ndarray, shape (L*M*N, 3)
        Velocity vectors at each grid node
    
    Returns
    -------
    interpolated_velocities : ndarray, shape (n_points, 3)
        Surface velocities
    """
    L, M, N = grid_params['shape']
    h = grid_params['h']
    min_bounds = grid_params['min_bounds']
    
    # Create coordinate arrays for each axis
    x = min_bounds[0] + np.arange(L) * h
    y = min_bounds[1] + np.arange(M) * h
    z = min_bounds[2] + np.arange(N) * h
    
    # Reshape velocity field to 3D grid
    velo_grid = velocity_field.reshape(L, M, N, 3)
    
    # Create interpolators for each component 
    interp_u = RegularGridInterpolator((x, y, z), velo_grid[:, :, :, 0], 
                                       bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((x, y, z), velo_grid[:, :, :, 1],
                                       bounds_error=False, fill_value=0)
    interp_w = RegularGridInterpolator((x, y, z), velo_grid[:, :, :, 2],
                                       bounds_error=False, fill_value=0)
    
    # Interpolate 
    u = interp_u(surface_points)
    v = interp_v(surface_points)
    w = interp_w(surface_points)
    
    interpolated_velocities = np.column_stack([u, v, w])
    
    return interpolated_velocities

def interpolate_to_surface(surface_points, grid_params, displacement_field):
    """
    Interpolate the grid based vector field to the surface point cloud 
    and return the deformed surface
    
    Parameters
    ----------
    surface_points : ndarray, shape (n_points, 3)
        Surface point cloud positions
    grid_params : dict
        Dictionary with 'shape', 'h', 'min_bounds'
    displacement_field : ndarray, shape (L*M*N, 3)
        Displacement vectors at each grid node
    
    Returns
    -------
    deformed_surface : ndarray, shape (n_points, 3)
        Surface points after applying interpolated displacements

    Examples
    --------
    >>> # After running VDERM
    >>> deformed_surface = interpolate_to_surface(original_surface, grid_params, vderm_grid.get_displacement_field())
    >>> surface_densities = interpolate_densities(deformed_surface, vderm_grid)
    >>> surface_velocities = interpolate_velocities(deformed_surface, grid_params, vderm_grid.velocities)
    >>> # Save surface with densities and velocities
    >>> write_xyz('surface_with_density.xyz', surface_pts, 
    ...          normals=surface_velocities, densities=surface_densities)
    """
    L, M, N = grid_params['shape']
    h = grid_params['h']
    min_bounds = grid_params['min_bounds']
    
    # Create coordinate arrays for each axis
    x = min_bounds[0] + np.arange(L) * h
    y = min_bounds[1] + np.arange(M) * h
    z = min_bounds[2] + np.arange(N) * h
    
    # Reshape displacement field to 3D grid
    disp_grid = displacement_field.reshape(L, M, N, 3)
    
    # Create interpolators for each component 
    interp_u = RegularGridInterpolator((x, y, z), disp_grid[:, :, :, 0], 
                                       bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((x, y, z), disp_grid[:, :, :, 1],
                                       bounds_error=False, fill_value=0)
    interp_w = RegularGridInterpolator((x, y, z), disp_grid[:, :, :, 2],
                                       bounds_error=False, fill_value=0)
    
    # Interpolate 
    u = interp_u(surface_points)
    v = interp_v(surface_points)
    w = interp_w(surface_points)
    
    interpolated_displacement = np.column_stack([u, v, w])
    
    return surface_points + interpolated_displacement

def export_mesh_file(filename, deformed_pcd, depth=8, fulldepth=5, scale=1.1):
    """
    Creates and exports a Poisson mesh from a deformed point cloud.
    
    Normals are automatically estimated from the deformed point cloud geometry
    using local neighborhood analysis, which is more accurate for deformed surfaces
    than using original normals.
    
    Parameters
    ----------
    filename : str
        Output file path (.ply, .stl, .obj, .off, or .gltf/.glb)
    deformed_pcd : ndarray, shape (n_points, 3)
        Deformed point cloud to remesh
    depth : int, default=8
        Poisson reconstruction octree depth (higher = more detail)
    fulldepth : int, default=5
        Depth below which octree will be complete
    scale : float, default=1.1
        Ratio between reconstruction cube diameter and samples' bounding cube diameter
    
    Returns
    -------
    result_mesh : pymeshlab Mesh
        The reconstructed mesh object
    
    Notes
    -----
    Poisson reconstruction default values from:
    https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version8.0/
    
    Normal estimation uses k=20 nearest neighbors with 2 smoothing iterations.
    Adjust these in the code if needed for your specific geometry.

    macOS + conda users may encounter an OpenMP conflict (OMP: Error #15) when
    this function is called, due to pymeshlab's bundled libomp conflicting with
    conda-forge's numpy/scipy. See the Known Issues section of the README for
    the fix. 
    
    Examples
    --------
    >>> # Basic usage
    >>> mesh = export_mesh_file('output.stl', deformed_points)
    
    >>> # Higher quality reconstruction
    >>> mesh = export_mesh_file('output.ply', deformed_points, depth=10)
    """
    _require_pymeshlab('create_pcd')
    
    # Create MeshSet and add point cloud
    ms = ml.MeshSet()
    point_cloud_mesh = ml.Mesh(vertex_matrix=deformed_pcd)
    ms.add_mesh(point_cloud_mesh)
    
    # Estimate normals from local geometry of deformed point cloud
    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    
    # Perform Poisson surface reconstruction using estimated normals
    ms.generate_surface_reconstruction_screened_poisson(
        depth=depth,
        fulldepth=fulldepth,
        scale=scale
    )
    
    # Compute normals for the reconstructed mesh (for smooth rendering)
    ms.compute_normal_for_point_clouds()
    
    # Save the mesh to file
    ms.save_current_mesh(filename)
    
    # Return the mesh object
    result_mesh = ms.current_mesh()
    
    return result_mesh
    
def export_mesh_vtk(filepath, deformed_pcd, densities, depth=8):
    """
    Create and export a Poisson mesh in VTK format with density vertex attribute.
    
    VTK format includes mesh geometry, vertex normals, and density scalar field.
    This is ideal for ParaView visualization with density coloring.
    
    Parameters
    ----------
    filepath : str
        Output file path (should end with .vtk)
    deformed_pcd : ndarray, shape (n_points, 3)
        Deformed point cloud to remesh
    densities : ndarray, shape (n_points,)
        Density values at each point
    depth : int, default=8
        Poisson reconstruction depth
    
    Returns
    -------
    mesh : pymeshlab Mesh object
        The reconstructed mesh

    Note:
        macOS + conda users may encounter an OpenMP conflict (OMP: Error #15) when
        this function is called, due to pymeshlab's bundled libomp conflicting with
        conda-forge's numpy/scipy. See the Known Issues section of the README for
        the fix. 
    """
    _require_pymeshlab('create_pcd')
    
    # Reconstruct mesh using Poisson
    ms = ml.MeshSet()
    point_cloud_mesh = ml.Mesh(vertex_matrix=deformed_pcd)
    ms.add_mesh(point_cloud_mesh)
    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    
    ms.generate_surface_reconstruction_screened_poisson(
        depth=depth,
        fulldepth=5,
        scale=1.1
    )
    
    ms.compute_normal_for_point_clouds()
    
    # Get mesh data
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    vertex_normals = mesh.vertex_normal_matrix()
    
    # Interpolate densities from point cloud to mesh vertices
    # (Poisson reconstruction creates new vertices, so we need to interpolate)
    interp = NearestNDInterpolator(deformed_pcd, densities)
    mesh_densities = interp(vertices)
    
    # Write VTK POLYDATA file with density attribute
    with open(filepath, 'w') as vtk:
        vtk.write('# vtk DataFile Version 3.0\n')
        vtk.write('VDERM Mesh with Density\n')
        vtk.write('ASCII\n')
        vtk.write('DATASET POLYDATA\n')
        
        # Write vertices
        vtk.write(f'POINTS {len(vertices)} float\n')
        for v in vertices:
            vtk.write(f'{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n')
        
        # Write triangular faces
        vtk.write(f'\nPOLYGONS {len(faces)} {len(faces) * 4}\n')
        for face in faces:
            vtk.write(f'3 {face[0]} {face[1]} {face[2]}\n')
        
        # Write vertex data
        vtk.write(f'\nPOINT_DATA {len(vertices)}\n')
        
        # Normals as vector field
        vtk.write('NORMALS normals float\n')
        for n in vertex_normals:
            vtk.write(f'{n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n')
        
        # Density as scalar field
        vtk.write('\nSCALARS density float 1\n')
        vtk.write('LOOKUP_TABLE default\n')
        for d in mesh_densities:
            vtk.write(f'{d:.6e}\n')
    
    return mesh
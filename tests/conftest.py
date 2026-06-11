"""
Shared pytest fixtures for diffusion-cartogram tests.
"""
import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def simple_grid():
    """Create a simple 10x10x10 grid for testing."""
    import diffusion_cartogram as vd
    return vd.VDERMGrid(shape=(10, 10, 10), h=0.1, min_bounds=[0, 0, 0])


@pytest.fixture
def medium_grid():
    """Create a medium 20x20x20 grid for testing."""
    import diffusion_cartogram as vd
    return vd.VDERMGrid(shape=(20, 20, 20), h=0.05, min_bounds=[0, 0, 0])


@pytest.fixture
def simple_density_function():
    """Simple Gaussian density function."""
    def density_func(x, y, z):
        r2 = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2
        return 1.0 + 2.0 * np.exp(-10 * r2)
    return density_func


@pytest.fixture
def temp_xyz_file():
    """Create a temporary .xyz file that gets cleaned up."""
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def temp_stl_file():
    """Create a temporary .stl file that gets cleaned up."""
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def temp_vtk_file():
    """Create a temporary .vtk file that gets cleaned up."""
    with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def sample_point_cloud():
    """Generate a simple point cloud for testing."""
    # Create points on a sphere
    n_points = 100
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    
    x = 0.5 * np.sin(phi) * np.cos(theta)
    y = 0.5 * np.sin(phi) * np.sin(theta)
    z = 0.5 * np.cos(phi)
    
    points = np.column_stack([x, y, z])
    
    # Normals point outward from origin (for sphere)
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points, normals


@pytest.fixture
def sample_densities():
    """Generate sample density values."""
    return np.random.uniform(0.5, 2.0, 100)
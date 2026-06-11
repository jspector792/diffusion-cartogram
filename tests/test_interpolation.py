"""
Tests for interpolation functions.
"""
import pytest
import numpy as np
import diffusion_cartogram as vderm


class TestInterpolation:
    """Tests for interpolation functions."""
    
    def test_interpolate_densities(self, simple_grid):
        """Test density interpolation to points."""
        # Set known density field
        simple_grid.set_density(lambda x, y, z: x + y + z)
        
        # Query points
        query_points = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5]
        ])
        
        densities = vderm.interpolate_densities(query_points, simple_grid)
        
        assert len(densities) == 2
        # Should be approximately x+y+z at query points
        assert densities[0] == pytest.approx(0.75, abs=0.1)
        assert densities[1] == pytest.approx(1.5, abs=0.1)
    
    def test_interpolate_to_surface(self, simple_grid):
        """Test displacement interpolation."""
        # Move grid uniformly
        simple_grid.positions += np.array([0.1, 0.2, 0.3])
        
        # Query points inside grid
        query_points = np.array([
            [0.5, 0.5, 0.5],
            [0.3, 0.4, 0.6]
        ])
        
        params = {
            'shape': (simple_grid.L, simple_grid.M, simple_grid.N),
            'h': simple_grid.h,
            'min_bounds': simple_grid.min_bounds
        }
        
        displacement_field = simple_grid.get_displacement_field()
        deformed = vderm.interpolate_to_surface(query_points, params, displacement_field)
        
        # Should be displaced by approximately [0.1, 0.2, 0.3]
        expected = query_points + np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(deformed, expected, decimal=2)
    
    def test_interpolate_velocities(self, simple_grid):
        """Test velocity interpolation."""
        # Set uniform velocities
        simple_grid.velocities[:] = np.array([1.0, 2.0, 3.0])
        
        query_points = np.array([[0.5, 0.5, 0.5]])
        
        params = {
            'shape': (simple_grid.L, simple_grid.M, simple_grid.N),
            'h': simple_grid.h,
            'min_bounds': simple_grid.min_bounds
        }
        
        velocities = vderm.interpolate_velocities(query_points, params, simple_grid.velocities)
        
        # Should get interpolated velocity close to [1, 2, 3]
        assert np.allclose(velocities[0], [1.0, 2.0, 3.0], atol=0.1)
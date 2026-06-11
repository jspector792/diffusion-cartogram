"""
Tests for core VDERM functionality.
"""
import pytest
import numpy as np
import diffusion_cartogram as vd


class TestVDERMGrid:
    """Tests for VDERMGrid class."""
    
    def test_initialization(self, simple_grid):
        """Test grid initializes with correct dimensions."""
        assert simple_grid.L == 10
        assert simple_grid.M == 10
        assert simple_grid.N == 10
        assert simple_grid.h == 0.1
        assert simple_grid.positions.shape == (1000, 3)
        assert simple_grid.velocities.shape == (1000, 3)
        assert simple_grid.rho.shape == (10, 10, 10)
    
    def test_initial_positions_correct(self):
        """Test that initial positions form a regular grid."""
        grid = vd.VDERMGrid(shape=(3, 3, 3), h=1.0, min_bounds=[0, 0, 0])
        
        # Check first point
        assert np.allclose(grid.positions[0], [0, 0, 0])
        
        # Check last point
        assert np.allclose(grid.positions[-1], [2, 2, 2])
        
        # Check a middle point (index for i=1, j=1, k=1)
        idx = grid._index_to_flat(1, 1, 1)
        assert np.allclose(grid.positions[idx], [1, 1, 1])
    
    def test_set_density_with_function(self, simple_grid, simple_density_function):
        """Test setting density with a callable."""
        simple_grid.set_density(simple_density_function)
        
        # Check density is not all ones
        assert not np.allclose(simple_grid.rho, 1.0)
        
        # Check density at center should be higher
        center_density = simple_grid.rho[5, 5, 5]
        edge_density = simple_grid.rho[0, 0, 0]
        assert center_density > edge_density
    
    def test_set_density_with_array(self, simple_grid):
        """Test setting density with a numpy array."""
        density_array = np.random.rand(10, 10, 10)
        simple_grid.set_density(density_array)
        
        np.testing.assert_array_equal(simple_grid.rho, density_array)
    
    def test_update_density(self, simple_grid):
        """Test density diffusion update."""
        # Set non-uniform density
        simple_grid.rho[5, 5, 5] = 10.0
        
        # Update
        simple_grid.update_density(dt=0.001)
        
        # Check that epsilon was computed
        assert simple_grid.epsilon is not None
        assert simple_grid.epsilon > 0
        
        # Check that density diffused (center should be less than 10)
        assert simple_grid.rho[5, 5, 5] < 10.0
    
    def test_update_velocities(self, simple_grid, simple_density_function):
        """Test velocity computation from density gradient."""
        simple_grid.set_density(simple_density_function)
        simple_grid.update_velocities()
        
        # Velocities should not all be zero
        assert not np.allclose(simple_grid.velocities, 0)
    
    def test_update_positions(self, simple_grid):
        """Test position update from velocities."""
        # Set some velocities
        simple_grid.velocities[:] = np.array([0.1, 0, 0])
        
        initial_positions = simple_grid.positions.copy()
        
        # Update positions
        simple_grid.update_positions(dt=0.1)
        
        # Check positions changed
        assert not np.allclose(simple_grid.positions, initial_positions)
        
        # Check displacement is correct
        displacement = simple_grid.positions - initial_positions
        assert np.allclose(displacement, [0.01, 0, 0])
    
    def test_get_displacement_field(self, simple_grid):
        """Test displacement field computation."""
        # Move grid
        simple_grid.positions += np.array([0.1, 0.2, 0.3])
        
        # Get displacement
        displacement = simple_grid.get_displacement_field()
        
        assert displacement.shape == (1000, 3)
        assert np.allclose(displacement, [0.1, 0.2, 0.3])
    
    def test_compute_timestep(self, simple_grid):
        """Test timestep computation."""
        simple_grid.velocities[:] = 1.0
        dt = simple_grid.compute_timestep()
        
        assert dt > 0
        assert dt <= 0.01  # Should cap at 0.01
    
    def test_compute_timestep_includes_diffusion_limit(self, simple_grid):
        """Test that timestep respects diffusion stability."""
        simple_grid.velocities[:] = 0.001  # Very small velocities
        dt = simple_grid.compute_timestep()
        
        # Should be limited by diffusion: dt <= h²/6
        dt_diffusion_limit = simple_grid.h**2 / 6
        assert dt <= dt_diffusion_limit * 0.9  # With safety factor


class TestRunVDERM:
    """Tests for run_VDERM function."""
    
    def test_basic_run_converges(self, simple_grid, simple_density_function):
        """Test that VDERM runs and converges."""
        simple_grid.set_density(simple_density_function)
        
        result = vd.run_VDERM(simple_grid, n_max=50, max_eps=0.05)
        
        assert result.epsilon is not None
        assert result.epsilon > 0
    
    def test_run_respects_max_iterations(self):
        """Test that run_VDERM stops at n_max."""
        grid = vd.VDERMGrid(shape=(5, 5, 5), h=0.2, min_bounds=[0, 0, 0])
        
        # Set density that won't converge quickly
        grid.set_density(lambda x, y, z: 1.0 + 10 * np.sin(10*x) * np.cos(10*y))
        
        result = vd.run_VDERM(grid, n_max=5, max_eps=0.001)
        
        # Should stop at max iterations, not converge
        assert result.epsilon > 0.001
    
    def test_manual_timestep(self, simple_grid, simple_density_function):
        """Test manual timestep override."""
        simple_grid.set_density(simple_density_function)
        
        # Use manually set small timestep
        result = vd.run_VDERM(simple_grid, n_max=20, dt=0.0001)
        
        assert result.epsilon is not None
    
    def test_instability_detection(self):
        """Test that instability is detected and raises error."""
        grid = vd.VDERMGrid(shape=(10, 10, 10), h=0.1, min_bounds=[0, 0, 0])
        
        # Create extreme gradient that will cause instability
        grid.set_density(lambda x, y, z: 1.0 + 100.0 * x)
        
        # Should raise RuntimeError due to instability
        with pytest.raises(RuntimeError, match="Numerical instability"):
            vd.run_VDERM(grid, n_max=100, dt=0.01)  # Too large dt


class TestRunVDERMWithTracking:
    """Tests for run_VDERM_with_tracking function."""
    
    def test_no_exports(self, simple_grid, sample_point_cloud):
        """Test tracking function with no exports enabled."""
        surface_pts, surface_norms = sample_point_cloud
        simple_grid.set_density(lambda x, y, z: 1.0)
        
        final_grid = vd.run_VDERM_with_tracking(
            simple_grid, surface_pts,
            n_max=10,
            export_grid=False,
            export_surface=False,
            export_mesh=False
        )
        
        assert final_grid is not None
    
    def test_grid_export_creates_files(self, simple_grid, sample_point_cloud, tmp_path):
        """Test that grid export creates files with correct format."""
        surface_pts, surface_norms = sample_point_cloud
        simple_grid.set_density(lambda x, y, z: 1.0)
        
        export_folder = str(tmp_path / 'test_exports')
        
        vd.run_VDERM_with_tracking(
            simple_grid, surface_pts,
            n_max=10,
            export_grid=True,
            export_grid_frequency=5,
            base_folder=export_folder
        )
        
        # Check that files were created
        grid_files = list((tmp_path / 'test_exports' / 'vderm_grid').glob('*.xyz'))
        assert len(grid_files) > 0
        
        # Check file format (should have 7 columns: x y z v_x v_y v_z rho)
        pos, vel, dens = vd.read_xyz(str(grid_files[0]))
        assert pos is not None
        assert vel is not None  # Should have velocities
        assert dens is not None  # Should have densities
    
    def test_surface_export_creates_files(self, simple_grid, sample_point_cloud, tmp_path):
        """Test that surface export creates files with correct format."""
        surface_pts, surface_norms = sample_point_cloud
        simple_grid.set_density(lambda x, y, z: 1.0)
        
        export_folder = str(tmp_path / 'test_exports')
        
        vd.run_VDERM_with_tracking(
            simple_grid, surface_pts,
            n_max=10,
            export_surface=True,
            export_surface_frequency=5,
            base_folder=export_folder
        )
        
        # Check that files were created
        surface_files = list((tmp_path / 'test_exports' / 'vderm_surface').glob('*.xyz'))
        assert len(surface_files) > 0
        
        # Check file format (should have 7 columns: x y z v_x v_y v_z rho)
        pos, vel, dens = vd.read_xyz(str(surface_files[0]))
        assert pos is not None
        assert vel is not None  # Should have velocities
        assert dens is not None  # Should have densities
    
    @pytest.mark.skipif(not vd.HAS_PYMESHLAB, reason="requires pymeshlab")
    def test_mesh_export_stl(self, simple_grid, sample_point_cloud, tmp_path):
        """Test that mesh export creates STL files."""
        surface_pts, surface_norms = sample_point_cloud
        simple_grid.set_density(lambda x, y, z: 1.0)
        
        export_folder = str(tmp_path / 'test_exports')
        
        vd.run_VDERM_with_tracking(
            simple_grid, surface_pts,
            n_max=5,
            export_mesh=True,
            export_mesh_frequency=5,
            mesh_format='stl',
            base_folder=export_folder
        )
        
        # Check that mesh files were created
        mesh_files = list((tmp_path / 'test_exports' / 'vderm_mesh').glob('*.stl'))
        assert len(mesh_files) > 0
    
    @pytest.mark.skipif(not vd.HAS_PYMESHLAB, reason="requires pymeshlab")
    def test_mesh_export_vtk(self, simple_grid, sample_point_cloud, tmp_path):
        """Test that mesh export creates VTK files with density."""
        surface_pts, surface_norms = sample_point_cloud
        simple_grid.set_density(lambda x, y, z: 1.0)
        
        export_folder = str(tmp_path / 'test_exports')
        
        vd.run_VDERM_with_tracking(
            simple_grid, surface_pts,
            n_max=5,
            export_mesh=True,
            export_mesh_frequency=5,
            mesh_format='vtk',
            base_folder=export_folder
        )
        
        # Check that VTK files were created
        vtk_files = list((tmp_path / 'test_exports' / 'vderm_mesh').glob('*.vtk'))
        assert len(vtk_files) > 0
        
        # Check that VTK file contains "density"
        with open(str(vtk_files[0]), 'r') as f:
            content = f.read()
            assert 'SCALARS density' in content
    
    def test_raises_without_pymeshlab_for_mesh_export(self, simple_grid, sample_point_cloud):
        """Test that mesh export raises error without pymeshlab."""
        if vd.HAS_PYMESHLAB:
            pytest.skip("Test only valid when pymeshlab is not installed")
        
        surface_pts, surface_norms = sample_point_cloud
        simple_grid.set_density(lambda x, y, z: 1.0)
        
        with pytest.raises(ImportError, match="requires pymeshlab"):
            vd.run_VDERM_with_tracking(
                simple_grid, surface_pts,
                n_max=5,
                export_mesh=True
            )


class TestGridUtilities:
    """Tests for grid creation utilities."""
    
    def test_compute_grid_dimensions(self):
        """Test grid dimension computation."""
        box_dims = np.array([2.0, 3.0, 4.0])
        shape, h = vd.compute_grid_dimensions(box_dims, max_points=1000)
        
        L, M, N = shape
        
        # Should have approximately 1000 points
        assert 800 < L * M * N < 1200
        
        # Spacing should be uniform
        assert h > 0
    
    def test_make_initial_grid_returns_correct_structure(self, sample_point_cloud):
        """Test that make_initial_grid returns correct params dict."""
        points, normals = sample_point_cloud
        
        params = vd.make_initial_grid(points, max_points=1000)
        
        # Check required keys
        assert 'shape' in params
        assert 'h' in params
        assert 'min_bounds' in params
        assert 'max_bounds' in params
        assert 'object_bounds' in params
        assert 'padding' in params
        
        # Check dimensions
        L, M, N = params['shape']
        assert L >= 3 and M >= 3 and N >= 3
    
    def test_make_initial_grid_padding_ratio(self, sample_point_cloud):
        """Test that grid respects padding ratio."""
        points, normals = sample_point_cloud
        
        params = vd.make_initial_grid(points, max_points=5000, padding=(3, 3, 3))
        
        obj_size = params['object_bounds']['size']
        grid_size = params['max_bounds'] - params['min_bounds']
        
        ratios = grid_size / obj_size
        
        # Should be close to 3.0 in each dimension
        assert np.allclose(ratios, 3.0, atol=0.3)
    
    def test_make_initial_grid_asymmetric_padding(self, sample_point_cloud):
        """Test asymmetric padding."""
        points, normals = sample_point_cloud
        
        params = vd.make_initial_grid(points, max_points=5000, padding=(2, 3, 5))
        
        obj_size = params['object_bounds']['size']
        grid_size = params['max_bounds'] - params['min_bounds']
        
        ratios = grid_size / obj_size
        
        # Should match requested padding
        assert ratios[0] == pytest.approx(2.0, abs=0.3)
        assert ratios[1] == pytest.approx(3.0, abs=0.3)
        assert ratios[2] == pytest.approx(5.0, abs=0.3)
    
    def test_make_initial_grid_centers_object(self, sample_point_cloud):
        """Test that object is centered in grid."""
        points, normals = sample_point_cloud
        
        params = vd.make_initial_grid(points, max_points=2000)
        
        obj_center = params['object_bounds']['center']
        grid_center = (params['min_bounds'] + params['max_bounds']) / 2
        
        # Object center should match grid center
        assert np.allclose(obj_center, grid_center, atol=0.01)
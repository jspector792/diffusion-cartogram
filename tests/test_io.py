"""
Tests for I/O functions.
"""
import pytest
import numpy as np
import tempfile
import os
import diffusion_cartogram as vd


class TestXYZIO:
    """Tests for unified XYZ file I/O."""
    
    def test_write_read_positions_only(self, temp_xyz_file):
        """Test writing and reading positions only (3 columns)."""
        positions = np.random.rand(100, 3)
        
        vd.write_xyz(temp_xyz_file, positions)
        pos_read, norms_read, dens_read = vd.read_xyz(temp_xyz_file)
        
        np.testing.assert_array_almost_equal(pos_read, positions)
        assert norms_read is None
        assert dens_read is None
    
    def test_write_read_with_densities(self, temp_xyz_file):
        """Test writing and reading positions + densities (4 columns)."""
        positions = np.random.rand(50, 3)
        densities = np.random.rand(50)
        
        vd.write_xyz(temp_xyz_file, positions, normals=None, densities=densities)
        pos_read, norms_read, dens_read = vd.read_xyz(temp_xyz_file)
        
        np.testing.assert_array_almost_equal(pos_read, positions)
        assert norms_read is None
        np.testing.assert_array_almost_equal(dens_read, densities)
    
    def test_write_read_with_normals(self, temp_xyz_file):
        """Test writing and reading positions + normals (6 columns)."""
        positions = np.random.rand(50, 3)
        normals = np.random.rand(50, 3)
        
        vd.write_xyz(temp_xyz_file, positions, normals=normals)
        pos_read, norms_read, dens_read = vd.read_xyz(temp_xyz_file)
        
        np.testing.assert_array_almost_equal(pos_read, positions)
        np.testing.assert_array_almost_equal(norms_read, normals)
        assert dens_read is None
    
    def test_write_read_complete(self, temp_xyz_file):
        """Test writing and reading all data (7 columns)."""
        positions = np.random.rand(50, 3)
        normals = np.random.rand(50, 3)
        densities = np.random.rand(50)
        
        vd.write_xyz(temp_xyz_file, positions, normals=normals, densities=densities)
        pos_read, norms_read, dens_read = vd.read_xyz(temp_xyz_file)
        
        np.testing.assert_array_almost_equal(pos_read, positions)
        np.testing.assert_array_almost_equal(norms_read, normals)
        np.testing.assert_array_almost_equal(dens_read, densities)
    
    def test_read_invalid_format(self):
        """Test that reading invalid format raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            # Write file with wrong number of columns
            f.write('1.0 2.0\n')  # Only 2 columns
            f.write('3.0 4.0\n')
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="Unrecognized file format"):
                vd.read_xyz(filepath)
        finally:
            os.unlink(filepath)
    
    def test_write_mismatched_normals_raises_error(self, temp_xyz_file):
        """Test that mismatched normals length raises error."""
        positions = np.random.rand(50, 3)
        normals = np.random.rand(40, 3)  # Wrong length!
        
        with pytest.raises(ValueError, match="must match positions"):
            vd.write_xyz(temp_xyz_file, positions, normals=normals)
    
    def test_write_mismatched_densities_raises_error(self, temp_xyz_file):
        """Test that mismatched densities length raises error."""
        positions = np.random.rand(50, 3)
        densities = np.random.rand(40)  # Wrong length!
        
        with pytest.raises(ValueError, match="must match positions"):
            vd.write_xyz(temp_xyz_file, positions, densities=densities)


@pytest.mark.skipif(not vd.HAS_PYMESHLAB, reason="requires pymeshlab")
class TestMeshIO:
    """Tests for mesh I/O operations (require pymeshlab)."""
    
    def test_export_mesh_file_creates_file(self, sample_point_cloud, temp_stl_file):
        """Test that export_mesh_file creates a valid STL file."""
        points, normals = sample_point_cloud
        
        mesh = vd.export_mesh_file(temp_stl_file, points, depth=6)
        
        assert mesh is not None
        assert os.path.exists(temp_stl_file)
        assert os.path.getsize(temp_stl_file) > 0
    
    def test_export_mesh_vtk_includes_density(self, sample_point_cloud, temp_vtk_file):
        """Test that VTK export includes density data."""
        points, normals = sample_point_cloud
        densities = np.random.rand(len(points))
        
        mesh = vd.export_mesh_vtk(temp_vtk_file, points, densities, depth=6)
        
        assert mesh is not None
        assert os.path.exists(temp_vtk_file)
        
        # Check file contains density
        with open(temp_vtk_file, 'r') as f:
            content = f.read()
            assert 'SCALARS density' in content
            assert 'NORMALS normals' in content
    
    def test_export_mesh_different_depths(self, sample_point_cloud):
        """Test mesh export with different reconstruction depths."""
        points, normals = sample_point_cloud
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f1:
            filepath1 = f1.name
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f2:
            filepath2 = f2.name
        
        try:
            mesh1 = vd.export_mesh_file(filepath1, points, depth=5)
            mesh2 = vd.export_mesh_file(filepath2, points, depth=8)
            
            assert mesh1 is not None
            assert mesh2 is not None
        finally:
            if os.path.exists(filepath1):
                os.unlink(filepath1)
            if os.path.exists(filepath2):
                os.unlink(filepath2)


class TestMeshIOWithoutPymeshlab:
    """Test that mesh functions fail gracefully without pymeshlab."""
    
    @pytest.mark.skipif(vd.HAS_PYMESHLAB, reason="only test when pymeshlab missing")
    def test_create_pcd_raises_without_pymeshlab(self):
        """Test that create_pcd raises helpful error without pymeshlab."""
        with pytest.raises(ImportError, match="requires pymeshlab"):
            vd.create_pcd('dummy.stl')
    
    @pytest.mark.skipif(vd.HAS_PYMESHLAB, reason="only test when pymeshlab missing")
    def test_export_mesh_raises_without_pymeshlab(self):
        """Test that export_mesh_file raises helpful error without pymeshlab."""
        points = np.random.rand(100, 3)
        
        with pytest.raises(ImportError, match="requires pymeshlab"):
            vd.export_mesh_file('dummy.stl', points)
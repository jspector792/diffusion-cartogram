"""
Tests for visualization functions.
"""
import pytest
import numpy as np
import os
import diffusion_cartogram as vd


class TestVisualization:
    """Tests for visualization functions."""
    
    @pytest.mark.skipif(not hasattr(vd, 'plot_pcd'), 
                        reason="visualization not available")
    def test_plot_pcd_3d(self, sample_point_cloud, tmp_path):
        """Test 3D point cloud plotting."""
        points, normals = sample_point_cloud
        
        output_file = str(tmp_path / 'test_3d.png')
        
        fig = vd.plot_pcd(points, view='3d', save_file=output_file)
        
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    @pytest.mark.skipif(not hasattr(vd, 'plot_pcd'),
                        reason="visualization not available")
    def test_plot_pcd_2d(self, sample_point_cloud, tmp_path):
        """Test 2D projection plotting."""
        points, normals = sample_point_cloud
        densities = np.random.rand(len(points))
        
        output_file = str(tmp_path / 'test_2d.png')
        
        fig = vd.plot_pcd(points, densities=densities, view='2d', save_file=output_file)
        
        assert os.path.exists(output_file)
    
    @pytest.mark.skipif(not hasattr(vd, 'animate_grid_deformation'), 
                        reason="visualization not available")
    def test_animate_grid_creates_file(self, tmp_path):
        """Test that animation creates output file."""
        # Create minimal test data
        export_folder = tmp_path / 'exports'
        grid_folder = export_folder / 'vderm_grid'
        grid_folder.mkdir(parents=True)
        
        # Create a few test grid files (with velocities and densities)
        positions = np.random.rand(100, 3)
        velocities = np.random.rand(100, 3) * 0.1
        densities = np.random.rand(100)
        
        for i in range(3):
            filepath = grid_folder / f'grid_iteration_{i:04d}.xyz'
            vd.write_xyz(str(filepath), positions + i*0.01, 
                        normals=velocities, densities=densities)
        
        # Create animation
        output_file = str(tmp_path / 'test.gif')
        
        vd.animate_grid_deformation(
            export_folder=str(export_folder),
            subfolder='vderm_grid',
            output_file=output_file,
            fps=2,
            subsample=None
        )
        
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    @pytest.mark.skipif(not hasattr(vd, 'plot_density_evolution'),
                        reason="visualization not available")
    def test_plot_density_evolution_creates_file(self, tmp_path):
        """Test that density plot creates output file."""
        # Create test data
        export_folder = tmp_path / 'exports'
        grid_folder = export_folder / 'vderm_grid'
        grid_folder.mkdir(parents=True)
        
        positions = np.random.rand(100, 3)
        velocities = np.random.rand(100, 3) * 0.1
        
        for i in range(5):
            densities = np.random.rand(100) * (1.0 + i*0.1)
            filepath = grid_folder / f'grid_iteration_{i:04d}.xyz'
            vd.write_xyz(str(filepath), positions, 
                        normals=velocities, densities=densities)
        
        output_file = str(tmp_path / 'density_plot.png')
        
        vd.plot_density_evolution(
            export_folder=str(export_folder),
            grid_folder='vderm_grid',
            output_file=output_file
        )
        
        assert os.path.exists(output_file)


class TestParaViewExport:
    """Tests for ParaView export functions."""
    
    @pytest.mark.skipif(not hasattr(vd, 'export_grid_to_paraview'),
                        reason="visualization not available")
    def test_export_grid_to_paraview(self, tmp_path):
        """Test grid to ParaView export."""
        # Create test grid data with velocities and densities
        export_folder = tmp_path / 'exports'
        grid_folder = export_folder / 'vderm_grid'
        grid_folder.mkdir(parents=True)
        
        positions = np.random.rand(50, 3)
        velocities = np.random.rand(50, 3) * 0.1
        densities = np.random.rand(50)
        
        xyz_file = grid_folder / 'grid_iteration_0000.xyz'
        vd.write_xyz(str(xyz_file), positions, normals=velocities, densities=densities)
        
        # Export to ParaView
        output_folder = str(tmp_path / 'paraview')
        
        vd.export_grid_to_paraview(
            export_folder=str(export_folder),
            subfolder='vderm_grid',
            output_folder=output_folder
        )
        
        # Check VTK file was created
        vtk_files = list(os.listdir(output_folder))
        assert len(vtk_files) > 0
        assert any(f.endswith('.vtk') for f in vtk_files)
        
        # Check VTK contains velocity and density
        vtk_path = os.path.join(output_folder, vtk_files[0])
        with open(vtk_path, 'r') as f:
            content = f.read()
            assert 'VECTORS velocity' in content
            assert 'SCALARS density' in content
    
    @pytest.mark.skipif(not hasattr(vd, 'export_surface_to_paraview'),
                        reason="visualization not available")
    def test_export_surface_to_paraview(self, tmp_path):
        """Test surface to ParaView export."""
        # Create test surface data
        export_folder = tmp_path / 'exports'
        surface_folder = export_folder / 'vderm_surface'
        surface_folder.mkdir(parents=True)
        
        positions = np.random.rand(50, 3)
        normals = np.random.rand(50, 3)
        densities = np.random.rand(50)
        
        xyz_file = surface_folder / 'surface_iteration_0000.xyz'
        vd.write_xyz(str(xyz_file), positions, normals=normals, densities=densities)
        
        # Export to ParaView
        output_folder = str(tmp_path / 'paraview')
        
        vd.export_surface_to_paraview(
            export_folder=str(export_folder),
            subfolder='vderm_surface',
            output_folder=output_folder
        )
        
        # Check VTK file was created with normals and density
        vtk_files = os.listdir(output_folder)
        assert len(vtk_files) > 0
        
        vtk_path = os.path.join(output_folder, vtk_files[0])
        with open(vtk_path, 'r') as f:
            content = f.read()
            assert 'NORMALS normals' in content
            assert 'SCALARS density' in content


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow_without_mesh(self, sample_point_cloud, tmp_path):
        """Test complete workflow using only point clouds (no pymeshlab)."""
        points, normals = sample_point_cloud
        
        # Create grid
        params = vd.make_initial_grid(points, max_points=1000)
        grid = vd.VDERMGrid(params['shape'], params['h'], params['min_bounds'])
        
        # Set simple density
        grid.set_density(lambda x, y, z: 1.0 + 0.5 * x)
        
        # Run VDERM
        final_grid = vd.run_VDERM(grid, n_max=10, max_eps=0.1)
        
        # Interpolate to surface
        final_surface = vd.interpolate_to_surface(
            points, params, final_grid.get_displacement_field()
        )
        
        # Save as XYZ
        output_file = str(tmp_path / 'deformed.xyz')
        final_densities = vd.interpolate_densities(points, final_grid)
        final_velocities = vd.interpolate_velocities(points, params, final_grid.velocities)
        
        vd.write_xyz(output_file, final_surface, 
                    normals=final_velocities, densities=final_densities)
        
        # Verify file
        assert os.path.exists(output_file)
        
        # Read it back
        pos, vel, dens = vd.read_xyz(output_file)
        assert len(pos) == len(points)
        assert vel is not None
        assert dens is not None
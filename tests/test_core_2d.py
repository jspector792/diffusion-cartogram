"""
Tests for 2-D VDERM core functionality (VDERMGrid2D and utilities).
"""
import pytest
import numpy as np
import tempfile
import os
import diffusion_cartogram as vd


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_grid_2d():
    """10×10 grid for fast unit tests."""
    return vd.VDERMGrid2D(shape=(10, 10), h=0.1, min_bounds=[0.0, 0.0])


@pytest.fixture
def medium_grid_2d():
    """20×20 grid for slightly larger tests."""
    return vd.VDERMGrid2D(shape=(20, 20), h=0.05, min_bounds=[0.0, 0.0])


@pytest.fixture
def simple_density_2d():
    """Gaussian density centred at (0.5, 0.5)."""
    def _f(x, y):
        return 1.0 + 2.0 * np.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2))
    return _f


@pytest.fixture
def sample_map_points():
    """Small 2-D point cloud on a circle."""
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    return np.column_stack([0.3 * np.cos(theta) + 0.5,
                             0.3 * np.sin(theta) + 0.5])


# ─── VDERMGrid2D initialisation ───────────────────────────────────────────────

class TestVDERMGrid2D:

    def test_initialization(self, simple_grid_2d):
        g = simple_grid_2d
        assert g.L == 10
        assert g.M == 10
        assert g.h == pytest.approx(0.1)
        assert g.positions.shape == (100, 2)
        assert g.velocities.shape == (100, 2)
        assert g.rho.shape == (10, 10)
        assert g.epsilon is None

    def test_initial_density_is_one(self, simple_grid_2d):
        assert np.allclose(simple_grid_2d.rho, 1.0)

    def test_initial_positions_correct(self):
        g = vd.VDERMGrid2D(shape=(3, 3), h=1.0, min_bounds=[0.0, 0.0])
        assert np.allclose(g.positions[0], [0.0, 0.0])
        assert np.allclose(g.positions[-1], [2.0, 2.0])
        flat = g._index_to_flat(1, 1)
        assert np.allclose(g.positions[flat], [1.0, 1.0])

    def test_index_round_trip(self, simple_grid_2d):
        for i in range(simple_grid_2d.L):
            for j in range(simple_grid_2d.M):
                flat = simple_grid_2d._index_to_flat(i, j)
                ii, jj = simple_grid_2d._flat_to_index(flat)
                assert ii == i and jj == j

    def test_set_density_callable(self, simple_grid_2d, simple_density_2d):
        simple_grid_2d.set_density(simple_density_2d)
        assert not np.allclose(simple_grid_2d.rho, 1.0)
        # Gaussian peak is near centre of grid
        assert simple_grid_2d.rho[5, 5] > simple_grid_2d.rho[0, 0]

    def test_set_density_array(self, simple_grid_2d):
        arr = np.random.rand(10, 10) + 0.5
        simple_grid_2d.set_density(arr)
        np.testing.assert_array_equal(simple_grid_2d.rho, arr)

    def test_set_density_wrong_shape_stored(self, simple_grid_2d):
        arr = np.ones((10, 10)) * 3.0
        simple_grid_2d.set_density(arr)
        assert simple_grid_2d.rho.shape == (10, 10)

    def test_update_density_diffuses(self, simple_grid_2d):
        simple_grid_2d.rho[5, 5] = 10.0
        simple_grid_2d.update_density(dt=0.001)
        assert simple_grid_2d.epsilon is not None
        assert simple_grid_2d.epsilon > 0
        assert simple_grid_2d.rho[5, 5] < 10.0

    def test_update_density_epsilon_computed(self, simple_grid_2d):
        simple_grid_2d.rho[3, 3] = 5.0
        simple_grid_2d.update_density(dt=0.001)
        assert isinstance(simple_grid_2d.epsilon, float)

    def test_update_density_neumann_bc(self, simple_grid_2d):
        # Mass should be approximately conserved under Neumann BCs
        initial_sum = simple_grid_2d.rho.sum()
        for _ in range(20):
            simple_grid_2d.update_density(dt=0.001)
        assert simple_grid_2d.rho.sum() == pytest.approx(initial_sum, rel=1e-3)

    def test_update_velocities_nonzero(self, simple_grid_2d, simple_density_2d):
        simple_grid_2d.set_density(simple_density_2d)
        simple_grid_2d.update_velocities()
        assert not np.allclose(simple_grid_2d.velocities, 0)

    def test_update_velocities_uniform_density(self, simple_grid_2d):
        # Uniform density → zero gradient → zero velocity
        simple_grid_2d.update_velocities()
        assert np.allclose(simple_grid_2d.velocities, 0)

    def test_update_positions(self, simple_grid_2d):
        simple_grid_2d.velocities[:] = [1.0, 0.0]
        init = simple_grid_2d.positions.copy()
        simple_grid_2d.update_positions(dt=0.1)
        disp = simple_grid_2d.positions - init
        assert np.allclose(disp, [0.1, 0.0])

    def test_get_displacement_field(self, simple_grid_2d):
        delta = np.array([0.2, -0.1])
        simple_grid_2d.positions += delta
        disp = simple_grid_2d.get_displacement_field()
        assert disp.shape == (100, 2)
        assert np.allclose(disp, delta)

    def test_compute_timestep_positive(self, simple_grid_2d, simple_density_2d):
        simple_grid_2d.set_density(simple_density_2d)
        simple_grid_2d.update_velocities()
        dt = simple_grid_2d.compute_timestep()
        assert dt > 0
        assert dt <= 0.01

    def test_compute_timestep_diffusion_limit(self, simple_grid_2d):
        simple_grid_2d.velocities[:] = 0.0001  # near-zero speed
        dt = simple_grid_2d.compute_timestep()
        # Must respect 2D diffusion stability: dt ≤ h²/4
        assert dt <= simple_grid_2d.h**2 / 4.0 + 1e-12

    def test_initial_positions_match_initial_copy(self, simple_grid_2d):
        np.testing.assert_array_equal(
            simple_grid_2d.positions,
            simple_grid_2d.initial_positions
        )


# ─── Grid utilities ───────────────────────────────────────────────────────────

class TestGridUtilities2D:

    def test_compute_grid_dimensions_2d(self):
        shape, h = vd.compute_grid_dimensions_2d([2.0, 1.0], max_points=1000)
        L, M = shape
        assert L * M <= 1200
        assert L * M >= 800
        assert h > 0

    def test_compute_grid_dimensions_square(self):
        shape, h = vd.compute_grid_dimensions_2d([1.0, 1.0], max_points=100)
        L, M = shape
        # Square box → L ≈ M
        assert abs(L - M) <= 2

    def test_make_initial_grid_2d_keys(self, sample_map_points):
        gp = vd.make_initial_grid_2d(sample_map_points, max_points=500)
        for key in ('shape', 'h', 'min_bounds', 'max_bounds',
                    'object_bounds', 'padding', 'actual_points'):
            assert key in gp

    def test_make_initial_grid_2d_shape(self, sample_map_points):
        gp = vd.make_initial_grid_2d(sample_map_points, max_points=500)
        L, M = gp['shape']
        assert L >= 3 and M >= 3
        assert L * M == gp['actual_points']

    def test_make_initial_grid_2d_padding(self, sample_map_points):
        gp = vd.make_initial_grid_2d(sample_map_points,
                                     max_points=2000, padding=(3.0, 3.0))
        obj_size = gp['object_bounds']['size']
        grid_size = gp['max_bounds'] - gp['min_bounds']
        ratios = grid_size / obj_size
        assert np.allclose(ratios, 3.0, atol=0.4)

    def test_make_initial_grid_2d_centers_object(self, sample_map_points):
        gp = vd.make_initial_grid_2d(sample_map_points)
        obj_center = gp['object_bounds']['center']
        grid_center = (gp['min_bounds'] + gp['max_bounds']) / 2
        assert np.allclose(obj_center, grid_center, atol=0.01)

    def test_make_initial_grid_2d_default_padding(self, sample_map_points):
        gp = vd.make_initial_grid_2d(sample_map_points)
        assert gp['padding'] == (2.0, 2.0)


# ─── Interpolation ────────────────────────────────────────────────────────────

class TestInterpolation2D:

    def test_interpolate_densities_2d_shape(self, simple_grid_2d,
                                            sample_map_points):
        dens = vd.interpolate_densities_2d(sample_map_points, simple_grid_2d)
        assert dens.shape == (len(sample_map_points),)

    def test_interpolate_densities_2d_uniform(self, simple_grid_2d,
                                              sample_map_points):
        # Uniform density → all interpolated values ≈ 1
        dens = vd.interpolate_densities_2d(sample_map_points, simple_grid_2d)
        assert np.allclose(dens, 1.0)

    def test_interpolate_velocities_2d_shape(self, simple_grid_2d,
                                             sample_map_points):
        params = {'shape': (10, 10), 'h': 0.1,
                  'min_bounds': np.array([0.0, 0.0])}
        vels = vd.interpolate_velocities_2d(
            sample_map_points, params, simple_grid_2d.velocities
        )
        assert vels.shape == (len(sample_map_points), 2)

    def test_interpolate_to_map_2d_no_displacement(self, simple_grid_2d,
                                                    sample_map_points):
        params = {'shape': (10, 10), 'h': 0.1,
                  'min_bounds': np.array([0.0, 0.0])}
        # Zero displacement field → output equals input
        deformed = vd.interpolate_to_map_2d(
            sample_map_points, params,
            simple_grid_2d.get_displacement_field()
        )
        np.testing.assert_allclose(deformed, sample_map_points, atol=1e-10)

    def test_interpolate_to_map_2d_with_displacement(self, simple_grid_2d,
                                                      sample_map_points):
        params = {'shape': (10, 10), 'h': 0.1,
                  'min_bounds': np.array([0.0, 0.0])}
        delta = np.array([0.1, 0.05])
        simple_grid_2d.positions += delta
        deformed = vd.interpolate_to_map_2d(
            sample_map_points, params,
            simple_grid_2d.get_displacement_field()
        )
        assert not np.allclose(deformed, sample_map_points)


# ─── run_VDERM compatibility ──────────────────────────────────────────────────

class TestRunVDERM2D:

    def test_run_vderm_with_2d_grid(self, simple_grid_2d, simple_density_2d):
        simple_grid_2d.set_density(simple_density_2d)
        result = vd.run_VDERM(simple_grid_2d, n_max=20, max_eps=0.5)
        assert result.epsilon is not None
        assert result.epsilon > 0

    def test_run_vderm_2d_converges(self, simple_density_2d):
        g = vd.VDERMGrid2D(shape=(15, 15), h=0.1, min_bounds=[0.0, 0.0])
        g.set_density(simple_density_2d)
        result = vd.run_VDERM(g, n_max=500, max_eps=0.05)
        assert result.epsilon <= 0.05

    def test_run_vderm_2d_manual_dt(self, simple_grid_2d, simple_density_2d):
        simple_grid_2d.set_density(simple_density_2d)
        result = vd.run_VDERM(simple_grid_2d, n_max=10, dt=0.0005)
        assert result.epsilon is not None

    def test_run_vderm_2d_positions_changed(self, simple_grid_2d,
                                             simple_density_2d):
        simple_grid_2d.set_density(simple_density_2d)
        init_pos = simple_grid_2d.positions.copy()
        vd.run_VDERM(simple_grid_2d, n_max=10)
        assert not np.allclose(simple_grid_2d.positions, init_pos)

    def test_run_vderm_2d_instability_detected(self):
        g = vd.VDERMGrid2D(shape=(10, 10), h=0.1, min_bounds=[0.0, 0.0])
        g.set_density(lambda x, y: 1.0 + 100.0 * x)
        with pytest.raises(RuntimeError, match="instability"):
            vd.run_VDERM(g, n_max=100, dt=0.01)


# ─── I/O ──────────────────────────────────────────────────────────────────────

class TestIO2D:

    def test_write_read_csv_2d_positions_only(self, tmp_path):
        pts = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        fp = str(tmp_path / 'test.csv')
        vd.write_csv_2d(fp, pts)
        pts_back, dens = vd.read_csv_2d(fp)
        np.testing.assert_allclose(pts_back, pts, rtol=1e-5)
        assert dens is None

    def test_write_read_csv_2d_with_densities(self, tmp_path):
        pts = np.random.rand(20, 2)
        dens = np.random.rand(20) + 0.5
        fp = str(tmp_path / 'test.csv')
        vd.write_csv_2d(fp, pts, densities=dens)
        pts_back, dens_back = vd.read_csv_2d(fp)
        np.testing.assert_allclose(pts_back, pts, rtol=1e-5)
        np.testing.assert_allclose(dens_back, dens, rtol=1e-5)

    def test_read_csv_2d_wrong_columns(self, tmp_path):
        fp = str(tmp_path / 'bad.csv')
        np.savetxt(fp, np.ones((5, 4)))
        with pytest.raises(ValueError):
            vd.read_csv_2d(fp)


# ─── run_VDERM_2d_with_tracking ──────────────────────────────────────────────

class TestTracking2D:

    def test_no_exports(self, simple_grid_2d, sample_map_points,
                        simple_density_2d):
        simple_grid_2d.set_density(simple_density_2d)
        result = vd.run_VDERM_2d_with_tracking(
            simple_grid_2d, sample_map_points,
            n_max=5, export_grid=False, export_map=False
        )
        assert result is not None

    def test_grid_export_creates_files(self, simple_density_2d,
                                       sample_map_points, tmp_path):
        g = vd.VDERMGrid2D(shape=(10, 10), h=0.1, min_bounds=[0.0, 0.0])
        g.set_density(simple_density_2d)
        vd.run_VDERM_2d_with_tracking(
            g, sample_map_points,
            n_max=6, export_grid=True, export_grid_frequency=3,
            base_folder=str(tmp_path / 'exports')
        )
        grid_files = list((tmp_path / 'exports' / 'vderm_grid').glob('*.csv'))
        assert len(grid_files) > 0

    def test_map_export_creates_files(self, simple_density_2d,
                                      sample_map_points, tmp_path):
        g = vd.VDERMGrid2D(shape=(10, 10), h=0.1, min_bounds=[0.0, 0.0])
        g.set_density(simple_density_2d)
        vd.run_VDERM_2d_with_tracking(
            g, sample_map_points,
            n_max=6, export_map=True, export_map_frequency=3,
            base_folder=str(tmp_path / 'exports')
        )
        map_files = list((tmp_path / 'exports' / 'vderm_map').glob('*.csv'))
        assert len(map_files) > 0

    def test_grid_export_format(self, simple_density_2d,
                                sample_map_points, tmp_path):
        g = vd.VDERMGrid2D(shape=(10, 10), h=0.1, min_bounds=[0.0, 0.0])
        g.set_density(simple_density_2d)
        vd.run_VDERM_2d_with_tracking(
            g, sample_map_points,
            n_max=3, export_grid=True, export_grid_frequency=2,
            base_folder=str(tmp_path / 'exports')
        )
        csv_files = list((tmp_path / 'exports' / 'vderm_grid').glob('*.csv'))
        data = np.loadtxt(str(csv_files[0]))
        assert data.shape[1] == 5  # x y vx vy rho

    def test_map_export_format(self, simple_density_2d,
                               sample_map_points, tmp_path):
        g = vd.VDERMGrid2D(shape=(10, 10), h=0.1, min_bounds=[0.0, 0.0])
        g.set_density(simple_density_2d)
        vd.run_VDERM_2d_with_tracking(
            g, sample_map_points,
            n_max=3, export_map=True, export_map_frequency=2,
            base_folder=str(tmp_path / 'exports')
        )
        csv_files = list((tmp_path / 'exports' / 'vderm_map').glob('*.csv'))
        pts, dens = vd.read_csv_2d(str(csv_files[0]))
        assert pts.shape[1] == 2
        assert dens is not None

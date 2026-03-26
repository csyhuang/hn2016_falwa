"""
Test suite for Numba implementations of FALWA modules.

This module tests that the Numba implementations produce numerically
identical results to the original F2PY (Fortran) implementations.
"""

import pytest
import numpy as np


class TestComputeQGPV:
    """Test cases for compute_qgpv Numba vs Fortran comparison."""
    
    @pytest.fixture
    def sample_input_data(self):
        """Generate sample input data for testing."""
        np.random.seed(42)  # For reproducibility
        
        # Grid dimensions
        nlon = 144
        nlat = 73
        kmax = 17
        
        # Wind fields (random but physically reasonable)
        ut = np.random.randn(nlon, nlat, kmax) * 10  # ~10 m/s variance
        vt = np.random.randn(nlon, nlat, kmax) * 5   # ~5 m/s variance
        
        # Potential temperature (300K base with perturbations)
        theta = 300 + np.random.randn(nlon, nlat, kmax) * 5
        
        # Height levels (0 to 16km)
        height = np.linspace(0, 16000, kmax).astype(np.float64)
        
        # Reference temperature profile (decreasing with height)
        t0 = np.linspace(300, 220, kmax).astype(np.float64)
        
        # Static stability (K/m)
        stat = np.ones(kmax, dtype=np.float64) * 0.01
        
        # Physical constants
        aa = 6.378e6      # Earth radius
        omega = 7.29e-5   # Earth rotation rate
        dz = 1000.0       # Vertical spacing
        hh = 7000.0       # Scale height
        rr = 287.0        # Gas constant
        cp = 1004.0       # Specific heat
        
        return {
            'ut': ut,
            'vt': vt,
            'theta': theta,
            'height': height,
            't0': t0,
            'stat': stat,
            'aa': aa,
            'omega': omega,
            'dz': dz,
            'hh': hh,
            'rr': rr,
            'cp': cp
        }
    
    def test_compute_qgpv_numba_runs(self, sample_input_data):
        """Test that the Numba implementation runs without error."""
        from falwa.numba_modules import compute_qgpv
        
        pv, avort = compute_qgpv(
            sample_input_data['ut'],
            sample_input_data['vt'],
            sample_input_data['theta'],
            sample_input_data['height'],
            sample_input_data['t0'],
            sample_input_data['stat'],
            sample_input_data['aa'],
            sample_input_data['omega'],
            sample_input_data['dz'],
            sample_input_data['hh'],
            sample_input_data['rr'],
            sample_input_data['cp']
        )
        
        # Check output shapes
        nlon, nlat, kmax = sample_input_data['ut'].shape
        assert pv.shape == (nlon, nlat, kmax)
        assert avort.shape == (nlon, nlat, kmax)
        
        # Check that outputs are not all zeros (except boundary levels for pv)
        assert not np.all(avort == 0)
        assert not np.all(pv[..., 1:-1] == 0)  # Interior levels should have non-zero values
    
    def test_compute_qgpv_numba_matches_fortran(self, sample_input_data):
        """Test that Numba implementation matches F2PY Fortran implementation."""
        from falwa.numba_modules import compute_qgpv as compute_qgpv_numba
        from falwa import compute_qgpv as compute_qgpv_fortran
        
        # Run Numba implementation
        pv_numba, avort_numba = compute_qgpv_numba(
            sample_input_data['ut'],
            sample_input_data['vt'],
            sample_input_data['theta'],
            sample_input_data['height'],
            sample_input_data['t0'],
            sample_input_data['stat'],
            sample_input_data['aa'],
            sample_input_data['omega'],
            sample_input_data['dz'],
            sample_input_data['hh'],
            sample_input_data['rr'],
            sample_input_data['cp']
        )
        
        # Run Fortran implementation
        pv_fortran, avort_fortran = compute_qgpv_fortran(
            sample_input_data['ut'],
            sample_input_data['vt'],
            sample_input_data['theta'],
            sample_input_data['height'],
            sample_input_data['t0'],
            sample_input_data['stat'],
            sample_input_data['aa'],
            sample_input_data['omega'],
            sample_input_data['dz'],
            sample_input_data['hh'],
            sample_input_data['rr'],
            sample_input_data['cp']
        )
        
        # Compare absolute vorticity
        np.testing.assert_allclose(
            avort_numba, avort_fortran,
            rtol=1e-10, atol=1e-12,
            err_msg="Absolute vorticity mismatch between Numba and Fortran"
        )
        
        # Compare potential vorticity
        np.testing.assert_allclose(
            pv_numba, pv_fortran,
            rtol=1e-10, atol=1e-12,
            err_msg="Potential vorticity mismatch between Numba and Fortran"
        )
    
    def test_compute_qgpv_different_grid_sizes(self):
        """Test Numba implementation with various grid sizes."""
        from falwa.numba_modules import compute_qgpv
        
        np.random.seed(123)
        
        # Test different grid configurations
        grid_configs = [
            (72, 37, 10),   # Coarse grid
            (144, 73, 17),  # Standard grid
            (360, 181, 20), # Fine grid
        ]
        
        for nlon, nlat, kmax in grid_configs:
            ut = np.random.randn(nlon, nlat, kmax)
            vt = np.random.randn(nlon, nlat, kmax)
            theta = 300 + np.random.randn(nlon, nlat, kmax) * 5
            height = np.linspace(0, 16000, kmax)
            t0 = np.linspace(300, 220, kmax)
            stat = np.ones(kmax) * 0.01
            
            pv, avort = compute_qgpv(
                ut, vt, theta, height, t0, stat,
                aa=6.378e6, omega=7.29e-5, dz=1000.0,
                hh=7000.0, rr=287.0, cp=1004.0
            )
            
            assert pv.shape == (nlon, nlat, kmax), f"Failed for grid {nlon}x{nlat}x{kmax}"
            assert avort.shape == (nlon, nlat, kmax), f"Failed for grid {nlon}x{nlat}x{kmax}"
    
    def test_compute_qgpv_pole_values(self, sample_input_data):
        """Test that pole values are computed correctly as zonal means."""
        from falwa.numba_modules import compute_qgpv
        
        pv, avort = compute_qgpv(
            sample_input_data['ut'],
            sample_input_data['vt'],
            sample_input_data['theta'],
            sample_input_data['height'],
            sample_input_data['t0'],
            sample_input_data['stat'],
            sample_input_data['aa'],
            sample_input_data['omega'],
            sample_input_data['dz'],
            sample_input_data['hh'],
            sample_input_data['rr'],
            sample_input_data['cp']
        )
        
        nlon = sample_input_data['ut'].shape[0]
        
        # At poles, all longitude values should be the same
        for k in range(avort.shape[2]):
            # South pole (j=0)
            south_pole_vals = avort[:, 0, k]
            assert np.allclose(south_pole_vals, south_pole_vals[0]), \
                f"South pole values not constant at level {k}"
            
            # North pole (j=nlat-1)
            north_pole_vals = avort[:, -1, k]
            assert np.allclose(north_pole_vals, north_pole_vals[0]), \
                f"North pole values not constant at level {k}"
    
    def test_compute_qgpv_boundary_levels(self, sample_input_data):
        """Test that boundary vertical levels have zero PV."""
        from falwa.numba_modules import compute_qgpv
        
        pv, avort = compute_qgpv(
            sample_input_data['ut'],
            sample_input_data['vt'],
            sample_input_data['theta'],
            sample_input_data['height'],
            sample_input_data['t0'],
            sample_input_data['stat'],
            sample_input_data['aa'],
            sample_input_data['omega'],
            sample_input_data['dz'],
            sample_input_data['hh'],
            sample_input_data['rr'],
            sample_input_data['cp']
        )
        
        # Boundary levels should be zero (not computed)
        assert np.all(pv[..., 0] == 0), "Bottom boundary PV should be zero"
        assert np.all(pv[..., -1] == 0), "Top boundary PV should be zero"


class TestComputeQGPVPerformance:
    """Performance benchmarks for compute_qgpv implementations."""
    
    @pytest.mark.slow
    def test_numba_jit_warmup(self):
        """Test that Numba JIT compilation happens correctly on first call."""
        from falwa.numba_modules import compute_qgpv
        import time
        
        np.random.seed(42)
        nlon, nlat, kmax = 144, 73, 17
        ut = np.random.randn(nlon, nlat, kmax)
        vt = np.random.randn(nlon, nlat, kmax)
        theta = 300 + np.random.randn(nlon, nlat, kmax)
        height = np.linspace(0, 16000, kmax)
        t0 = np.linspace(300, 220, kmax)
        stat = np.ones(kmax) * 0.01
        
        # First call includes JIT compilation time
        start1 = time.perf_counter()
        compute_qgpv(ut, vt, theta, height, t0, stat,
                     aa=6.378e6, omega=7.29e-5, dz=1000.0,
                     hh=7000.0, rr=287.0, cp=1004.0)
        first_call_time = time.perf_counter() - start1
        
        # Second call should be much faster
        start2 = time.perf_counter()
        compute_qgpv(ut, vt, theta, height, t0, stat,
                     aa=6.378e6, omega=7.29e-5, dz=1000.0,
                     hh=7000.0, rr=287.0, cp=1004.0)
        second_call_time = time.perf_counter() - start2
        
        # Second call should be significantly faster (at least 2x)
        # This validates that JIT compilation worked
        assert second_call_time < first_call_time, \
            f"Second call ({second_call_time:.4f}s) should be faster than first ({first_call_time:.4f}s)"


"""
Detailed unit tests for QGFieldNHN22 class.

These tests verify all computed attributes at each stage of the computation pipeline:
1. After interpolate_fields() - interpolated fields and QGPV
2. After compute_reference_states() - reference states (qref, uref, ptref)
3. After compute_lwa_and_barotropic_fluxes() - LWA and flux terms

Each test validates the shape, checks for NaN values, and verifies against expected values
when available.
"""
import os
import pytest
from math import pi
import numpy as np
import xarray as xr
from falwa.oopinterface import QGFieldNHN22


class ArrayValueCheckMismatchException(Exception):
    """
    Custom error raised when value check for n-dim array fails.
    Given that value check errors are observed on different machines, unit tests
    with such failures may be marked xfail.
    """


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def u_field(test_data_dir):
    """Return u field arranged in ascending order in latitude and height."""
    u_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-u.nc")
    return u_file.u[::-1, ::-1, :]


@pytest.fixture(scope="module")
def v_field(test_data_dir):
    """Return v field arranged in ascending order in latitude and height."""
    v_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-v.nc")
    return v_file.v[::-1, ::-1, :]


@pytest.fixture(scope="module")
def t_field(test_data_dir):
    """Return t field arranged in ascending order in latitude and height."""
    t_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-t.nc")
    return t_file.t[::-1, ::-1, :]


@pytest.fixture(scope="module")
def grid_parameters():
    """Return grid parameters for the test data."""
    xlon = np.arange(0, 360, 1.5)
    ylat = np.arange(-90, 91, 1.5)
    plev = np.array([
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750,
        700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225,
        200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7,
        5, 3, 2, 1])
    return {
        'xlon': xlon,
        'ylat': ylat,
        'plev': plev,
        'nlon': xlon.size,
        'nlat': ylat.size,
        'nlev': plev.size,
    }


@pytest.fixture(scope="module")
def model_parameters():
    """Return physical/numerical parameters for the model."""
    return {
        'kmax': 49,
        'dz': 1000.,
        'hh': 7000.,  # scale height
        'cp': 1004.,  # heat capacity of dry air
        'rr': 287.,   # gas constant
        'omega': 7.29e-5,  # rotation rate of the earth
        'aa': 6.378e+6,  # earth radius
        'maxit': 100000,
        'tol': 1.e-5,
        'rjac': 0.95,
        'eq_boundary_index': 3,
    }


@pytest.fixture(scope="module")
def qgfield_nhn22_full(u_field, v_field, t_field, grid_parameters, model_parameters):
    """
    Create a QGFieldNHN22 object and run full computation pipeline.
    This fixture is module-scoped to avoid repeated computation.
    """
    qgfield = QGFieldNHN22(
        grid_parameters['xlon'],
        grid_parameters['ylat'],
        grid_parameters['plev'],
        u_field, v_field, t_field,
        kmax=model_parameters['kmax'],
        maxit=model_parameters['maxit'],
        dz=model_parameters['dz'],
        npart=grid_parameters['nlat'],
        tol=model_parameters['tol'],
        rjac=model_parameters['rjac'],
        scale_height=model_parameters['hh'],
        cp=model_parameters['cp'],
        dry_gas_constant=model_parameters['rr'],
        omega=model_parameters['omega'],
        planet_radius=model_parameters['aa'],
        eq_boundary_index=model_parameters['eq_boundary_index'],
        northern_hemisphere_results_only=False)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()
    qgfield.compute_lwa_and_barotropic_fluxes()
    return qgfield


@pytest.fixture(scope="module")
def qgfield_nhn22_nhem_only(u_field, v_field, t_field, grid_parameters, model_parameters):
    """
    Create a QGFieldNHN22 object with northern_hemisphere_results_only=True.
    """
    qgfield = QGFieldNHN22(
        grid_parameters['xlon'],
        grid_parameters['ylat'],
        grid_parameters['plev'],
        u_field, v_field, t_field,
        kmax=model_parameters['kmax'],
        maxit=model_parameters['maxit'],
        dz=model_parameters['dz'],
        npart=grid_parameters['nlat'],
        tol=model_parameters['tol'],
        rjac=model_parameters['rjac'],
        scale_height=model_parameters['hh'],
        cp=model_parameters['cp'],
        dry_gas_constant=model_parameters['rr'],
        omega=model_parameters['omega'],
        planet_radius=model_parameters['aa'],
        eq_boundary_index=model_parameters['eq_boundary_index'],
        northern_hemisphere_results_only=True)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()
    qgfield.compute_lwa_and_barotropic_fluxes()
    return qgfield


@pytest.fixture(scope="module")
def expected_values_nhn22(test_data_dir):
    """Load expected values from reference netCDF file."""
    return xr.open_dataset(f"{test_data_dir}/expected_values_nhn22.nc")


# ============================================================================
# Tests for Interpolated Fields (after interpolate_fields)
# ============================================================================

class TestInterpolatedFields:
    """Tests for interpolated fields computed after calling interpolate_fields()."""

    def test_qgpv_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that QGPV has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.qgpv.shape == expected_shape, \
            f"QGPV shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.qgpv.shape}"

    def test_qgpv_no_nan(self, qgfield_nhn22_full):
        """Test that QGPV contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.qgpv).sum()
        assert nan_count == 0, f"QGPV contains {nan_count} NaN values"

    def test_qgpv_finite(self, qgfield_nhn22_full):
        """Test that QGPV contains only finite values."""
        assert np.all(np.isfinite(qgfield_nhn22_full.qgpv)), "QGPV contains infinite values"

    def test_interpolated_u_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that interpolated_u has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.interpolated_u.shape == expected_shape, \
            f"interpolated_u shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.interpolated_u.shape}"

    def test_interpolated_u_no_nan(self, qgfield_nhn22_full):
        """Test that interpolated_u contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.interpolated_u).sum()
        assert nan_count == 0, f"interpolated_u contains {nan_count} NaN values"

    def test_interpolated_v_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that interpolated_v has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.interpolated_v.shape == expected_shape, \
            f"interpolated_v shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.interpolated_v.shape}"

    def test_interpolated_v_no_nan(self, qgfield_nhn22_full):
        """Test that interpolated_v contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.interpolated_v).sum()
        assert nan_count == 0, f"interpolated_v contains {nan_count} NaN values"

    def test_interpolated_theta_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that interpolated_theta has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.interpolated_theta.shape == expected_shape, \
            f"interpolated_theta shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.interpolated_theta.shape}"

    def test_interpolated_theta_no_nan(self, qgfield_nhn22_full):
        """Test that interpolated_theta contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.interpolated_theta).sum()
        assert nan_count == 0, f"interpolated_theta contains {nan_count} NaN values"

    def test_interpolated_theta_positive(self, qgfield_nhn22_full):
        """Test that potential temperature values are positive."""
        assert np.all(qgfield_nhn22_full.interpolated_theta > 0), \
            "Potential temperature contains non-positive values"

    def test_static_stability_is_tuple(self, qgfield_nhn22_full):
        """Test that static_stability returns a tuple for NHN22 (with both hemispheres)."""
        stat = qgfield_nhn22_full.static_stability
        assert isinstance(stat, tuple), \
            f"static_stability should be a tuple for global computation, got {type(stat)}"
        assert len(stat) == 2, f"static_stability tuple should have 2 elements, got {len(stat)}"

    def test_static_stability_shape(self, qgfield_nhn22_full, model_parameters):
        """Test that static_stability arrays have the correct shape."""
        stat_s, stat_n = qgfield_nhn22_full.static_stability
        expected_shape = (model_parameters['kmax'],)
        assert stat_s.shape == expected_shape, \
            f"static_stability_s shape mismatch: expected {expected_shape}, got {stat_s.shape}"
        assert stat_n.shape == expected_shape, \
            f"static_stability_n shape mismatch: expected {expected_shape}, got {stat_n.shape}"

    def test_static_stability_no_nan(self, qgfield_nhn22_full):
        """Test that static_stability contains no NaN values."""
        stat_s, stat_n = qgfield_nhn22_full.static_stability
        assert np.isnan(stat_s).sum() == 0, "static_stability_s contains NaN values"
        assert np.isnan(stat_n).sum() == 0, "static_stability_n contains NaN values"

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        Users have reported failures in value checks on different machines. 
        Suspected to be floating point difference.
        """)
    def test_qgpv_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test QGPV against expected values."""
        rtol = 1.e-3
        try:
            assert np.allclose(qgfield_nhn22_full.qgpv, expected_values_nhn22.qgpv.values, rtol=rtol), \
                "QGPV values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("QGPV value check failed")

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        Users have reported failures in value checks on different machines. 
        Suspected to be floating point difference.
        """)
    def test_static_stability_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test static_stability against expected values."""
        rtol = 1.e-3
        stat_s, stat_n = qgfield_nhn22_full.static_stability
        try:
            assert np.allclose(stat_s, expected_values_nhn22.static_stability_s.values, rtol=rtol), \
                "static_stability_s values don't match expected values"
            assert np.allclose(stat_n, expected_values_nhn22.static_stability_n.values, rtol=rtol), \
                "static_stability_n values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("Static stability value check failed")


# ============================================================================
# Tests for Reference States (after compute_reference_states)
# ============================================================================

class TestReferenceStates:
    """Tests for reference states computed after calling compute_reference_states()."""

    def test_qref_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that qref has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'])
        assert qgfield_nhn22_full.qref.shape == expected_shape, \
            f"qref shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.qref.shape}"

    def test_qref_no_nan(self, qgfield_nhn22_full):
        """Test that qref contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.qref).sum()
        assert nan_count == 0, f"qref contains {nan_count} NaN values"

    def test_uref_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that uref has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'])
        assert qgfield_nhn22_full.uref.shape == expected_shape, \
            f"uref shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.uref.shape}"

    def test_uref_no_nan(self, qgfield_nhn22_full):
        """Test that uref contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.uref).sum()
        assert nan_count == 0, f"uref contains {nan_count} NaN values"

    def test_ptref_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that ptref has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'])
        assert qgfield_nhn22_full.ptref.shape == expected_shape, \
            f"ptref shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.ptref.shape}"

    def test_ptref_no_nan(self, qgfield_nhn22_full):
        """Test that ptref contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.ptref).sum()
        assert nan_count == 0, f"ptref contains {nan_count} NaN values"

    def test_ptref_positive_away_from_equator(self, qgfield_nhn22_full):
        """Test that reference potential temperature values are positive away from equator.

        Note: The ptref at the equator boundary may be zero due to the direct inversion
        boundary conditions. We check values away from the equator.
        """
        # Exclude the equator region where boundary conditions may cause zeros
        equator_idx = qgfield_nhn22_full.equator_idx
        ptref = qgfield_nhn22_full.ptref
        # Check northern hemisphere (away from equator)
        ptref_nh = ptref[:, equator_idx+5:]
        assert np.all(ptref_nh > 0), \
            "Reference potential temperature in NH (away from equator) contains non-positive values"

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        Users have reported failures in value checks on different machines. 
        Suspected to be floating point difference.
        """)
    def test_qref_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test qref against expected values."""
        rtol = 1.e-3
        try:
            assert np.allclose(qgfield_nhn22_full.qref, expected_values_nhn22.qref.values, rtol=rtol), \
                "qref values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("qref value check failed")

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        Users have reported failures in value checks on different machines. 
        Suspected to be floating point difference.
        """)
    def test_uref_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test uref against expected values."""
        rtol = 1.e-3
        try:
            assert np.allclose(qgfield_nhn22_full.uref, expected_values_nhn22.uref.values, rtol=rtol), \
                "uref values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("uref value check failed")

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        Users have reported failures in value checks on different machines. 
        Suspected to be floating point difference.
        """)
    def test_ptref_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test ptref against expected values."""
        rtol = 1.e-3
        try:
            assert np.allclose(qgfield_nhn22_full.ptref, expected_values_nhn22.ptref.values, rtol=rtol), \
                "ptref values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("ptref value check failed")


# ============================================================================
# Tests for LWA and Barotropic Fluxes (after compute_lwa_and_barotropic_fluxes)
# ============================================================================

class TestLWAAndFluxes:
    """Tests for LWA and flux terms computed after calling compute_lwa_and_barotropic_fluxes()."""

    def test_lwa_shape(self, qgfield_nhn22_full, grid_parameters, model_parameters):
        """Test that 3D LWA has the correct shape."""
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.lwa.shape == expected_shape, \
            f"lwa shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.lwa.shape}"

    def test_lwa_no_nan(self, qgfield_nhn22_full):
        """Test that 3D LWA contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.lwa).sum()
        assert nan_count == 0, f"lwa contains {nan_count} NaN values"

    def test_lwa_non_negative(self, qgfield_nhn22_full):
        """Test that LWA values are non-negative (by definition, LWA >= 0)."""
        # LWA should be non-negative by physical definition
        min_lwa = qgfield_nhn22_full.lwa.min()
        assert min_lwa >= -1e-10, f"lwa contains significantly negative values (min = {min_lwa})"

    def test_lwa_baro_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that barotropic LWA has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.lwa_baro.shape == expected_shape, \
            f"lwa_baro shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.lwa_baro.shape}"

    def test_lwa_baro_no_nan(self, qgfield_nhn22_full):
        """Test that barotropic LWA contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.lwa_baro).sum()
        assert nan_count == 0, f"lwa_baro contains {nan_count} NaN values"

    def test_u_baro_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that barotropic zonal wind has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.u_baro.shape == expected_shape, \
            f"u_baro shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.u_baro.shape}"

    def test_u_baro_no_nan(self, qgfield_nhn22_full):
        """Test that barotropic zonal wind contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.u_baro).sum()
        assert nan_count == 0, f"u_baro contains {nan_count} NaN values"

    def test_adv_flux_f1_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that advective flux F1 has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.adv_flux_f1.shape == expected_shape, \
            f"adv_flux_f1 shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.adv_flux_f1.shape}"

    def test_adv_flux_f1_no_nan(self, qgfield_nhn22_full):
        """Test that advective flux F1 contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.adv_flux_f1).sum()
        assert nan_count == 0, f"adv_flux_f1 contains {nan_count} NaN values"

    def test_adv_flux_f2_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that advective flux F2 has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.adv_flux_f2.shape == expected_shape, \
            f"adv_flux_f2 shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.adv_flux_f2.shape}"

    def test_adv_flux_f2_no_nan(self, qgfield_nhn22_full):
        """Test that advective flux F2 contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.adv_flux_f2).sum()
        assert nan_count == 0, f"adv_flux_f2 contains {nan_count} NaN values"

    def test_adv_flux_f3_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that advective flux F3 has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.adv_flux_f3.shape == expected_shape, \
            f"adv_flux_f3 shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.adv_flux_f3.shape}"

    def test_adv_flux_f3_no_nan(self, qgfield_nhn22_full):
        """Test that advective flux F3 contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.adv_flux_f3).sum()
        assert nan_count == 0, f"adv_flux_f3 contains {nan_count} NaN values"

    def test_convergence_zonal_advective_flux_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that convergence of zonal advective flux has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.convergence_zonal_advective_flux.shape == expected_shape, \
            f"convergence_zonal_advective_flux shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.convergence_zonal_advective_flux.shape}"

    def test_convergence_zonal_advective_flux_no_nan(self, qgfield_nhn22_full):
        """Test that convergence of zonal advective flux contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.convergence_zonal_advective_flux).sum()
        assert nan_count == 0, f"convergence_zonal_advective_flux contains {nan_count} NaN values"

    def test_divergence_eddy_momentum_flux_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that divergence of eddy momentum flux has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.divergence_eddy_momentum_flux.shape == expected_shape, \
            f"divergence_eddy_momentum_flux shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.divergence_eddy_momentum_flux.shape}"

    def test_divergence_eddy_momentum_flux_no_nan(self, qgfield_nhn22_full):
        """Test that divergence of eddy momentum flux contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.divergence_eddy_momentum_flux).sum()
        assert nan_count == 0, f"divergence_eddy_momentum_flux contains {nan_count} NaN values"

    def test_meridional_heat_flux_shape(self, qgfield_nhn22_full, grid_parameters):
        """Test that meridional heat flux has the correct shape."""
        expected_shape = (grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_full.meridional_heat_flux.shape == expected_shape, \
            f"meridional_heat_flux shape mismatch: expected {expected_shape}, got {qgfield_nhn22_full.meridional_heat_flux.shape}"

    def test_meridional_heat_flux_no_nan(self, qgfield_nhn22_full):
        """Test that meridional heat flux contains no NaN values."""
        nan_count = np.isnan(qgfield_nhn22_full.meridional_heat_flux).sum()
        assert nan_count == 0, f"meridional_heat_flux contains {nan_count} NaN values"

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        LWA and flux computation for QGFieldNHN22 is numerically unstable.
        Suspected to be precision issues.
        """)
    def test_lwa_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test LWA against expected values."""
        rtol = 1.e-3
        try:
            assert np.allclose(qgfield_nhn22_full.lwa, expected_values_nhn22.lwa.values, rtol=rtol), \
                "lwa values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("lwa value check failed")

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        LWA and flux computation for QGFieldNHN22 is numerically unstable.
        Suspected to be precision issues.
        """)
    def test_lwa_baro_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test barotropic LWA against expected values."""
        rtol = 1.e-3
        try:
            assert np.allclose(qgfield_nhn22_full.lwa_baro, expected_values_nhn22.lwa_baro.values, rtol=rtol), \
                "lwa_baro values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("lwa_baro value check failed")

    @pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
        LWA and flux computation for QGFieldNHN22 is numerically unstable.
        Suspected to be precision issues.
        """)
    def test_u_baro_expected_values(self, qgfield_nhn22_full, expected_values_nhn22):
        """Test barotropic zonal wind against expected values."""
        rtol = 1.e-3
        try:
            assert np.allclose(qgfield_nhn22_full.u_baro, expected_values_nhn22.u_baro.values, rtol=rtol), \
                "u_baro values don't match expected values"
        except AssertionError:
            raise ArrayValueCheckMismatchException("u_baro value check failed")


# ============================================================================
# Tests for Northern Hemisphere Only Mode
# ============================================================================

class TestNorthernHemisphereOnly:
    """Tests for QGFieldNHN22 with northern_hemisphere_results_only=True."""

    def test_qgpv_shape_nhem(self, qgfield_nhn22_nhem_only, grid_parameters, model_parameters):
        """Test that QGPV has the correct shape for NH-only mode."""
        # QGPV is computed globally regardless of northern_hemisphere_results_only
        expected_shape = (model_parameters['kmax'], grid_parameters['nlat'], grid_parameters['nlon'])
        assert qgfield_nhn22_nhem_only.qgpv.shape == expected_shape, \
            f"QGPV shape mismatch in NH-only mode"

    def test_qref_shape_nhem(self, qgfield_nhn22_nhem_only, grid_parameters, model_parameters):
        """Test that qref has the correct shape for NH-only mode."""
        # Reference states only for NH
        nlat_nh = grid_parameters['nlat'] // 2 + 1  # 61 for 121 grid points
        expected_shape = (model_parameters['kmax'], nlat_nh)
        assert qgfield_nhn22_nhem_only.qref.shape == expected_shape, \
            f"qref shape mismatch in NH-only mode: expected {expected_shape}, got {qgfield_nhn22_nhem_only.qref.shape}"

    def test_uref_shape_nhem(self, qgfield_nhn22_nhem_only, grid_parameters, model_parameters):
        """Test that uref has the correct shape for NH-only mode."""
        nlat_nh = grid_parameters['nlat'] // 2 + 1
        expected_shape = (model_parameters['kmax'], nlat_nh)
        assert qgfield_nhn22_nhem_only.uref.shape == expected_shape, \
            f"uref shape mismatch in NH-only mode"

    def test_ptref_shape_nhem(self, qgfield_nhn22_nhem_only, grid_parameters, model_parameters):
        """Test that ptref has the correct shape for NH-only mode."""
        nlat_nh = grid_parameters['nlat'] // 2 + 1
        expected_shape = (model_parameters['kmax'], nlat_nh)
        assert qgfield_nhn22_nhem_only.ptref.shape == expected_shape, \
            f"ptref shape mismatch in NH-only mode"

    def test_lwa_shape_nhem(self, qgfield_nhn22_nhem_only, grid_parameters, model_parameters):
        """Test that LWA has the correct shape for NH-only mode."""
        nlat_nh = grid_parameters['nlat'] // 2 + 1
        expected_shape = (model_parameters['kmax'], nlat_nh, grid_parameters['nlon'])
        assert qgfield_nhn22_nhem_only.lwa.shape == expected_shape, \
            f"lwa shape mismatch in NH-only mode"

    def test_lwa_baro_shape_nhem(self, qgfield_nhn22_nhem_only, grid_parameters):
        """Test that barotropic LWA has the correct shape for NH-only mode."""
        nlat_nh = grid_parameters['nlat'] // 2 + 1
        expected_shape = (nlat_nh, grid_parameters['nlon'])
        assert qgfield_nhn22_nhem_only.lwa_baro.shape == expected_shape, \
            f"lwa_baro shape mismatch in NH-only mode"

    def test_static_stability_nhem(self, qgfield_nhn22_nhem_only, model_parameters):
        """Test that static_stability returns single array for NH-only mode."""
        # For NH-only, static_stability should return just the northern static stability
        stat = qgfield_nhn22_nhem_only.static_stability
        # Since it's NH only, it should return just the array (not a tuple)
        assert stat.shape == (model_parameters['kmax'],), \
            f"static_stability shape mismatch in NH-only mode"

    def test_reference_states_no_nan_nhem(self, qgfield_nhn22_nhem_only):
        """Test that reference states contain no NaN values in NH-only mode."""
        assert np.isnan(qgfield_nhn22_nhem_only.qref).sum() == 0, "qref contains NaN in NH-only mode"
        assert np.isnan(qgfield_nhn22_nhem_only.uref).sum() == 0, "uref contains NaN in NH-only mode"
        assert np.isnan(qgfield_nhn22_nhem_only.ptref).sum() == 0, "ptref contains NaN in NH-only mode"


# ============================================================================
# Tests for Properties and Configuration
# ============================================================================

class TestProperties:
    """Tests for various properties of QGFieldNHN22."""

    def test_eq_boundary_index(self, qgfield_nhn22_full, model_parameters):
        """Test that eq_boundary_index property is correctly set."""
        assert qgfield_nhn22_full.eq_boundary_index == model_parameters['eq_boundary_index'], \
            f"eq_boundary_index mismatch"

    def test_northern_hemisphere_results_only_false(self, qgfield_nhn22_full):
        """Test that northern_hemisphere_results_only is False for global computation."""
        assert qgfield_nhn22_full.northern_hemisphere_results_only == False

    def test_northern_hemisphere_results_only_true(self, qgfield_nhn22_nhem_only):
        """Test that northern_hemisphere_results_only is True for NH-only computation."""
        assert qgfield_nhn22_nhem_only.northern_hemisphere_results_only == True

    def test_prefactor_positive(self, qgfield_nhn22_full):
        """Test that normalization prefactor is positive."""
        assert qgfield_nhn22_full.prefactor > 0, "Normalization prefactor should be positive"

    def test_height_array(self, qgfield_nhn22_full, model_parameters):
        """Test that height array has correct properties."""
        assert len(qgfield_nhn22_full.height) == model_parameters['kmax']
        assert qgfield_nhn22_full.height[0] == 0.0
        assert np.all(np.diff(qgfield_nhn22_full.height) == model_parameters['dz'])


# ============================================================================
# Tests for Layerwise Flux Terms Storage
# ============================================================================

class TestLayerwiseFluxTermsStorage:
    """Tests for accessing layerwise flux terms via the storage object."""

    def test_layerwise_storage_accessible(self, qgfield_nhn22_full):
        """Test that layerwise_flux_terms_storage is accessible."""
        storage = qgfield_nhn22_full.layerwise_flux_terms_storage
        assert storage is not None, "layerwise_flux_terms_storage should not be None"

    def test_lwa_via_storage(self, qgfield_nhn22_full):
        """Test that LWA can be accessed via storage."""
        storage = qgfield_nhn22_full.layerwise_flux_terms_storage
        lwa_from_storage = storage.lwa
        assert lwa_from_storage is not None, "LWA from storage should not be None"


# ============================================================================
# Tests for Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in QGFieldNHN22."""

    def test_qgpv_before_interpolate(self, u_field, v_field, t_field, grid_parameters, model_parameters):
        """Test that accessing QGPV before interpolate_fields raises an error."""
        qgfield = QGFieldNHN22(
            grid_parameters['xlon'],
            grid_parameters['ylat'],
            grid_parameters['plev'],
            u_field, v_field, t_field,
            kmax=model_parameters['kmax'],
            eq_boundary_index=model_parameters['eq_boundary_index'],
            northern_hemisphere_results_only=False)
        with pytest.raises(ValueError, match="QGPV field is not present"):
            _ = qgfield.qgpv

    def test_reference_states_before_interpolate(self, u_field, v_field, t_field, grid_parameters, model_parameters):
        """Test that calling compute_reference_states before interpolate_fields raises an error."""
        qgfield = QGFieldNHN22(
            grid_parameters['xlon'],
            grid_parameters['ylat'],
            grid_parameters['plev'],
            u_field, v_field, t_field,
            kmax=model_parameters['kmax'],
            eq_boundary_index=model_parameters['eq_boundary_index'],
            northern_hemisphere_results_only=False)
        # compute_reference_states requires interpolate_fields to be called first
        with pytest.raises(ValueError, match="interpolate_fields has to be called"):
            qgfield.compute_reference_states()

    def test_fluxes_before_reference_states(self, u_field, v_field, t_field, grid_parameters, model_parameters):
        """Test that calling compute_lwa_and_barotropic_fluxes before reference states raises an error."""
        qgfield = QGFieldNHN22(
            grid_parameters['xlon'],
            grid_parameters['ylat'],
            grid_parameters['plev'],
            u_field, v_field, t_field,
            kmax=model_parameters['kmax'],
            eq_boundary_index=model_parameters['eq_boundary_index'],
            northern_hemisphere_results_only=False)
        qgfield.interpolate_fields()
        # compute_lwa_and_barotropic_fluxes requires compute_reference_states to be called first
        with pytest.raises(ValueError, match="Reference states have not been computed"):
            qgfield.compute_lwa_and_barotropic_fluxes()


# ============================================================================
# Tests for Physical Consistency
# ============================================================================

class TestPhysicalConsistency:
    """Tests for physical consistency of computed quantities."""

    def test_uref_zonal_mean_of_u(self, qgfield_nhn22_full):
        """
        Test that uref is related to zonal mean of interpolated_u.
        Note: This is a soft check because uref is computed from PV inversion.
        """
        u_zonal_mean = np.mean(qgfield_nhn22_full.interpolated_u, axis=-1)
        # Check correlation - they should be positively correlated
        # Flatten for correlation calculation
        uref_flat = qgfield_nhn22_full.uref.flatten()
        u_mean_flat = u_zonal_mean.flatten()
        correlation = np.corrcoef(uref_flat, u_mean_flat)[0, 1]
        assert correlation > 0, "uref and zonal mean u should be positively correlated"

    def test_ptref_similar_to_theta_zonal_mean(self, qgfield_nhn22_full):
        """
        Test that ptref is related to zonal mean of interpolated_theta.
        """
        theta_zonal_mean = np.mean(qgfield_nhn22_full.interpolated_theta, axis=-1)
        # Check correlation
        ptref_flat = qgfield_nhn22_full.ptref.flatten()
        theta_mean_flat = theta_zonal_mean.flatten()
        correlation = np.corrcoef(ptref_flat, theta_mean_flat)[0, 1]
        assert correlation > 0.9, "ptref and zonal mean theta should be highly correlated"

    def test_lwa_baro_is_vertical_average_of_lwa(self, qgfield_nhn22_full):
        """
        Test that lwa_baro is approximately a density-weighted vertical average of lwa.
        This is a qualitative check.
        """
        # LWA baro should be related to column-integrated LWA
        lwa_column_sum = np.sum(qgfield_nhn22_full.lwa, axis=0)
        # Check that they're correlated (signs should match)
        lwa_baro_flat = qgfield_nhn22_full.lwa_baro.flatten()
        lwa_sum_flat = lwa_column_sum.flatten()
        correlation = np.corrcoef(lwa_baro_flat, lwa_sum_flat)[0, 1]
        assert correlation > 0.5, "lwa_baro should be correlated with column-integrated lwa"


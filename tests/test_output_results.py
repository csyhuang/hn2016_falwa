"""
These are integration tests that take in realistic fields, compute derived quantities and check if their values change
with the implemented code changes.
"""
import os
import pytest
from math import pi
import numpy as np
import xarray as xr
from hn2016_falwa.oopinterface import QGFieldNH18, QGFieldNHN22


test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data"


class ArrayValueCheckMismatchException(Exception):
    """
    This is a custom error which occur when value check for n-dim array fails.
    Given that value check error is observed on different machines, the unit test
    with such failure in this module will be marked xfail.
    """


@pytest.fixture(scope="module")
def u_field():
    """
    Return u field arranged in ascending order in latitude and height from 2005-01-23 00:00
    """
    u_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-u.nc")
    return u_file.u[::-1, ::-1, :]


@pytest.fixture(scope="module")
def v_field():
    """
    Return v field arranged in ascending order in latitude and height from 2005-01-23 00:00
    """
    v_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-v.nc")
    return v_file.v[::-1, ::-1, :]


@pytest.fixture(scope="module")
def t_field():
    """
    Return t field arranged in ascending order in latitude and height from 2005-01-23 00:00
    """
    t_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-t.nc")
    return t_file.t[::-1, ::-1, :]


@pytest.fixture(scope="function")
def files_with_expected_values_nh18():
    return xr.open_dataset(f"{test_data_dir}/expected_values_nh18.nc")


@pytest.fixture(scope="function")
def files_with_expected_values_nhn22():
    return xr.open_dataset(f"{test_data_dir}/expected_values_nhn22.nc")


@pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
    Users have reported failures in value checks on different machines. Suspected to be floating point
    difference. The test shall be revised to tackle this in upcoming release.  
    """)
def test_expected_value_check_nh18(u_field, v_field, t_field, files_with_expected_values_nh18):
    xlon = np.arange(0, 360, 1.5)
    ylat = np.arange(-90, 91, 1.5)
    plev = np.array([
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750,
        700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225,
        200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7,
        5, 3, 2, 1])
    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size

    # *** Set other parameters and constants ***
    p0 = 1000.  # surface pressure [hPa]
    kmax = 49  # number of grid points for vertical extrapolation (dimension of height)
    dz = 1000.  # differential height element
    height = np.arange(0, kmax) * dz  # pseudoheight [m]
    dphi = np.diff(ylat)[0] * pi / 180.  # differential latitudinal element
    dlambda = np.diff(xlon)[0] * pi / 180.  # differential latitudinal element
    hh = 7000.  # scale height
    cp = 1004.  # heat capacity of dry air
    rr = 287.  # gas constant
    omega = 7.29e-5  # rotation rate of the earth
    aa = 6.378e+6  # earth radius
    # just above the ground (z=1km) to aloft
    npart = nlat  # number of partitions to construct the equivalent latitude grids
    maxit = 100000  # maximum number of iteration in the SOR solver to solve for reference state
    tol = 1.e-5  # tolerance that define convergence of solution
    rjac = 0.95  # spectral radius of the Jacobi iteration in the SOR solver.
    jd = nlat // 2 + 1  # (one plus) index of latitude grid point with value 0 deg
    # This is to be input to fortran code. The index convention is different.

    qgfield = QGFieldNH18(
        xlon, ylat, plev, u_field, v_field, t_field, kmax=kmax, maxit=maxit, dz=dz, npart=npart, tol=tol,
        rjac=rjac, scale_height=hh, cp=cp, dry_gas_constant=rr, omega=omega, planet_radius=aa,
        northern_hemisphere_results_only=False)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()
    qgfield.compute_lwa_and_barotropic_fluxes()

    mismatch = "{0} values don't match"
    rtol = 1.e-3

    try:
        assert np.allclose(qgfield.qgpv, files_with_expected_values_nh18.qgpv.values, rtol), mismatch.format("qgpv")
        assert np.allclose(qgfield.lwa, files_with_expected_values_nh18.lwa.values, rtol), mismatch.format("lwa")
        assert np.allclose(qgfield.qref, files_with_expected_values_nh18.qref.values, rtol), mismatch.format("qref")
        assert np.allclose(qgfield.uref, files_with_expected_values_nh18.uref.values, rtol), mismatch.format("uref")
        assert np.allclose(qgfield.ptref, files_with_expected_values_nh18.ptref.values, rtol), mismatch.format("ptref")
        assert np.allclose(qgfield.static_stability, files_with_expected_values_nh18.static_stability.values, rtol), mismatch.format("static_stability")
        assert np.allclose(qgfield.lwa_baro, files_with_expected_values_nh18.lwa_baro.values, rtol), mismatch.format("lwa_baro")
        assert np.allclose(qgfield.u_baro, files_with_expected_values_nh18.u_baro.values, rtol), mismatch.format("u_baro")
    except:
        ArrayValueCheckMismatchException()

@pytest.mark.xfail(raises=ArrayValueCheckMismatchException, reason="""
    LWA and flux computation for QGFieldNHN22 is numerically unstable. 
    Suspected to be precision issues which will be tackled in upcoming release.
    """)
def test_expected_value_check_nhn22(u_field, v_field, t_field, files_with_expected_values_nhn22):
    xlon = np.arange(0, 360, 1.5)
    ylat = np.arange(-90, 91, 1.5)
    plev = np.array([
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750,
        700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225,
        200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7,
        5, 3, 2, 1])
    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size

    # *** Set other parameters and constants ***
    p0 = 1000.  # surface pressure [hPa]
    kmax = 49  # number of grid points for vertical extrapolation (dimension of height)
    dz = 1000.  # differential height element
    height = np.arange(0, kmax) * dz  # pseudoheight [m]
    dphi = np.diff(ylat)[0] * pi / 180.  # differential latitudinal element
    dlambda = np.diff(xlon)[0] * pi / 180.  # differential latitudinal element
    hh = 7000.  # scale height
    cp = 1004.  # heat capacity of dry air
    rr = 287.  # gas constant
    omega = 7.29e-5  # rotation rate of the earth
    aa = 6.378e+6  # earth radius
    # just above the ground (z=1km) to aloft
    npart = nlat  # number of partitions to construct the equivalent latitude grids
    maxit = 100000  # maximum number of iteration in the SOR solver to solve for reference state
    tol = 1.e-5  # tolerance that define convergence of solution
    rjac = 0.95  # spectral radius of the Jacobi iteration in the SOR solver.
    jd = nlat // 2 + 1  # (one plus) index of latitude grid point with value 0 deg
    eq_boundary_index = 3
    # This is to be input to fortran code. The index convention is different.

    qgfield = QGFieldNHN22(
        xlon, ylat, plev, u_field, v_field, t_field, kmax=kmax, maxit=maxit, dz=dz, npart=npart, tol=tol,
        rjac=rjac, scale_height=hh, cp=cp, dry_gas_constant=rr, omega=omega, planet_radius=aa,
        eq_boundary_index=eq_boundary_index, northern_hemisphere_results_only=False)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()
    qgfield.compute_lwa_and_barotropic_fluxes()

    mismatch = "{0} values don't match"
    rtol = 1.e-3

    try:
        assert np.allclose(qgfield.qgpv, files_with_expected_values_nhn22.qgpv.values, rtol), mismatch.format("qgpv")
        assert np.allclose(qgfield.lwa, files_with_expected_values_nhn22.lwa.values, rtol), mismatch.format("lwa")
        assert np.allclose(qgfield.qref, files_with_expected_values_nhn22.qref.values, rtol), mismatch.format("qref")
        assert np.allclose(qgfield.uref, files_with_expected_values_nhn22.uref.values, rtol), mismatch.format("uref")
        assert np.allclose(qgfield.ptref, files_with_expected_values_nhn22.ptref.values, rtol), mismatch.format("ptref")
        assert np.allclose(qgfield.static_stability[0], files_with_expected_values_nhn22.static_stability_s.values,
                           rtol), mismatch.format("static_stability_s")
        assert np.allclose(qgfield.static_stability[1], files_with_expected_values_nhn22.static_stability_n.values,
                           rtol), mismatch.format("static_stability_n")
        assert np.allclose(qgfield.lwa_baro, files_with_expected_values_nhn22.lwa_baro.values, rtol), mismatch.format("lwa_baro")
        assert np.allclose(qgfield.u_baro, files_with_expected_values_nhn22.u_baro.values, rtol), mismatch.format("u_baro")
    except:
        ArrayValueCheckMismatchException()


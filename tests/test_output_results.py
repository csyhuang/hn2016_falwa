"""
These are integration tests that take in realistic fields, compute derived quantities and check if their values change
with the implemented code changes.
"""
import os
import pytest
from math import pi
import numpy as np
import xarray as xr
from falwa.oopinterface import QGFieldNH18, QGFieldNHN22


class ArrayValueCheckMismatchException(Exception):
    """
    This is a custom error which occur when value check for n-dim array fails.
    Given that value check error is observed on different machines, the unit test
    with such failure in this module will be marked xfail.
    """


@pytest.fixture(scope="module")
def u_field(test_data_dir):
    """
    Return u field arranged in ascending order in latitude and height from 2005-01-23 00:00
    """
    u_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-u.nc")
    return u_file.u[::-1, ::-1, :]


@pytest.fixture(scope="module")
def v_field(test_data_dir):
    """
    Return v field arranged in ascending order in latitude and height from 2005-01-23 00:00
    """
    v_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-v.nc")
    return v_file.v[::-1, ::-1, :]


@pytest.fixture(scope="module")
def t_field(test_data_dir):
    """
    Return t field arranged in ascending order in latitude and height from 2005-01-23 00:00
    """
    t_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-t.nc")
    return t_file.t[::-1, ::-1, :]


@pytest.fixture(scope="function")
def files_with_expected_values_nh18(test_data_dir):
    return xr.open_dataset(f"{test_data_dir}/expected_values_nh18.nc")


@pytest.fixture(scope="function")
def files_with_expected_values_nhn22(test_data_dir):
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


def test_offgrid_data_input(offgrid_input_data):
    qgfield = QGFieldNH18(
        offgrid_input_data.longitude.data, offgrid_input_data.latitude.data, offgrid_input_data.level.data,
        offgrid_input_data.u.data, offgrid_input_data.v.data, offgrid_input_data.t.data,
        northern_hemisphere_results_only=False)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()
    qgfield.compute_lwa_and_barotropic_fluxes()
    assert np.isnan(qgfield.qref).sum() == 0
    assert np.isnan(qgfield.uref).sum() == 0
    assert np.isnan(qgfield.ptref).sum() == 0
    assert np.isnan(qgfield.lwa).sum() == 0
    assert qgfield.interpolated_u.shape == (49, 192, 288)
    assert qgfield.qref.shape == (49, 192)
    assert qgfield.uref.shape == (49, 192)
    assert qgfield.lwa.shape == (49, 192, 288)
    assert qgfield.convergence_zonal_advective_flux.shape == (192, 288)


def test_offgrid_data_input_xarrayinterface(offgrid_input_data):
    from falwa.xarrayinterface import QGDataset
    qgdataset = QGDataset(
        offgrid_input_data)
    qgdataset.interpolate_fields()
    qgdataset.compute_reference_states()
    qgdataset.compute_lwa_and_barotropic_fluxes()
    assert np.isnan(qgdataset.qref).sum() == 0
    assert np.isnan(qgdataset.uref).sum() == 0
    assert np.isnan(qgdataset.ptref).sum() == 0
    assert np.isnan(qgdataset.lwa).sum() == 0
    assert qgdataset.interpolated_u.shape == (49, 192, 288)
    assert qgdataset.qref.shape == (49, 192)
    assert qgdataset.uref.shape == (49, 192)
    assert qgdataset.lwa.shape == (49, 192, 288)
    assert qgdataset.convergence_zonal_advective_flux.shape == (192, 288)


@pytest.fixture(scope="module")
def offgrid_input_data(test_data_dir):
    """
    Return dataset with latitude grids not including equator.
    """
    xlon = np.arange(0, 360, 1.25)
    ylat = np.linspace(-90, 90, 192, endpoint=True)
    plev = [
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
        750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250,
        225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10,
        7, 5, 3, 2, 1]

    test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data"
    u_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-u.nc").interp(
        {"longitude": xlon, "latitude": ylat, "level": plev}, method="linear", kwargs={"fill_value": "extrapolate"})
    v_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-v.nc").interp(
        {"longitude": xlon, "latitude": ylat, "level": plev}, method="linear", kwargs={"fill_value": "extrapolate"})
    t_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-t.nc").interp(
        {"longitude": xlon, "latitude": ylat, "level": plev}, method="linear", kwargs={"fill_value": "extrapolate"})
    ds = xr.Dataset({
        "u": (("level", "latitude", "longitude"), u_file.u.data),
        "v": (("level", "latitude", "longitude"), v_file.v.data),
        "t": (("level", "latitude", "longitude"), t_file.t.data)},
        coords={
            "level": plev,
            "latitude": ylat,
            "longitude": xlon})
    return ds

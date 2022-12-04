import os
import pytest

import numpy as np
from math import pi
from scipy.interpolate import interp1d

from hn2016_falwa.constant import *
from hn2016_falwa.oopinterface import QGField

# === Parameters specific for testing the qgfield class ===
nlat = 121
nlon = 240
xlon = np.linspace(0, 2. * pi, nlon, endpoint=False)
ylat = np.linspace(-90., 90., nlat, endpoint=True)
plev = np.array([1000,  900,  800,  700,  600,  500,  400,  300,  200, 100,   10,    1])
nlev = plev.size
kmax = 49
get_cwd = os.path.dirname(os.path.abspath(__file__))
u_field = np.reshape(np.loadtxt(get_cwd + '/test_data/global_demo_u.txt'), [nlev, nlat, nlon])
v_field = np.reshape(np.loadtxt(get_cwd + '/test_data/global_demo_v.txt'), [nlev, nlat, nlon])
t_field = np.reshape(np.loadtxt(get_cwd + '/test_data/global_demo_t.txt'), [nlev, nlat, nlon])
theta_field = t_field * (plev[:, np.newaxis, np.newaxis] / P_GROUND) ** (-DRY_GAS_CONSTANT / CP)


def test_qgfield():

    # Create a QGField object out of some masked array for testing. The test below is to ensure
    # warning is raised for masked array.
    with pytest.warns(UserWarning):
        qgfield = QGField(xlon=xlon, ylat=np.ma.masked_equal(ylat, 0.0), plev=plev,
                          u_field=np.ma.masked_equal(u_field, 0.0), v_field=np.ma.masked_equal(v_field, 0.0),
                          t_field=np.ma.masked_equal(t_field, 0.0), kmax=kmax, maxit=100000, dz=1000., npart=None,
                          tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
                          omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS)

    # Check that the input fields are interpolated onto a grid of correct dimension
    # and the interpolated values are bounded.
    qgpv, interpolated_u, interpolated_v, interpolated_theta, static_stability = \
        qgfield.interpolate_fields()

    # Check that the dimensions of the interpolated fields are correct
    assert (kmax, nlat, nlon) == qgpv.shape
    assert (kmax, nlat, nlon) == interpolated_u.shape
    assert (kmax, nlat, nlon) == interpolated_v.shape
    assert (kmax, nlat, nlon) == interpolated_theta.shape
    assert (kmax,) == static_stability.shape

    assert (interpolated_u[1:-1, :, :].max() <= u_field.max()) & \
           (interpolated_u[1:-1, :, :].max() >= u_field.min())
    assert (interpolated_u[1:-1, :, :].min() <= u_field.max()) & \
           (interpolated_u[1:-1, :, :].min() >= u_field.min())
    assert (interpolated_v[1:-1, :, :].max() <= v_field.max()) & \
           (interpolated_v[1:-1, :, :].max() >= v_field.min())
    assert (interpolated_v[1:-1, :, :].min() <= v_field.max()) & \
           (interpolated_v[1:-1, :, :].min() >= v_field.min())
    assert (interpolated_theta[1:-1, :, :].max() <= theta_field.max()) & \
           (interpolated_theta[1:-1, :, :].max() >= theta_field.min())
    assert (interpolated_theta[1:-1, :, :].min() <= theta_field.max()) & \
           (interpolated_theta[1:-1, :, :].min() >= theta_field.min())
    assert 0 == np.isnan(qgpv).sum()
    assert 0 == (qgpv == float('Inf')).sum()

    # Check that the output reference states are of correct dimension, and
    # the QGPV reference state is non-decreasing.
    qref_north_hem, uref_north_hem, ptref_north_hem = \
        qgfield.compute_reference_states(northern_hemisphere_results_only=True)

    # Check dimension of the input field
    assert (kmax, nlat // 2 + 1) == qref_north_hem.shape
    assert (kmax, nlat // 2 + 1) == uref_north_hem.shape
    assert (kmax, nlat // 2 + 1) == ptref_north_hem.shape

    # Check if qref is monotonically increasing in N. Hem
    assert (np.diff(qref_north_hem, axis=-1)[1:-1, 1:-1] >= 0.).all()

    # Check LWA and fluxes
    lwa_and_fluxes = qgfield.compute_lwa_and_barotropic_fluxes(northern_hemisphere_results_only=True)
    assert lwa_and_fluxes.adv_flux_f1.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.adv_flux_f2.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.adv_flux_f3.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.convergence_zonal_advective_flux.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.divergence_eddy_momentum_flux.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.meridional_heat_flux.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.lwa_baro.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.u_baro.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.lwa.shape == (kmax, nlat // 2 + 1, nlon)


def test_qgfield_full_globe():

    # Create a QGField object for testing
    qgfield = QGField(xlon=xlon, ylat=ylat, plev=plev, u_field=u_field, v_field=v_field, t_field=t_field, kmax=kmax,
                      maxit=100000, dz=1000., npart=None, tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP,
                      dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS)

    # Check that the input fields are interpolated onto a grid of correct dimension
    # and the interpolated values are bounded.
    interpolated_fields = qgfield.interpolate_fields()

    # Check that the output reference states are of correct dimension, and
    # the QGPV reference state is non-decreasing.
    qref_full_hem, uref_full_hem, ptref_full_hem = qgfield.compute_reference_states(
        northern_hemisphere_results_only=False)

    # Check dimension of the input field
    assert (kmax, nlat) == qref_full_hem.shape
    assert (kmax, nlat) == uref_full_hem.shape
    assert (kmax, nlat) == ptref_full_hem.shape

    # Check if qref is monotonically increasing in both hemisphere (exclude equator)
    assert (np.diff(qref_full_hem, axis=-1)[1:-1, 1:nlat//2-1] >= 0.).all()  # South Hem
    assert (np.diff(qref_full_hem, axis=-1)[1:-1, nlat // 2 + 1:-1] >= 0.).all()  # North Hem

    # Check LWA and fluxes
    lwa_and_fluxes = qgfield.compute_lwa_and_barotropic_fluxes(northern_hemisphere_results_only=False)
    assert lwa_and_fluxes.adv_flux_f1.shape == (nlat, nlon)
    assert lwa_and_fluxes.adv_flux_f2.shape == (nlat, nlon)
    assert lwa_and_fluxes.adv_flux_f3.shape == (nlat, nlon)
    assert lwa_and_fluxes.convergence_zonal_advective_flux.shape == (nlat, nlon)
    assert lwa_and_fluxes.divergence_eddy_momentum_flux.shape == (nlat, nlon)
    assert lwa_and_fluxes.meridional_heat_flux.shape == (nlat, nlon)
    assert lwa_and_fluxes.lwa_baro.shape == (nlat, nlon)
    assert lwa_and_fluxes.u_baro.shape == (nlat, nlon)
    assert lwa_and_fluxes.lwa.shape == (kmax, nlat, nlon)


def test_raise_error_for_unrealistic_fields():
    """
    Error shall be raised if the SOR algorithm for computing reference state does not converge.
    """
    qgfield = QGField(xlon=xlon, ylat=ylat, plev=plev, u_field=u_field, v_field=u_field, t_field=u_field, kmax=kmax,
                      maxit=100000, dz=1000., npart=None, tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP,
                      dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS)
    qgfield.interpolate_fields()
    with pytest.raises(ValueError):
        qgfield.compute_reference_states()


def test_raise_error_for_unrealistic_kmax():
    """
    Error shall be raised if kmax is out of bound of the pressure grid provided in the input data

    .. versionadded:: 0.4.0
    """
    too_large_kmax = 1000
    with pytest.raises(ValueError):
        QGField(xlon=xlon, ylat=ylat, plev=plev, u_field=u_field, v_field=v_field, t_field=t_field, kmax=too_large_kmax)


def test_interpolate_fields_even_grids():
    """
    To test whether the new features of even-to-odd grid interpolation works well.

    .. versionadded:: 0.3.5

    """
    ylat_even = np.linspace(-90., 90., nlat + 1, endpoint=True)[1:-1]
    u_field_even = interp1d(ylat, u_field, axis=1,
                            fill_value="extrapolate")(ylat_even)
    v_field_even = interp1d(ylat, v_field, axis=1,
                            fill_value="extrapolate")(ylat_even)
    t_field_even = interp1d(ylat, t_field, axis=1,
                            fill_value="extrapolate")(ylat_even)

    # Create a QGField object for testing
    qgfield_even = QGField(xlon=xlon, ylat=ylat_even, plev=plev, u_field=u_field_even, v_field=v_field_even,
                           t_field=t_field_even, kmax=kmax, maxit=100000, dz=1000., npart=None, tol=1.e-5, rjac=0.95,
                           scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA,
                           planet_radius=EARTH_RADIUS)

    qgpv, interpolated_u, interpolated_v, interpolated_theta, static_stability = \
        qgfield_even.interpolate_fields()

    assert kmax == qgfield_even.kmax
    assert nlat-1 == qgfield_even.get_latitude_dim()
    assert nlon == qgfield_even.nlon

    # Check that the dimensions of the interpolated fields are correct
    assert (kmax, nlat-1, nlon) == qgpv.shape
    assert (kmax, nlat-1, nlon) == interpolated_u.shape
    assert (kmax, nlat-1, nlon) == interpolated_v.shape
    assert (kmax, nlat-1, nlon) == interpolated_theta.shape
    assert (kmax,) == static_stability.shape

    # Check that at the interior grid points, the interpolated fields
    # are bounded
    assert (interpolated_u[1:-1, 1:-1, 1:-1].max() <= u_field.max()) & \
           (interpolated_u[1:-1, 1:-1, 1:-1].max() >= u_field.min())

    assert (interpolated_u[1:-1, 1:-1, 1:-1].min() <= u_field.max()) & \
           (interpolated_u[1:-1, 1:-1, 1:-1].min() >= u_field.min())

    assert (interpolated_v[1:-1, 1:-1, 1:-1].max() <= v_field.max()) & \
           (interpolated_v[1:-1, 1:-1, 1:-1].max() >= v_field.min())

    assert (interpolated_v[1:-1, 1:-1, 1:-1].min() <= v_field.max()) & \
           (interpolated_v[1:-1, 1:-1, 1:-1].min() >= v_field.min())

    assert (interpolated_theta[1:-1, 1:-1, 1:-1].max() <= theta_field.max()) & \
           (interpolated_theta[1:-1, 1:-1, 1:-1].max() >= theta_field.min())

    assert (interpolated_theta[1:-1, 1:-1, 1:-1].min() <= theta_field.max()) & \
           (interpolated_theta[1:-1, 1:-1, 1:-1].min() >= theta_field.min())

    assert 0 == np.isnan(qgpv).sum()
    assert 0 == (qgpv == float('Inf')).sum()


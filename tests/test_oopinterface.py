import os
import pytest

import numpy as np
from math import pi
from scipy.interpolate import interp1d

from hn2016_falwa.constant import *
from hn2016_falwa.oopinterface import QGFieldNH18

# === Parameters specific for testing the qgfield class ===
nlat = 121
nlon = 240
xlon = np.linspace(0, 2. * pi, nlon, endpoint=False)
ylat = np.linspace(-90., 90., nlat, endpoint=True)
plev = np.array([1000,  900,  800,  700,  600,  500,  400,  300,  200, 100,   10,    1])
nlev = plev.size
kmax = 49
get_cwd = os.path.dirname(os.path.abspath(__file__))
u_field = np.reshape(np.loadtxt(get_cwd + '/data/global_demo_u.txt'), [nlev, nlat, nlon])
v_field = np.reshape(np.loadtxt(get_cwd + '/data/global_demo_v.txt'), [nlev, nlat, nlon])
t_field = np.reshape(np.loadtxt(get_cwd + '/data/global_demo_t.txt'), [nlev, nlat, nlon])
theta_field = t_field * (plev[:, np.newaxis, np.newaxis] / P_GROUND) ** (-DRY_GAS_CONSTANT / CP)


def test_qgfield():

    # Create a QGFieldNH18 object out of some masked array for testing. The test below is to ensure
    # warning is raised for masked array.
    with pytest.warns(UserWarning):
        qgfield = QGFieldNH18(
            xlon=xlon, ylat=np.ma.masked_equal(ylat, 0.0), plev=plev,
            u_field=np.ma.masked_equal(u_field, 0.0), v_field=np.ma.masked_equal(v_field, 0.0),
            t_field=np.ma.masked_equal(t_field, 0.0), kmax=kmax, maxit=100000, dz=1000., npart=None,
            tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
            omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=True)

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
        qgfield.compute_reference_states()

    # Check dimension of the input field
    assert (kmax, nlat // 2 + 1) == qref_north_hem.shape
    assert (kmax, nlat // 2 + 1) == uref_north_hem.shape
    assert (kmax, nlat // 2 + 1) == ptref_north_hem.shape

    # Check if qref is monotonically increasing in N. Hem
    assert (np.diff(qref_north_hem, axis=-1)[1:-1, 1:-1] >= 0.).all()

    # Check numerical values at maybe 45N
    np.testing.assert_allclose(qref_north_hem[1:-1, 30], np.array([
       1.38131594e-04, 1.19859978e-04, 1.05491689e-04, 9.93276096e-05,
       9.22312472e-05, 1.15428025e-04, 1.76390117e-04, 2.59869051e-04,
       2.68715711e-04, 2.38162745e-04, 2.18186744e-04, 1.86275673e-04,
       1.68824676e-04, 1.56962190e-04, 9.88995968e-05, 7.78025576e-05,
       7.11415702e-05, 8.75867385e-05, 8.90332624e-05, 8.90447767e-05,
       8.88902142e-05, 8.85380061e-05, 8.81466043e-05, 8.77005638e-05,
       8.73810977e-05, 8.70058817e-05, 8.67982961e-05, 8.65848763e-05,
       8.64754241e-05, 8.63178982e-05, 9.19433192e-05, 9.59023756e-05,
       9.09137157e-05, 8.63107880e-05, 8.63246689e-05, 8.76919204e-05,
       8.96391234e-05, 9.18857274e-05, 9.40389724e-05, 9.59951480e-05,
       9.83253872e-05, 1.00259563e-04, 1.02466469e-04, 1.04852489e-04,
       1.06957177e-04, 1.09106934e-04, 1.10848619e-04]), rtol=1.e-4)
    np.testing.assert_allclose(uref_north_hem[1:-1, 30], np.array([
       4.0792756,  7.9502687, 11.6558895, 15.18368, 18.527643,
       21.687384, 24.602285, 27.075981, 28.873444, 29.869736,
       30.134935, 30.241474, 30.34288, 30.507023, 30.719505,
       30.744917, 30.567951, 30.155119, 29.801447, 29.569159,
       29.49587, 29.563114, 29.779781, 30.163935, 30.708176,
       31.40156, 32.236263, 33.202354, 34.27791, 35.433662,
       36.633213, 38.06651, 39.856266, 41.948883, 44.131187,
       46.426163, 48.819324, 51.259815, 53.726505, 56.18757,
       58.665966, 61.160248, 63.64206, 66.105194, 68.56923,
       71.03153, 73.52619]), rtol=1.e-4)
    np.testing.assert_allclose(ptref_north_hem[1:-1, 30], np.array([
        282.9861, 289.11166, 294.43488, 298.87775, 302.74335,
        305.4141, 307.44733, 310.67563, 318.58322, 329.07108,
        339.64813, 352.70694, 366.25687, 379.23846, 389.83984,
        408.4669, 442.00266, 466.65228, 492.23962, 518.2976,
        544.6688, 571.2527, 597.9376, 624.6638, 651.401,
        678.0977, 704.8375, 731.67053, 758.4729, 785.2347,
        811.9185, 836.7763, 887.0521, 945.8203, 1005.51154,
        1065.4585, 1125.3812, 1185.4506, 1245.7306, 1306.2148,
        1366.9077, 1427.694, 1488.4418, 1549.4957, 1610.9202,
        1672.3165, 1733.3434]), rtol=1.e-4)

    # Check LWA and fluxes
    lwa_and_fluxes = qgfield.compute_lwa_and_barotropic_fluxes()
    assert lwa_and_fluxes.adv_flux_f1.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.adv_flux_f2.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.adv_flux_f3.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.convergence_zonal_advective_flux.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.divergence_eddy_momentum_flux.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.meridional_heat_flux.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.lwa_baro.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.u_baro.shape == (nlat // 2 + 1, nlon)
    assert lwa_and_fluxes.lwa.shape == (kmax, nlat // 2 + 1, nlon)

    # Check values at midlatitudes, maybe an average of 30-60 deg
    np.testing.assert_allclose(
        lwa_and_fluxes.lwa_baro[20:40, :].mean(axis=-1), np.array([
            19.630522, 21.613638, 23.790615, 25.932142, 28.172373, 30.730598, 33.59619, 36.23143, 38.318214, 40.161243,
            42.13259, 44.062523, 45.303726, 45.888157, 46.33659, 46.821716, 46.8923, 46.248104, 45.019547, 43.42369]),
        rtol=1.e-4)
    np.testing.assert_allclose(
        lwa_and_fluxes.meridional_heat_flux[20:40, :].mean(axis=-1), np.array([
            1.1194464e-05, 1.6376511e-05, 2.3858540e-05, 3.2586056e-05, 3.9103314e-05, 4.2148498e-05, 4.5597390e-05,
            5.1362793e-05, 5.7720707e-05, 6.1747254e-05, 6.7472894e-05, 7.2880815e-05, 6.9412905e-05, 6.0302500e-05,
            5.6556517e-05, 5.2262596e-05, 4.7150170e-05, 4.1664603e-05, 3.7316819e-05, 3.1581640e-05]), atol=1.e-8)
    np.testing.assert_allclose(
            lwa_and_fluxes.divergence_eddy_momentum_flux[20:40, :].mean(axis=-1), np.array([
            -2.2376915e-05, -2.8482620e-05, -2.9509038e-05, -2.8688662e-05,
            -3.0097528e-05, -2.2539192e-05, -1.0400371e-05, -4.7039898e-06,
            -4.3675382e-06, -7.4572531e-06, -7.7944615e-06, -5.3719723e-06,
            -1.5880020e-07,  8.9033156e-06,  2.3703995e-05,  3.7202812e-05,
            3.2651566e-05,  1.3700411e-05, -4.6122295e-08,  4.2839401e-06]), atol=1.e-8)


def test_qgfield_full_globe():

    # Create a QGFieldNH18 object for testing
    qgfield = QGFieldNH18(
        xlon=xlon, ylat=ylat, plev=plev, u_field=u_field, v_field=v_field, t_field=t_field, kmax=kmax,
        maxit=100000, dz=1000., npart=None, tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP,
        dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS,
        northern_hemisphere_results_only=False)

    # Check that the input fields are interpolated onto a grid of correct dimension
    # and the interpolated values are bounded.
    interpolated_fields = qgfield.interpolate_fields()

    # Check that the output reference states are of correct dimension, and
    # the QGPV reference state is non-decreasing.
    qref_full_hem, uref_full_hem, ptref_full_hem = qgfield.compute_reference_states()

    # Check dimension of the input field
    assert (kmax, nlat) == qref_full_hem.shape
    assert (kmax, nlat) == uref_full_hem.shape
    assert (kmax, nlat) == ptref_full_hem.shape

    # Check if qref is monotonically increasing in both hemisphere (exclude equator)
    assert (np.diff(qref_full_hem, axis=-1)[1:-1, 1:nlat//2-1] >= 0.).all()  # South Hem
    assert (np.diff(qref_full_hem, axis=-1)[1:-1, nlat // 2 + 1:-1] >= 0.).all()  # North Hem

    # Check LWA and fluxes
    lwa_and_fluxes = qgfield.compute_lwa_and_barotropic_fluxes()
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
    qgfield = QGFieldNH18(
        xlon=xlon, ylat=ylat, plev=plev, u_field=u_field, v_field=u_field, t_field=u_field, kmax=kmax,
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
        QGFieldNH18(xlon=xlon, ylat=ylat, plev=plev, u_field=u_field, v_field=v_field, t_field=t_field, kmax=too_large_kmax)


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

    # Create a QGFieldNH18 object for testing
    qgfield_even = QGFieldNH18(
        xlon=xlon, ylat=ylat_even, plev=plev, u_field=u_field_even, v_field=v_field_even,
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


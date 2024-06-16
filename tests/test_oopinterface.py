import os
import pytest

import numpy as np
from math import pi
from scipy.interpolate import interp1d

from falwa.constant import *
from falwa.oopinterface import QGFieldNH18

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
        1.38362630e-04, 1.20079159e-04, 1.05550326e-04, 9.96177671e-05,
        9.35690857e-05, 1.15448156e-04, 1.72288336e-04, 2.56547577e-04,
        2.69101661e-04, 2.39006153e-04, 2.19803634e-04, 1.82800421e-04,
        1.65751667e-04, 1.56776645e-04, 1.17109288e-04, 7.60578398e-05,
        7.00473866e-05, 8.18840656e-05, 8.36232019e-05, 8.38833570e-05,
        8.39077404e-05, 8.37530951e-05, 8.35119526e-05, 8.33404699e-05,
        8.33300787e-05, 8.34126185e-05, 8.34538208e-05, 8.33546219e-05,
        8.34762132e-05, 8.36767065e-05, 8.43329752e-05, 8.64855846e-05,
        8.84983337e-05, 8.85507322e-05, 8.90751658e-05, 8.99565385e-05,
        9.13786476e-05, 9.34255690e-05, 9.55669331e-05, 9.72306210e-05,
        9.92810512e-05, 1.01078442e-04, 1.02833549e-04, 1.04835256e-04,
        1.06896784e-04, 1.08497952e-04, 1.09184584e-04]), rtol=1.e-4)
    np.testing.assert_allclose(uref_north_hem[1:-1, 30], np.array([
        4.05332613, 7.8968749, 11.5826273, 15.0907135, 18.4161644,
        21.55954552, 24.46743965, 26.92323112, 28.70576477, 29.70732689,
        29.95204926, 30.04506302, 30.1388092, 30.25588036, 30.3896122,
        30.47213364, 30.31372643, 29.99117279, 29.65275574, 29.3494072,
        29.10631561, 28.95228767, 28.91378212, 29.02213669, 29.28206825,
        29.70830345, 30.3042717, 31.06984711, 32.01585388, 33.14328003,
        34.44922638, 35.93932724, 37.64125824, 39.56806564, 41.68249893,
        43.94068909, 46.3024559, 48.74007416, 51.21943665, 53.71104431,
        56.19809723, 58.69459152, 61.19521713, 63.6696434, 66.10819244,
        68.54389954, 70.9834137]), rtol=1.e-4)

    np.testing.assert_allclose(ptref_north_hem[1:-1, 30], np.array([
        283.13015747, 289.28668213, 294.64263916, 299.12219238,
        303.00302124, 305.73303223, 307.87088013, 311.10409546,
        319.0144043, 329.7913208, 340.84194946, 353.95306396,
        367.5017395, 381.01858521, 394.40744019, 408.45413208,
        428.76165771, 448.32330322, 468.36608887, 489.41296387,
        511.50256348, 534.64575195, 558.87097168, 584.21966553,
        610.65496826, 638.32421875, 667.36999512, 697.70269775,
        729.35955811, 762.3651123, 796.74871826, 832.36737061,
        871.67785645, 913.37677002, 957.04296875, 1002.74676514,
        1050.68945312, 1101.1854248, 1154.52209473, 1210.76513672,
        1270.12329102, 1332.7376709, 1398.59362793, 1468.07580566,
        1541.81811523, 1619.97949219, 1702.31713867]), rtol=1.e-4)

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
            18.89754418, 20.87051008, 23.05533319, 25.2151785, 27.47119569,
            30.05935184, 32.97341561, 35.66663705, 37.78897341, 39.66149794,
            41.66640906, 43.62471202, 44.91178335, 45.53234882, 46.03212018,
            46.55321029, 46.65420873, 46.04706782, 44.85904265, 43.29714241]),
        rtol=1.e-4)
    np.testing.assert_allclose(
        lwa_and_fluxes.meridional_heat_flux[20:40, :].mean(axis=-1), np.array([
            1.12029810e-05, 1.64006721e-05, 2.39006440e-05, 3.26379003e-05,
            3.91442227e-05, 4.22063488e-05, 4.56718689e-05, 5.14516640e-05,
            5.78098681e-05, 6.17617342e-05, 6.74995267e-05, 7.29475429e-05,
            6.94604844e-05, 6.03513984e-05, 5.66071426e-05, 5.23191117e-05,
            4.72185161e-05, 4.17106714e-05, 3.73196665e-05, 3.15979413e-05]), atol=1.e-8)
    np.testing.assert_allclose(
            lwa_and_fluxes.divergence_eddy_momentum_flux[20:40, :].mean(axis=-1), np.array([
            -2.24915863e-05, -2.83135124e-05, -2.94437864e-05, -2.86393852e-05,
            -2.99478112e-05, -2.22916394e-05, -1.02538906e-05, -4.70246948e-06,
            -4.35666305e-06, -7.40037880e-06, -7.76735834e-06, -5.40079576e-06,
            -2.30816429e-07, 8.78221903e-06, 2.35922609e-05, 3.71332814e-05,
            3.26047562e-05, 1.36971351e-05, -1.89297898e-08, 4.30947320e-06]), atol=1.e-8)


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


def test_skip_vertical_interpolation():
    """
    To test whether the new features of even-to-odd grid interpolation works well.

    .. versionadded:: 1.3

    """
    from scipy.interpolate import interp1d

    zlev = -SCALE_HEIGHT * np.log(plev / P_GROUND)
    height = np.arange(0, kmax) * 1000.
    new_plev = P_GROUND * np.exp(-height/SCALE_HEIGHT)
    interp_to_regular_zgrid = lambda field_to_interp: interp1d(
        zlev, field_to_interp, axis=0, kind='linear', fill_value="extrapolate")(height)

    # *** Do interpolation within QGField object ***
    qgfield = QGFieldNH18(
        xlon=xlon, ylat=np.ma.masked_equal(ylat, 0.0), plev=plev,
        u_field=np.ma.masked_equal(u_field, 0.0),
        v_field=np.ma.masked_equal(v_field, 0.0),
        t_field=np.ma.masked_equal(t_field, 0.0),
        kmax=kmax, maxit=100000, dz=1000., npart=None,
        tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
        omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=True,
        data_on_even_spaced_pseudoheight_grid=False)
    qgfield.interpolate_fields(return_named_tuple=False)

    # *** Do interpolation outside QGField object ***
    interp_u_field = interp_to_regular_zgrid(u_field)
    interp_v_field = interp_to_regular_zgrid(v_field)
    interp_t_field = interp_to_regular_zgrid(t_field)
    qgfield_intact = QGFieldNH18(
        xlon=xlon, ylat=np.ma.masked_equal(ylat, 0.0), plev=new_plev,
        u_field=np.ma.masked_equal(interp_u_field, 0.0),
        v_field=np.ma.masked_equal(interp_v_field, 0.0),
        t_field=np.ma.masked_equal(interp_t_field, 0.0),
        kmax=kmax, maxit=100000, dz=1000., npart=None,
        tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
        omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=True,
        data_on_even_spaced_pseudoheight_grid=True)
    qgfield_intact.interpolate_fields(return_named_tuple=False)
    assert np.allclose(qgfield_intact.height, qgfield.height, rtol=1e-03, atol=1e-05)
    assert np.allclose(qgfield_intact.interpolated_u, qgfield.interpolated_u, rtol=1e-03, atol=1e-05)
    assert np.allclose(qgfield_intact.interpolated_v, qgfield.interpolated_v, rtol=1e-03, atol=1e-05)
    assert np.allclose(qgfield_intact.interpolated_theta, qgfield.interpolated_theta, rtol=1e-03, atol=1e-05)
import os
import warnings
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
        1.04599619e-04, 1.26467340e-04, 1.23562040e-04, 1.24151703e-04,
        1.24635008e-04, 1.27913052e-04, 1.41169843e-04, 1.90680647e-04,
        2.56034862e-04, 2.58298929e-04, 2.37191924e-04, 2.05968954e-04,
        1.61239287e-04, 1.22308016e-04, 9.59203573e-05, 1.10673090e-04,
        1.30930879e-04, 1.07213596e-04, 8.59187627e-05, 7.08199864e-05,
        6.05890785e-05, 5.35338937e-05, 4.89931242e-05, 4.63480628e-05,
        4.51836632e-05, 4.50701218e-05, 4.57926276e-05, 4.71210529e-05,
        4.90499747e-05, 5.14934368e-05, 5.42651868e-05, 7.73193121e-05,
        1.05260626e-04, 1.04041714e-04, 9.61731261e-05, 9.01177221e-05,
        8.59375849e-05, 8.33739664e-05, 8.14663892e-05, 7.99478404e-05,
        7.97097273e-05, 7.98133076e-05, 8.05693774e-05, 8.20351837e-05,
        8.33117174e-05, 8.50354973e-05, 8.73589012e-05]), rtol=1.e-4)
    np.testing.assert_allclose(uref_north_hem[1:-1, 30], np.array([
        4.32260895, 8.03401756, 11.49249077, 14.84575176, 18.14178658,
        21.38952827, 24.32489014, 26.79528427, 28.50146103, 29.43782425,
        29.66746521, 29.75590515, 29.79162407, 29.77187347, 29.70695496,
        29.52400589, 29.33841705, 29.29707146, 29.37574387, 29.56597519,
        29.82884598, 30.15755844, 30.54856682, 31.00727463, 31.54107857,
        32.15468597, 32.85074997, 33.63311005, 34.49978256, 35.43934631,
        36.44489288, 37.48699188, 38.69142151, 40.16823578, 41.91610336,
        43.89514923, 46.0021019, 48.15969467, 50.32623672, 52.46518707,
        54.58965302, 56.70991516, 58.83708954, 60.98814011, 63.19325638,
        65.51700592, 67.96997833]), rtol=1.e-4)

    np.testing.assert_allclose(ptref_north_hem[1:-1, 30], np.array([
        283.26266479, 289.47640991, 294.85855103, 299.38513184,
        303.27514648, 306.07662964, 308.74539185, 312.09375,
        319.14794922, 329.62298584, 341.22283936, 355.05941772,
        369.15429688, 382.63134766, 395.67318726, 407.88659668,
        433.66488647, 460.43347168, 487.07714844, 513.67218018,
        540.27307129, 566.92840576, 593.70532227, 620.63946533,
        647.71960449, 674.94067383, 702.33789062, 729.89660645,
        757.58825684, 785.45550537, 813.57403564, 840.40252686,
        892.54760742, 951.70996094, 1010.70678711, 1070.02416992,
        1129.44519043, 1189.15979004, 1249.25280762, 1309.56311035,
        1370.06018066, 1430.67907715, 1491.31774902, 1552.30883789,
        1613.36242676, 1673.77954102, 1733.34326172]), rtol=1.e-4)

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
            18.55747378, 20.52898478, 22.71621467, 24.93939907, 27.245788,
            29.94535787, 32.92075474, 35.61772485, 37.74264333, 39.65523785,
            41.7474895, 43.67650872, 44.91942116, 45.41484521, 45.70745524,
            46.03775218, 45.96004931, 45.15501773, 43.80189754, 42.14999322]),
        rtol=1.e-4)
    np.testing.assert_allclose(
        lwa_and_fluxes.meridional_heat_flux[20:40, :].mean(axis=-1), np.array([
            1.11954738e-05, 1.63745787e-05, 2.38665466e-05, 3.25914959e-05,
            3.90815204e-05, 4.22089427e-05, 4.57247919e-05, 5.15018898e-05,
            5.78318250e-05, 6.16741914e-05, 6.74087075e-05, 7.28844771e-05,
            6.93942122e-05, 6.03139378e-05, 5.65834689e-05, 5.22989994e-05,
            4.72295911e-05, 4.17550597e-05, 3.73351161e-05, 3.16298564e-05]), rtol=0.03)
    np.testing.assert_allclose(
            lwa_and_fluxes.divergence_eddy_momentum_flux[20:40, :].mean(axis=-1), np.array([
            -2.25946059e-05, -2.83075488e-05, -2.94427754e-05, -2.86319011e-05,
            -2.99386824e-05, -2.21416886e-05, -1.01041360e-05, -4.78678997e-06,
            -4.56198587e-06, -7.51141807e-06, -7.73846396e-06, -5.31851630e-06,
            -1.79050173e-07, 8.74939467e-06, 2.35134712e-05, 3.70757625e-05,
            3.25983249e-05, 1.37527849e-05, 6.34945922e-08, 4.44805497e-06]), rtol=1.e-4)


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
        dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS,
        raise_error_for_nonconvergence=True)
    qgfield.interpolate_fields()
    assert not qgfield.nonconvergent_uref  # before calling compute_reference_states
    with pytest.raises(ValueError):
        qgfield.compute_reference_states()
    assert qgfield.nonconvergent_uref  # after calling compute_reference_states

    # Check the case when error is suppressed for unrealistic fields
    qgfield._raise_error_for_nonconvergence = False
    with pytest.warns(UserWarning):
        qgfield.compute_reference_states()
    assert qgfield.nonconvergent_uref
    qgfield.compute_lwa_only()
    assert qgfield.lwa.sum() > 0


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
        data_on_evenly_spaced_pseudoheight_grid=False)
    qgfield.interpolate_fields(return_named_tuple=False)

    # *** Do interpolation outside QGField object ***
    interp_u_field = interp_to_regular_zgrid(u_field)
    interp_v_field = interp_to_regular_zgrid(v_field)
    interp_t_field = interp_to_regular_zgrid(t_field)
    qgfield_skip_interp = QGFieldNH18(
        xlon=xlon, ylat=np.ma.masked_equal(ylat, 0.0), plev=new_plev,
        u_field=np.ma.masked_equal(interp_u_field, 0.0),
        v_field=np.ma.masked_equal(interp_v_field, 0.0),
        t_field=np.ma.masked_equal(interp_t_field, 0.0),
        kmax=kmax, maxit=100000, dz=1000., npart=None,
        tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
        omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=True,
        data_on_evenly_spaced_pseudoheight_grid=True)
    qgfield_skip_interp.interpolate_fields(return_named_tuple=False)
    assert np.allclose(qgfield_skip_interp.height, qgfield.height, rtol=1e-03, atol=1e-05)
    assert np.allclose(qgfield_skip_interp.interpolated_u, qgfield.interpolated_u, rtol=1e-03, atol=1e-05)
    assert np.allclose(qgfield_skip_interp.interpolated_v, qgfield.interpolated_v, rtol=1e-03, atol=1e-05)
    # TODO: Below is something to be fixed - to let user input theta instead of t
    assert not np.allclose(qgfield_skip_interp.interpolated_theta, qgfield.interpolated_theta, rtol=1e-03, atol=1e-05)


def test_layerwise_flux_guard():
    """Test that layerwise flux properties raise ValueError before computation."""
    qgfield = QGFieldNH18(
        xlon=xlon, ylat=ylat, plev=plev,
        u_field=u_field, v_field=v_field, t_field=t_field,
        kmax=kmax, maxit=100000, dz=1000., npart=None,
        tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP,
        dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA,
        planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=True)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()

    # Guard test: properties should raise before computation
    for prop_name in ['ua1', 'ua2', 'ep1', 'ep2', 'ep3', 'stretch_term']:
        with pytest.raises(ValueError):
            getattr(qgfield, prop_name)


def test_layerwise_flux_properties_full_globe():
    """Test layerwise flux properties with northern_hemisphere_results_only=False."""
    qgfield = QGFieldNH18(
        xlon=xlon, ylat=ylat, plev=plev,
        u_field=u_field, v_field=v_field, t_field=t_field,
        kmax=kmax, maxit=100000, dz=1000., npart=None,
        tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP,
        dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA,
        planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=False)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()
    qgfield.compute_layerwise_lwa_fluxes()

    expected_shape = (kmax, nlat, nlon)
    for prop_name in ['ua1', 'ua2', 'ep1', 'ep2', 'ep3', 'stretch_term']:
        prop = getattr(qgfield, prop_name)
        assert prop.shape == expected_shape, f"{prop_name} shape mismatch: {prop.shape} != {expected_shape}"
        assert np.isnan(prop).sum() == 0, f"{prop_name} contains NaN"
        assert np.abs(prop).sum() > 0, f"{prop_name} is all zeros"


def test_layerwise_flux_properties_nh_only():
    """Regression test: compute_layerwise_lwa_fluxes must not crash with northern_hemisphere_results_only=True."""
    qgfield = QGFieldNH18(
        xlon=xlon, ylat=ylat, plev=plev,
        u_field=u_field, v_field=v_field, t_field=t_field,
        kmax=kmax, maxit=100000, dz=1000., npart=None,
        tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP,
        dry_gas_constant=DRY_GAS_CONSTANT, omega=EARTH_OMEGA,
        planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=True)
    qgfield.interpolate_fields()
    qgfield.compute_reference_states()
    qgfield.compute_layerwise_lwa_fluxes()

    expected_shape = (kmax, nlat // 2 + 1, nlon)
    for prop_name in ['ua1', 'ua2', 'ep1', 'ep2', 'ep3', 'stretch_term']:
        prop = getattr(qgfield, prop_name)
        assert prop.shape == expected_shape, f"{prop_name} shape mismatch: {prop.shape} != {expected_shape}"
        assert np.isnan(prop).sum() == 0, f"{prop_name} contains NaN"
        assert np.abs(prop).sum() > 0, f"{prop_name} is all zeros"

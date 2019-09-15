import os
import unittest
from math import pi

import numpy as np
from scipy.interpolate import interp1d

from hn2016_falwa.oopinterface import QGField


class OOPParams(object):
    def __init__(self):
        # Define physical constants
        p0 = 1000.  # Ground pressure level. Unit: hPa
        self.scale_height = 7000.  # Unit: m
        self.kmax = 49
        self.dz = 1000.
        self.cp = 1004.
        self.dry_gas_constant = 287.
        self.planet_radius = 6.378e+6  # Unit: m

        # Define parameters
        self.nlev, self.nlat, self.nlon = 12, 31, 60

        self.xlon = np.linspace(0, 2. * pi, self.nlon, endpoint=False)
        ylat = np.linspace(-90., 90., self.nlat, endpoint=True)
        self.plev = np.array([1000, 900, 800, 700, 600, 500, 400,
                              300, 200, 100, 10, 1])

        dir_path = os.path.dirname(__file__)
        self.u_field = np.reshape(
            np.loadtxt(dir_path + '/test_data/demo_u.txt'),
            [self.nlev, self.nlat, self.nlon]
        )
        self.v_field = np.reshape(
            np.loadtxt(dir_path + '/test_data/demo_u.txt'),
            [self.nlev, self.nlat, self.nlon]
        )
        self.t_field = np.reshape(
            np.loadtxt(dir_path + '/test_data/demo_u.txt'),
            [self.nlev, self.nlat, self.nlon]
        )

        # Compute the potential temperature to check whether the interpolation
        # method returns a bounded potential temperature field
        self.theta_field = self.t_field * (self.plev[:, np.newaxis, np.newaxis]/p0)**(-self.dry_gas_constant / self.cp)

        # Create a QGField object for testing
        self.qgfield = QGField(
            self.xlon, ylat, self.plev,
            self.u_field, self.v_field, self.t_field,
            kmax=self.kmax,
            dz=self.dz,
            scale_height=self.scale_height,
            cp=self.cp,
            dry_gas_constant=self.dry_gas_constant,
            planet_radius=self.planet_radius)


def test_interpolate_fields(self):
    """
    Check that the input fields are interpolated onto a grid of correct dimension and the interpolated values are bounded.
    """

    qgpv, interpolated_u, interpolated_v, interpolated_theta, static_stability = \
        OOPParams.qgfield.interpolate_fields()

    kmax, nlat, nlon = \
        OOPParams.qgfield.kmax, \
        OOPParams.qgfield.get_latitude_dim(), \
        OOPParams.qgfield.nlon

    # Check that the dimensions of the interpolated fields are correct
    assert (kmax, nlat, nlon) == qgpv.shape
    assert (kmax, nlat, nlon) == interpolated_u.shape
    assert (kmax, nlat, nlon) == interpolated_v.shape
    assert (kmax, nlat, nlon) == interpolated_theta.shape
    assert (kmax,) == static_stability.shape

    assert (interpolated_u[1:-1, :, :].max() <= OOPParams.u_field.max()) &\
        (interpolated_u[1:-1, :, :].max() >= OOPParams.u_field.min())
    assert (interpolated_u[1:-1, :, :].min() <= OOPParams.u_field.max()) &\
        (interpolated_u[1:-1, :, :].min() >= OOPParams.u_field.min())
    assert (interpolated_v[1:-1, :, :].max() <= OOPParams.v_field.max()) &\
        (interpolated_v[1:-1, :, :].max() >= OOPParams.v_field.min())
    assert (interpolated_v[1:-1, :, :].min() <= OOPParams.v_field.max()) &\
        (interpolated_v[1:-1, :, :].min() >= OOPParams.v_field.min())
    assert (interpolated_theta[1:-1, :, :].max() <= OOPParams.theta_field.max()) &\
        (interpolated_theta[1:-1, :, :].max() >= OOPParams.theta_field.min())
    assert (interpolated_theta[1:-1, :, :].min() <= OOPParams.theta_field.max()) &\
        (interpolated_theta[1:-1, :, :].min() >=OOPParams.theta_field.min())
    assert 0 == np.isnan(qgpv).sum()
    assert 0 == (qgpv == float('Inf')).sum()


def test_compute_reference_states(self):
    """
    Check that the output reference states are of correct dimension, and the QGPV reference state is non-decreasing.
    """
    qref_north_hem, uref_north_hem, ptref_north_hem = \
        OOPParams.qgfield.compute_reference_states(
            northern_hemisphere_results_only=True
        )
    kmax, nlat, nlon = \
        OOPParams.qgfield.kmax, OOPParams.qgfield.nlat, OOPParams.qgfield.nlon

    # Check dimension of the input field
    assert (kmax, nlat//2+1) == qref_north_hem.shape
    assert (kmax, nlat//2+1) == uref_north_hem.shape
    assert (kmax, nlat//2+1) == ptref_north_hem.shape
    assert (np.diff(qref_north_hem, axis=-1)[:, :] >= 0.).all()


def test_interpolate_fields_even_grids(self):
    """
    To test whether the new features of even-to-odd grid interpolation works well.

    .. versionadded:: 0.3.5

    """
    ylat = np.linspace(-90., 90., OOPParams.nlat, endpoint=True)
    ylat_even = np.linspace(-90., 90., OOPParams.nlat+1, endpoint=True)[1:-1]
    u_field_even = interp1d(ylat, OOPParams.u_field, axis=1,
                            fill_value="extrapolate")(ylat_even)
    v_field_even = interp1d(ylat, OOPParams.v_field, axis=1,
                            fill_value="extrapolate")(ylat_even)
    t_field_even = interp1d(ylat, OOPParams.t_field, axis=1,
                            fill_value="extrapolate")(ylat_even)

    # Create a QGField object for testing
    OOPParams.qgfield_even = QGField(
        OOPParams.xlon, ylat_even, OOPParams.plev,
        u_field_even, v_field_even, t_field_even,
        kmax=OOPParams.kmax,
        dz=OOPParams.dz,
        scale_height=OOPParams.scale_height,
        cp=OOPParams.cp,
        dry_gas_constant=OOPParams.dry_gas_constant,
        planet_radius=OOPParams.planet_radius)

    qgpv, interpolated_u, interpolated_v, interpolated_theta, static_stability = \
        OOPParams.qgfield_even.interpolate_fields()

    kmax, nlat, nlon = \
        OOPParams.qgfield_even.kmax, \
        OOPParams.qgfield_even.get_latitude_dim(), \
        OOPParams.qgfield_even.nlon

    # Check that the dimensions of the interpolated fields are correct
    assert (kmax, nlat, nlon) == qgpv.shape
    assert (kmax, nlat, nlon) == interpolated_u.shape
    assert (kmax, nlat, nlon) == interpolated_v.shape
    assert (kmax, nlat, nlon) == interpolated_theta.shape
    assert (kmax,) == static_stability.shape

    # Check that at the interior grid points, the interpolated fields
    # are bounded
    assert (interpolated_u[1:-1, 1:-1, 1:-1].max() <= OOPParams.u_field.max()) &\
        (interpolated_u[1:-1, 1:-1, 1:-1].max() >= OOPParams.u_field.min())

    assert (interpolated_u[1:-1, 1:-1, 1:-1].min() <= OOPParams.u_field.max()) &\
        (interpolated_u[1:-1, 1:-1, 1:-1].min() >= OOPParams.u_field.min())

    assert (interpolated_v[1:-1, 1:-1, 1:-1].max() <= OOPParams.v_field.max()) &\
        (interpolated_v[1:-1, 1:-1, 1:-1].max() >= OOPParams.v_field.min())

    assert (interpolated_v[1:-1, 1:-1, 1:-1].min() <= OOPParams.v_field.max()) &\
        (interpolated_v[1:-1, 1:-1, 1:-1].min() >= OOPParams.v_field.min())

    assert (interpolated_theta[1:-1, 1:-1, 1:-1].max() <= OOPParams.theta_field.max()) &\
        (interpolated_theta[1:-1, 1:-1, 1:-1].max() >= OOPParams.theta_field.min())

    assert interpolated_theta[1:-1, 1:-1, 1:-1].min() <= OOPParams.theta_field.max() &\
        (interpolated_theta[1:-1, 1:-1, 1:-1].min() >= OOPParams.theta_field.min())

    assert 0 == np.isnan(qgpv).sum()
    assert 0 == (qgpv == float('Inf')).sum()


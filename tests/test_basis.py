import pytest
import numpy as np
from math import pi
import hn2016_falwa.basis as basis
from hn2016_falwa.constant import *

# === Parameters specific for testing the basis module ===
nlat = 241
nlon = 480
planet_radius = 1.
ylat = np.linspace(-90, 90, nlat, endpoint=True)
clat = np.abs(np.cos(np.deg2rad(ylat)))
xlon = np.linspace(0, 360, nlon, endpoint=False)
vort = np.sin(3. * np.deg2rad(xlon[np.newaxis, :])) \
            * np.cos(np.deg2rad(ylat[:, np.newaxis]))
dummy_vgrad = 3. / (planet_radius * np.cos(np.deg2rad(ylat[:, np.newaxis]))) \
                   * np.cos(np.deg2rad(3. * xlon[np.newaxis, :])) * np.cos(
    np.deg2rad(ylat[:, np.newaxis])) \
                   - 1. / planet_radius * np.sin(np.deg2rad(ylat[:, np.newaxis])) * np.sin(
    np.deg2rad(3. * xlon[np.newaxis, :]))
dphi = np.deg2rad(np.diff(ylat)[0])
area = 2. * pi * planet_radius ** 2 \
            * (np.cos(np.deg2rad(ylat[:, np.newaxis])) * dphi) \
            / float(nlon) * np.ones((nlat, nlon))


def test_lwa():
    '''
    To assert that the lwa function in basis.py produce the expect results -
    lwa shall be all zero when the there is no meridional component in the
    wind field.
    '''

    test_vort = (np.ones((5, 5)) * np.array([1, 2, 3, 4, 5]))\
        .swapaxes(0, 1)
    test_q_part = np.array([1, 2, 3, 4, 5])
    input_result, _ = basis.lwa(5, 5, test_vort, test_q_part, np.ones(5))
    assert np.array_equal(input_result, np.zeros((5, 5)))


def test_eqvlat():
    '''
    To test whether the eqvlat function in basis.py produce a reference state of vorticity non-decreasing with latitude, given a random vorticity field.
    '''
    q_part1, vgrad = basis.eqvlat(
        ylat, vort, area, nlat,
        planet_radius=EARTH_RADIUS,
        vgrad=dummy_vgrad
    )
    q_part2, _ = basis.eqvlat(
        ylat, vort, area, nlat,
        planet_radius=EARTH_RADIUS,
        vgrad=None
    )
    assert np.all(np.diff(q_part1) >= 0.)
    assert q_part1.tolist() == q_part2.tolist()
    assert vgrad is not None
    assert vgrad.shape == q_part1.shape


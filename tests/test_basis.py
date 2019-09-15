import pytest
import numpy as np
from math import pi
import hn2016_falwa.basis as basis


class BasisParams(object):
    def __init__(self):
        self.nlat = 241
        self.nlon = 480
        self.planet_radius = 1.
        self.ylat = np.linspace(-90, 90, self.nlat, endpoint=True)
        self.clat = np.abs(np.cos(np.deg2rad(self.ylat)))
        self.xlon = np.linspace(0, 360, self.nlon, endpoint=False)
        self.vort = np.sin(3. * np.deg2rad(self.xlon[np.newaxis, :])) \
                    * np.cos(np.deg2rad(self.ylat[:, np.newaxis]))
        self.dummy_vgrad = 3. / (self.planet_radius * np.cos(np.deg2rad(self.ylat[:, np.newaxis]))) \
                           * np.cos(np.deg2rad(3. * self.xlon[np.newaxis, :])) * np.cos(
            np.deg2rad(self.ylat[:, np.newaxis])) \
                           - 1. / self.planet_radius * np.sin(np.deg2rad(self.ylat[:, np.newaxis])) * np.sin(
            np.deg2rad(3. * self.xlon[np.newaxis, :]))
        self.dphi = np.deg2rad(np.diff(self.ylat)[0])
        self.area = 2. * pi * self.planet_radius ** 2 \
                    * (np.cos(np.deg2rad(self.ylat[:, np.newaxis])) * self.dphi) \
                    / float(self.nlon) * np.ones((self.nlat, self.nlon))


def test_lwa(self):
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


def test_eqvlat(self):
    '''
    To test whether the eqvlat function in basis.py produce a reference state of vorticity non-decreasing with latitude, given a random vorticity field.
    '''
    q_part1, vgrad = basis.eqvlat(
        BasisParams.ylat, BasisParams.vort, BasisParams.area, BasisParams.nlat,
        planet_radius=BasisParams.planet_radius,
        vgrad=BasisParams.dummy_vgrad
    )
    q_part2, _ = basis.eqvlat(
        BasisParams.ylat, BasisParams.vort, BasisParams.area, BasisParams.nlat,
        planet_radius=BasisParams.planet_radius,
        vgrad=None
    )
    assert np.all(np.diff(q_part1) >= 0.)
    assert q_part1.tolist() == q_part2.tolist()
    assert vgrad is not None
    assert vgrad.shape == q_part1.shape


import pytest
from enum import Enum

def setUp(self):
    '''
    Set up a hypothetical vorticity fields with uniform longitude and latitude grids
    to test the functions in basis.py and wrapper.py
    '''
    self.nlat, self.nlon = 241, 480
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
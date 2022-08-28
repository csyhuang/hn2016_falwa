"""
------------------------------------------
File name: barotropic_field.py
Author: Clare Huang
"""
from math import pi

import numpy as np

from hn2016_falwa import basis


class BarotropicField(object):

    """
    An object that deals with barotropic (2D) wind and/or PV fields

    Parameters
    ----------
        xlon : np.array
            Longitude array in degree with dimension = nlon.

        ylat : np.array
            Latitude array in degree with dimension = nlat.

        area : np.ndarray
            Differential area at each lon-lat grid points with dimension (nlat,nlon). If 'area=None': it will be initiated as area of uniform grid (in degree) on a spherical surface. Dimension = [nlat, nlon]

        dphi : np.array
            Differential length element along the lat grid with dimension = nlat.

        pv_field : np.ndarray
            Absolute vorticity field with dimension [nlat x nlon]. If 'pv_field=None': pv_field is expected to be computed with u,v,t field.


    Example
    ---------
    >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)

    """

    def __init__(self, xlon, ylat, pv_field, area=None, dphi=None,
                 n_partitions=None, planet_radius=6.378e+6):

        """
        Create a BarotropicField object.

        Parameters
        ----------
            xlon : np.array
                Longitude array in degree with dimension = nlon.

            ylat : np.array
                Latitude array in degree with dimension = nlat.

            area : np.ndarray
                Differential area at each lon-lat grid points with dimension (nlat,nlon). If 'area=None': it will be initiated as area of uniform grid (in degree) on a spherical surface. Dimension = [nlat, nlon]

            dphi : np.array
                Differential length element along the lat grid with dimension = nlat.

            pv_field : np.ndarray
                Absolute vorticity field with dimension = [nlat, nlon].
                If none, pv_field is expected to be computed with u,v,t field.

        """

        self.xlon = xlon
        self.ylat = ylat
        self.clat = np.abs(np.cos(np.deg2rad(ylat)))
        self.nlon = xlon.size
        self.nlat = ylat.size
        self.planet_radius = planet_radius
        if dphi is None:
            self.dphi = pi/(self.nlat-1) * np.ones((self.nlat))
        else:
            self.dphi = dphi

        if area is None:
            self.area = 2. * pi * planet_radius ** 2 * \
                        (np.cos(ylat[:, np.newaxis] * pi/180.) * self.dphi[:, np.newaxis])\
                        / float(self.nlon)*np.ones((self.nlat, self.nlon))
        else:
            self.area = area

        self.pv_field = pv_field

        if n_partitions is None:
            self.n_partitions = self.nlat
        else:
            self.n_partitions = n_partitions

        # Quantities that are computed with the methods below
        self._eqvlat = None
        self._lwa = None

    def _compute_eqvlat(self):
        """
        Private function. Compute equivalent latitude if it has not been computed yet.
        """
        self.eqvlat, _ = basis.eqvlat(
            self.ylat, self.pv_field, self.area, self.n_partitions,
            planet_radius=self.planet_radius
        )
        return self.eqvlat

    def _compute_lwa(self):
        """
        Private function. Compute equivalent latitude if it has not been computed yet.
        """
        eqvlat = self.equivalent_latitudes
        if self._lwa is None:
            self._lwa, dummy = basis.lwa(
                self.nlon, self.nlat, self.pv_field, eqvlat,
                self.planet_radius * self.clat * self.dphi
            )
        return self.lwa

    @property
    def equivalent_latitudes(self):
        """
        Return the computd quivalent latitude with the *pv_field* stored in the object.

        Return
        ----------
        An numpy array with dimension (nlat) of equivalent latitude array.

        Example
        ----------
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> eqv_lat = barofield1.equivalent_latitudes

        """
        if self._eqvlat is None:
            return self._compute_eqvlat()
        return self._eqvlat

    @property
    def lwa(self):

        """
        Compute the finite-amplitude local wave activity based on the *equivalent_latitudes* and the *pv_field* stored in the object.

        Return
        ----------
        An 2-D numpy array with dimension [nlat,nlon] of local wave activity values.

        Example
        ----------
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> lwa = barofield1.lwa

        """
        if self._lwa is None:
            return self._compute_lwa()
        return self._lwa

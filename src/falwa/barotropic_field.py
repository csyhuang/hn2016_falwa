"""
------------------------------------------
File name: barotropic_field.py
Author: Clare Huang
"""
from math import pi
from typing import Optional

import numpy as np

from falwa import basis


class BarotropicField(object):
    """An object that deals with barotropic (2D) wind and/or PV fields.

    Parameters
    ----------
    xlon : np.ndarray
        Longitude array in degree with dimension (nlon,).
    ylat : np.ndarray
        Latitude array in degree with dimension (nlat,).
    pv_field : np.ndarray
        Absolute vorticity field with dimension (nlat, nlon).
    area : np.ndarray, optional
        Differential area at each lon-lat grid points with dimension (nlat, nlon).
        If None, it will be initiated as area of uniform grid (in degree) on a
        spherical surface.
    dphi : np.ndarray, optional
        Differential length element along the lat grid with dimension (nlat,).
        If None, it is computed for a uniform grid.
    n_partitions : int, optional
        Number of partitions for equivalent latitude calculation.
        Defaults to `nlat`.
    planet_radius : float, optional
        Radius of the planet, defaults to 6.378e+6 (Earth's radius in meters).
    return_partitioned_lwa : bool, optional
        If True, return local wave activity as a stacked field of cyclonic and
        anticyclonic components. If False, return a single field of local wave
        activity of dimension (nlat, nlon). Default is False.

    Examples
    --------
    >>> from falwa.barotropic_field import BarotropicField
    >>> import numpy as np
    >>> xlon = np.arange(0, 360, 1.5)
    >>> ylat = np.arange(-90, 91, 1.5)
    >>> abs_vorticity = np.random.rand(len(ylat), len(xlon))
    >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
    """

    def __init__(self, xlon: np.ndarray, ylat: np.ndarray, pv_field: np.ndarray,
                 area: Optional[np.ndarray] = None, dphi: Optional[np.ndarray] = None,
                 n_partitions: Optional[int] = None, planet_radius: float = 6.378e+6,
                 return_partitioned_lwa: bool = False):
        """
        Create a BarotropicField object.
        """
        self.xlon = xlon
        self.ylat = ylat
        self.clat = np.abs(np.cos(np.deg2rad(ylat)))
        self.nlon = xlon.size
        self.nlat = ylat.size
        self.planet_radius = planet_radius
        self.return_partitioned_lwa = return_partitioned_lwa
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

    def _compute_eqvlat(self) -> np.ndarray:
        """
        Private function. Compute equivalent latitude if it has not been computed yet.
        """
        self.eqvlat, _ = basis.eqvlat_fawa(
            self.ylat, self.pv_field, self.area, self.n_partitions,
            planet_radius=self.planet_radius, output_fawa=False
        )
        return self.eqvlat

    def _compute_lwa(self) -> np.ndarray:
        """
        Private function. Compute equivalent latitude if it has not been computed yet.
        """
        eqvlat = self.equivalent_latitudes
        if self._lwa is None:
            self._lwa, dummy = basis.lwa(
                self.nlon, self.nlat, self.pv_field, eqvlat,
                self.planet_radius * self.clat * self.dphi,
                return_partitioned_lwa=self.return_partitioned_lwa
            )
        return self._lwa

    @property
    def equivalent_latitudes(self) -> np.ndarray:
        """Return the computed equivalent latitude with the `pv_field` stored in the object.

        Returns
        -------
        np.ndarray
            An array with dimension (nlat,) of equivalent latitude.

        Examples
        --------
        >>> from falwa.barotropic_field import BarotropicField
        >>> import numpy as np
        >>> xlon = np.arange(0, 360, 1.5)
        >>> ylat = np.arange(-90, 91, 1.5)
        >>> abs_vorticity = np.random.rand(len(ylat), len(xlon))
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> eqv_lat = barofield1.equivalent_latitudes
        """
        if self._eqvlat is None:
            self._eqvlat = self._compute_eqvlat()
        return self._eqvlat

    @property
    def lwa(self) -> np.ndarray:
        """Compute the finite-amplitude local wave activity.

        This is based on the `equivalent_latitudes` and the `pv_field`
        stored in the object.

        Returns
        -------
        np.ndarray
            A 2D array with dimension (nlat, nlon) of local wave activity values.

        Examples
        --------
        >>> from falwa.barotropic_field import BarotropicField
        >>> import numpy as np
        >>> xlon = np.arange(0, 360, 1.5)
        >>> ylat = np.arange(-90, 91, 1.5)
        >>> abs_vorticity = np.random.rand(len(ylat), len(xlon))
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> lwa = barofield1.lwa
        """
        if self._lwa is None:
            self._lwa = self._compute_lwa()
        return self._lwa

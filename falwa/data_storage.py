"""
------------------------------------------
File name: data_storage.py
Author: Clare Huang
"""
from typing import Tuple, Optional, Union, List, NamedTuple
import numpy as np


class InvalidCallOfSHemVariables(Exception):
    """
    northern_hemisphere_results_only = True. 
    Southern hemispheric variables are not computed.
    """


class HemisphericProperty:

    def __init__(self, attr, ndims_fill=(0, 0), doc=None):
        self.attr = attr
        self.__doc__ = doc
        # Dimensions to fill in pre- and post-latitude
        npre, npost = ndims_fill
        self.slc_pre  = tuple(slice(None) for _ in range(npre))
        self.slc_post = tuple(slice(None) for _ in range(npost))

    @property
    def ndim_j(self):
        return len(self.slc_pre)


class NHemProperty(HemisphericProperty):

    def __get__(self, obj, objtype=None):
        if obj.northern_hemisphere_results_only:
            return getattr(obj, self.attr)
        # Assemble full slice from slices of pre-latitude dimensions, latitude
        # dimension slice and post-latitude dimensions slices
        slc = (*self.slc_pre, slice(-(obj.nlat//2+1), None), *self.slc_post)
        return getattr(obj, self.attr)[slc]

    def __set__(self, obj, value):
        jdim = value.shape[self.ndim_j]
        slc = (*self.slc_pre, slice(-jdim, None), *self.slc_post)
        getattr(obj, self.attr)[slc] = value


class SHemProperty(HemisphericProperty):

    def __get__(self, obj, objtype=None):
        if obj.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables()
        slc = (*self.slc_pre, slice(None, obj.nlat//2+1), *self.slc_post)
        return getattr(obj, self.attr)[slc]

    def __set__(self, obj, value):
        jdim = value.shape[self.ndim_j]
        if obj.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables()
        slc = (*self.slc_pre, slice(None, jdim), *self.slc_post)
        vlc = (*self.slc_pre, slice(None, None, -1), *self.slc_post)
        getattr(obj, self.attr)[slc] = value[vlc]



class DerivedQuantityStorage:
    """
    This class manages the storage of derived variables in :py:class:`oopinterface.QGField`.

    Variables are stored in fortran indexing order for easy communication with
    f2py modules. To return variables in python indexing order, use the method
    :py:meth:`fortran_to_python` to swap the axes.
    """
    def __init__(
        self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
        swapaxis_1: int, swapaxis_2: int, northern_hemisphere_results_only: bool):
        self._pydim = pydim  # python dimension
        self._fdim = fdim    # fortran dimension
        self._swapaxis_1 = swapaxis_1
        self._swapaxis_2 = swapaxis_2
        self._northern_hemisphere_results_only = northern_hemisphere_results_only

    @property
    def pydim(self):
        return self._pydim

    @property
    def fdim(self):
        return self._fdim

    @property
    def swapaxis_1(self):
        return self._swapaxis_1

    @property
    def swapaxis_2(self):
        return self._swapaxis_2

    @property
    def northern_hemisphere_results_only(self):
        return self._northern_hemisphere_results_only

    def fortran_to_python(self, phy_field: np.ndarray):
        return np.swapaxes(phy_field, self.swapaxis_1, self.swapaxis_2)

    def python_to_fortran(self, phy_field: np.ndarray):  # This may not be necessary
        return np.swapaxes(phy_field, self.swapaxis_2, self.swapaxis_1)


class DomainAverageStorage(DerivedQuantityStorage):
    """
    This stores global/hemispheric averaged potential temperature and static stability.

    Fortran dimension: (kmax)
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int, northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.t0: Optional[np.array] = None
        self.tn0: Optional[np.array] = None
        self.ts0: Optional[np.array] = None
        self.static_stability: Optional[np.array] = None
        self.static_stability_n: Optional[np.array] = None
        self.static_stability_s: Optional[np.array] = None


class InterpolatedFieldsStorage(DerivedQuantityStorage):
    """
    This class stores 3D fields on interpolated grids, including:

    - u
    - v
    - theta (potential temperature)
    - avort (absolute vorticity, used as boundary condition to solve for reference state in NHN22)

    Fortran dimension: (nlon, nlat, kmax)
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int, northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.interpolated_u: Optional[np.ndarray] = None
        self.interpolated_v: Optional[np.ndarray] = None
        self.interpolated_theta: Optional[np.ndarray] = None
        self.interpolated_avort: Optional[np.ndarray] = None
        self.qgpv: Optional[np.ndarray] = None


class ReferenceStatesStorage(DerivedQuantityStorage):
    """
    This class stores height-latitude 2D fields on interpolated grids, including:

    - uref
    - qref (it actually stores qref/f, where f is Coriolis paramter)
    - tref (potential temperature reference state)

    Fortran dimension: (nlat, kmax)
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int,
                 northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.qref = np.zeros(self.fdim)  # This is to substitute self._qref_ntemp
        self.uref = np.zeros(self.fdim)
        self.ptref = np.zeros(self.fdim)
        kmax, self.nlat = self.pydim

    qref_nhem = NHemProperty("qref", (0, 1))
    qref_shem = SHemProperty("qref", (0, 1))

    def qref_correct_unit(self, ylat, omega, python_indexing=True):
        """
        This returns Qref of the correct unit to the user.
        TODO: encapsulate this elsewhere to avoid potential error
        """
        qref_right_unit = \
            self.qref * 2 * omega * np.sin(np.deg2rad(ylat[:, np.newaxis]))
        if python_indexing:
            return self.fortran_to_python(qref_right_unit) # (kmax, nlat)
        return qref_right_unit

    uref_nhem = NHemProperty("uref", (0, 1))
    uref_shem = SHemProperty("uref", (0, 1))

    ptref_nhem = NHemProperty("ptref", (0, 1))
    ptref_shem = SHemProperty("ptref", (0, 1))


class LWAStorage(DerivedQuantityStorage):
    """
    This class stores 3D LWA field on interpolated grids.

    Fortran dimension: (nlon, nlat, kmax)
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int,
                 northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.lwa = np.zeros(self.fdim)
        self.nlat = self.fdim[1]

    lwa_nhem = NHemProperty("lwa", (1, 1))
    lwa_shem = SHemProperty("lwa", (1, 1))


class OutputBarotropicFluxTermsStorage(DerivedQuantityStorage):
    """
    This class stores vertically integrated derived quantities in latitude-longitude 2D grid, including:

    - adv_flux_f1
    - adv_flux_f2
    - adv_flux_f3
    - zonal_adv_flux
    - convergence_zonal_advective_flux
    - divergence_eddy_momentum_flux
    - meridional_heat_flux

    Variables are stored in **python indexing order**: (nlat, nlon)
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int,
                 northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        # *** variables below are computed all at once after getting flux terms above?
        self.adv_flux_f1 = np.zeros(self.pydim)
        self.adv_flux_f2 = np.zeros(self.pydim)
        self.adv_flux_f3 = np.zeros(self.pydim)
        self.zonal_adv_flux = np.zeros(self.pydim)
        self.convergence_zonal_advective_flux = np.zeros(self.pydim)
        self.divergence_eddy_momentum_flux = np.zeros(self.pydim)
        self.meridional_heat_flux = np.zeros(self.pydim)


class BarotropicFluxTermsStorage(DerivedQuantityStorage):
    """
    This class stores intermediate computed quantities in latitude-longitude 2D grid.

    Variables are stored in fortran indexing order: (nlon, nlat)
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int,
                 northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.nlat = fdim[1]
        self.ua1baro = np.zeros(self.fdim)
        self.ua2baro = np.zeros(self.fdim)
        self.ep1baro = np.zeros(self.fdim)
        self.ep2baro = np.zeros(self.fdim)
        self.ep3baro = np.zeros(self.fdim)
        self.ep4 = np.zeros(self.fdim)
        self.u_baro = np.zeros(self.fdim)
        self.lwa_baro = np.zeros(self.fdim)  # This is barotropic LWA (astarbaro)

    ua1baro_nhem = NHemProperty("ua1baro", (1, 0))
    ua1baro_shem = SHemProperty("ua1baro", (1, 0))

    ua2baro_nhem = NHemProperty("ua2baro", (1, 0))
    ua2baro_shem = SHemProperty("ua2baro", (1, 0))

    ep1baro_nhem = NHemProperty("ep1baro", (1, 0))
    ep1baro_shem = SHemProperty("ep1baro", (1, 0))

    ep2baro_nhem = NHemProperty("ep2baro", (1, 0))
    ep2baro_shem = SHemProperty("ep2baro", (1, 0))

    ep3baro_nhem = NHemProperty("ep3baro", (1, 0))
    ep3baro_shem = SHemProperty("ep3baro", (1, 0))

    ep4_nhem = NHemProperty("ep4", (1, 0))
    ep4_shem = SHemProperty("ep4", (1, 0))

    u_baro_nhem = NHemProperty("u_baro", (1, 0))
    u_baro_shem = SHemProperty("u_baro", (1, 0))

    lwa_baro_nhem = NHemProperty("lwa_baro", (1, 0))
    lwa_baro_shem = SHemProperty("lwa_baro", (1, 0))


"""
------------------------------------------
File name: data_storage.py
Author: Clare Huang
"""
from typing import Tuple, Optional, Union, List, NamedTuple
import numpy as np


class DerivedQuantityStorage:
    """
    Variables are stored in **fortran dimension ordering**.
    It is converted to python when output to users.
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
    This class stores 3D fields on interpolated grids
    Python dimension: (kmax, nlat, nlon)
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
    If northern_hemisphere_results_only=False:
        return dimension (kmax, nlat)
    If northern_hemisphere_results_only=True:
        return dimension (kmax, nlat//2+1)

    Note that qref here is not yet multiplied by 2*omega*sin(phi).
    This is for downstream computation.
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int,
                 northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.qref = np.zeros(self.fdim)  # This is to substitute self._qref_ntemp
        self.uref = np.zeros(self.fdim)
        self.ptref = np.zeros(self.fdim)
        kmax, self.nlat = self.pydim

    # *** Qref ***
    @property
    def qref_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.qref
        return self.qref[-(self.nlat//2+1):, :]

    @qref_nhem.setter
    def qref_nhem(self, value):
        if self.northern_hemisphere_results_only:
            self.qref[:, :] = value
        else:
            self.qref[-(self.nlat//2+1):, :] = value

    @property
    def qref_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.qref[:(self.nlat//2+1), :]

    @qref_shem.setter
    def qref_shem(self, value):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.qref[self.nlat//2::-1, :] = value  # running from equator to pole

    def qref_correct_unit(self, ylat, omega):
        """
        This returns Qref of the correct unit to the user
        TODO: encapsulate this elsewhere to avoid potential error
        """
        qref_stemp_right_unit = \
            self.qref * 2 * omega * np.sin(np.deg2rad(ylat[:, np.newaxis]))
        return self.fortran_to_python(qref_stemp_right_unit)  # (kmax, nlat)

    # *** Uref ***
    @property
    def uref_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.uref
        return self.uref[-(self.nlat//2+1):, :]

    @uref_nhem.setter
    def uref_nhem(self, value):
        # input uref would be of dim (jd, kmax)
        jdim = value.shape[0]
        self.uref[-jdim:, :] = value

    @property
    def uref_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.uref[:(self.nlat//2+1), :]

    @uref_shem.setter
    def uref_shem(self, value):
        jdim = value.shape[0]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.uref[:jdim, :] = value[::-1, :]  # running from equator to pole

    # *** PTref (reference potential temperature) ***
    @property
    def ptref_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ptref
        return self.ptref[-(self.nlat//2+1):, :]

    @ptref_nhem.setter
    def ptref_nhem(self, value):
        jdim = value.shape[0]
        self.ptref[-jdim:, :] = value

    @property
    def ptref_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ptref[:(self.nlat//2+1), :]

    @ptref_shem.setter
    def ptref_shem(self, value):
        jdim = value.shape[0]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ptref[:jdim, :] = value[::-1, :]  # running from equator to pole  # running from equator to pole


class LWAStorage(DerivedQuantityStorage):
    """
    LWA has python dim (kmax, nlat, nlon) / fortran dim (nlon, nlat, kmax)
    """
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int,
                 northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.lwa = np.zeros(self.fdim)
        self.nlat = self.fdim[1]

    @property
    def lwa_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.lwa
        return self.lwa[:, -(self.nlat//2+1):, :]

    @lwa_nhem.setter
    def lwa_nhem(self, value):
        jdim = value.shape[1]
        self.lwa[:, -jdim:, :] = value

    @property
    def lwa_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.lwa[:, :(self.nlat//2+1), :]

    @lwa_shem.setter
    def lwa_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.lwa[:, :jdim, :] = value[:, ::-1, :]  # running from equator to pole  # running from equator to pole


class OutputBarotropicFluxTermsStorage(DerivedQuantityStorage):
    """
    This is operating in python dimension only
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
    Barotropic flux terms has python dim (nlat, nlon) / fortran dim (nlon, nlat)
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
        self.ubaro = np.zeros(self.fdim)
        self.lwa_baro = np.zeros(self.fdim)  # This is barotropic LWA (astarbaro)

    @property
    def ua1baro_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ua1baro
        return self.ua1baro[:, -(self.nlat//2+1):]

    @ua1baro_nhem.setter
    def ua1baro_nhem(self, value):
        jdim = value.shape[1]
        self.ua1baro[:, -jdim:] = value

    @property
    def ua1baro_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ua1baro[:, :(self.nlat//2+1)]

    @ua1baro_shem.setter
    def ua1baro_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ua1baro[:, :jdim] = value[:, ::-1]

    @property
    def ua2baro_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ua2baro
        return self.ua2baro[:, -(self.nlat//2+1):]

    @ua2baro_nhem.setter
    def ua2baro_nhem(self, value):
        jdim = value.shape[1]
        self.ua2baro[:, -jdim:] = value

    @property
    def ua2baro_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ua2baro[:, :(self.nlat//2+1)]

    @ua2baro_shem.setter
    def ua2baro_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ua2baro[:, :jdim] = value[:, ::-1]

    @property
    def ep1baro_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ep1baro
        return self.ep1baro[:, -(self.nlat//2+1):]

    @ep1baro_nhem.setter
    def ep1baro_nhem(self, value):
        jdim = value.shape[1]
        self.ep1baro[:, -jdim:] = value

    @property
    def ep1baro_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ep1baro[:, :(self.nlat//2+1)]

    @ep1baro_shem.setter
    def ep1baro_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ep1baro[:, :jdim] = value[:, ::-1]

    @property
    def ep2baro_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ep2baro
        return self.ep2baro[:, -(self.nlat//2+1):]

    @ep2baro_nhem.setter
    def ep2baro_nhem(self, value):
        jdim = value.shape[1]
        self.ep2baro[:, -jdim:] = value

    @property
    def ep2baro_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ep2baro[:, :(self.nlat//2+1)]

    @ep2baro_shem.setter
    def ep2baro_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ep2baro[:, :jdim] = value[:, ::-1]

    @property
    def ep3baro_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ep3baro
        return self.ep3baro[:, -(self.nlat//2+1):]

    @ep3baro_nhem.setter
    def ep3baro_nhem(self, value):
        jdim = value.shape[1]
        self.ep3baro[:, -jdim:] = value

    @property
    def ep3baro_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ep3baro[:, :(self.nlat//2+1)]

    @ep3baro_shem.setter
    def ep3baro_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ep3baro[:, :jdim] = value[:, ::-1]

    @property
    def ep4_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ep4
        return self.ep4[:, -(self.nlat//2+1):]

    @ep4_nhem.setter
    def ep4_nhem(self, value):
        jdim = value.shape[1]
        self.ep4[:, -jdim:] = value

    @property
    def ep4_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ep4[:, :(self.nlat//2+1)]

    @ep4_shem.setter
    def ep4_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ep4[:, :jdim] = value[:, ::-1]

    @property
    def ubaro_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ubaro
        return self.ubaro[:, -(self.nlat//2+1):]

    @ubaro_nhem.setter
    def ubaro_nhem(self, value):
        jdim = value.shape[1]
        self.ubaro[:, -jdim:] = value

    @property
    def ubaro_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ubaro[:, :(self.nlat//2+1)]

    @ubaro_shem.setter
    def ubaro_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ubaro[:, :jdim] = value[:, ::-1]

    @property
    def lwa_baro_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.lwa_baro
        return self.lwa_baro[:, -(self.nlat // 2 + 1):]

    @lwa_baro_nhem.setter
    def lwa_baro_nhem(self, value):
        print(f"Debug. value.shape = {value.shape}")
        jdim = value.shape[1]
        self.lwa_baro[:, -jdim:] = value

    @property
    def lwa_baro_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.lwa_baro[:, :(self.nlat // 2 + 1)]

    @lwa_baro_shem.setter
    def lwa_baro_shem(self, value):
        jdim = value.shape[1]
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.lwa_baro[:, :jdim] = value[:, ::-1]


class InvalidCallOfSHemVariables(Exception):
    """
    northern_hemisphere_results_only = True. 
    Southern hemispheric variables are not computed.
    """
    pass


if __name__ == "__main__":

    # Do simple experiment to test setter method
    print("instance.pp_var")

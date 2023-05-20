"""
------------------------------------------
File name: data_storage.py
Author: Clare Huang
"""
from typing import Tuple, Optional, Union, List, NamedTuple
from dataclasses import dataclass, fields, field
import numpy as np
from collections import namedtuple


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
    def __init__(self, pydim: Union[int, Tuple], fdim: Union[int, Tuple],
                 swapaxis_1: int, swapaxis_2: int, northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.interpolated_u: Optional[np.ndarray] = None
        self.interpolated_v: Optional[np.ndarray] = None
        self.interpolated_theta: Optional[np.ndarray] = None
        self.interpolated_avort: Optional[np.ndarray] = None
        self.qgpv: Optional[np.ndarray] = None

    def to_python_indexing(self):  # TODO: may not be necessary
        Interpolated_fields = namedtuple('Interpolated_fields', ['U', 'V', 'Theta', 'AbsVort', 'QGPV'])
        interpolated_fields = Interpolated_fields(
            self.fortran_to_python(self.interpolated_u),
            self.fortran_to_python(self.interpolated_v),
            self.fortran_to_python(self.interpolated_theta),
            self.fortran_to_python(self.interpolated_avort),
            self.fortran_to_python(self.qgpv))
        return interpolated_fields


@dataclass
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
                 swapaxis_1: int, swapaxis_2: int, northern_hemisphere_results_only: bool):
        super().__init__(pydim, fdim, swapaxis_1, swapaxis_2, northern_hemisphere_results_only)
        self.qref = np.empty(self.fdim)  # This is to substitute self._qref_ntemp
        self.uref = np.empty(self.fdim)
        self.ptref = np.empty(self.fdim)
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
        if self.northern_hemisphere_results_only:
            self.uref[:, :] = value
        else:
            self.uref[-(self.nlat//2+1):, :] = value

    @property
    def uref_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.uref[:(self.nlat//2+1), :]

    @uref_shem.setter
    def uref_shem(self, value):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.uref[self.nlat//2::-1, :] = value  # running from equator to pole

    # *** PTref (reference potential temperature) ***
    @property
    def ptref_nhem(self):
        if self.northern_hemisphere_results_only:
            return self.ptref
        return self.ptref[-(self.nlat//2+1):, :]

    @ptref_nhem.setter
    def ptref_nhem(self, value):
        if self.northern_hemisphere_results_only:
            self.ptref[:, :] = value
        else:
            self.ptref[-(self.nlat//2+1):, :] = value

    @property
    def ptref_shem(self):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        return self.ptref[:(self.nlat//2+1), :]

    @ptref_shem.setter
    def ptref_shem(self, value):
        if self.northern_hemisphere_results_only:
            raise InvalidCallOfSHemVariables
        else:
            self.ptref[self.nlat//2::-1, :] = value  # running from equator to pole


class InvalidCallOfSHemVariables(Exception):
    f"""
    northern_hemisphere_results_only = True. 
    qref_shem is not available.
    """
    pass


if __name__ == "__main__":

    # Do simple experiment to test setter method
    print("instance.pp_var")

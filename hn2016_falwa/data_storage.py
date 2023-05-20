"""
------------------------------------------
File name: data_storage.py
Author: Clare Huang
"""
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass, fields, field
import numpy as np
from collections import namedtuple


@dataclass(match_args=True, kw_only=True)
class DerivedQuantityStorage:
    """
    Variables are stored in fortran dimension ordering.
    It is converted to python when output to users.
    """
    pydim: Tuple  # python dimension
    fdim: Tuple  # fortran dimension
    swapaxis_1: int
    swapaxis_2: int

    def fortran_to_python(self, phy_field: np.ndarray):
        return np.swapaxes(phy_field, self.swapaxis_1, self.swapaxis_2)

    def python_to_fortran(self, phy_field: np.ndarray):  # This may not be necessary
        return np.swapaxes(phy_field, self.swapaxis_2, self.swapaxis_1)


class DomainAverageStorage(DerivedQuantityStorage):
    def __post_init__(self):
        self.tn0: Optional[np.array] = None
        self.ts0: Optional[np.array] = None
        self.static_stability_n: Optional[np.array] = None
        self.static_stability_s: Optional[np.array] = None


class InterpolatedFieldsStorage(DerivedQuantityStorage):
    def __post_init__(self):
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


class ReferenceStatesStorage(DerivedQuantityStorage):
    def __post_init__(self):
        self.qref = np.empty(self.fdim)
        self.uref = np.empty(self.fdim)
        self.tref = np.empty(self.fdim)


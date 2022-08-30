from enum import Enum
import math
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg.lapack import dgetrf, dgetri
from hn2016_falwa import utilities
from hn2016_falwa.constant import SCALE_HEIGHT, CP, DRY_GAS_CONSTANT, EARTH_RADIUS, EARTH_OMEGA
from collections import namedtuple


class EquatorwardBC(Enum):
    """
    Equatorward Boundary conditions
    """
    NS11 = 'Nakamura and Solomon (JAS, 2011)'
    NH18 = 'Nakamura and Huang (Science, 2018)'
    NHN22 = 'Neal, Huang and Nakamura (GRL, 2022)'


class IntegrationScheme(Enum):
    """
    Equatorward Boundary conditions
    """
    RECTANGULAR = 'rectangular rule'


class QGFormalism(ABC):
    def __init__(self, xlon, ylat, plev, equatorward_bc: EquatorwardBC, kmax=49, maxit=100000, dz=1000., npart=None,
                 tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
                 omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, prefactor=None, eq_boundary_index=5):
        self._xlon = xlon
        self._ylat = ylat
        self._plev = plev
        self._equatorward_bc = equatorward_bc
        self._kmax = kmax
        self._dz = dz
        self._scale_height = scale_height
        self._prefactor = None

    def _compute_prefactor(self):
        """
        Private function. Compute prefactor for normalization by evaluating
            int^{z=kmax*dz}_0 e^{-z/H} dz
        using rectangular rule consistent with the integral evaluation in compute_lwa_and_barotropic_fluxes.f90.
        TODO: add integration scheme
        """
        self._prefactor = sum([math.exp(-k * self._dz / self._scale_height) * self._dz for k in range(1, self._kmax-1)])

    def _check_valid_plev(self, plev, scale_height, kmax, dz):
        pass

    @abstractmethod
    def interpolate_fields(self):
        pass

    @abstractmethod
    def compute_reference_states(self):
        pass

    @abstractmethod
    def compute_lwa_and_barotropic_fluxes(self):
        pass


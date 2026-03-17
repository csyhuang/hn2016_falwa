"""
------------------------------------------
File name: hemisphere_strategy.py
Author: Clare Huang

Hemisphere computation strategy classes.

This module provides strategy objects that encapsulate hemisphere-specific
logic for array slicing, dimension calculation, and computation control.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class HemisphereStrategy(ABC):
    """
    Abstract base class for hemisphere computation strategies.

    Encapsulates logic for:
    - Determining latitude dimensions for storage arrays
    - Array slicing for hemisphere-specific data extraction
    - Controlling whether Southern Hemisphere computations should run
    - Latitude coordinate selection for reference states
    """

    @abstractmethod
    def should_compute_shem(self) -> bool:
        """Return True if Southern Hemisphere should be computed."""
        pass

    @abstractmethod
    def get_lat_dim(self, nlat_analysis: int, equator_idx: int) -> int:
        """
        Return the latitude dimension for storage arrays.

        Parameters
        ----------
        nlat_analysis : int
            Total number of latitude points in analysis grid
        equator_idx : int
            Index of equator in the latitude array (1-based, Fortran convention)

        Returns
        -------
        int
            Number of latitude points for storage
        """
        pass

    @abstractmethod
    def get_ylat_for_ref_states(
        self, input_ylat: np.ndarray, internal_ylat: np.ndarray,
        nlat: int, nlat_analysis: int, need_interp: bool
    ) -> np.ndarray:
        """
        Return latitude array for reference state output.

        Parameters
        ----------
        input_ylat : np.ndarray
            Original latitude array provided by user
        internal_ylat : np.ndarray
            Internal analysis latitude array (may include interpolated equator)
        nlat : int
            Number of original latitude points
        nlat_analysis : int
            Number of internal analysis latitude points
        need_interp : bool
            Whether latitude interpolation was needed

        Returns
        -------
        np.ndarray
            Latitude array for reference states
        """
        pass

    @abstractmethod
    def get_ylat_for_analysis(
        self, internal_ylat: np.ndarray, nlat_analysis: int
    ) -> np.ndarray:
        """
        Return latitude array for internal analysis computations.
        """
        pass

    @abstractmethod
    def get_interp_slice(
        self, internal_ylat: np.ndarray, input_ylat: np.ndarray,
        nlat: int, nlat_analysis: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (source_lat, target_lat) for interpolation back to user grid.
        """
        pass

    @abstractmethod
    def get_clat_slice(self, clat: np.ndarray, equator_idx: int) -> np.ndarray:
        """
        Return cosine(latitude) array slice for flux computations.
        """
        pass

    @abstractmethod
    def slice_3d_for_flux(
        self, equator_idx: int, nlat_analysis: int
    ) -> Tuple[slice, slice, slice]:
        """
        Return 3D slicer tuple for flux vector computations.
        Format: (lon_slice, lat_slice, height_slice)
        """
        pass

    @abstractmethod
    def get_ptref_for_stretch(
        self, ptref: np.ndarray, nlat_analysis: int, kmax: int, equator_idx: int
    ) -> np.ndarray:
        """
        Prepare ptref array for stretch term computation.
        May need padding for NH-only case.
        """
        pass

    @property
    @abstractmethod
    def is_northern_only(self) -> bool:
        """Return True if only Northern Hemisphere is computed."""
        pass


class GlobalHemisphereStrategy(HemisphereStrategy):
    """
    Strategy for computing both Northern and Southern Hemispheres.
    """

    def should_compute_shem(self) -> bool:
        return True

    def get_lat_dim(self, nlat_analysis: int, equator_idx: int) -> int:
        return nlat_analysis

    def get_ylat_for_ref_states(
        self, input_ylat, internal_ylat, nlat, nlat_analysis, need_interp
    ):
        return input_ylat

    def get_ylat_for_analysis(self, internal_ylat, nlat_analysis):
        return internal_ylat

    def get_interp_slice(self, internal_ylat, input_ylat, nlat, nlat_analysis):
        return internal_ylat, input_ylat

    def get_clat_slice(self, clat, equator_idx):
        return clat

    def slice_3d_for_flux(self, equator_idx, nlat_analysis):
        return (slice(None), slice(None), slice(None))

    def get_ptref_for_stretch(self, ptref, nlat_analysis, kmax, equator_idx):
        return ptref

    @property
    def is_northern_only(self) -> bool:
        return False


class NorthernHemisphereOnlyStrategy(HemisphereStrategy):
    """
    Strategy for computing Northern Hemisphere only.

    When this strategy is active:
    - Storage arrays have reduced latitude dimension (equator to pole)
    - Southern Hemisphere computations are skipped
    - Output arrays contain only NH data
    """

    def should_compute_shem(self) -> bool:
        return False

    def get_lat_dim(self, nlat_analysis: int, equator_idx: int) -> int:
        return equator_idx

    def get_ylat_for_ref_states(
        self, input_ylat, internal_ylat, nlat, nlat_analysis, need_interp
    ):
        if need_interp:
            return input_ylat[-(nlat // 2):]
        else:
            return input_ylat[-(nlat // 2 + 1):]

    def get_ylat_for_analysis(self, internal_ylat, nlat_analysis):
        return internal_ylat[-(nlat_analysis // 2 + 1):]

    def get_interp_slice(self, internal_ylat, input_ylat, nlat, nlat_analysis):
        return (
            internal_ylat[-(nlat_analysis // 2 + 1):],
            input_ylat[-(nlat // 2):]
        )

    def get_clat_slice(self, clat, equator_idx):
        return clat[-equator_idx:]

    def slice_3d_for_flux(self, equator_idx, nlat_analysis):
        return (slice(None), slice(equator_idx - 1, nlat_analysis), slice(None))

    def get_ptref_for_stretch(self, ptref, nlat_analysis, kmax, equator_idx):
        # Pad ptref to full-globe so shapes broadcast; SH values are discarded
        ptref_full = np.zeros((nlat_analysis, kmax))
        ptref_full[-equator_idx:, :] = ptref
        return ptref_full

    @property
    def is_northern_only(self) -> bool:
        return True


def create_hemisphere_strategy(northern_hemisphere_results_only: bool) -> HemisphereStrategy:
    """
    Factory function to create appropriate hemisphere strategy.

    Parameters
    ----------
    northern_hemisphere_results_only : bool
        If True, returns NorthernHemisphereOnlyStrategy
        If False, returns GlobalHemisphereStrategy

    Returns
    -------
    HemisphereStrategy
        The appropriate strategy instance
    """
    if northern_hemisphere_results_only:
        return NorthernHemisphereOnlyStrategy()
    return GlobalHemisphereStrategy()


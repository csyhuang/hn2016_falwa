# Refactoring Plan: Strategy Pattern for Hemisphere Handling

## Overview

This document outlines a refactoring plan to replace the `northern_hemisphere_results_only` boolean flag with a **Strategy Pattern** implementation. The goal is to encapsulate hemisphere-specific logic into dedicated strategy classes while maintaining backward compatibility with the existing API.

---

## Current State Analysis

### How `northern_hemisphere_results_only` is currently used:

| Location | Usage |
|----------|-------|
| `__init__` (line 169) | Stored as `self._northern_hemisphere_results_only` |
| `_initialize_storage` (lines 194-239) | Determines `lat_dim` for storage arrays |
| `_return_interp_variables` (lines 394-403) | Different interpolation slices |
| `compute_lwa_only` (lines 643, 670) | Skip SHem computation |
| `_prepare_coordinates_and_ref_states` (line 711) | Select latitude input |
| `_compute_intermediate_barotropic_flux_terms` (line 764) | Skip SHem flux computation |
| `_compute_flux_vector_phi_baro` (lines 974-982) | Different array slicing |
| `_compute_stretch_term` (lines 1033-1061) | Array padding for NH-only |
| `_compute_layerwise_lwa_fluxes_wrapper` (line 1117) | Skip SHem computation |
| `ylat_ref_states` property (lines 1183-1188) | Return different lat subsets |
| `ylat_ref_states_analysis` property (lines 1196-1198) | Return different lat subsets |
| `QGFieldNH18._compute_reference_states` (line 1621) | Skip SHem computation |
| `QGFieldNHN22._compute_reference_states` (line 1794) | Skip SHem computation |
| `QGFieldNHN22.static_stability` property (lines 1895-1898) | Return different values |

### Files affected:
- `src/falwa/oopinterface.py`
- `src/falwa/data_storage.py` (minimal changes - already uses descriptors)

---

## Strategy Pattern Design

### 1. Create Strategy Classes

Create a new file: `src/falwa/hemisphere_strategy.py`

```python
"""
Hemisphere computation strategy classes.

This module provides strategy objects that encapsulate hemisphere-specific
logic for array slicing, dimension calculation, and computation control.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
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
```

---

### 2. Modify `QGFieldBase.__init__`

```python
# In oopinterface.py

from falwa.hemisphere_strategy import create_hemisphere_strategy, HemisphereStrategy

class QGFieldBase(ABC):
    
    def __init__(self, ..., northern_hemisphere_results_only=False, ...):
        # ... existing code ...
        
        # === Create hemisphere strategy (replaces boolean flag) ===
        self._hemisphere_strategy: HemisphereStrategy = create_hemisphere_strategy(
            northern_hemisphere_results_only
        )
        
        # Keep the property for backward compatibility
        self._northern_hemisphere_results_only = northern_hemisphere_results_only
        
        # ... rest of __init__ ...
```

---

### 3. Refactor Methods to Use Strategy

#### 3.1 `_initialize_storage`

**Before:**
```python
lat_dim = self.equator_idx if self.northern_hemisphere_results_only else self._nlat_analysis
```

**After:**
```python
lat_dim = self._hemisphere_strategy.get_lat_dim(self._nlat_analysis, self.equator_idx)
```

#### 3.2 `_return_interp_variables`

**Before:**
```python
if self.need_latitude_interpolation:
    if self.northern_hemisphere_results_only:
        return self._interp_back(
            variable, self._ylat[-(self._nlat_analysis // 2 + 1):],
            self._input_ylat[-(self.nlat // 2):],
            which_axis=interp_axis)
    else:
        return self._interp_back(variable, self._ylat, self._input_ylat, which_axis=interp_axis)
```

**After:**
```python
if self.need_latitude_interpolation:
    source_lat, target_lat = self._hemisphere_strategy.get_interp_slice(
        self._ylat, self._input_ylat, self.nlat, self._nlat_analysis)
    return self._interp_back(variable, source_lat, target_lat, which_axis=interp_axis)
```

#### 3.3 `compute_lwa_only` and similar computation methods

**Before:**
```python
if not self.northern_hemisphere_results_only:
    # Compute SHem...
```

**After:**
```python
if self._hemisphere_strategy.should_compute_shem():
    # Compute SHem...
```

#### 3.4 `_compute_flux_vector_phi_baro`

**Before:**
```python
if self._northern_hemisphere_results_only:
    slicer = [slice(None)] * 3
    slicer[1] = slice(self.equator_idx-1, self._nlat_analysis)
else:
    slicer = [slice(None)] * 3
```

**After:**
```python
slicer = self._hemisphere_strategy.slice_3d_for_flux(
    self.equator_idx, self._nlat_analysis)
```

#### 3.5 `_compute_stretch_term`

**Before:**
```python
if self.northern_hemisphere_results_only:
    ptref_full = np.zeros((self._nlat_analysis, self.kmax))
    ptref_full[-self.equator_idx:, :] = self._reference_states_storage.ptref
else:
    ptref_full = self._reference_states_storage.ptref
```

**After:**
```python
ptref_full = self._hemisphere_strategy.get_ptref_for_stretch(
    self._reference_states_storage.ptref,
    self._nlat_analysis, self.kmax, self.equator_idx)
```

#### 3.6 Properties `ylat_ref_states` and `ylat_ref_states_analysis`

**Before:**
```python
@property
def ylat_ref_states(self) -> np.array:
    if self.northern_hemisphere_results_only:
        if self.need_latitude_interpolation:
            return self._input_ylat[-(self.nlat // 2):]
        else:
            return self._input_ylat[-(self.nlat // 2 + 1):]
    return self._input_ylat
```

**After:**
```python
@property
def ylat_ref_states(self) -> np.array:
    return self._hemisphere_strategy.get_ylat_for_ref_states(
        self._input_ylat, self._ylat, self.nlat,
        self._nlat_analysis, self.need_latitude_interpolation)
```

---

### 4. Keep Backward Compatibility

The `northern_hemisphere_results_only` property must continue to work:

```python
@property
def northern_hemisphere_results_only(self) -> bool:
    """
    Even though a global field is required for input, whether ref state 
    and fluxes are computed for northern hemisphere only.
    """
    return self._hemisphere_strategy.is_northern_only
```

---

## Implementation Steps

### Phase 1: Create Strategy Module (Low Risk) ✅ COMPLETED
- [x] Create `src/falwa/hemisphere_strategy.py` with strategy classes
- [x] Add unit tests for strategy classes in `tests/test_hemisphere_strategy.py`
- [x] Verify strategy logic matches current boolean conditionals

### Phase 2: Integrate into QGFieldBase (Medium Risk) ✅ COMPLETED
- [x] Add strategy creation in `__init__`
- [x] Keep `_northern_hemisphere_results_only` for backward compatibility
- [x] Update `_initialize_storage` to use strategy
- [x] Run existing tests to ensure no regression

### Phase 3: Refactor Computation Methods (Medium Risk) ✅ COMPLETED
- [x] Update `_return_interp_variables`
- [x] Update `compute_lwa_only`
- [x] Update `_compute_intermediate_barotropic_flux_terms`
- [x] Update `_compute_layerwise_lwa_fluxes_wrapper`
- [x] Update `_compute_flux_vector_phi_baro`
- [x] Update `_compute_stretch_term`
- [x] Run tests after each method update

### Phase 4: Refactor Properties (Low Risk) ✅ COMPLETED
- [x] Update `ylat_ref_states`
- [x] Update `ylat_ref_states_analysis`
- [x] Update `northern_hemisphere_results_only` property

### Phase 5: Update Child Classes (Medium Risk) ✅ COMPLETED
- [x] Update `QGFieldNH18._compute_reference_states`
- [x] Update `QGFieldNHN22._compute_reference_states`
- [x] Update `QGFieldNHN22.static_stability`

### Phase 6: Cleanup and Documentation ✅ COMPLETED
- [x] Keep `_northern_hemisphere_results_only` for backward compatibility with storage classes
- [x] All 81 tests passing
- [x] Strategy pattern fully integrated

---

## Testing Strategy

### Unit Tests for Strategy Classes

```python
# tests/test_hemisphere_strategy.py

import pytest
import numpy as np
from falwa.hemisphere_strategy import (
    GlobalHemisphereStrategy,
    NorthernHemisphereOnlyStrategy,
    create_hemisphere_strategy
)


class TestGlobalStrategy:
    
    def test_should_compute_shem(self):
        strategy = GlobalHemisphereStrategy()
        assert strategy.should_compute_shem() is True
    
    def test_get_lat_dim(self):
        strategy = GlobalHemisphereStrategy()
        assert strategy.get_lat_dim(nlat_analysis=181, equator_idx=91) == 181
    
    def test_is_northern_only(self):
        strategy = GlobalHemisphereStrategy()
        assert strategy.is_northern_only is False


class TestNorthernOnlyStrategy:
    
    def test_should_compute_shem(self):
        strategy = NorthernHemisphereOnlyStrategy()
        assert strategy.should_compute_shem() is False
    
    def test_get_lat_dim(self):
        strategy = NorthernHemisphereOnlyStrategy()
        assert strategy.get_lat_dim(nlat_analysis=181, equator_idx=91) == 91
    
    def test_is_northern_only(self):
        strategy = NorthernHemisphereOnlyStrategy()
        assert strategy.is_northern_only is True


class TestFactoryFunction:
    
    def test_creates_global_strategy(self):
        strategy = create_hemisphere_strategy(False)
        assert isinstance(strategy, GlobalHemisphereStrategy)
    
    def test_creates_northern_only_strategy(self):
        strategy = create_hemisphere_strategy(True)
        assert isinstance(strategy, NorthernHemisphereOnlyStrategy)
```

### Integration Tests

Run all existing notebook examples and test suites with both:
- `northern_hemisphere_results_only=False`
- `northern_hemisphere_results_only=True`

Compare results before and after refactoring to ensure numerical equivalence.

---

## Benefits of This Refactoring

1. **Single Responsibility**: Each strategy class has one job - handle hemisphere-specific logic
2. **Open/Closed Principle**: Easy to add new strategies (e.g., `SouthernHemisphereOnlyStrategy`) without modifying existing code
3. **Testability**: Strategy logic can be unit tested in isolation
4. **Readability**: Method code becomes cleaner - no more scattered `if self.northern_hemisphere_results_only:` checks
5. **Type Safety**: IDE autocomplete and type checking work better with strategy objects

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing user code | Keep `northern_hemisphere_results_only` constructor param and property |
| Numerical differences | Comprehensive regression tests comparing outputs |
| Performance overhead | Strategy methods are simple; overhead is negligible |
| Increased complexity | Clear documentation and single-purpose strategy classes |

---

## Timeline Estimate

| Phase | Estimated Time |
|-------|----------------|
| Phase 1: Create Strategy Module | 2 hours |
| Phase 2: Integrate into QGFieldBase | 2 hours |
| Phase 3: Refactor Computation Methods | 4 hours |
| Phase 4: Refactor Properties | 1 hour |
| Phase 5: Update Child Classes | 2 hours |
| Phase 6: Cleanup and Documentation | 2 hours |
| **Total** | **~13 hours** |

---

## Future Extensions

Once the strategy pattern is in place, it becomes easy to add:

1. **`SouthernHemisphereOnlyStrategy`** - for studies focused on Antarctic dynamics
2. **`TropicsOnlyStrategy`** - for tropical wave activity analysis
3. **`CustomLatitudeRangeStrategy`** - allow user-defined latitude bounds

Each would simply implement the `HemisphereStrategy` interface without touching core computation logic.


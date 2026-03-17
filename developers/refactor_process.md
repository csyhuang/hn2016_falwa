# Refactoring Process: Strategy Pattern for Hemisphere Handling

**Date Completed:** March 16, 2026

## Overview

This document summarizes the refactoring of the `northern_hemisphere_results_only` logic in `QGFieldBase` and its child classes from scattered boolean conditionals to a Strategy Pattern implementation.

## Motivation

The original implementation had 15+ locations where `if self.northern_hemisphere_results_only:` conditionals controlled:
- Array dimension calculations
- Whether to compute Southern Hemisphere results
- Latitude slicing for data extraction
- Interpolation ranges

This scattered logic was difficult to maintain and extend. The Strategy Pattern encapsulates all hemisphere-specific logic into dedicated strategy classes.

---

## Files Created

### 1. `src/falwa/hemisphere_strategy.py`

New module containing:

| Class/Function | Description |
|----------------|-------------|
| `HemisphereStrategy` | Abstract base class defining the interface |
| `GlobalHemisphereStrategy` | Strategy for computing both hemispheres |
| `NorthernHemisphereOnlyStrategy` | Strategy for NH-only computation |
| `create_hemisphere_strategy()` | Factory function to create appropriate strategy |

**Key Strategy Methods:**
- `should_compute_shem()` → Returns whether SHem should be computed
- `get_lat_dim()` → Returns latitude dimension for storage arrays
- `get_ylat_for_ref_states()` → Returns latitude array for reference state output
- `get_ylat_for_analysis()` → Returns latitude array for internal analysis
- `get_interp_slice()` → Returns source/target lat for interpolation
- `get_clat_slice()` → Returns cos(lat) array slice for flux computations
- `slice_3d_for_flux()` → Returns 3D slicer for flux vector computations
- `get_ptref_for_stretch()` → Prepares ptref array (with padding for NH-only)
- `is_northern_only` → Property indicating if only NH is computed

### 2. `tests/test_hemisphere_strategy.py`

24 unit tests covering:
- `TestGlobalStrategy` (9 tests)
- `TestNorthernOnlyStrategy` (10 tests)
- `TestFactoryFunction` (3 tests)
- `TestStrategyConsistency` (2 tests)

---

## Files Modified

### 1. `src/meson.build`

Added `hemisphere_strategy.py` to install sources:

```meson
py.install_sources(
    ...
    'falwa/hemisphere_strategy.py',
    ...
)
```

### 2. `src/falwa/oopinterface.py`

#### Import Added
```python
from falwa.hemisphere_strategy import create_hemisphere_strategy, HemisphereStrategy
```

#### `QGFieldBase.__init__` Changes
```python
# Added strategy creation
self._hemisphere_strategy: HemisphereStrategy = create_hemisphere_strategy(
    northern_hemisphere_results_only
)
```

#### Methods Refactored

| Method | Change |
|--------|--------|
| `_initialize_storage` | `lat_dim = self._hemisphere_strategy.get_lat_dim(...)` |
| `_return_interp_variables` | Uses `strategy.get_interp_slice()` |
| `compute_lwa_only` | Uses `strategy.get_ylat_for_analysis()` and `strategy.should_compute_shem()` |
| `_prepare_coordinates_and_ref_states` | Signature changed to accept `hemisphere_strategy` parameter |
| `_compute_intermediate_barotropic_flux_terms` | Uses `strategy.should_compute_shem()` |
| `compute_lwa_and_barotropic_fluxes` | Uses `strategy.get_clat_slice()` |
| `_compute_flux_vector_phi_baro` | Uses `strategy.slice_3d_for_flux()` |
| `_compute_stretch_term` | Uses `strategy.get_ptref_for_stretch()` and `strategy.is_northern_only` |
| `_compute_layerwise_lwa_fluxes_wrapper` | Uses `strategy.should_compute_shem()` |

#### Properties Refactored

| Property | Change |
|----------|--------|
| `ylat_ref_states` | Uses `strategy.get_ylat_for_ref_states()` |
| `ylat_ref_states_analysis` | Uses `strategy.get_ylat_for_analysis()` |
| `northern_hemisphere_results_only` | Returns `strategy.is_northern_only` |

#### Child Class Changes

| Class | Method | Change |
|-------|--------|--------|
| `QGFieldNH18` | `_compute_reference_states` | Uses `strategy.should_compute_shem()` |
| `QGFieldNHN22` | `_compute_reference_states` | Uses `strategy.should_compute_shem()` |
| `QGFieldNHN22` | `static_stability` | Uses `strategy.is_northern_only` |

### 3. `developers/strategy_pattern.md`

Updated to mark all implementation phases as completed.

---

## Backward Compatibility

The refactoring maintains **full backward compatibility**:

1. **Constructor API unchanged**: `northern_hemisphere_results_only=False` parameter works as before
2. **Property unchanged**: `QGField.northern_hemisphere_results_only` returns the same boolean
3. **All existing tests pass**: 81 tests + 2 xpassed

The `_northern_hemisphere_results_only` attribute is retained for:
- Passing to storage classes which still use it internally
- Backward compatibility during transition period

---

## Test Results

```
================== 81 passed, 2 xpassed, 8 warnings in 18.47s ==================
```

- **Original tests**: 57 passed, 2 xpassed
- **New strategy tests**: 24 passed
- **Total**: 83 tests (81 passed + 2 xpassed)

---

## Code Quality Improvements

### Before (scattered conditionals)
```python
# Example from _return_interp_variables (before)
if self.need_latitude_interpolation:
    if self.northern_hemisphere_results_only:
        return self._interp_back(
            variable, self._ylat[-(self._nlat_analysis // 2 + 1):],
            self._input_ylat[-(self.nlat // 2):],
            which_axis=interp_axis)
    else:
        return self._interp_back(variable, self._ylat, self._input_ylat, ...)
```

### After (strategy pattern)
```python
# Example from _return_interp_variables (after)
if self.need_latitude_interpolation:
    source_lat, target_lat = self._hemisphere_strategy.get_interp_slice(
        self._ylat, self._input_ylat, self.nlat, self._nlat_analysis)
    return self._interp_back(variable, source_lat, target_lat, which_axis=interp_axis)
```

---

## Future Extensions

The Strategy Pattern now enables easy addition of new computation modes:

1. **`SouthernHemisphereOnlyStrategy`** - For Antarctic-focused studies
2. **`TropicsOnlyStrategy`** - For tropical wave activity analysis  
3. **`CustomLatitudeRangeStrategy`** - User-defined latitude bounds

Each would simply implement the `HemisphereStrategy` interface without modifying core computation logic.

---

## Summary of Changes by Line Count

| File | Lines Added | Lines Removed | Net Change |
|------|-------------|---------------|------------|
| `hemisphere_strategy.py` | 211 | 0 | +211 |
| `test_hemisphere_strategy.py` | 211 | 0 | +211 |
| `oopinterface.py` | ~30 | ~45 | -15 |
| `meson.build` | 1 | 0 | +1 |
| **Total** | ~453 | ~45 | +408 |

The increase in line count is due to:
1. New strategy module with comprehensive documentation
2. New test file with 24 unit tests
3. Extracted logic that was previously inline


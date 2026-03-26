# F-Order to C-Order Refactoring Plan

## Overview

This document outlines the plan to refactor all Numba modules and related code from Fortran-order (column-major, F-order) indexing to C-order (row-major) indexing. The goal is to eliminate the need for axis swapping between Python/NumPy operations and the computational kernels, resulting in cleaner code and potentially better performance.

## Background

### Current State
- The Numba modules in `src/falwa/numba_modules/` were translated from F2PY/Fortran modules
- Arrays currently use F-order indexing: `[i, j, k]` where i=longitude, j=latitude, k=height
- The `data_storage.py` module handles conversions between Python dimension (`pydim`) and Fortran dimension (`fdim`)
- `oopinterface.py` uses `np.swapaxes()` to convert between Python and Fortran indexing

### Target State
- All Numba modules will use C-order indexing: `[k, j, i]` where k=height, j=latitude, i=longitude
- The `data_storage.py` conversion logic can be simplified or removed
- No axis swapping needed in `oopinterface.py`

## Dimension Mapping

| Array Type | Current F-Order | Target C-Order | Variables |
|------------|-----------------|----------------|-----------|
| 3D fields  | `[i, j, k]` (nlon, nlat, kmax) | `[k, j, i]` (kmax, nlat, nlon) | `pv`, `uu`, `vv`, `pt`, `theta`, `avort`, `qgpv`, `astar1`, `astar2`, `ua1`, `ua2`, `ep1`, `ep2`, `ep3`, `ncforce` |
| 2D lat-height | `[j, k]` (nlat, kmax) | `[k, j]` (kmax, nlat) | `qref`, `uref`, `tref`, `ptref`, `qbar`, `tbar`, `ubar`, `zmav` |
| 2D lon-lat | `[i, j]` (nlon, nlat) | `[j, i]` (nlat, nlon) | `astarbaro`, `ubaro`, `ep4`, `ua1baro`, `ua2baro`, `ep1baro`, `ep2baro`, `ep3baro`, `lwa_baro`, `u_baro` |
| 1D fields | `[k]` (kmax,) | `[k]` (kmax,) | `height`, `t0`, `stat`, `z` (no change) |

## Files to Modify

### 1. Numba Modules (`src/falwa/numba_modules/`)

#### 1.1 `compute_qgpv.py`
**Current shapes:**
- Input: `ut[i, j, k]`, `vt[i, j, k]`, `theta[i, j, k]` → shape `(nlon, nlat, kmax)`
- Output: `pv[i, j, k]`, `avort[i, j, k]` → shape `(nlon, nlat, kmax)`
- Intermediate: `zmav[j, k]` → shape `(nlat, kmax)`

**Changes needed:**
- [ ] Change input shapes to `(kmax, nlat, nlon)`
- [ ] Change output shapes to `(kmax, nlat, nlon)`
- [ ] Change `zmav` shape to `(kmax, nlat)`
- [ ] Update all loop indices from `[i, j, kk]` to `[kk, j, i]`
- [ ] Update array initialization: `np.zeros((nlon, nlat, kmax))` → `np.zeros((kmax, nlat, nlon))`
- [ ] Update docstrings with new shapes

#### 1.2 `compute_qgpv_direct_inv.py`
**Changes needed:**
- [ ] Same pattern as `compute_qgpv.py`
- [ ] Update all 3D array indexing

#### 1.3 `compute_reference_states.py`
**Current shapes:**
- Input: `pv[i, j, k]`, `uu[i, j, k]`, `pt[i, j, k]` → shape `(nlon, nlat, kmax)`
- Output: `qref[j, k]`, `uref[j, k]`, `tref[j, k]` → shape `(jd, kmax)` or `(nlat, kmax)`

**Changes needed:**
- [ ] Change 3D input shapes to `(kmax, nlat, nlon)`
- [ ] Change 2D output shapes to `(kmax, jd)` or `(kmax, nlat)`
- [ ] Update loop indexing in `_compute_zonal_means()`
- [ ] Update loop indexing in `_compute_hemispheric_mean_temperature()`
- [ ] Update loop indexing in `_area_analysis()`
- [ ] Update docstrings

#### 1.4 `compute_qref_and_fawa_first.py`
**Changes needed:**
- [ ] Change 3D input shapes to `(kmax, nlat, nlon)`
- [ ] Change 2D output shapes to `(kmax, nd)`
- [ ] Update `_compute_zonal_means()` indexing
- [ ] Update `_area_analysis_qref()` indexing
- [ ] Update all intermediate array shapes

#### 1.5 `matrix_b4_inversion.py`
**Changes needed:**
- [ ] Change `qref[j, k]` → `qref[k, j]`
- [ ] Change `ckref[j, k]` → `ckref[k, j]`
- [ ] Change `sjk[j1, j2, k]` → `sjk[k, j1, j2]`
- [ ] Update all array accesses

#### 1.6 `matrix_after_inversion.py`
**Changes needed:**
- [ ] Update all 2D/3D array indexing patterns
- [ ] Change output array shapes

#### 1.7 `upward_sweep.py`
**Changes needed:**
- [ ] Update indexing for sweep arrays
- [ ] Change output shapes

#### 1.8 `compute_flux_dirinv.py`
**Current shapes:**
- Input 3D: `pv`, `uu`, `vv`, `pt`, `ncforce` → shape `(imax, jmax, kmax)`
- Input 2D: `qref`, `uref`, `tref` → shape `(nd, kmax)`
- Output 3D: `astar1`, `astar2`, `ua1`, `ua2`, `ep1`, `ep2`, `ep3`, `ncforce3d` → shape `(imax, nd, kmax)`
- Output 2D: `ep4` → shape `(imax, nd)`

**Changes needed:**
- [ ] Change 3D input shapes to `(kmax, jmax, imax)`
- [ ] Change 2D input shapes to `(kmax, nd)`
- [ ] Change 3D output shapes to `(kmax, nd, imax)`
- [ ] Change 2D output shapes to `(nd, imax)`
- [ ] Update all loop indices
- [ ] Update intermediate arrays (`qe`, `ue`, `ncforce2d`)

#### 1.9 `compute_lwa_only_nhn22.py`
**Current shapes:**
- Input: `pv[i, j, k]`, `uu[i, j, k]` → shape `(imax, jmax, kmax)`
- Input: `qref[j, k]` → shape `(nd, kmax)`
- Output: `astarbaro[i, j]`, `ubaro[i, j]` → shape `(imax, nd)`
- Output: `astar1[i, j, k]`, `astar2[i, j, k]` → shape `(imax, nd, kmax)`

**Changes needed:**
- [ ] Change all shapes following the mapping table
- [ ] Update all array accesses in the computational kernel

### 2. Data Storage (`src/falwa/data_storage.py`)

#### 2.1 `DerivedQuantityStorage` base class
**Changes needed:**
- [ ] Update `pydim` and `fdim` to be identical (no conversion needed)
- [ ] Update `swapaxis_1` and `swapaxis_2` (may no longer be needed)
**Skip the following changes as user will do that by hand:**
- [ ] Simplify or remove `fortran_to_python()` method (becomes identity or removed)
- [ ] Simplify or remove `python_to_fortran()` methoda

#### 2.2 `InterpolatedFieldsStorage`
**Current:**
- `pydim=(kmax, nlat, nlon)`, `fdim=(nlon, nlat, kmax)`

**Changes needed:**
- [ ] Set `pydim = fdim = (kmax, nlat, nlon)`
- [ ] Remove axis swap logic

#### 2.3 `ReferenceStatesStorage`
**Current:**
- `pydim=(kmax, nlat)`, `fdim=(nlat, kmax)`

**Changes needed:**
- [ ] Set `pydim = fdim = (kmax, nlat)`
- [ ] Update `qref_correct_unit()` method

#### 2.4 `LayerwiseFluxTermsStorage`
**Current:**
- `pydim=(kmax, nlat, nlon)`, `fdim=(nlon, nlat, kmax)`

**Changes needed:**
- [ ] Set `pydim = fdim = (kmax, nlat, nlon)`

#### 2.5 `BarotropicFluxTermsStorage`
**Current:**
- `pydim=(nlat, nlon)`, `fdim=(nlon, nlat)`

**Changes needed:**
- [ ] Set `pydim = fdim = (nlat, nlon)`

#### 2.6 `OutputBarotropicFluxTermsStorage`
**Changes needed:**
- [ ] Already uses `pydim`, verify consistency

#### 2.7 `HemisphericProperty`, `NHemProperty`, `SHemProperty`
**Changes needed:**
- [ ] Update `ndims_fill` tuples for new axis positions
- [ ] Update slicing logic for latitude dimension

### 3. OOP Interface (`src/falwa/oopinterface.py`) ✅ COMPLETED

#### 3.1 `_initialize_storage()` method ✅
**Changes made:**
- [x] Updated all `pydim` and `fdim` arguments to use C-order (now identical)
- [x] Set `swapaxis_1=0, swapaxis_2=0` for all storage classes (no swap needed)

#### 3.2 `interpolate_fields()` method ✅
**Changes made:**
- [x] Removed `np.swapaxes(interpolated_u, 0, 2)` calls
- [x] Store interpolated fields directly without axis swapping

#### 3.3 `_compute_qgpv()` implementations ✅
**Changes made:**
- [x] No axis swapping needed - Numba modules already return C-order arrays

#### 3.4 `_compute_reference_states()` implementations ✅
**Changes made:**
- [x] Updated docstring to reflect C-order output dimensions `[kmax, nlat]`

#### 3.5 `compute_lwa_only()` method ✅
**Changes made:**
- [x] Updated qref slicing from `qref[-equator_idx:, :]` to `qref[:, -equator_idx:]` for C-order `[k, j]`
- [x] Updated hemispheric flip slicing for C-order

#### 3.6 `_compute_lwa_and_barotropic_fluxes_wrapper()` method ✅
**Changes made:**
- [x] Updated `uref.shape[0]` to `uref.shape[1]` for C-order `[k, j]`
- [x] Updated vertical averaging calls with `height_axis=0`
- [x] Updated uref_baro calculation for C-order broadcasting

#### 3.7 `_compute_intermediate_barotropic_flux_terms()` method ✅
**Changes made:**
- [x] Updated qref slicing from `qref[-equator_idx:, :]` to `qref[:, -equator_idx:]`
- [x] Updated uref and tref slicing from `[j, :]` to `[:, j]` indexing

#### 3.8 `compute_lwa_and_barotropic_fluxes()` method ✅
**Changes made:**
- [x] Removed `np.swapaxes()` calls for ncforce initialization
- [x] Updated ncforce array initialization to C-order `(kmax, nlat, nlon)`
- [x] Removed axis swapping for divergence_eddy_momentum_flux calculation
- [x] Updated clat broadcasting to use `clat[:, np.newaxis]` for 2D arrays

#### 3.9 `_vertical_average()` method ✅
**Changes made:**
- [x] Updated default `height_axis` parameter from -1 to 0

#### 3.10 `_compute_flux_vector_phi_baro()` method ✅
**Changes made:**
- [x] Updated uref broadcasting from `[np.newaxis, :, :]` to `[:, :, np.newaxis]` for C-order
- [x] Removed `.T` transpose from return statement

#### 3.11 `_compute_stretch_term()` method ✅
**Changes made:**
- [x] Updated ptref_full shape from `(nlat, kmax)` to `(kmax, nlat)` for C-order
- [x] Updated ptref broadcasting from `[np.newaxis, :, :]` to `[:, :, np.newaxis]`
- [x] Removed `np.swapaxes(inner_ep4, 0, 2)` from result

#### 3.12 `_compute_layerwise_lwa_fluxes_wrapper()` method ✅
**Changes made:**
- [x] Updated ncforce initialization to C-order `(kmax, nlat, nlon)`
- [x] Removed axis swapping for ncforce
- [x] Updated qref, uref, tref slicing for C-order `[k, j]`

### 4. Utilities (`src/falwa/utilities.py`)

**Changes needed:**
- [ ] Review `zonal_convergence()` function
- [ ] Review `z_derivative_of_prod()` function
- [ ] Update any axis-specific operations

### 5. Tests

#### 5.1 Update test files
**Changes needed:**
- [ ] Update expected array shapes in all tests
- [ ] Update array creation in test fixtures
- [ ] Verify numerical results remain identical (within floating-point tolerance)

#### 5.2 Create validation tests
**Changes needed:**
- [ ] Create tests that compare C-order results with known-good F-order results
- [ ] Add regression tests for key outputs

## Implementation Strategy

### Phase 1: Numba Module Updates
1. Start with the simplest module (`compute_lwa_only_nhn22.py`)
2. Create a parallel C-order version with `_corder` suffix for testing
3. Validate outputs match F-order version
4. Proceed to more complex modules

### Phase 2: Data Storage Updates
1. Create new storage classes with C-order (or update existing ones)
2. Update dimension parameters
3. Simplify/remove axis swapping methods

### Phase 3: OOP Interface Updates ✅ COMPLETED
1. ✅ Update `_initialize_storage()` first - set pydim=fdim for all storage classes
2. ✅ Update each computational method - removed axis swapping
3. ✅ Remove all `np.swapaxes()` calls

**Changes made in Phase 3:**
- `_initialize_storage()`: Updated all storage classes to use identical `pydim` and `fdim` in C-order
- `_vertical_average()`: Changed default `height_axis` from -1 to 0 for C-order arrays
- `interpolate_fields()`: Removed `np.swapaxes()` calls, arrays stored directly in C-order
- `_compute_lwa_and_barotropic_fluxes_wrapper()`: Updated slicing for C-order 2D arrays `[k, j]`
- `_compute_intermediate_barotropic_flux_terms()`: Updated reference state slicing from `[j, :]` to `[:, j]`
- `compute_lwa_only()`: Updated qref slicing for C-order
- `compute_lwa_and_barotropic_fluxes()`: Removed axis swapping, arrays already in C-order
- `_compute_flux_vector_phi_baro()`: Updated broadcasting for C-order arrays
- `_compute_stretch_term()`: Updated ptref_full shape and removed axis swapping
- `_compute_layerwise_lwa_fluxes_wrapper()`: Removed axis swapping, updated slicing

### Phase 4: Testing and Validation
1. Run all existing tests
2. Compare outputs with reference data
3. Performance benchmarking

## Detailed Index Mapping Reference

### 3D Array Loop Transformations

**F-order (current):**
```python
for k in range(kmax):
    for j in range(nlat):
        for i in range(nlon):
            arr[i, j, k] = ...
```

**C-order (target):**
```python
for k in range(kmax):
    for j in range(nlat):
        for i in range(nlon):
            arr[k, j, i] = ...
```

### 2D Lat-Height Array Transformations

**F-order (current):**
```python
for k in range(kmax):
    for j in range(nlat):
        arr[j, k] = ...
```

**C-order (target):**
```python
for k in range(kmax):
    for j in range(nlat):
        arr[k, j] = ...
```

### 2D Lon-Lat Array Transformations

**F-order (current):**
```python
for j in range(nlat):
    for i in range(nlon):
        arr[i, j] = ...
```

**C-order (target):**
```python
for j in range(nlat):
    for i in range(nlon):
        arr[j, i] = ...
```

## Key Considerations

### 1. Hemispheric Slicing
Current code uses slicing like `arr[:, ::-1, :]` for latitude reversal. After conversion:
- 3D arrays: `arr[:, ::-1, :]` remains valid (latitude is still axis 1)
- 2D lat-height: `arr[:, ::-1]` → verify axis positions

### 2. Vertical Averaging
The `_vertical_average()` method uses `height_axis=-1`. After conversion:
- Height will be axis 0, so update to `height_axis=0`

### 3. Broadcasting
NumPy broadcasting rules apply differently:
- F-order: `clat[np.newaxis, :, np.newaxis]` for shape `(1, nlat, 1)`
- C-order: may need `clat[np.newaxis, :, np.newaxis]` or similar

### 4. Contiguous Arrays
After refactoring, arrays will naturally be C-contiguous, improving Numba performance:
- Remove explicit `np.ascontiguousarray()` calls where unnecessary
- Or keep them for safety/clarity

## Checklist Summary

### Numba Modules (9 files)
- [ ] `compute_qgpv.py`
- [ ] `compute_qgpv_direct_inv.py`
- [ ] `compute_reference_states.py`
- [ ] `compute_qref_and_fawa_first.py`
- [ ] `matrix_b4_inversion.py`
- [ ] `matrix_after_inversion.py`
- [ ] `upward_sweep.py`
- [ ] `compute_flux_dirinv.py`
- [ ] `compute_lwa_only_nhn22.py`

### Data Storage (1 file)
- [ ] `data_storage.py` - all storage classes

### OOP Interface (1 file)
- [x] `oopinterface.py` - all methods

### Tests
- [ ] Unit tests
- [ ] Integration tests
- [ ] Regression tests

## Notes

- The loop order in Numba JIT functions doesn't need to change (loops iterate the same way), only the array indexing changes
- This refactoring should be done incrementally with validation at each step
- Keep the F-order versions available during transition for comparison
- Document any performance differences observed


# F2PY to Numba Migration Plan for `hn2016_falwa`

## Overview

This document outlines a comprehensive plan for migrating the F2PY (Fortran-to-Python) modules in `src/falwa/f90_modules/` to Numba JIT-compiled Python functions. This will eliminate the need for the Meson build system and Fortran compiler.

### Goals
1. **Eliminate Meson build system** - No compilation step required during installation
2. **Pure Python distribution** - Package can be distributed as a pure Python wheel
3. **Maintain performance** - Numba JIT compilation provides near-Fortran performance
4. **Improve maintainability** - Python code is easier to read, modify, and debug
5. **Better cross-platform support** - No Fortran compiler required
6. **Simplified CI/CD** - No complex build matrix for different platforms

### Why Numba over Cython?

| Aspect | Numba | Cython |
|--------|-------|--------|
| **Build step** | None (JIT compiled) | Requires compilation |
| **Distribution** | Pure Python wheel | Platform-specific wheels |
| **Syntax** | Pure Python | Mixed Python/C syntax |
| **NumPy support** | Excellent, native | Requires memoryviews |
| **Debugging** | Standard Python tools | More complex |
| **First-call overhead** | ~100-500ms JIT warm-up | None |
| **Installation** | `pip install numba` | Requires C compiler |

---

## F2PY Modules to Migrate

The following **9 F2PY modules** in `src/falwa/f90_modules/` need migration:

| # | Module Name | Source File | Lines | Complexity | Priority |
|---|-------------|-------------|-------|------------|----------|
| 1 | `compute_qgpv` | `compute_qgpv.f90` | 137 | Medium | High |
| 2 | `compute_qgpv_direct_inv` | `compute_qgpv_direct_inv.f90` | 113 | Medium | High |
| 3 | `compute_reference_states` | `compute_reference_states.f90` | 262 | High | High |
| 4 | `compute_qref_and_fawa_first` | `compute_qref_and_fawa_first.f90` | 177 | High | High |
| 5 | `matrix_b4_inversion` | `matrix_b4_inversion.f90` | 89 | Medium | Medium |
| 6 | `matrix_after_inversion` | `matrix_after_inversion.f90` | 40 | Low | Medium |
| 7 | `upward_sweep` | `upward_sweep.f90` | 93 | Medium | Medium |
| 8 | `compute_flux_dirinv_nshem` | `compute_flux_dirinv.f90` | 183 | High | High |
| 9 | `compute_lwa_only_nhn22` | `compute_lwa_only_nhn22.f90` | 108 | Medium | Medium |

**Total: ~1,202 lines of Fortran → ~1,500-2,000 lines of Python/Numba**

---

## Migration Strategy

### Phase 1: Setup and Infrastructure (Week 1)

#### 1.1 Create New Module Structure
```
src/falwa/
├── numba_modules/
│   ├── __init__.py
│   ├── compute_qgpv.py
│   ├── compute_qgpv_direct_inv.py
│   ├── compute_reference_states.py
│   ├── compute_qref_and_fawa_first.py
│   ├── matrix_operations.py      # Combined: matrix_b4_inversion, matrix_after_inversion
│   ├── upward_sweep.py
│   ├── compute_flux_dirinv.py
│   ├── compute_lwa_only_nhn22.py
│   └── utils.py                  # Shared helper functions
├── ...
```

#### 1.2 Add Numba Dependency
Update `pyproject.toml`:
```toml
[project]
dependencies = [
    "numpy>=1.20",
    "numba>=0.56",
    ...
]
```

#### 1.3 Create Testing Infrastructure
- Create `tests/test_numba_modules.py` with comparison tests
- Ensure outputs match F2PY modules within numerical tolerance

---

### Phase 2: Core QGPV Computation (Week 2)

#### 2.1 Migrate `compute_qgpv.py`

**Fortran Signature:**
```fortran
SUBROUTINE compute_qgpv(nlon, nlat, kmax, ut, vt, theta, height, t0, stat, &
                        aa, omega, dz, hh, rr, cp, &
                        pv, avort)
```

**Numba Implementation Pattern:**
```python
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def compute_qgpv(
    ut: np.ndarray,      # (nlon, nlat, kmax) zonal wind
    vt: np.ndarray,      # (nlon, nlat, kmax) meridional wind
    theta: np.ndarray,   # (nlon, nlat, kmax) potential temperature
    height: np.ndarray,  # (kmax,) height levels
    t0: np.ndarray,      # (kmax,) reference temperature
    stat: np.ndarray,    # (kmax,) static stability
    aa: float,           # Earth radius
    omega: float,        # Earth rotation rate
    dz: float,           # vertical grid spacing
    hh: float,           # scale height
    rr: float,           # gas constant
    cp: float            # specific heat
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute quasi-geostrophic potential vorticity and absolute vorticity.
    
    Returns:
        pv: (nlon, nlat, kmax) potential vorticity
        avort: (nlon, nlat, kmax) absolute vorticity
    """
    nlon, nlat, kmax = ut.shape
    rkappa = rr / cp
    pi = np.pi
    dphi = pi / (nlat - 1)
    
    # Initialize output arrays
    pv = np.zeros((nlon, nlat, kmax), dtype=np.float64)
    avort = np.zeros((nlon, nlat, kmax), dtype=np.float64)
    
    # Compute zonal means (vectorized)
    tzd = np.zeros((nlat, kmax), dtype=np.float64)
    for j in range(nlat):
        for k in range(kmax):
            for i in range(nlon):
                tzd[j, k] += theta[i, j, k] / nlon
    
    # Interior absolute vorticity
    for kk in prange(kmax):
        for j in range(1, nlat - 1):
            phi0 = -np.pi/2 + (j) * np.pi / (nlat - 1)
            phim = -np.pi/2 + (j - 1) * np.pi / (nlat - 1)
            phip = -np.pi/2 + (j + 1) * np.pi / (nlat - 1)
            
            for i in range(1, nlon - 1):
                av1 = 2.0 * omega * np.sin(phi0)
                av2 = (vt[i+1, j, kk] - vt[i-1, j, kk]) / (2.0 * aa * np.cos(phi0) * dphi)
                av3 = -(ut[i, j+1, kk] * np.cos(phip) - ut[i, j-1, kk] * np.cos(phim)) / \
                      (2.0 * aa * np.cos(phi0) * dphi)
                avort[i, j, kk] = av1 + av2 + av3
            
            # Periodic boundary in longitude
            # ... (handle i=0 and i=nlon-1 cases)
    
    # Interior PV with stretching term
    # ... (similar pattern)
    
    return pv, avort
```

#### 2.2 Migrate `compute_qgpv_direct_inv.py`

Similar structure but with hemispheric separation for static stability.

---

### Phase 3: Reference State Computation (Week 3)

#### 3.1 Migrate `compute_reference_states.py` (Most Complex)

**Key Challenges:**
- Iterative SOR (Successive Over-Relaxation) solver
- Convergence checking
- Area analysis algorithm

**Numba Implementation Notes:**
```python
@njit(cache=True)
def sor_solver(
    qref: np.ndarray,
    u: np.ndarray,
    tref: np.ndarray,
    # ... other parameters
    maxits: int,
    eps: float,
    rjac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Successive Over-Relaxation elliptic solver.
    
    Note: Numba supports while loops and convergence checking.
    """
    for iteration in range(maxits):
        max_change = 0.0
        for j in range(1, jd - 1):
            for k in range(1, kmax - 1):
                # SOR update formula
                new_val = (1 - omega_sor) * u[j, k] + omega_sor * (...)
                max_change = max(max_change, abs(new_val - u[j, k]))
                u[j, k] = new_val
        
        if max_change < eps:
            return qref, u, tref, iteration + 1
    
    return qref, u, tref, maxits
```

#### 3.2 Migrate `compute_qref_and_fawa_first.py`

**Key Algorithm: Area Analysis**
```python
@njit(cache=True)
def area_analysis(
    pv2: np.ndarray,      # 2D PV field
    alat: np.ndarray,     # latitude area thresholds
    npart: int,           # number of partitions
    a: float,             # Earth radius
    dphi: float,
    dlambda: float
) -> np.ndarray:
    """
    Compute equivalent latitude via area analysis.
    """
    qmax = np.max(pv2)
    qmin = np.min(pv2)
    dq = (qmax - qmin) / (npart - 1)
    
    qn = np.zeros(npart)
    an = np.zeros(npart)
    
    for nn in range(npart):
        qn[nn] = qmax - dq * nn
    
    # Area accumulation
    imax, jmax = pv2.shape
    for j in range(jmax):
        phi0 = -np.pi/2 + dphi * j
        for i in range(imax):
            ind = int((qmax - pv2[i, j]) / dq)
            ind = min(max(ind, 0), npart - 1)
            da = a * a * dphi * dlambda * np.cos(phi0)
            an[ind] += da
    
    # Cumulative sum
    aan = np.cumsum(an)
    
    # Interpolate to get qref
    # ...
    return qref
```

---

### Phase 4: Matrix Operations (Week 4)

#### 4.1 Migrate `matrix_b4_inversion.py` and `matrix_after_inversion.py`

**Combined into `matrix_operations.py`:**
```python
@njit(cache=True)
def setup_matrices_before_inversion(
    k: int,
    jmax: int,
    kmax: int,
    nd: int,
    jb: int,
    jd: int,
    z: np.ndarray,
    statn: np.ndarray,
    qref: np.ndarray,
    ckref: np.ndarray,
    sjk: np.ndarray,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Setup Q, C, D, r matrices for LU decomposition."""
    # ... implementation
    return qjj, djj, cjj, rj


@njit(cache=True)
def process_matrices_after_inversion(
    k: int,
    kmax: int,
    jd: int,
    qjj: np.ndarray,
    djj: np.ndarray,
    cjj: np.ndarray,
    rj: np.ndarray,
    sjk: np.ndarray,
    tjk: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Post-process LU-inverted matrices."""
    # Matrix multiplication using loops (Numba-friendly)
    xjj = np.zeros((jd - 2, jd - 2))
    for i in range(jd - 2):
        for j in range(jd - 2):
            for kk in range(jd - 2):
                xjj[i, j] += qjj[i, kk] * djj[kk, j]
            sjk[i, j, k - 1] = -xjj[i, j]
    # ...
    return sjk, tjk
```

**Note:** For matrix inversion, use `np.linalg.solve` or `np.linalg.inv` (supported by Numba).

#### 4.2 Migrate `upward_sweep.py`

```python
@njit(cache=True)
def upward_sweep(
    jmax: int,
    kmax: int,
    nd: int,
    jb: int,
    jd: int,
    sjk: np.ndarray,
    tjk: np.ndarray,
    ckref: np.ndarray,
    tb: np.ndarray,
    qref_over_cor: np.ndarray,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform upward sweep to compute Uref and Tref."""
    # ...
    return qref, tref, u
```

---

### Phase 5: Flux Computation (Week 5)

#### 5.1 Migrate `compute_flux_dirinv.py`

**This is the most performance-critical module.**

```python
@njit(parallel=True, fastmath=True, cache=True)
def compute_flux_dirinv_nshem(
    pv: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    pt: np.ndarray,
    ncforce: np.ndarray,
    tn0: np.ndarray,
    qref: np.ndarray,
    uref: np.ndarray,
    tref: np.ndarray,
    imax: int,
    jmax: int,
    kmax: int,
    nd: int,
    jb: int,
    jd: int,
    is_nhem: bool,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float,
    prefac: float
) -> tuple[np.ndarray, ...]:
    """
    Compute LWA and flux terms for a hemisphere.
    
    Uses parallel loops for the main k-i-j iteration.
    """
    # Initialize outputs
    astar1 = np.zeros((imax, nd, kmax))
    astar2 = np.zeros((imax, nd, kmax))
    # ... other outputs
    
    # Main computation loop (parallelized over k)
    for k in prange(1, kmax - 1):
        zk = dz * (k)
        for i in range(imax):
            for j in range(jstart, jend):
                # LWA computation
                # ... (transcribe Fortran logic)
                pass
    
    return astar1, astar2, ncforce3d, ua1, ua2, ep1, ep2, ep3, ep4
```

#### 5.2 Migrate `compute_lwa_only_nhn22.py`

Simplified version of flux computation.

---

### Phase 6: Integration and Testing (Week 6)

#### 6.1 Create Wrapper Module

Update `src/falwa/__init__.py`:
```python
# Try Numba modules first, fall back to F2PY if available
try:
    from falwa.numba_modules import (
        compute_qgpv,
        compute_qgpv_direct_inv,
        compute_reference_states,
        compute_qref_and_fawa_first,
        matrix_b4_inversion,
        matrix_after_inversion,
        upward_sweep,
        compute_flux_dirinv_nshem,
        compute_lwa_only_nhn22
    )
    _USE_NUMBA = True
except ImportError:
    # Fall back to F2PY modules
    from falwa._fortran_modules import (...)
    _USE_NUMBA = False
```

#### 6.2 Comprehensive Testing

Create `tests/test_numba_vs_fortran.py`:
```python
import pytest
import numpy as np

# Import both implementations
from falwa.numba_modules import compute_qgpv as compute_qgpv_numba
from falwa import compute_qgpv as compute_qgpv_f2py  # Original

class TestNumbaVsFortran:
    @pytest.fixture
    def sample_data(self):
        """Load test data."""
        # Use existing test data from tests/data/
        ...
    
    def test_compute_qgpv_equivalence(self, sample_data):
        """Verify Numba output matches F2PY output."""
        result_numba = compute_qgpv_numba(...)
        result_f2py = compute_qgpv_f2py(...)
        
        np.testing.assert_allclose(
            result_numba[0], result_f2py[0],  # pv
            rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            result_numba[1], result_f2py[1],  # avort
            rtol=1e-10, atol=1e-12
        )
    
    # Similar tests for all other modules...
```

#### 6.3 Performance Benchmarking

Create `benchmarks/benchmark_numba.py`:
```python
import time
import numpy as np

def benchmark_module(func_numba, func_f2py, args, n_runs=10):
    """Compare execution times."""
    # Warm-up JIT
    _ = func_numba(*args)
    
    # Benchmark
    numba_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func_numba(*args)
        numba_times.append(time.perf_counter() - start)
    
    f2py_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func_f2py(*args)
        f2py_times.append(time.perf_counter() - start)
    
    print(f"Numba: {np.mean(numba_times)*1000:.2f} ± {np.std(numba_times)*1000:.2f} ms")
    print(f"F2PY:  {np.mean(f2py_times)*1000:.2f} ± {np.std(f2py_times)*1000:.2f} ms")
    print(f"Ratio: {np.mean(numba_times)/np.mean(f2py_times):.2f}x")
```

---

### Phase 7: Documentation and Cleanup (Week 7)

#### 7.1 Update Build Configuration

Update `pyproject.toml` to remove Meson:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hn2016_falwa"
dependencies = [
    "numpy>=1.20",
    "numba>=0.56",
    "scipy>=1.7",
    # ... other deps
]
```

#### 7.2 Update Documentation

- Update `readme.md` with new installation instructions
- Add Numba-specific notes to docstrings
- Update API documentation

#### 7.3 Deprecation Strategy

1. **Version X.Y.0**: Include both F2PY and Numba modules, Numba as default
2. **Version X.Y+1.0**: Deprecation warning when using F2PY fallback
3. **Version X.Y+2.0**: Remove F2PY modules entirely

---

## Numba Best Practices for This Migration

### 1. Array Memory Layout

```python
# Fortran uses column-major (F) order
# Ensure arrays are contiguous
@njit
def process_array(arr):
    # Force Fortran order if needed
    arr = np.asfortranarray(arr)
    # Or ensure C-contiguous
    arr = np.ascontiguousarray(arr)
```

### 2. Avoid Python Objects in Hot Loops

```python
# BAD - creates Python objects
@njit
def bad_example():
    result = []  # Python list
    for i in range(100):
        result.append(i)

# GOOD - use NumPy arrays
@njit
def good_example():
    result = np.empty(100)
    for i in range(100):
        result[i] = i
```

### 3. Use `prange` for Parallelization

```python
from numba import prange

@njit(parallel=True)
def parallel_loop(arr):
    result = np.zeros_like(arr)
    for i in prange(arr.shape[0]):  # Parallel loop
        for j in range(arr.shape[1]):  # Sequential inner loop
            result[i, j] = process(arr[i, j])
    return result
```

### 4. Cache Compiled Functions

```python
@njit(cache=True)  # Cache to disk
def cached_function(...):
    ...
```

### 5. Handle Complex Branching

```python
# Numba handles if/else well, but avoid dynamic typing
@njit
def conditional_logic(is_nhem: bool, value: float):
    if is_nhem:
        return value * 2.0
    else:
        return value * -2.0
```

---

## Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical differences | Medium | High | Extensive comparison testing with tolerance checks |
| Performance regression | Low | Medium | Benchmark each module; optimize hot paths |
| JIT compilation overhead | Medium | Low | Use `cache=True`; document warm-up time |
| Numba version compatibility | Low | Medium | Pin minimum version; test with multiple versions |
| Complex Fortran features | Low | Medium | Manual translation; extensive testing |

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Setup | Module structure, dependencies, test infrastructure |
| 2 | Core QGPV | `compute_qgpv.py`, `compute_qgpv_direct_inv.py` |
| 3 | Reference States | `compute_reference_states.py`, `compute_qref_and_fawa_first.py` |
| 4 | Matrix Operations | `matrix_operations.py`, `upward_sweep.py` |
| 5 | Flux Computation | `compute_flux_dirinv.py`, `compute_lwa_only_nhn22.py` |
| 6 | Integration | Testing, benchmarking, CI/CD updates |
| 7 | Documentation | Docs, cleanup, release preparation |

---

## Appendix: Module Dependencies

```
oopinterface.py
├── QGFieldNH18
│   ├── _compute_qgpv() → compute_qgpv
│   ├── _compute_reference_state_wrapper() → compute_reference_states
│   └── compute_lwa_and_barotropic_fluxes() → compute_flux_dirinv_nshem
│
├── QGFieldNHN22
│   ├── _compute_qgpv() → compute_qgpv_direct_inv
│   ├── _compute_reference_states_nhn22_hemispheric_wrapper()
│   │   ├── compute_qref_and_fawa_first
│   │   ├── matrix_b4_inversion (loop)
│   │   ├── np.linalg.inv (Python)
│   │   ├── matrix_after_inversion (loop)
│   │   └── upward_sweep
│   └── compute_lwa_and_barotropic_fluxes() → compute_flux_dirinv_nshem
│
└── QGFieldBase
    └── compute_lwa_only() → compute_lwa_only_nhn22
```

---

## References

1. [Numba Documentation](https://numba.pydata.org/numba-doc/latest/index.html)
2. [Numba for CUDA](https://numba.pydata.org/numba-doc/latest/cuda/index.html) (future GPU acceleration)
3. [F2PY User Guide](https://numpy.org/doc/stable/f2py/)
4. Existing Cython migration plan: `developers/cython_migration_plan.md`


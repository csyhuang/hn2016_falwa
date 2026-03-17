# F2PY to Numba Migration Plan for `falwa`

## Overview

This document outlines the plan for migrating the F2PY (Fortran-to-Python) modules in `src/falwa/f90_modules/` to **Numba JIT-compiled Python** to eliminate the need for the Meson build system and Fortran compiler.

### Goals
1. **Eliminate Meson build system** - No compilation step during installation
2. **No Fortran/C compiler required** - Pure Python that JIT-compiles at runtime
3. **Maintain performance** - Numba achieves near-Fortran speeds for numerical code
4. **Improve maintainability** - Standard Python/NumPy syntax
5. **Easier debugging** - Can run without JIT for debugging purposes
6. **Cross-platform compatibility** - Works on any platform with Python + LLVM

---

## Comparison: Numba vs Other Alternatives

| Feature | F2PY (Current) | Cython | **Numba** |
|---------|----------------|--------|-----------|
| Build required | Yes (Meson + Fortran) | Yes (C compiler) | **No** (JIT) |
| Compiler needed | Fortran + C | C only | **None** |
| Performance | Excellent | Excellent | Excellent |
| Debugging | Difficult | Difficult | **Easy** (disable JIT) |
| Install complexity | High | Medium | **Low** |
| GPU support | No | Limited | **Yes** (CUDA) |
| Array syntax | Fortran | Mixed | **NumPy-native** |

---

## F2PY Modules to Migrate (9 Total)

### Module Inventory

| # | Module Name | Source File | Lines | Complexity | Priority |
|---|-------------|-------------|-------|------------|----------|
| 1 | `compute_qgpv` | `compute_qgpv.f90` | 137 | Medium | P2 |
| 2 | `compute_qgpv_direct_inv` | `compute_qgpv_direct_inv.f90` | 113 | Medium | P2 |
| 3 | `compute_qref_and_fawa_first` | `compute_qref_and_fawa_first.f90` | 177 | High | P3 |
| 4 | `matrix_b4_inversion` | `matrix_b4_inversion.f90` | 89 | Low | P1 |
| 5 | `matrix_after_inversion` | `matrix_after_inversion.f90` | 40 | Low | P1 |
| 6 | `upward_sweep` | `upward_sweep.f90` | 93 | Low | P1 |
| 7 | `compute_reference_states` | `compute_reference_states.f90` | 262 | High | P4 |
| 8 | `compute_flux_dirinv_nshem` | `compute_flux_dirinv.f90` | 183 | High | P3 |
| 9 | `compute_lwa_only_nhn22` | `compute_lwa_only_nhn22.f90` | 108 | Medium | P2 |

---

## Migration Architecture

### New Directory Structure

```
src/falwa/
├── __init__.py              # Update imports
├── oopinterface.py          # No changes needed
├── numba_modules/           # NEW: Numba implementations
│   ├── __init__.py
│   ├── compute_qgpv.py
│   ├── compute_qgpv_direct_inv.py
│   ├── compute_qref_and_fawa_first.py
│   ├── matrix_b4_inversion.py
│   ├── matrix_after_inversion.py
│   ├── upward_sweep.py
│   ├── compute_reference_states.py
│   ├── compute_flux_dirinv.py
│   └── compute_lwa_only_nhn22.py
├── f90_modules/             # ARCHIVE: Keep for reference
│   └── ...
└── ...
```

---

## Phase 1: Infrastructure Setup (Week 1)

### 1.1 Update `pyproject.toml`

Replace Meson build with pure Python:

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "falwa"
# ... existing metadata ...
dependencies = [
    "numpy>=1.22",
    "scipy",
    "xarray",
    "numba>=0.57"  # Add Numba dependency
]
```

### 1.2 Create `src/falwa/numba_modules/__init__.py`

```python
"""
Numba JIT-compiled implementations of computational kernels.
These replace the F2PY Fortran modules.
"""
from .compute_qgpv import compute_qgpv
from .compute_qgpv_direct_inv import compute_qgpv_direct_inv
from .compute_qref_and_fawa_first import compute_qref_and_fawa_first
from .matrix_b4_inversion import matrix_b4_inversion
from .matrix_after_inversion import matrix_after_inversion
from .upward_sweep import upward_sweep
from .compute_reference_states import compute_reference_states
from .compute_flux_dirinv import compute_flux_dirinv_nshem
from .compute_lwa_only_nhn22 import compute_lwa_only_nhn22

__all__ = [
    'compute_qgpv',
    'compute_qgpv_direct_inv', 
    'compute_qref_and_fawa_first',
    'matrix_b4_inversion',
    'matrix_after_inversion',
    'upward_sweep',
    'compute_reference_states',
    'compute_flux_dirinv_nshem',
    'compute_lwa_only_nhn22',
]
```

### 1.3 Update `src/falwa/__init__.py`

```python
"""
File name: __init__.py
Author: Clare Huang, Christopher Polster
"""

__version__ = "3.0.0"  # Major version bump for Numba migration

# Import from Numba modules instead of F2PY
from .numba_modules import (
    compute_qgpv,
    compute_qgpv_direct_inv,
    compute_qref_and_fawa_first,
    matrix_b4_inversion,
    matrix_after_inversion,
    upward_sweep,
    compute_reference_states,
    compute_flux_dirinv_nshem,
    compute_lwa_only_nhn22,
)
```

### 1.4 Remove Build Files

- Delete `meson.build` (root)
- Delete `src/meson.build`
- Delete `src/falwa/meson.build`
- Delete `src/falwa/f90_modules/meson.build`

---

## Phase 2: Module Migration (Weeks 2-5)

### Numba Module Template

Each module will follow this template pattern:

```python
"""
Numba-accelerated implementation of [MODULE_NAME].

Translated from: src/falwa/f90_modules/[FORTRAN_FILE].f90
"""
import numpy as np
from numba import njit, prange
from math import sin, cos, acos, exp, pi

# Optional: Pre-compile with specific signatures for faster first call
# @njit('Tuple((f8[:,:,:], f8[:,:,:]))(f8[:,:,:], f8[:,:,:], ...)', cache=True)

@njit(cache=True, parallel=True)
def compute_kernel(...):
    """
    Internal JIT-compiled kernel.
    """
    # Loop-heavy computations here
    pass

def compute_function(input_arrays, params):
    """
    Public interface matching F2PY function signature.
    
    Parameters
    ----------
    ... (document all parameters)
    
    Returns
    -------
    ... (document all outputs)
    """
    # Ensure arrays are contiguous and correct dtype
    input_arrays = np.ascontiguousarray(input_arrays, dtype=np.float64)
    
    # Pre-allocate output arrays
    output = np.zeros(...)
    
    # Call JIT kernel
    compute_kernel(input_arrays, output, params)
    
    return output
```

### Migration Order by Priority

#### Priority 1: Simple Matrix Operations (Week 2)
These have no dependencies and simple loop structures:

1. **`matrix_after_inversion.py`** (40 lines)
   - Simple matrix multiply and vector operations
   - Good starting point to validate approach

2. **`matrix_b4_inversion.py`** (89 lines)
   - Matrix coefficient computation
   - Moderate loop complexity

3. **`upward_sweep.py`** (93 lines)
   - Sequential vertical sweep
   - Array operations with boundary conditions

#### Priority 2: QGPV Computation (Week 3)

4. **`compute_qgpv.py`** (137 lines)
   - Zonal mean calculations
   - Interior vorticity computation
   - Periodic boundary handling

5. **`compute_qgpv_direct_inv.py`** (113 lines)
   - Similar to compute_qgpv
   - Hemispheric-specific static stability

6. **`compute_lwa_only_nhn22.py`** (108 lines)
   - LWA computation kernel
   - Uses similar patterns to flux computation

#### Priority 3: LWA and Reference State Setup (Week 4)

7. **`compute_qref_and_fawa_first.py`** (177 lines)
   - Area analysis algorithm
   - Matrix setup for inversion

8. **`compute_flux_dirinv.py`** (183 lines)
   - Most complex flux computation
   - Multiple output arrays
   - Hemispheric handling

#### Priority 4: SOR Solver (Week 5)

9. **`compute_reference_states.py`** (262 lines)
   - Iterative SOR (Successive Over-Relaxation) solver
   - Convergence criteria
   - Most complex module

---

## Phase 3: Detailed Migration Specifications

### 3.1 `matrix_after_inversion.py` (Example Implementation)

```python
"""
Numba implementation of matrix operations after LU inversion.

Translated from: src/falwa/f90_modules/matrix_after_inversion.f90
"""
import numpy as np
from numba import njit

@njit(cache=True)
def _matrix_after_inversion_kernel(k, kmax, jd, qjj, djj, cjj, rj, sjk, tjk):
    """
    Internal kernel for matrix operations after inversion.
    """
    jd_minus_2 = jd - 2
    
    # Get tj slice
    tj = tjk[:, k].copy()
    
    # xjj = qjj @ djj
    xjj = np.zeros((jd_minus_2, jd_minus_2), dtype=np.float64)
    for i in range(jd_minus_2):
        for j in range(jd_minus_2):
            for kk in range(jd_minus_2):
                xjj[i, j] += qjj[i, kk] * djj[kk, j]
            # sjk[:,:,k-1] = -xjj
            sjk[i, j, k-1] = -xjj[i, j]
    
    # yj = rj - cjj @ tj
    yj = np.zeros(jd_minus_2, dtype=np.float64)
    for i in range(jd_minus_2):
        for kk in range(jd_minus_2):
            yj[i] += cjj[i, kk] * tj[kk]
        yj[i] = rj[i] - yj[i]
    
    # tjk[:,k-1] = qjj @ yj
    for i in range(jd_minus_2):
        tjk[i, k-1] = 0.0
        for kk in range(jd_minus_2):
            tjk[i, k-1] += qjj[i, kk] * yj[kk]

def matrix_after_inversion(k, kmax, jd, qjj, djj, cjj, rj, sjk, tjk):
    """
    Post-process matrices after LU inversion for upward sweep.
    
    Parameters
    ----------
    k : int
        Current vertical level index
    kmax : int
        Total number of vertical levels
    jd : int
        Meridional grid dimension
    qjj : ndarray, shape (jd-2, jd-2)
        Q matrix (inverted)
    djj : ndarray, shape (jd-2, jd-2)
        D matrix
    cjj : ndarray, shape (jd-2, jd-2)
        C matrix
    rj : ndarray, shape (jd-2,)
        r vector
    sjk : ndarray, shape (jd-2, jd-2, kmax-1)
        S matrix (modified in-place)
    tjk : ndarray, shape (jd-2, kmax-1)
        t vector (modified in-place)
    
    Returns
    -------
    sjk : ndarray
        Updated S matrix
    tjk : ndarray
        Updated t vector
    
    Notes
    -----
    This function modifies sjk and tjk in-place, matching F2PY behavior.
    """
    # Ensure correct dtypes and contiguity
    qjj = np.ascontiguousarray(qjj, dtype=np.float64)
    djj = np.ascontiguousarray(djj, dtype=np.float64)
    cjj = np.ascontiguousarray(cjj, dtype=np.float64)
    rj = np.ascontiguousarray(rj, dtype=np.float64)
    sjk = np.ascontiguousarray(sjk, dtype=np.float64)
    tjk = np.ascontiguousarray(tjk, dtype=np.float64)
    
    _matrix_after_inversion_kernel(k, kmax, jd, qjj, djj, cjj, rj, sjk, tjk)
    
    return sjk, tjk
```

### 3.2 Key Fortran-to-Numba Translation Patterns

#### Array Indexing
```fortran
! Fortran (1-based)
do j = 1, nd
    result(j) = input(j)
enddo
```
```python
# Python/Numba (0-based)
for j in range(nd):
    result[j] = input[j]
```

#### Array Slicing
```fortran
! Fortran column-major order
array(i, j, k)
```
```python
# NumPy row-major order (C-contiguous)
# May need to transpose when porting
array[k, j, i]  # or rearrange loops
```

#### Trigonometric Functions
```fortran
pi = acos(-1.)
sin(phi0)
cos(phi0)
```
```python
from math import sin, cos, acos, pi
# or use np.sin/np.cos for arrays
```

#### Conditional Logic
```fortran
if((qe(i,jj).le.0.).and.(jj.ge.j)) then
```
```python
if qe[i, jj] <= 0.0 and jj >= j:
```

### 3.3 Handling Fortran-Specific Patterns

#### In-Place Array Modification
Fortran `INTENT(INOUT)` arrays modify in place. In Numba:
```python
@njit
def modify_array(arr):
    # arr is modified in place
    arr[0] = 1.0
    return arr  # Return for clarity
```

#### Multi-dimensional Array Initialization
```python
# Pre-allocate output arrays before calling kernel
result = np.zeros((imax, nd, kmax), dtype=np.float64, order='F')
```

#### Periodic Boundary Conditions
Handle explicitly in loops:
```python
for i in range(nlon):
    ip1 = (i + 1) % nlon
    im1 = (i - 1 + nlon) % nlon
```

---

## Phase 4: Testing Strategy

### 4.1 Unit Tests Per Module

Create `tests/test_numba_modules.py`:

```python
"""
Unit tests comparing Numba implementations to F2PY reference.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Test tolerance
RTOL = 1e-10
ATOL = 1e-12

class TestMatrixAfterInversion:
    """Test matrix_after_inversion against F2PY reference."""
    
    def test_basic_operation(self):
        """Test basic matrix computation."""
        # Setup test data
        kmax, jd = 49, 86
        k = 10
        
        qjj = np.random.rand(jd-2, jd-2)
        djj = np.random.rand(jd-2, jd-2)
        cjj = np.random.rand(jd-2, jd-2)
        rj = np.random.rand(jd-2)
        sjk = np.zeros((jd-2, jd-2, kmax-1))
        tjk = np.random.rand(jd-2, kmax-1)
        
        # Run both implementations
        from falwa.numba_modules import matrix_after_inversion as numba_impl
        from falwa.f90_modules import matrix_after_inversion as f2py_impl
        
        # Copy arrays for F2PY (modifies in place)
        sjk_f2py = sjk.copy()
        tjk_f2py = tjk.copy()
        f2py_impl(k+1, kmax, jd, qjj, djj, cjj, rj, sjk_f2py, tjk_f2py)  # 1-based
        
        # Numba implementation
        sjk_numba = sjk.copy()
        tjk_numba = tjk.copy()
        numba_impl(k, kmax, jd, qjj, djj, cjj, rj, sjk_numba, tjk_numba)  # 0-based
        
        assert_allclose(sjk_numba, sjk_f2py, rtol=RTOL, atol=ATOL)
        assert_allclose(tjk_numba, tjk_f2py, rtol=RTOL, atol=ATOL)
```

### 4.2 Integration Tests

Run existing tests in `tests/test_oopinterface.py` with Numba backend:

```python
@pytest.fixture(params=['numba', 'f2py'])
def backend(request):
    """Fixture to test both backends."""
    if request.param == 'numba':
        # Use Numba modules
        import falwa.numba_modules as modules
    else:
        # Use F2PY modules
        import falwa.f90_modules as modules
    return modules
```

### 4.3 Performance Benchmarks

Create `benchmarks/benchmark_numba.py`:

```python
"""
Performance comparison: Numba vs F2PY.
"""
import time
import numpy as np

def benchmark_compute_qgpv():
    """Benchmark QGPV computation."""
    # Setup realistic data dimensions
    nlon, nlat, kmax = 360, 181, 49
    
    # Generate test data
    ut = np.random.rand(nlon, nlat, kmax).astype(np.float64)
    vt = np.random.rand(nlon, nlat, kmax).astype(np.float64)
    theta = np.random.rand(nlon, nlat, kmax).astype(np.float64) * 300
    height = np.linspace(0, 48000, kmax)
    t0 = np.ones(kmax) * 250
    stat = np.ones(kmax) * 0.01
    
    # Warmup Numba JIT
    from falwa.numba_modules import compute_qgpv as numba_qgpv
    _ = numba_qgpv(ut[:10], vt[:10], theta[:10], height, t0, stat, 
                   6.378e6, 7.29e-5, 1000., 7000., 287., 1004.)
    
    # Benchmark Numba
    t0_numba = time.perf_counter()
    for _ in range(10):
        pv_numba, avort_numba = numba_qgpv(
            ut, vt, theta, height, t0, stat,
            6.378e6, 7.29e-5, 1000., 7000., 287., 1004.
        )
    t_numba = (time.perf_counter() - t0_numba) / 10
    
    # Benchmark F2PY
    from falwa import compute_qgpv as f2py_qgpv
    t0_f2py = time.perf_counter()
    for _ in range(10):
        pv_f2py, avort_f2py = f2py_qgpv(
            ut, vt, theta, height, t0, stat,
            6.378e6, 7.29e-5, 1000., 7000., 287., 1004.
        )
    t_f2py = (time.perf_counter() - t0_f2py) / 10
    
    print(f"Numba: {t_numba:.4f}s, F2PY: {t_f2py:.4f}s")
    print(f"Speedup: {t_f2py/t_numba:.2f}x" if t_numba < t_f2py else f"Slowdown: {t_numba/t_f2py:.2f}x")
```

---

## Phase 5: Optimization Techniques

### 5.1 Numba-Specific Optimizations

#### Parallelization with `prange`
```python
from numba import njit, prange

@njit(parallel=True, cache=True)
def parallel_loop(arr):
    result = np.zeros_like(arr)
    for i in prange(arr.shape[0]):  # Parallel
        for j in range(arr.shape[1]):  # Sequential
            result[i, j] = compute_something(arr[i, j])
    return result
```

#### Loop Fusion
Combine multiple loops that iterate over the same data:
```python
# Instead of:
for i in range(n):
    a[i] = compute_a(x[i])
for i in range(n):
    b[i] = compute_b(x[i])

# Use:
for i in range(n):
    a[i] = compute_a(x[i])
    b[i] = compute_b(x[i])
```

#### Avoid Python Objects in Hot Loops
```python
# Bad - creates Python objects
for i in range(n):
    result.append(x[i])  # append creates objects

# Good - pre-allocate
result = np.empty(n)
for i in range(n):
    result[i] = x[i]
```

### 5.2 Memory Layout Optimization

Ensure arrays are contiguous for cache efficiency:
```python
# Check/convert to C-contiguous
arr = np.ascontiguousarray(arr)

# Or use Fortran order if that matches loop structure
arr = np.asfortranarray(arr)
```

### 5.3 Caching JIT Compilation

Use `cache=True` to avoid recompilation:
```python
@njit(cache=True)
def cached_function(...):
    ...
```

---

## Phase 6: Deployment and Maintenance

### 6.1 Update CI/CD Pipeline

`.github/workflows/test.yml`:
```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[test]"
    
    - name: Run tests
      run: pytest tests/ -v
```

### 6.2 Documentation Updates

Update `readme.md`:
```markdown
## Installation

```bash
pip install falwa
```

No Fortran compiler required! The package uses Numba JIT compilation.

### First-Time Use

The first call to computational functions may take a few seconds 
as Numba compiles the code. Subsequent calls will be fast.
```

### 6.3 Version Strategy

- **v2.x** - Current F2PY implementation (maintenance only)
- **v3.0** - Numba implementation (recommended)
- Provide migration guide for users upgrading from v2.x

---

## Timeline Summary

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | Infrastructure | Update pyproject.toml, create numba_modules/, remove meson files |
| 2 | Migration P1 | matrix_after_inversion, matrix_b4_inversion, upward_sweep |
| 3 | Migration P2 | compute_qgpv, compute_qgpv_direct_inv, compute_lwa_only_nhn22 |
| 4 | Migration P3 | compute_qref_and_fawa_first, compute_flux_dirinv |
| 5 | Migration P4 | compute_reference_states (SOR solver) |
| 6 | Testing | Unit tests, integration tests, benchmarks |
| 7 | Optimization | Performance tuning, parallel optimization |
| 8 | Release | Documentation, CI/CD, release v3.0 |

---

## Risk Mitigation

### Potential Issues and Solutions

| Risk | Mitigation |
|------|------------|
| Performance regression | Benchmark each module; use `prange` for parallelization |
| Numerical precision differences | Use `np.float64`; validate with tolerance tests |
| Array layout incompatibility | Handle C vs Fortran order explicitly |
| First-call JIT overhead | Document; use `cache=True`; consider AOT compilation |
| Numba version compatibility | Pin minimum version; test across versions |

---

## Appendix: Quick Reference

### Fortran to Numba Cheat Sheet

| Fortran | Numba/Python |
|---------|--------------|
| `REAL` | `np.float64` |
| `INTEGER` | `np.int64` |
| `LOGICAL` | `bool` or `np.bool_` |
| `DO i=1,n` | `for i in range(n)` |
| `IF (cond) THEN` | `if cond:` |
| `array(i,j,k)` | `array[i-1,j-1,k-1]` or transpose |
| `INTENT(IN)` | Regular parameter |
| `INTENT(OUT)` | Return value or pre-allocated array |
| `INTENT(INOUT)` | Pass array, modify in place |
| `acos(-1.)` | `np.pi` or `math.pi` |
| `exp(x)` | `math.exp(x)` or `np.exp(x)` |

### Numba Decorator Reference

```python
@njit                    # Basic JIT compilation
@njit(cache=True)        # Cache compiled code to disk
@njit(parallel=True)     # Enable automatic parallelization
@njit(fastmath=True)     # Allow fast-math optimizations
@njit(nogil=True)        # Release GIL (for multi-threading)
```

---

## References

- [Numba Documentation](https://numba.readthedocs.io/)
- [Numba Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- [Existing Cython Migration Plan](./cython_migration_plan.md)
- [Original F2PY Modules](../src/falwa/f90_modules/)


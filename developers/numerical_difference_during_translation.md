# Numerical Differences During F2PY to Numba Translation

When translating code from F2PY (Fortran) to Numba (Python), small numerical differences are expected and normal, even when the implementations are logically equivalent. This document explains the reasons and provides guidance on acceptable tolerances.

## Reasons for Small Numerical Differences

### 1. Floating-Point Arithmetic is NOT Associative

```python
# In floating-point, (a + b) + c ≠ a + (b + c)
a = 1e-16
b = 1.0
c = -1.0
print((a + b) + c)  # might give 0.0
print(a + (b + c))  # might give 1e-16
```

### 2. Loop Order and Memory Access Patterns

- Fortran uses **column-major (F-order)** memory layout
- NumPy/Numba default to **row-major (C-order)**
- Even with correct results, the order of accumulating sums can differ, leading to different rounding errors

### 3. Compiler Optimizations

- Fortran compilers may use **Fused Multiply-Add (FMA)** operations: `a*b + c` computed in one step with single rounding
- Python/Numba may compute `a*b` first, round, then add `c` and round again
- This can introduce ~1 ULP (unit in last place) differences

### 4. Reduction Operations (Sums)

When summing many values:

```fortran
! Fortran might process: sum = ((v1 + v2) + v3) + v4 ...
```

```python
# NumPy/Numba might use different chunking or SIMD vectorization
# sum = (v1 + v2 + v3 + v4) + (v5 + v6 + v7 + v8) ...
```

### 5. Math Library Implementations

Functions like `exp()`, `log()`, `sin()` may have implementation differences between:
- Fortran's intrinsic math library
- NumPy's underlying C library (OpenLibm or system libm)

---

## What Tolerance is Acceptable?

For climate/atmospheric science applications, differences of **O(1e-4)** or smaller are typically considered **numerically equivalent** because:

1. Input data uncertainty is usually much larger
2. Physical interpretation remains unchanged
3. These are within expected floating-point precision limits

### Recommended Tolerances

| Scenario | Relative Tolerance (`rtol`) | Absolute Tolerance (`atol`) |
|----------|-----------------------------|-----------------------------|
| Strict validation | 1e-4 | 1e-10 |
| Scientific equivalence | 1e-3 | 1e-8 |
| Relaxed (edge cases) | 1e-2 | 1e-6 |

---

## Investigating Larger Differences

Points with larger differences (>1e-2) typically occur at:

1. **Near-polar latitudes**: Small denominators in spherical geometry calculations
2. **Near-zero reference values**: Division by small numbers amplifies relative errors
3. **Domain boundaries**: Edge effects in finite difference schemes
4. **Transition regions**: Where physical quantities change sign

### Comparison Script

Use the comparison script to validate translations:

```bash
cd scripts/numba_migration

python compare_nc_files.py \
    --reference output_plots_v2.3.3_NH18/lwa_reference_output_v2.3.3_NH18.nc \
    --translated output_plots_ce9e383_NH18/lwa_reference_output_NH18.nc \
    --rtol 1.e-4 \
    --atol 1.e-10 \
    --output_file comparison_report.txt
```

### Options

- `--rtol`: Relative tolerance (default: 1e-4)
- `--atol`: Absolute tolerance for near-zero values (default: 0.0)
- `--max_mismatches`: Maximum mismatches to display per variable (default: 50)
- `--output_file`: Save detailed report to file

---

## References

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [NumPy floating-point documentation](https://numpy.org/doc/stable/user/basics.types.html)
- [Numba documentation on numerical precision](https://numba.readthedocs.io/)


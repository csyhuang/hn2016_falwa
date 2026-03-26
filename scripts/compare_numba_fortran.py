#!/usr/bin/env python
"""Quick comparison script between Numba and Fortran compute_qgpv implementations."""

import numpy as np
import xarray as xr
from falwa.numba_modules import compute_qgpv as compute_qgpv_numba
from falwa.numba_modules import compute_qgpv_direct_inv as compute_qgpv_direct_inv_numba
from falwa import compute_qgpv as compute_qgpv_fortran
from falwa import compute_qgpv_direct_inv as compute_qgpv_direct_inv_fortran

# ============================================================================
# Load real test data (same approach as analyze_nc_dataset.py)
# ============================================================================
print("Loading test data...")
data_dir = "/Users/clarehuang/Library/CloudStorage/Dropbox/GitHub/hn2016_falwa/tests/data"
u_data = xr.open_dataset(f"{data_dir}/2005-01-23-0000-u.nc")
v_data = xr.open_dataset(f"{data_dir}/2005-01-23-0000-v.nc")
t_data = xr.open_dataset(f"{data_dir}/2005-01-23-0000-t.nc")

# Grid setup (following analyze_nc_dataset.py conventions)
xlon = u_data.longitude.values
ylat = u_data.latitude.values
plev = u_data.level.values

# latitude has to be in ascending order
if np.diff(ylat)[0] < 0:
    print("  Flipping latitude to ascending order.")
    ylat = ylat[::-1]

# pressure level has to be in descending order (ascending height)
if np.diff(plev)[0] > 0:
    print("  Flipping pressure levels to descending order.")
    plev = plev[::-1]

nlon = xlon.size
nlat = ylat.size
nlev = plev.size
print(f"  Grid dimensions: {nlon} lon x {nlat} lat x {nlev} levels")

# Vertical grid parameters for compute_qgpv
kmax = 49  # number of grid points for vertical extrapolation
dz = 1000.  # differential height element [m]
height = np.arange(0, kmax) * dz  # pseudoheight [m]
hh = 7000.  # scale height [m]

# Physical constants
aa = 6.378e6      # Earth radius [m]
omega = 7.29e-5   # Earth rotation rate [rad/s]
rr = 287.0        # Dry gas constant [J/(kg·K)]
cp = 1004.0       # Specific heat at constant pressure [J/(kg·K)]

# Extract fields for time step 0 (flip latitude to ascending order)
tstep = 0
uu = u_data.u.values[:, ::-1, :]  # (plev, lat, lon)
vv = v_data.v.values[:, ::-1, :]
tt = t_data.t.values[:, ::-1, :]

print(f"  Raw data shape (plev, lat, lon): {uu.shape}")

# For compute_qgpv, we need (nlon, nlat, kmax) order
# The Fortran code expects already-interpolated fields on the height grid
# For testing, we'll create synthetic interpolated data with correct shape
# Using the actual data dimensions but transposed to (lon, lat, lev)
ut = np.ascontiguousarray(np.transpose(uu, (2, 1, 0)))  # (nlon, nlat, nlev)
vt = np.ascontiguousarray(np.transpose(vv, (2, 1, 0)))
theta = np.ascontiguousarray(np.transpose(tt, (2, 1, 0)))

print(f"  Transposed data shape (lon, lat, lev): {ut.shape}")

# Reference temperature and static stability profiles
# Using simple profiles for testing (these would normally come from domain averages)
t0 = np.linspace(300, 220, nlev).astype(np.float64)
stat = np.ones(nlev, dtype=np.float64) * 0.01  # Static stability [K/m]

# Update kmax to match actual data
kmax = nlev
height = np.linspace(0, 16000, kmax).astype(np.float64)

# ============================================================================
# Run both implementations
# ============================================================================
print('\nRunning Numba implementation...')
pv_numba, avort_numba = compute_qgpv_numba(
    ut, vt, theta, height, t0, stat,
    aa=aa, omega=omega, dz=dz,
    hh=hh, rr=rr, cp=cp
)

print('Running Fortran implementation...')
pv_fortran, avort_fortran = compute_qgpv_fortran(
    ut, vt, theta, height, t0, stat,
    aa=aa, omega=omega, dz=dz,
    hh=hh, rr=rr, cp=cp
)

print(f'Numba PV shape: {pv_numba.shape}')
print(f'Fortran PV shape: {pv_fortran.shape}')

# Compare
print('\nComparing results...')
print(f'Max abs diff (avort): {np.max(np.abs(avort_numba - avort_fortran)):.6e}')
print(f'Max abs diff (pv): {np.max(np.abs(pv_numba - pv_fortran)):.6e}')

# Relative differences (avoid division by very small numbers)
avort_mask = np.abs(avort_fortran) > 1e-15
if np.any(avort_mask):
    max_rel_avort = np.max(np.abs((avort_numba[avort_mask] - avort_fortran[avort_mask]) / avort_fortran[avort_mask]))
    print(f'Max rel diff (avort): {max_rel_avort:.6e}')

pv_interior = pv_fortran[..., 1:-1]
pv_mask = np.abs(pv_interior) > 1e-15
if np.any(pv_mask):
    pv_numba_interior = pv_numba[..., 1:-1]
    max_rel_pv = np.max(np.abs((pv_numba_interior[pv_mask] - pv_interior[pv_mask]) / pv_interior[pv_mask]))
    print(f'Max rel diff (pv interior): {max_rel_pv:.6e}')

# Check if close
try:
    np.testing.assert_allclose(avort_numba, avort_fortran, rtol=1e-5, atol=1e-9)
    print('\n✓ Absolute vorticity: PASS (within tolerance)')
except AssertionError as e:
    print(f'\n✗ Absolute vorticity: FAIL')
    print(str(e)[:500])

try:
    np.testing.assert_allclose(pv_numba, pv_fortran, rtol=1e-5, atol=1e-9)
    print('✓ Potential vorticity: PASS (within tolerance)')
except AssertionError as e:
    print(f'✗ Potential vorticity: FAIL')
    print(str(e)[:500])


# ============================================================================
# Test compute_qgpv_direct_inv
# ============================================================================
print('\n' + '=' * 70)
print('Testing compute_qgpv_direct_inv')
print('=' * 70)

# Additional parameters for direct_inv version
# Use separate reference states for southern and northern hemispheres
ts0 = np.linspace(300, 220, kmax).astype(np.float64)  # Southern hemisphere ref temp
tn0 = np.linspace(298, 218, kmax).astype(np.float64)  # Northern hemisphere ref temp
stats = np.ones(kmax, dtype=np.float64) * 0.01  # Southern hemisphere static stability
statn = np.ones(kmax, dtype=np.float64) * 0.012  # Northern hemisphere static stability
jd = nlat // 2  # Equatorial boundary index (1-indexed in Fortran convention)

print(f'  jd (equatorial boundary): {jd}')

print('\nRunning Numba implementation (direct_inv)...')
pv_numba_di, avort_numba_di = compute_qgpv_direct_inv_numba(
    jd, ut, vt, theta, height, ts0, tn0, stats, statn,
    aa=aa, omega=omega, dz=dz,
    hh=hh, rr=rr, cp=cp
)

print('Running Fortran implementation (direct_inv)...')
pv_fortran_di, avort_fortran_di = compute_qgpv_direct_inv_fortran(
    jd, ut, vt, theta, height, ts0, tn0, stats, statn,
    aa=aa, omega=omega, dz=dz,
    hh=hh, rr=rr, cp=cp
)

print(f'Numba PV shape: {pv_numba_di.shape}')
print(f'Fortran PV shape: {pv_fortran_di.shape}')

# Compare
print('\nComparing results...')
print(f'Max abs diff (avort): {np.max(np.abs(avort_numba_di - avort_fortran_di)):.6e}')
print(f'Max abs diff (pv): {np.max(np.abs(pv_numba_di - pv_fortran_di)):.6e}')

# Relative differences
avort_mask_di = np.abs(avort_fortran_di) > 1e-15
if np.any(avort_mask_di):
    max_rel_avort_di = np.max(np.abs((avort_numba_di[avort_mask_di] - avort_fortran_di[avort_mask_di]) / avort_fortran_di[avort_mask_di]))
    print(f'Max rel diff (avort): {max_rel_avort_di:.6e}')

pv_interior_di = pv_fortran_di[..., 1:-1]
pv_mask_di = np.abs(pv_interior_di) > 1e-15
if np.any(pv_mask_di):
    pv_numba_interior_di = pv_numba_di[..., 1:-1]
    max_rel_pv_di = np.max(np.abs((pv_numba_interior_di[pv_mask_di] - pv_interior_di[pv_mask_di]) / pv_interior_di[pv_mask_di]))
    print(f'Max rel diff (pv interior): {max_rel_pv_di:.6e}')

# Check if close
try:
    np.testing.assert_allclose(avort_numba_di, avort_fortran_di, rtol=1e-5, atol=1e-9)
    print('\n✓ Absolute vorticity (direct_inv): PASS (within tolerance)')
except AssertionError as e:
    print(f'\n✗ Absolute vorticity (direct_inv): FAIL')
    print(str(e)[:500])

try:
    np.testing.assert_allclose(pv_numba_di, pv_fortran_di, rtol=1e-5, atol=1e-9)
    print('✓ Potential vorticity (direct_inv): PASS (within tolerance)')
except AssertionError as e:
    print(f'✗ Potential vorticity (direct_inv): FAIL')
    print(str(e)[:500])

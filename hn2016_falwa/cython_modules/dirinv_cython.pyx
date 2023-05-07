"""
-------------------------------------------------------------------------------------------------------------------
File name: dirinv_cython.pyx
Author: Clare Huang
Created on: 2023/4/29
Description:
    Execute in cython_modules/ : python setup_cython.py build_ext --inplace
    Migrate f2py modules to Cython.
    This file contains all modules for the direct inversion algorithm in NHN22. Not yet validated/optimized.

Calling sin:
https://cython.readthedocs.io/en/stable/src/tutorial/pure.html?highlight=sin#calling-c-functions

Pycharm cython support:
https://www.jetbrains.com/help/pycharm/cython.html#cython-support

Optimization:
https://people.duke.edu/~ccc14/sta-663-2016/18D_Cython.html

The version of cython in MDTF environment is 0.29. This is the latest stable version:
https://cython.readthedocs.io/en/stable/src/quickstart/overview.html
Working with Numpy:
https://cython.readthedocs.io/en/stable/src/tutorial/numpy.html?highlight=numpy

Working with cython array
https://github.com/cython/cython/issues/2678
Not sure if useful: https://stackoverflow.com/questions/68004787/cython-effectively-using-numpy-in-pure-python-mode
Cython's pure python syntax: https://www.infoworld.com/article/3670116/use-cython-to-accelerate-array-iteration-in-numpy.html
Cython for numpy users: https://cython.readthedocs.io/en/stable/src/userguide/numpy_tutorial.html
Cython compiler directives: https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html?highlight=wraparound#compiler-directives
Cython: Blend the Best of Python and C++ | SciPy 2015 Tutorial | Kurt Smith: https://youtu.be/gMvkiQ-gOW8
-------------------------------------------------------------------------------------------------------------------
"""

import cython
from libc.math cimport sin, cos, pi, exp
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from hn2016_falwa.constant import P_GROUND, SCALE_HEIGHT, CP, DRY_GAS_CONSTANT, EARTH_RADIUS, EARTH_OMEGA
import numpy as np
cimport numpy as np


cpdef int x_sq_minus_x(int x):
    """Test function from cython documentation page"""
    return x**2 - x


def sin_func(x):
    """Test function from cython documentation page"""
    return sin(x * x)


def compute(array_1: cython.int[:,:]):
    view2d: int[:,:] = array_1
    for i in range(3):
        for j in range(4):
            print(view2d[i, j].astype('int'))


@cython.boundscheck(False)
@cython.wraparound(False)
def top_boundary_condition(int kmax, int jd, int jb, int nd, float dp, cython.float[:] clat, cython.float[:] slat, float dz, float exp_upper, cython.float[:, :] tbar, float h, float a, float rr, float omega, cython.float[:,:,:] sjk, cython.float[:,:] tjk):
    """
    exp_upper = exp(-z(kmax - 1) * rkappa / h)
    This is the last chunk in f90_modules/compute_qref_and_fawa_first.f90
    float a = EARTH_RADIUS,
    float rr = DRY_GAS_CONSTANT,
    float omega = EARTH_OMEGA

    Operations taken away - as input.
    tjk = np.zeros((kmax-1, jd-2))
    sjk = np.zeros((kmax-1, jd-2, jd-2))
    tjk[:, :] = 0.
    sjk[:, :, :] = 0.
    """
    sjk_view3d: float[:, :, :] = sjk
    tjk_view2d: float[:, :] = tjk

    for jj in range(jb+1, nd-1):
        j = jj - jb
        # phi0 = float(jj - 1) * dp
        cos0 = clat[jj]
        sin0 = slat[jj]
        tjk[kmax - 1, j - 1] = -dz * rr * cos0 * exp_upper
        tjk[kmax - 1, j - 1] = tjk[kmax - 1, j - 1] * (tbar[kmax, j + 1] - tbar[kmax, j - 1])
        tjk[kmax - 1, j - 1] = tjk[kmax - 1, j - 1] / (4. * omega * sin0 * dp * h * a)
        sjk[kmax - 1, j - 1, j - 1] = 1.
    return sjk, tjk

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_b4_inversion_cython(
    cython.double[:,:,:] sjk,
    cython.double[:,:] tjk,
    double rkappa,
    double dp,
    cython.double[:] z,
    int k,
    int jd,
    cython.double[:] statn,
    int jb,
    int nd,
    double om,
    double scale_height,
    double planet_radius,
    double dz,
    double dry_gas_constant,
    cython.double[:,:] qref,
    cython.double[:,:] u,
    cython.double[:,:] ckref):
    """
    Args:
        sjk:
        tjk:
        rkappa(float): rr/cp
        dp: pi/float(jmax-1) d(lat) in radian
        z: height array (with 1km spacing)
        k: height level index
        jd: self.nlat//2 + self.nlat % 2 - self.eq_boundary_index
        statn:
        jb: self.eq_boundary_index (5?)
        nd: self.nlat//2 + self.nlat % 2
        om: earth rotation rate
        scale_height: 
        planet_radius:
        dz:
        dry_gas_constant:
        qref:
        u:
        ckref:

    Returns:
        qref[kmax,nd], u[kmax,jd], ckref[kmax,nd]

    Taken away from operation:
        rkappa = rr/cp
        pi = acos(-1.)
        dp = pi/double(jmax-1)

        qjj = np.zeros((jd-2, jd-2))
        djj = np.zeros((jd-2, jd-2))
        cjj = np.zeros((jd-2, jd-2))
        xjj = np.zeros((jd-2, jd-2))
        sjj = np.zeros((jd-2, jd-2))
        rj = np.zeros(jd - 2)
        tj = np.zeros(jd - 2)
    """

    sjk_view3d: double[:, :, :] = sjk
    tjk_view2d: double[:, :] = tjk
    qref_view2d: double[:,:] = qref
    u_view2d: double[:, :] = u
    ckref_view2d: double[:, :] = ckref
    cdef double[:] rj = np.empty(jd-2, dtype=float)
    cdef double[:, :] cjj = np.empty((jd-2, jd-2), dtype=float)
    cdef double[:, :] djj = np.empty((jd-2, jd-2), dtype=float)
    cdef double[:, :] qjj = np.empty((jd-2, jd-2), dtype=float)
    cdef double[:, :] xjj = np.empty((jd-2, jd-2), dtype=float)
    cdef double ajk, jbk, cjk, djk, ejk, fjk, amp, amm, fact
    cdef int i, j, jj, kk

    zp = 0.5 * (z[k + 1] + z[k])
    zm = 0.5 * (z[k - 1] + z[k])
    statp = 0.5*(statn[k+1]+statn[k])
    statm = 0.5*(statn[k-1]+statn[k])

    sjj = sjk_view3d[k, :, :]
    tj = tjk_view2d[k, :]

    for jj in range(jb + 1, nd - 1):
        j = jj - jb
        phi0 = float(jj) * dp  # jj - 1 -> jj
        phip = (float(jj) + 0.5) * dp   # jj -> jj + 1
        phim = (float(jj) - 0.5) * dp   # jj -> jj + 1
        cos0 = cos(phi0)
        cosp = cos(phip)
        cosm = cos(phim)
        sin0 = sin(phi0)
        sinp = sin(phip)
        sinm = sin(phim)

        fact = 4. * om * om * scale_height * planet_radius * planet_radius * sin0 * dp * dp / (dz * dz * dry_gas_constant * cos0)
        amp = exp(-zp / scale_height) * exp(rkappa * zp / scale_height) / statp
        amp = amp*fact*exp(z[k] / scale_height)
        amm = exp(-zm / scale_height) * exp(rkappa * zm / scale_height) / statm
        amm = amm*fact*exp(z[k] / scale_height)

        # ***** Specify A, B, C, D, E, F (Eqs. 4-9) *****
        ajk = 1./(sinp*cosp)
        bjk = 1./(sinm*cosm)
        cjk = amp
        djk = amm
        ejk = ajk + bjk + cjk + djk
        fjk = -0.5 * planet_radius * dp * (qref_view2d[k, jj + 1] - qref_view2d[k, jj - 1])

        # ***** Specify rk (Eq. 15) ****
        # **** North-south boundary conditions ****
        u_view2d[k, jd-1] = 0. # jd -> jd-1
        phi0 = dp * float(jb+1)  # jb -> jb+1
        u_view2d[k, 0] = ckref_view2d[k, jb] / (2. * pi * planet_radius) \
                         - om * planet_radius * cos(phi0)
        rj[j-2] = fjk
        if j == 1:
            rj[j - 1] = fjk - bjk * u_view2d[k, 0]
        if j == jd-2:
            rj[j - 1] = fjk - ajk * u_view2d[k, jd - 1]
        # ***** Specify Ck & Dk (Eqs. 18-19) *****
        cjj[j - 2, j - 2] = cjk
        djj[j - 2, j - 2] = djk

        # **** Specify Qk (Eq. 17) *****
        qjj[j - 2, j - 2] = -ejk

        if j - 1 >= 0 & j - 1 < (jd - 2) - 1:  # f2py: if(j-1.ge.1.and.j-1.lt.jd-2)
            qjj[j - 1, j] = ajk

        if j - 1 > 0 & j - 1 <= (jd - 2) - 1:  # f2py: if(j-1.gt.1.and.j-1.le.jd-2)
            qjj[j - 1, j - 2] = bjk

    # **** Compute Qk + Ck Sk *******
    for i in range(0, jd-2):
        for j in range(0, jd-2):
            xjj[i, j] = 0
            for kk in range(0, jd-2):
                xjj[i, j] = xjj[i, j] + cjj[i, kk] * sjj[kk, j]
            qjj[i, j] = qjj[i, j] + xjj[i, j]

    return np.asarray(qjj), np.asarray(djj), np.asarray(cjj), np.asarray(rj), np.asarray(tj)


def matrix_after_inversion(
    k: int, kmax: int, jd: int, qjj: np.ndarray,
    djj: np.ndarray, cjj: np.ndarray, rj: np.array, sjk: np.ndarray,
    tjk: np.ndarray, tj: np.array):
    xjj = np.zeros((jd-2, jd-2))
    yj = np.zeros(jd-2)
    """
    sjk [k,j,i] >.<
    """

    for i in range(0, jd-2):
        for j in range(0, jd-2):
            xjj[i, j] = 0.
            for kk in range(0, jd-2):
                xjj[i, j] = xjj[i, j] + qjj[i, kk] * djj[kk, j]
            sjk[k-1, j, i] = -xjj[i, j]

    #  **** Evaluate rk - Ck Tk ****
    for i in range(0, jd-2):
        yj[i] = 0.
        for kk in range(0, jd-2):
            yj[i] = yj[i] + cjj[i, kk] * tj[kk]
        yj[i] = rj[i] - yj[i]

    # ***** Evaluate Eq. 23 *******
    for i in range(0, jd-2):
        tj[i] = 0.
        for kk in range(0, jd-2):
            tj[i] = tj[i] + qjj[i, kk] * yj[kk]
        tjk[i, k-1] = tj[i]

    return sjk, tjk, tj


def upward_sweep(
    jmax: int, kmax: int, nd: int, jb: int, jd: int,
    sjk: np.ndarray, tjk: np.ndarray, ckref: np.ndarray, tb: np.array, qref_over_cor: np.ndarray,
    a: float, om: float, dz: float, h: float, rr: float, cp: float, dp: float, rkappa: float):

    pjk = np.zeros((jd-2, kmax))
    tj = np.zeros(jd - 2)
    yj = np.zeros(jd - 2)
    sjj = np.zeros((jd - 2, jd-2))
    pj = np.zeros(jd - 2)
    qref = np.zeros((nd, kmax))
    tref = np.zeros((jd, kmax))
    u = np.zeros((jd, kmax))
    tg = np.zeros(kmax)

    for k in range(0, kmax-1):
        pj[:] = pjk[:, k]
        sjj[:, :] = sjk[:, :, k]
        tj[:] = tjk[:, k]

        for i in range(0, jd-2):
            yj[i] = 0.
            for kk in range(0, jd-2):
                yj[i] = yj[i] + sjj[i, kk] * pj[kk]
            pjk[i, k + 1] = yj[i] + tj[i]

    # **** Recover u *****
    for k in range(0, kmax):
        for j in range(1, jd-1):
            u[j, k] = pjk[j - 1, k]

    # *** Corner boundary conditions ***
    u[0, 0] = 0.
    u[jd - 1, 1] = 0.
    u[0, kmax - 1] = ckref[jb, kmax - 1] / (2. * pi * a) - om * a * cos(dp * float(jb))
    u[jd - 1, kmax - 1] = 0.

    # *** Divide by cos phi to revover Uref ****
    for jj in range(jb, nd-1):
        j = jj - jb
        phi0 = dp * float(jj)
        u[j, :] = u[j, :] / cos(phi0)

    u[jd - 1, :] = 2. * u[jd - 2, :] - u[jd - 3, :]

    # ******* compute tref *******
    qref = qref_over_cor
    for k in range(1, kmax-1):
        t00 = 0.
        zz = dz * float(k - 1)
        tref[0, k] = t00
        tref[1, k] = t00
        for j in range(i, jd-1):
            phi0 = dp * float(j + jb)
            cor = 2. * om * sin(phi0)
            uz = (u[j, k + 1] - u[j, k - 1]) / (2. * dz)
            ty = -cor * uz * a * h * exp(rkappa * zz / h)
            ty = ty / rr
            tref[j + 1, k] = tref[j - 1, k] + 2. * ty * dp
        for j in range(0, nd):
            phi0 = dp * float(j)
            qref[j, k] = qref_over_cor[j, k] * sin(phi0)

        tg[k] = 0.
        wt = 0.
        for jj in range(jb, nd):
            j = jj - jb
            phi0 = dp * float(jj)
            tg[k] = tg[k] + cos(phi0) * tref[j, k]
            wt = wt + cos(phi0)
        tg[k] = tg[k] / wt
        tres = tb[k] - tg[k]
        tref[:, k] = tref[:, k] + tres
    tref[:, 0] = tref[:, 1]-tb[1] + tb[0]
    tref[:, kmax - 1] = tref[:, kmax - 2] - tb[kmax - 2] + tb[kmax - 1]

    return qref, tref, u


def compute_flux_dirinv(
    pv, uu, vv, pt, tn0, qref, uref, tref,
    imax: int, jmax: int, kmax: int,
    nd: int, jb: int, jd: int,
    a: float, om: float, dz: float, dp: float, h: float, rr: float, rkappa: float,
    cp: float, prefac: float, z: np.array):

    tg = np.zeros(kmax)
    ua1 = np.zeros((imax, nd))
    ua2 = np.zeros((imax, nd))
    ep1 = np.zeros((imax, nd))
    ep2 = np.zeros((imax, nd))
    ep3 = np.zeros((imax, nd))
    ep4 = np.zeros((imax, nd))
    qe = np.zeros((imax, nd))
    ue = np.zeros((imax, nd))

    astarbaro = np.zeros((imax, nd))
    ubaro = np.zeros((imax, nd))
    urefbaro = np.zeros(nd)
    ua1baro = np.zeros((imax, nd))
    ua2baro = np.zeros((imax, nd))
    ep1baro = np.zeros((imax, nd))
    ep2baro = np.zeros((imax, nd))
    ep3baro = np.zeros((imax, nd))
    ep4baro = np.zeros((imax, nd))
    astar1 = np.zeros((imax, nd, kmax))
    astar2 = np.zeros((imax, nd, kmax))

    tg = tn0
    dc = dz / prefac

    for k in range(1, kmax-1):
        zk = dz * k
        for i in range(0, imax):
            for j in range(jb, nd-1):
                astar1[i, j, k] = 0.       # LWA * cos(phi)
                astar2[i, j, k] = 0.       # LWA * cos(phi)
                ua2[i, j] = 0.             # F2
                phi0 = dp * float(j - 1)   # latitude
                cor = 2. * om * sin(phi0)  # Coriolis parameter
                for jj in range(0, nd):
                    phi1 = dp * float(jj)
                    qe[i, jj] = pv[i, jj + nd - 1, k] - qref[j, k]
                    ue[i, jj] = uu[i, jj + nd - 1, k] - uref[j - jb, k]  # ue; shift uref 5N
                    aa = a * dp * cos(phi1)
                    if qe[i, jj] <= 0.0 & jj >= j:
                        astar2[i, j, k] = astar2[i, j, k] - qe[i, jj] * aa  # anticyclonic
                        ua2[i, j] = ua2[i, j] - qe[i, jj] * ue[i, jj] * aa
                    if qe[i, jj] >= 0.0 & jj < j:
                        astar1[i, j, k] = astar1[i, j, k] + qe[i, jj] * aa  # cyclonic
                        ua2[i, j] = ua2[i, j] + qe[i, jj] * ue[i, jj] * aa

                #  ******** Other fluxes ********
                ua1[i, j] = uref[j - jb, k] * (astar1[i, j, k] + astar2[i, j, k])            #F1
                ep1[i, j] = -0.5 * (uu[i, j + nd - 1, k] - uref[j - jb, k]) ** 2  # F3a
                ep1[i, j] = ep1[i, j] + 0.5 * vv[i, j + nd - 1, k] ** 2    # F3a + b
                ep11 = 0.5 * (pt[i, j + nd - 1, k] - tref[j - jb, k]) ** 2         # F3c
                zz = dz * float(k - 1)
                ep11 = ep11 * (rr / h) * exp(-rkappa * zz / h)
                ep11 = ep11 * 2. * dz / (tg[k + 1] - tg[k - 1])
                ep1[i, j] = ep1[i, j] - ep11                   # F3
                phip = dp*float(j+1)
                phi0 = dp*float(j)
                phim = dp*float(j-1)
                cosp = cos(phip)          # cosine for one grid north
                cos0 = cos(phi0)          # cosine for latitude grid
                cosm = cos(phim)          # cosine for one grid south
                sin0 = sin(phi0)          # sine for latitude grid
                ep1[i, j] = ep1[i, j] * cos0  # correct for cosine factor
                # meridional eddy momentum flux one grid north and south
                ep2[i, j] = (uu[i, j + nd, k] - uref[j - jb + 1, k]) * cosp * cosp * vv[i, j + nd, k]
                ep3[i, j] = (uu[i, j + nd - 2, k] - uref[j - jb - 1, k]) * cosm * cosm * vv[i, j + nd - 2, k]

                # low-level meridional eddy heat flux
                if k == 1:     # (26) of SI-HN17
                    ep41 = 2. * om * sin0 * cos0 * dz / prefac # prefactor
                    ep42 = exp(-dz / h) * vv[i, j + nd - 1, 1] * (pt[i, j + nd - 1, 1] - tref[j - jb, 1])
                    ep42 = ep42 / (tg[2] - tg[0])
                    ep43 = vv[i, j + nd - 1, 0] * (pt[i, j + nd - 1, 0] - tref[j - jb, 0])
                    ep43 = 0.5 * ep43 / (tg[1] - tg[0])
                    ep4[i, j] = ep41 * (ep42 + ep43)   # low - level heat flux
            phip = dp * jb
            phi0 = dp * (jb - 1)
            cosp = cos(phip)          # cosine for one grid north
            cos0 = cos(phi0)          # cosine for latitude grid
            ep2[i, jb] = (uu[i, nd + jb + 1, k] - uref[2, k]) * cosp * cosp * vv[i, nd + jb + 1, k]
            ep3[i, jb] = (uu[i, nd + jb, k] - uref[1, k]) * cos0 * cos0 * vv[i, nd + jb, k]

        # ******** Column average: (25) of SI-HN17 ********
        astarbaro = astarbaro + (astar1[:, :, k] + astar2[:, :, k]) * exp(-zk / h) * dc
        ua1baro = ua1baro + ua1 * exp(-zk / h) * dc
        ua2baro = ua2baro + ua2 * exp(-zk / h) * dc
        ep1baro = ep1baro + ep1 * exp(-zk / h) * dc
        ep2baro = ep2baro + ep2 * exp(-zk / h) * dc
        ep3baro = ep3baro + ep3 * exp(-zk / h) * dc

        for j in range(jb, nd):
            ubaro[:, j] = ubaro[:, j]+uu[:, j + nd - 1, k]*exp(-zk / h) * dc
            urefbaro[j] = urefbaro[j] + uref[j - jb, k] * exp(-zk / h) * dc

    return astarbaro, ubaro, urefbaro, ua1baro, ua2baro, \
           ep1baro, ep2baro, ep3baro, ep4, astar1, astar2


if __name__ == "__main__":
    ans = sin_func(0.444)
    print(ans)







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
import copy
import cython
from libc.math cimport sin, cos, pi, exp
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
    cdef int jj

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
    cython.double[:,:] qref_over_cor,
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
        qref_over_cor:
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
    qref_view2d: double[:,:] = qref_over_cor
    u_view2d: double[:, :] = u
    ckref_view2d: double[:, :] = ckref
    cdef double[:] rj = np.zeros(jd-2, dtype=float)
    cdef double[:, :] cjj = np.zeros((jd-2, jd-2), dtype=float)
    cdef double[:, :] djj = np.zeros((jd-2, jd-2), dtype=float)
    cdef double[:, :] qjj = np.zeros((jd-2, jd-2), dtype=float)
    cdef double[:, :] xjj = np.zeros((jd-2, jd-2), dtype=float)
    cdef double ajk, jbk, cjk, djk, ejk, fjk, amp, amm, fact
    cdef int i, j, jj, kk

    zp = 0.5 * (z[k] + z[k - 1])
    zm = 0.5 * (z[k - 2] + z[k - 1])
    statp = 0.5 * (statn[k] + statn[k - 1])
    statm = 0.5 * (statn[k - 2] + statn[k - 1])

    sjj = sjk_view3d[:, :, k - 1]

    for jj in range(jb + 2, nd):
        j = jj - jb
        phi0 = float(jj - 1) * dp
        phip = (float(jj) - 0.5) * dp
        phim = (float(jj) - 1.5) * dp
        cos0 = cos(phi0)
        cosp = cos(phip)
        cosm = cos(phim)
        sin0 = sin(phi0)
        sinp = sin(phip)
        sinm = sin(phim)

        fact = 4. * om * om * scale_height * planet_radius * planet_radius * sin0 * dp * dp / (dz * dz * dry_gas_constant * cos0)
        amp = exp(-zp / scale_height) * exp(rkappa * zp / scale_height) / statp
        amp = amp * fact * exp(z[k - 1] / scale_height)
        amm = exp(-zm / scale_height) * exp(rkappa * zm / scale_height) / statm
        amm = amm * fact * exp(z[k - 1] / scale_height)

        # ***** Specify A, B, C, D, E, F (Eqs. 4-9) *****
        ajk = 1./(sinp*cosp)
        bjk = 1./(sinm*cosm)
        cjk = amp
        djk = amm
        ejk = ajk + bjk + cjk + djk
        fjk = -0.5 * planet_radius * dp * (qref_view2d[jj, k - 1] - qref_view2d[jj - 2, k - 1])

        # ***** Specify rk (Eq. 15) ****
        # **** North-south boundary conditions ****
        u_view2d[k - 1, jd - 1] = 0.
        phi0 = dp * float(jb)
        u_view2d[k - 1, 0] = ckref_view2d[k - 1, jb] / (2. * pi * planet_radius) \
                             - om * planet_radius * cos(phi0)
        rj[j-2] = fjk

        if j == 2:
            rj[j - 2] = fjk - bjk * u_view2d[k - 1, 0]
        if j == jd-1:
            rj[j - 2] = fjk - ajk * u_view2d[k - 1, jd - 1]

        # ***** Specify Ck & Dk (Eqs. 18-19) *****
        cjj[j - 2, j - 2] = cjk
        djj[j - 2, j - 2] = djk

        # **** Specify Qk (Eq. 17) *****
        qjj[j - 2, j - 2] = -ejk

        if (j - 1 >= 1) & (j - 1 < jd - 2):  # f2py: if(j-1.ge.1.and.j-1.lt.jd-2)
            qjj[j - 2, j - 1] = ajk

        if (j - 1 > 1) & (j - 1 <= jd - 2):  # f2py: if(j-1.gt.1.and.j-1.le.jd-2)
            qjj[j - 2, j - 3] = bjk

    # **** Compute Qk + Ck Sk *******
    for i in range(1, jd-1):
        for j in range(1, jd-1):
            xjj[i - 1, j - 1] = 0
            for kk in range(1, jd-1):
                xjj[i - 1, j - 1] = xjj[i - 1, j - 1] + cjj[i - 1, kk - 1] * sjj[kk - 1, j - 1]
            qjj[i - 1, j - 1] = qjj[i - 1, j - 1] + xjj[i - 1, j - 1]

    return np.asarray(qjj), np.asarray(djj), np.asarray(cjj), np.asarray(rj)

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_after_inversion_cython(
    int k,
    int kmax,
    int jd,
    cython.double[:, :] qjj,
    cython.double[:, :] djj,
    cython.double[:, :] cjj,
    cython.double[:] rj,
    cython.double[:, :, :] sjk,  # (jd-2, jd-2, kmax-1)
    cython.double[:, :] tjk):

    # *** Create memory view ***
    qjj_view2d: double[:, :] = qjj
    djj_view2d: double[:, :] = djj
    cjj_view2d: double[:, :] = cjj
    sjk_view3d: double[:, :, :] = sjk
    tjk_view2d: double[:, :] = tjk  # tjk does not match

    cdef double[:, :] xjj = np.zeros((jd-2, jd-2), dtype=float)
    cdef double[:] yj = np.zeros(jd-2, dtype=float)
    cdef double[:] tj = np.zeros(jd-2, dtype=float)

    tj[:] = tjk_view2d[:, k-1]
    for i in range(1, jd-1):
        for j in range(1, jd-1):
            xjj[i-1, j-1] = 0.
            for kk in range(1, jd-1):
                xjj[i-1, j-1] = xjj[i-1, j-1] + qjj_view2d[i-1, kk-1] * djj_view2d[kk-1, j-1]
            sjk_view3d[i-1,j-1,k-2] = -xjj[i-1, j-1]

    #  **** Evaluate rk - Ck Tk ****
    for i in range(1, jd-1):
        yj[i-1] = 0.
        for kk in range(1, jd-1):
            yj[i-1] = yj[i-1] + cjj_view2d[i-1, kk-1] * tj[kk-1]
        yj[i-1] = rj[i-1] - yj[i-1]

    # ***** Evaluate Eq. 23 *******
    for i in range(1, jd-1):
        tj[i-1] = 0.
        for kk in range(1, jd-1):
            tj[i-1] = tj[i-1] + qjj_view2d[i-1, kk-1] * yj[kk-1]
        tjk_view2d[i-1, k-2] = tj[i-1]

    return np.asarray(sjk), np.asarray(tjk)


@cython.boundscheck(False)
@cython.wraparound(False)
def upward_sweep(
    int jmax,
    int kmax,
    int nd,
    int jb,
    int jd,
    cython.double[:, :, :] sjk,  # sjk(jd-2,jd-2,kmax-1)
    cython.double[:, :] tjk,     # tjk(jd-2,kmax-1)
    cython.double[:,:] ckref,    # ckref(nd,kmax)
    cython.double[:] tb,         # tb(kmax)
    cython.double[:, :] qref_over_cor, # qref_over_cor(nd,kmax)
    double a,
    double om,
    double dz,
    double h,
    double rr,
    double cp,
    double dp,
    double rkappa):
    """TODO: debugging not yet finished. tref does not match original code"""

    # *** Create memory view ***
    sjk_view3d: double[:, :, :] = sjk
    tjk_view2d: double[:, :] = tjk
    ckref_view2d: double[:, :] = ckref
    qref_over_cor_view2d: double[:, :] = qref_over_cor

    # *** Output ***
    cdef double[:, :] qref = np.zeros((nd, kmax), dtype=float)
    cdef double[:, :] tref = np.zeros((jd, kmax), dtype=float)
    cdef double[:, :] u = np.zeros((jd, kmax), dtype=float)

    # *** Dummy arrays ***
    cdef double[:] tg = np.zeros(kmax, dtype=float)
    cdef double[:, :] pjk = np.zeros((jd-2, kmax), dtype=float)
    cdef double[:] tj = np.zeros(jd-2, dtype=float)
    cdef double[:] yj = np.zeros(jd-2, dtype=float)
    cdef double[:, :] sjj = np.zeros((jd-2, jd-2), dtype=float)
    cdef double[:] pj = np.zeros(jd-2, dtype=float)

    # *** indices and dummy variables ***
    cdef int i, j, jj, k
    cdef double uz, ty, wt, tres, phi0

    pjk[:, 0] = 0.0
    for k in range(1, kmax):
        pj[:] = pjk[:, k-1]
        tj[:] = tjk_view2d[:, k-1]

        for i in range(1, jd-1):
            yj[i-1] = 0.
            for kk in range(1, jd-1):
                yj[i-1] = yj[i-1] + sjk_view3d[i-1, kk-1, k-1] * pj[kk-1]
            pjk[i-1, k] = yj[i-1] + tj[i-1]

    # **** Recover u *****
    for k in range(1, kmax+1):
        for j in range(2, jd):
            u[j-1, k-1] = pjk[j-2, k-1]

    # *** Corner boundary conditions ***
    u[0, 0] = 0.
    u[jd - 1, 0] = 0.
    u[0, kmax - 1] = ckref_view2d[jb, kmax - 1] / (2. * pi * a) - om * a * cos(dp * float(jb))
    u[jd - 1, kmax - 1] = 0.

    # *** Divide by cos phi to recover Uref ****
    for jj in range(jb+1, nd):
        j = jj - jb
        phi0 = dp * float(jj-1)
        for k in range(1, kmax+1):
            u[j-1, k-1] = u[j-1, k-1] / cos(phi0)

    for k in range(1, kmax+1):
        u[jd-1, k-1] = 2. * u[jd-2, k-1] - u[jd-3, k-1]

    # ******* compute tref *******
    for k in range(2, kmax):
        t00 = 0.
        zz = dz * float(k - 1)
        tref[0, k-1] = t00
        tref[1, k-1] = t00
        for j in range(2, jd):
            phi0 = dp*float(j-1+jb)
            cor = 2. * om * sin(phi0)
            uz = (u[j-1,k] - u[j-1,k-2]) / (2. * dz)
            ty = -cor * uz * a * h * exp(rkappa * zz / h)
            ty = ty / rr
            tref[j, k-1] = tref[j-2, k-1] + 2. * ty * dp
        for j in range(1, nd+1):
            phi0 = dp * float(j-1)
            qref[j-1, k-1] = qref_over_cor_view2d[j-1, k-1] * sin(phi0)

        tg[k-1] = 0.
        wt = 0.
        for jj in range(jb+1, nd+1):
            j = jj - jb
            phi0 = dp * float(jj-1)
            tg[k-1] = tg[k-1] + cos(phi0) * tref[j-1, k-1]
            wt = wt + cos(phi0)
        tg[k-1] = tg[k-1] / wt
        tres = tb[k-1] - tg[k-1]
        for jj in range(1, jd+1):
            tref[jj-1, k-1] = tref[jj-1, k-1] + tres

    for j in range(1, jd+1):
        tref[j-1, 0] = tref[j-1, 1]-tb[1] + tb[0]
        tref[j-1, kmax-1] = tref[j-1, kmax-2] - tb[kmax-2] + tb[kmax-1]

    return np.asarray(tref), np.asarray(qref), np.asarray(u)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_flux_dirinv(
    int imax,
    int jmax,
    int kmax,
    int nd,
    int jb,
    int jd,
    cython.double[:,:,:] pv,  # pv(imax,jmax,kmax)
    cython.double[:,:,:] uu,  # uu(imax,jmax,kmax)
    cython.double[:,:,:] vv,  # vv(imax,jmax,kmax)
    cython.double[:,:,:] pt,  # pt(imax,jmax,kmax)
    cython.double[:,:] qref,  # qref(nd,kmax)
    cython.double[:,:] uref,  # uref(jd,kmax)
    cython.double[:,:] tref,  # tref(jd,kmax)
    cython.double[:] tn0,     # tn0(kmax)
    double a,
    double om,
    double dz,
    double dp,
    double h,
    double rr,
    double rkappa,
    double cp,
    double prefac,
    cython.double[:] z):

    # *** Create memory view ***
    pv_view3d: double[:, :, :] = pv
    uu_view3d: double[:, :, :] = uu
    vv_view3d: double[:, :, :] = vv
    pt_view3d: double[:, :, :] = pt
    qref_view2d: double[:, :] = qref
    uref_view2d: double[:, :] = uref
    tref_view2d: double[:, :] = tref

    # *** Output ***
    cdef double[:, :] astarbaro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ubaro = np.zeros((imax,nd), dtype=float)
    cdef double[:] urefbaro = np.zeros(nd, dtype=float)
    cdef double[:, :] ua1baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ua2baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep1baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep2baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep3baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep4 = np.zeros((imax,nd), dtype=float)
    cdef double[:, :, :] astar1 = np.zeros((imax,nd,kmax), dtype=float)
    cdef double[:, :, :] astar2 = np.zeros((imax,nd,kmax), dtype=float)

    # *** Dummy arrays ***
    cdef double[:] tg = np.zeros(kmax, dtype=float)
    cdef double[:, :] ua1 = np.zeros((imax, nd), dtype=float)
    cdef double[:, :] ua2 = np.zeros((imax, nd), dtype=float)
    cdef double[:, :] ep1 = np.zeros((imax, nd), dtype=float)
    cdef double[:, :] ep2 = np.zeros((imax, nd), dtype=float)
    cdef double[:, :] ep3 = np.zeros((imax, nd), dtype=float)
    cdef double[:, :] qe = np.zeros((imax, nd), dtype=float)
    cdef double[:, :] ue = np.zeros((imax, nd), dtype=float)

    # *** dummy variables ***
    cdef double dc, aa, zk, phip, phi0, phim, phi1, ep11, ep41, ep42, ep43, cosp, cos0

    tg = tn0
    dc = dz / prefac

    # The chunk of code below will be translated different compared to other files
    for k in range(1, kmax-1):     # do k = 2,kmax-1
        zk = dz * float(k)         # zk = dz*float(k-1)
        for i in range(0, imax):   # do i = 1,imax
            for j in range(jb, nd-1):  # do j = jb+1,nd-1
                astar1[i, j, k] = 0.   # LWA * cos(phi)
                astar2[i, j, k] = 0.   # LWA * cos(phi)
                ua2[i, j] = 0.         # F2
                phi0 = dp * float(j)
                cor = 2. * om * sin(phi0)
                for jj in range(0, nd): # do jj = 1,nd
                    phi1 = dp*float(jj)
                    qe[i, jj] = pv_view3d[i, jj + nd - 1, k] - qref_view2d[j, k]   #qe; Q = qref
                    ue[i, jj] = uu_view3d[i, jj + nd - 1, k] - uref_view2d[j - jb, k]  #ue; shift uref 5N
                    aa = a*dp*cos(phi1)
                    if (qe[i,jj] <= 0.0) & (jj >= j):
                        astar2[i, j, k] = astar2[i, j, k] - qe[i, jj] * aa  # anticyclonic
                        ua2[i, j] = ua2[i, j] - qe[i, jj] * ue[i, jj] * aa
                    if (qe[i,jj] > 0.0) & (jj < j):
                        astar1[i, j, k] = astar1[i, j, k] + qe[i, jj] * aa  # cyclonic
                        ua2[i, j] = ua2[i, j] + qe[i, jj] * ue[i, jj] * aa

                #  ******** Other fluxes ********

                ua1[i,j] = uref_view2d[j-jb,k]*(astar1[i,j,k] + astar2[i,j,k])     # F1
                ep1[i, j] = -0.5 * (uu_view3d[i, j+nd-1, k] - uref_view2d[j-jb, k]) ** 2  # F3a
                ep1[i, j] = ep1[i, j] + 0.5 * vv_view3d[i, j+nd-1, k] ** 2         # F3a + b
                ep11 = 0.5 * (pt_view3d[i, j+nd-1, k] - tref_view2d[j-jb, k]) ** 2        # F3c
                zz = dz * float(k)
                ep11 = ep11 * (rr / h) * exp(-rkappa * zz / h)
                ep11 = ep11 * 2. * dz / (tg[k + 1] - tg[k - 1])
                ep1[i, j] = ep1[i, j] - ep11                                # F3
                phip = dp*float(j+1)
                phi0 = dp*float(j)
                phim = dp*float(j-1)
                cosp = cos(phip)          # cosine for one grid north
                cos0 = cos(phi0)          # cosine for latitude grid
                cosm = cos(phim)          # cosine for one grid south
                sin0 = sin(phi0)          # sine for latitude grid
                ep1[i, j] = ep1[i, j] * cos0  # correct for cosine factor
                # meridional eddy momentum flux one grid north and south
                ep2[i, j] = (uu_view3d[i, j + nd, k] - uref_view2d[j - jb + 1, k]) * cosp * cosp * vv_view3d[i, j + nd, k]
                ep3[i, j] = (uu_view3d[i, j + nd - 2, k] - uref_view2d[j - jb - 1, k]) * cosm * cosm * vv_view3d[i, j + nd - 2, k]

                # low-level meridional eddy heat flux
                if k == 1:     # (26) of SI-HN17
                    ep41 = 2. * om * sin0 * cos0 * dz / prefac # prefactor
                    ep42 = exp(-dz / h) * vv_view3d[i, j + nd - 1, 1] * (pt_view3d[i, j + nd - 1, 1] - tref_view2d[j - jb, 1])
                    ep42 = ep42 / (tg[2] - tg[0])
                    ep43 = vv_view3d[i, j+nd-1, 0] * (pt_view3d[i, j+nd-1, 0] - tref_view2d[j - jb, 0])
                    ep43 = 0.5 * ep43 / (tg[1] - tg[0])
                    ep4[i, j] = ep41 * (ep42 + ep43)   # low - level heat flux

            phip = dp * jb
            phi0 = dp * (jb - 1)
            cosp = cos(phip)          # cosine for one grid north
            cos0 = cos(phi0)          # cosine for latitude grid
            ep2[i, jb-1] = (uu_view3d[i, nd+jb, k] - uref_view2d[1, k]) * cosp * cosp * vv_view3d[i, nd+jb, k]
            ep3[i, jb-1] = (uu_view3d[i, nd+jb-1, k] - uref_view2d[0, k]) * cos0 * cos0 * vv_view3d[i, nd+jb-1, k]

        # ******** Column average: (25) of SI-HN17 ********
        for i in range(0, imax):
            for j in range(0, nd):
                astarbaro[i, j] = astarbaro[i, j] + (astar1[i, j, k] + astar2[i, j, k]) * exp(-zk / h) * dc
                ua1baro[i, j] = ua1baro[i, j] + ua1[i, j] * exp(-zk / h) * dc
                ua2baro[i, j] = ua2baro[i, j] + ua2[i, j] * exp(-zk / h) * dc
                ep1baro[i, j] = ep1baro[i, j] + ep1[i, j] * exp(-zk / h) * dc
                ep2baro[i, j] = ep2baro[i, j] + ep2[i, j] * exp(-zk / h) * dc
                ep3baro[i, j] = ep3baro[i, j] + ep3[i, j] * exp(-zk / h) * dc

            for j in range(jb, nd):
                ubaro[i, j] = ubaro[i, j]+uu_view3d[i, j+nd-1, k] * exp(-zk / h) * dc

        for j in range(jb, nd):
            urefbaro[j] = urefbaro[j] + uref_view2d[j-jb, k] * exp(-zk / h) * dc

    return np.asarray(astarbaro), np.asarray(ubaro), np.asarray(urefbaro), np.asarray(ua1baro), np.asarray(ua2baro),\
        np.asarray(ep1baro), np.asarray(ep2baro), np.asarray(ep3baro), np.asarray(ep4), \
        np.asarray(astar1), np.asarray(astar2)


if __name__ == "__main__":
    ans = sin_func(0.444)
    print(ans)


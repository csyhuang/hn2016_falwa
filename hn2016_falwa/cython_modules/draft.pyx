import cython
from libc.math cimport sin, cos, pi, exp
import numpy as np
cimport numpy as np

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
    pv_view3d: double[:,:,:] = pv
    uu_view3d: double[:, :, :] = uu
    vv_view3d: double[:, :, :] = vv
    pt_view3d: double[:, :, :] = pt
    qref_view2d: double[:, :] = qref
    uref_view2d: double[:, :] = uref
    tref_view2d: double[:, :] = tref

    # *** Output ***
    cdef double[:, :] astarbaro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ubaro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] urefbaro = np.zeros(nd, dtype=float)
    cdef double[:, :] ua1baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ua2baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep1baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep2baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep3baro = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] ep4 = np.zeros((imax,nd), dtype=float)
    cdef double[:, :] astar1 = np.zeros((imax,nd,kmax), dtype=float)
    cdef double[:, :] astar2 = np.zeros((imax,nd,kmax), dtype=float)

    # *** Dummy arrays ***
    cdef double[:] tg = np.zeros(kmax, dtype=float)
    cdef double[:] ua1 = np.zeros((imax, nd), dtype=float)
    cdef double[:] ua2 = np.zeros((imax, nd), dtype=float)
    cdef double[:] ep1 = np.zeros((imax, nd), dtype=float)
    cdef double[:] ep2 = np.zeros((imax, nd), dtype=float)
    cdef double[:] ep3 = np.zeros((imax, nd), dtype=float)
    cdef double[:] qe = np.zeros((imax, nd), dtype=float)
    cdef double[:] ue = np.zeros((imax, nd), dtype=float)

    # *** dummy variables ***
    cdef double dc, aa, zk, phi0, phi1, ep11

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
                    qe[i, jj] = pv[i, jj + nd - 1, k] - qref[j, k]   #qe; Q = qref
                    ue[i, jj] = uu[i, jj + nd - 1, k] - uref[j - jb, k]  #ue; shift uref 5N
                    aa = a*dp*cos(phi1)
                    if (qe(i,jj) <= 0.0) & (jj >= j):
                        astar2[i, j, k] = astar2[i, j, k] - qe[i, jj] * aa  # anticyclonic
                        ua2[i, j] = ua2[i, j] - qe[i, jj] * ue[i, jj] * aa
                    if (qe(i,jj) > 0.0) & (jj < j):
                        astar1[i, j, k] = astar1[i, j, k] + qe[i, jj] * aa  # cyclonic
                        ua2[i, j] = ua2[i, j] + qe[i, jj] * ue[i, jj] * aa

                #  ******** Other fluxes ********

                ua1[i,j] = uref[j-jb,k]*(astar1[i,j,k] + astar2[i,j,k])     # F1
                ep1[i, j] = -0.5 * (uu[i, j+nd-1, k] - uref[j-jb, k]) ** 2  # F3a
                ep1[i, j] = ep1[i, j] + 0.5 * vv[i, j+nd-1, k] ** 2         # F3a + b
                ep11 = 0.5 * (pt[i, j+nd-1, k] - tref[j-jb, k]) ** 2        # F3c
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
                ep2[i, j] = (uu[i, j + nd, k] - uref[j - jb + 1, k]) * cosp * cosp * vv[i, j + nd, k]
                ep3[i, j] = (uu[i, j + nd - 2, k] - uref[j - jb - 1, k]) * cosm * cosm * vv[i, j + nd - 2, k]

                # low-level meridional eddy heat flux
                if k == 1:     # (26) of SI-HN17
                    ep41 = 2. * om * sin0 * cos0 * dz / prefac # prefactor
                    ep42 = exp(-dz / h) * vv[i, j + nd - 1, 1] * (pt[i, j + nd - 1, 1] - tref[j - jb, 1])
                    ep42 = ep42 / (tg[2] - tg[0])
                    ep43 = vv[i, j+nd-1, 0] * (pt[i, j+nd-1, 0] - tref[j - jb, 0])
                    ep43 = 0.5 * ep43 / (tg[1] - tg[0])
                    ep4[i, j] = ep41 * (ep42 + ep43)   # low - level heat flux
            # TODO: below not yet checked
            phip = dp * jb
            phi0 = dp * (jb - 1)
            cosp = cos(phip)          # cosine for one grid north
            cos0 = cos(phi0)          # cosine for latitude grid
            ep2[i, jb-1] = (uu[i, nd+jb, k] - uref[1, k]) * cosp * cosp * vv[i, nd+jb, k]
            ep3[i, jb-1] = (uu[i, nd+jb-1, k] - uref[0, k]) * cos0 * cos0 * vv[i, nd+jb-1, k]

        # ******** Column average: (25) of SI-HN17 ********
        astarbaro = astarbaro + (astar1[:, :, k] + astar2[:, :, k]) * exp(-zk / h) * dc
        ua1baro = ua1baro + ua1 * exp(-zk / h) * dc
        ua2baro = ua2baro + ua2 * exp(-zk / h) * dc
        ep1baro = ep1baro + ep1 * exp(-zk / h) * dc
        ep2baro = ep2baro + ep2 * exp(-zk / h) * dc
        ep3baro = ep3baro + ep3 * exp(-zk / h) * dc

        for j in range(jb, nd):
            ubaro[:, j] = ubaro[:, j]+uu[:, j+nd-1, k]*exp(-zk / h) * dc
            urefbaro[j] = urefbaro[j] + uref[j-jb, k] * exp(-zk / h) * dc

    return astarbaro, ubaro, urefbaro, ua1baro, ua2baro, ep1baro, ep2baro, ep3baro, ep4, astar1, astar2
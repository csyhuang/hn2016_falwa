from math import *
import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import eig,eigvals,det
from scipy import arange, array, exp
from copy import copy
import itertools

def load_pickle(fname):
    f = open(fname)
    var = pickle.load(f)
    f.close()
    return var

def dump_pickle(variable,fname):
    f = open(fname, 'w')
    pickle.dump(variable, f)
    f.close()

def input_jk_output_index(j,k,kmax):
    return j*(kmax) + k


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

# Constants in Noboru's code
nlon = 240
nlat = 121
jmax = 73
jmax1 = nlat #nlat+20
dz = 1000.          # vertical z spacing (m)
aa = 6378000.     # planetary radius
grav = 9.81       # gravity
dm = 1./float(jmax1+1)  # gaussian latitude spacing
p0 = 1000.          # reference pressure (hPa)
r0 = 287.           # gas constant
hh = 7000.          # scale height
cp = 1004.          # specific heat
rkappa = r0/cp
om = 7.29e-5          # angular velocity of the earth

# Constant arrays
xlon = np.linspace(0,360,nlon,endpoint=False)

# **** Define gaussian latitude grids in radian ****
gl = np.array([(j+1)*dm for j in range(jmax1)]) # This is sin / mu
gl_2 = np.array([j*dm for j in range(jmax1+2)]) # This is sin / mu
cosl = np.sqrt(1.-gl**2)
cosl_2 = np.sqrt(1.-gl_2**2)
alat = np.arcsin(gl)*180./pi
alat_2 = np.arcsin(gl_2)*180./pi
dmdz = (dm/dz)

# **** Function for solving the reference state in hemispheric domain ****
# tstamp: string; time-stamp for the snapshot (not used)
# zmum: numpy array with shape [kmax x nlat]; zonal mean zonal wind x cos(lat)
# FAWA_cos: numpy array with shape [kmax x nlat]; zonal mean finite-amplitude wave activity x cos(lat)
# ylat: numpy array with shape [nlat]; latitude in degree
# ephalf2: epsilon in Nakamura and Solomon (2010) at half-step. See Huang and Nakamura, submitted (supplementary materials)
# Delta_PT: numpy array with shape [nlat]; theta(kmax-2,j) - theta_ref(kmax-2) (obtained by box-counting)
# zm_PT: numpy array with shape [kmax,nlat]; zonal mean potential temperature (theta)

def Solve_Uref_noslip(tstamp,zmum,FAWA_cos,ylat,ephalf2,Delta_PT,zm_PT,use_real_Data=True):

    # **** Get from input these parameters ****
    kmax = FAWA_cos.shape[0]
    height = np.array([i for i in range(kmax)]) # in [km]

    # **** Initialize Coefficients ****
    c_a = np.zeros((jmax1,kmax))
    c_b = np.zeros((jmax1,kmax))
    c_c = np.zeros((jmax1,kmax))
    c_d = np.zeros((jmax1,kmax))
    c_e = np.zeros((jmax1,kmax))
    c_f = np.zeros((jmax1,kmax))

    # --- Initialize interpolated variables ---
    zmu1 = np.zeros((jmax1,kmax))
    cx1 = np.zeros((jmax1,kmax))
    cor1 = np.zeros((jmax1,kmax))
    ephalf = np.zeros((jmax1,kmax))
    Delta_PT1 = np.zeros((jmax1+2))
    zm_PT1 = np.zeros((jmax1,kmax))

    # --- Define Epsilon as a function of y and z ---
    # **** Interpolate to gaussian latitude ****
    if use_real_Data:
        for vv1,vvm in zip([zmu1,cx1,zm_PT1] , [zmum,FAWA_cos,zm_PT]):
            f_toGaussian = interpolate.interp1d(ylat[:],vvm[:,:].T,axis=0, kind='linear')    #[jmax x kmax]
            vv1[:,:] = f_toGaussian(alat[:])

        # --- Interpolation of ephalf ---
        f_ep_toGaussian = interpolate.interp1d(ylat[:],ephalf2[:,:].T,axis=0, kind='linear')    #[jmax x kmax]
        ephalf[:,:] = f_ep_toGaussian(alat[:])
        # --- Interpolation of Delta_PT ---
        f_DT_toGaussian = interpolate.interp1d(ylat[:],Delta_PT[:], kind='linear')    # This is txt in Noboru's code
        Delta_PT1[:] = f_DT_toGaussian(alat_2[:])
    else:
        # Use random matrix here just to test!
        zmu1 = np.random.rand(jmax1,kmax)+np.ones((jmax1,kmax))*1.e-8
        cx1 = np.random.rand(jmax1,kmax)+np.ones((jmax1,kmax))*1.e-8


    # --- Added on Aug 1, 2016 ---
    cor1 = 2.*om*gl[:,np.newaxis] * np.ones((jmax1,kmax))

    qxx0 = -cx1/cor1 # Input of LWA has cosine.
    c_f[0,:] = qxx0[1,:] - 2*qxx0[0,:]
    c_f[-1,:] = qxx0[-2,:] - 2*qxx0[-1,:]
    c_f[1:-1,:] = qxx0[:-2,:] + qxx0[2:,:] - 2*qxx0[1:-1,:]
    #c_f[:,0] = 0.0

    # **** Lower Adiabatic boundary conditions ****
    #uz1 = np.zeros((jmax1))
    #uz1[:] = - r0 * cosl[:]**2 * Input_dB1[:] * 2*dz / (cor1[:,1]**2 * aa**2 * hh * dm**2) * exp(-rkappa*(1.)/7.) \
    #- r0 * cosl[:]**2 * Input_dB0[:] * 2*dz / (cor1[:,0]**2 * aa**2 * hh * dm**2) * exp(-rkappa*(0.)/7.)

    #  **** Upper Boundary Condition (Come back later) ****
    uz2 = np.zeros((jmax1))
    dDelta_PT1 = (Delta_PT1[2:]-Delta_PT1[:-2]) # Numerical trick: Replace uz2[1] with an extrapolated value
    uz2[:] = - r0 * cosl[:]**2 * exp(-rkappa*(kmax-2.)/7.) * dDelta_PT1 / (cor1[:,-2]**2 * aa * hh * dmdz)

    #  **** Initialize the coefficients a,b,c,d,e,f ****
    c_a[:,:] = 1.0
    c_b[:,:] = 1.0
    c_c[:,1:-1] = dmdz**2 *ephalf[:,1:-1]*exp(-dz/(2*hh))  # This one should be correct
    c_d[:,1:-1] = dmdz**2 *ephalf[:,0:-2]*exp(dz/(2*hh)) # Check convention of ephalf
    c_e[:,1:-1] = -(c_a[:,1:-1]+c_b[:,1:-1]+c_c[:,1:-1]+c_d[:,1:-1])

    b = np.zeros((jmax1*kmax))
    row_index=[]
    col_index=[]
    coeff = []

    jrange = range(jmax1)
    krange = range(1,kmax-1)
    for j, k in itertools.product(jrange, krange):
        ind = input_jk_output_index(j,k,kmax)
        b[ind] = c_f[j,k]
        if (j<jmax1-1):
            row_index.append(ind)
            col_index.append(input_jk_output_index(j+1,k,kmax))
            coeff.append(c_a[j,k])
        if (j>0):
            row_index.append(ind)
            col_index.append(input_jk_output_index(j-1,k,kmax))
            coeff.append(c_b[j,k])
        row_index.append(ind)
        col_index.append(input_jk_output_index(j,k+1,kmax))
        coeff.append(c_c[j,k])
        row_index.append(ind)
        col_index.append(input_jk_output_index(j,k-1,kmax))
        coeff.append(c_d[j,k])
        row_index.append(ind)
        col_index.append(input_jk_output_index(j,k,kmax))
        coeff.append(c_e[j,k])

    # ==== Upper boundary condition - thermal wind ====
    for j in range(jmax1):
        ind1 = input_jk_output_index(j,kmax-1,kmax)
        b[ind1] = uz2[j] #- r0 * cosl[j]**2  * exp(-rkappa*(kmax-2.)/7.) * (Delta_PT1[j+1]-Delta_PT1[j-1])/ (cor1[j,-2]**2 * aa * hh * dmdz)
        row_index.append(ind1)
        col_index.append(ind1)
        coeff.append(1.0)
        row_index.append(ind1)
        col_index.append(input_jk_output_index(j,kmax-3,kmax))
        coeff.append(-1.0)

    # ==== Lower boundary condition - no-slip (k=0) ====
    for j in range(jmax1):
        ind = input_jk_output_index(j,0,kmax)
        b[ind] = zmu1[j,0]*cosl[j]/cor1[j,0]
        # A[ind,ind] = 1.0
        row_index.append(ind)
        col_index.append(ind)
        coeff.append(1.0)

    A = csc_matrix((coeff, (row_index, col_index)), shape=(jmax1*kmax,jmax1*kmax))

    # === Solving the linear system ===
    u2 = spsolve(A, b)

    # === Mapping back to 2D matrix ===
    u = np.zeros((jmax1+2,kmax))
    for j in range(jmax1):
        for k in range(kmax):
            u[j+1,k] = u2[j*kmax + k]

    u_MassCorr_noslip = np.zeros_like(u)
    u_MassCorr_noslip[1:-1,:] = u[1:-1,:] * cor1 / cosl[:,np.newaxis]

    # --- Initialize T_MassCorr to be output ---
    u_Ref_regular_noslip = np.zeros_like(zmum)
    T_Ref_regular_noslip = np.zeros_like(zmum)

    u_MassCorr = u_MassCorr_noslip
    u_Ref_regular = u_Ref_regular_noslip
    T_Ref_regular = T_Ref_regular_noslip
    BCstring = 'Noslip'

    # ---- Back out temperature correction here -----
    T_MassCorr = np.zeros_like(u_MassCorr)
    for k in range(1,kmax-2):
        for j in range(2,jmax1,2):
            T_MassCorr[j,k] = T_MassCorr[j-2,k] - (2.*om*gl[j-1])*aa*hh*dmdz / (r0 * cosl[j-1]) * (u_MassCorr[j-1,k+1]-u_MassCorr[j-1,k-1])
        # ---- First do interpolation (gl is regular grid) ----
        f_Todd = interpolate.interp1d(gl_2[::2],T_MassCorr[::2,k])    #[jmax x kmax]
        f_Todd_ex = extrap1d(f_Todd)
        T_MassCorr[:,k] = f_Todd_ex(gl_2[:]) # Get all the points interpolated

        # ---- Then do domain average ----
        T_MC_mean = np.mean(T_MassCorr[:,k])
        T_MassCorr[:,k] -= T_MC_mean

    # --- First, interpolate MassCorr back to regular grid first ---
    f_u_MassCorr = interpolate.interp1d(alat_2,u_MassCorr,axis=0, kind='linear')    #[jmax x kmax]
    u_MassCorr_regular = f_u_MassCorr(ylat[-nlat/2:]).T
    f_T_MassCorr = interpolate.interp1d(alat_2,T_MassCorr,axis=0, kind='linear')    #[jmax x kmax]
    T_MassCorr_regular = f_T_MassCorr(ylat[-nlat/2:]).T

    u_Ref = zmum[:,-nlat/2:] - u_MassCorr_regular
    T_ref = zm_PT[:,-nlat/2:] * np.exp(-np.arange(kmax)/7. * rkappa)[:,np.newaxis] - T_MassCorr_regular

    u_Ref_regular[:,-nlat/2:] = u_Ref
    T_Ref_regular[:,-nlat/2:] = T_ref

        # print 'u_Ref_regular.shape=',u_Ref_regular.shape

    # return u_Ref_regular,T_Ref_regular
    return u_Ref_regular_noslip,T_Ref_regular_noslip

# --- As a test whether the function Solve_Uref is working ---
if __name__ == "__main__":
    #numpy.random.rand(d0, d1, ..., dn)
    t1 = np.random.rand(nlat,kmax)+np.ones((nlat,kmax))*0.001
    t2 = np.random.rand(nlat,kmax)+np.ones((nlat,kmax))*0.001
    t3 = np.random.rand(nlat,kmax)+np.ones((nlat,kmax))*0.001
    eh = np.random.rand(jmax1,kmax)+np.ones((jmax1,kmax))*0.001
    Delta_PT = np.sort(np.random.rand(jmax1))
    use_real_Data = False
    uutest = Solve_Uref(t1,t2,t3,eh,Delta_PT)

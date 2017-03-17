from math import *
import numpy as np

def Compute_QGPV_GivenVort(omega,nlat,nlon,kmax,unih,ylat,avort,PT3,t0_cN,t0_cS,stat_cN,stat_cS):

    '''
    Input variables:
        nlat: integer; dimension of latitude grid
		nlon: integer; dimension of longitude grid
		kmax: integer; dimension of height grid
        unih: 1-D numpy array of height in [meters]; size = kmax
		ylat: 1-D numpy array of ascending latitude in [degree]; size = nlat
		avort: 3-D numpy array of absolute vorticity (i.e. relative vorticity + 2*Omega*sin(lat)) in [1/s];
		       shape = [kmax x nlat x nlon]
		PT3: 3-D numpy array of potential temperature in [K]; shape = [kmax x nlat x nlon]
		t0_N (t0_S): 1-D numpy array area-weighted average of potential temperature (\tilde{\theta} in HN16) 
		             in the Northern (Southern) hemispheric domain; size = kmax
		stat_cN (stat_cS): 1-D numpy array of static stability (d\tilde{\theta}/dz in HN16) in the Northern (Southern) 
		         hemispheric domain; size = kmax
		Note: The 4 arrays, t0_cN, t0_cS, stat_cN, stat_cS, are output from function static_stability
        
    Output variables:
	QGPV, dzdiv
        QGPV: 3-d numpy array of quasi-geostrophic potential vorticity; shape = [kmax x nlat x nlon]
        dzdiv: 3-d numpy array of the stretching term in QGPV; shape = [kmax x nlat x nlon]
		
	Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
    '''


    clat = np.cos(ylat*pi/180.)
    clat = np.abs(clat) # Just to avoid the negative value at poles

    # --- Next, calculate PV ---
    av2 = np.empty_like(PT3) # dv/d(lon)
    av3 = np.empty_like(PT3) # du/d(lat)
    QGPV = np.empty_like(PT3) # av1+av2+av3+dzdiv

    av1 = np.ones((kmax,nlat,nlon)) * 2*omega*np.sin(ylat[np.newaxis,:,np.newaxis]*pi/180.)

    # Calculate the z-divergence term
    zdiv = np.empty_like(PT3)
    dzdiv = np.empty_like(PT3)
    for kk in range(kmax): # This is more efficient
        zdiv[kk,:60,:] = exp(-unih[kk]/7000.)*(PT3[kk,:60,:]-t0_cS[kk])/stat_cS[kk]
        zdiv[kk,60:,:] = exp(-unih[kk]/7000.)*(PT3[kk,60:,:]-t0_cN[kk])/stat_cN[kk]

    dzdiv[1:kmax-1,:,:] = np.exp(unih[1:kmax-1,np.newaxis,np.newaxis]/7000.)*(zdiv[2:kmax,:,:]-zdiv[0:kmax-2,:,:])/2000. # 2dz = 2000.

    dzdiv[0,:,:] = exp(unih[0]/7000.)*(zdiv[1,:,:]-zdiv[0,:,:])/1000. # 2dz = 2000.
    dzdiv[kmax-1,:,:] = exp(unih[kmax-1]/7000.)*(zdiv[kmax-1,:,:]-zdiv[kmax-2,:,:])/1000. # 2dz = 2000.

    QGPV = avort+dzdiv * av1
    return QGPV, dzdiv


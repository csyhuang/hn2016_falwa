from math import pi,exp
import numpy as np

def compute_qgpv_givenvort(omega,nlat,nlon,kmax,unih,ylat,avort,potential_temp,t0_cn,t0_cs,stat_cn,stat_cs,scale_height=7000.):

    '''
    Input variables:
        nlat: integer; dimension of latitude grid
		nlon: integer; dimension of longitude grid
		kmax: integer; dimension of height grid
        unih: 1-D numpy array of height in [meters]; size = kmax
		ylat: 1-D numpy array of ascending latitude in [degree]; size = nlat
		avort: 3-D numpy array of absolute vorticity (i.e. relative vorticity + 2*Omega*sin(lat)) in [1/s];
		       shape = [kmax x nlat x nlon]
		potential_temp: 3-D numpy array of potential temperature in [K]; shape = [kmax x nlat x nlon]
		t0_N (t0_S): 1-D numpy array area-weighted average of potential temperature (\tilde{\theta} in HN16) 
		             in the Northern (Southern) hemispheric domain; size = kmax
		stat_cn (stat_cs): 1-D numpy array of static stability (d\tilde{\theta}/dz in HN16) in the Northern (Southern) 
		         hemispheric domain; size = kmax
		Note: The 4 arrays, t0_cn, t0_cs, stat_cn, stat_cs, are output from function static_stability
		scale_height: scale_height of the atmosphere in [m]
        
    Output variables:
	QGPV, dzdiv
        QGPV: 3-d numpy array of quasi-geostrophic potential vorticity; shape = [kmax x nlat x nlon]
        dzdiv: 3-d numpy array of the stretching term in QGPV; shape = [kmax x nlat x nlon]
		
	Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
    '''


    clat = np.cos(ylat*pi/180.)
    clat = np.abs(clat) # Just to avoid the negative value at poles

    # --- Next, calculate PV ---
    av2 = np.empty_like(potential_temp) # dv/d(lon)
    av3 = np.empty_like(potential_temp) # du/d(lat)
    qgpv = np.empty_like(potential_temp) # av1+av2+av3+dzdiv

    av1 = np.ones((kmax,nlat,nlon)) * 2*omega*np.sin(ylat[np.newaxis,:,np.newaxis]*pi/180.)

    # Calculate the z-divergence term
    zdiv = np.empty_like(potential_temp)
    dzdiv = np.empty_like(potential_temp)
    for kk in range(kmax): # This is more efficient
        zdiv[kk,:60,:] = exp(-unih[kk]/scale_height)*(potential_temp[kk,:60,:]-t0_cs[kk])/stat_cs[kk]
        zdiv[kk,60:,:] = exp(-unih[kk]/scale_height)*(potential_temp[kk,60:,:]-t0_cn[kk])/stat_cn[kk]

    dzdiv[1:kmax-1,:,:] = np.exp(unih[1:kmax-1,np.newaxis,np.newaxis]/scale_height)* \
    (zdiv[2:kmax,:,:]-zdiv[0:kmax-2,:,:]) \
    /(unih[2:kmax,np.newaxis,np.newaxis]-unih[0:kmax-2,np.newaxis,np.newaxis])

    dzdiv[0,:,:] = exp(unih[0]/scale_height)*(zdiv[1,:,:]-zdiv[0,:,:])/ \
    (unih[1,np.newaxis,np.newaxis]-unih[0,np.newaxis,np.newaxis])
    dzdiv[kmax-1,:,:] = exp(unih[kmax-1]/scale_height)*(zdiv[kmax-1,:,:]-zdiv[kmax-2,:,:])/ \
    (unih[kmax-1,np.newaxis,np.newaxis]-unih[kmax-2,np.newaxis,np.newaxis])

    qgpv = avort+dzdiv * av1
    return qgpv, dzdiv


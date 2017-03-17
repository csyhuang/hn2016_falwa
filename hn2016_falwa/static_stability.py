import numpy as np
from math import *


def static_stability(height,area,theta,S_ET=None,N_ET=None):
    ''' The function "static_stability" computes the vertical gradient (z-derivative)
    of hemispheric-averaged potential temperature, i.e. d\tilde{theta}/dz in the def-
    inition of QGPV in eq.(3) of Huang and Nakamura (2016), by central differencing.
    At the boundary, the static stability is estimated by forward/backward differen-
    cing involving two adjacent z-grid points:
        
        i.e. stat_N[0] = (t0_N[1]-t0_N[0])/(height[1]-height[0])
            stat_N[-1] = (t0_N[-2]-t0_N[-1])/(height[-2]-height[-1])
    
    Input variables:
        - height: array of z-coordinate [in meters] with size = kmax, equally spaced
        - ylat: array of ascending y-coordinate [latitude, in degree] with size = nlat, equally spaced
        - theta: matrix of potential temperature [K] with shape = [kmax,nlat,nlon]
    
    Output variables:
        - t0_N (t0_S): array area-weighted average of potential temperature (\tilde{\theta} in HN16) 
        in the Northern (Southern) hemispheric domain; size = kmax
        - stat_N (stat_S): array of static stability (d\tilde{\theta}/dz in HN16) in the Northern 
        (Southern) hemispheric domain; size = kmax
            
    
    Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues
    '''
    

    nlat = theta.shape[1]
    if S_ET==None:
        S_ET = nlat/2
    if N_ET==None:
        N_ET = nlat/2
    
    stat_N = np.zeros(theta.shape[0])
    stat_S = np.zeros(theta.shape[0])

    zonal_mean = np.mean(theta,axis=-1)
    area_zonal_mean = np.mean(area,axis=-1)

    csm_N_ET = np.sum(area_zonal_mean[-N_ET:])
    csm_S_ET = np.sum(area_zonal_mean[:S_ET])

    t0_N = np.sum(zonal_mean[:,-N_ET:]*area_zonal_mean[np.newaxis,-N_ET:],axis=-1)/csm_N_ET
    t0_S = np.sum(zonal_mean[:,:S_ET]*area_zonal_mean[np.newaxis,:S_ET],axis=-1)/csm_S_ET

    stat_N[1:-1] = (t0_N[2:]-t0_N[:-2])/(height[2:]-height[:-2])
    stat_S[1:-1] = (t0_S[2:]-t0_S[:-2])/(height[2:]-height[:-2])
    stat_N[0] = (t0_N[1]-t0_N[0])/(height[1]-height[0])
    stat_N[-1] = (t0_N[-2]-t0_N[-1])/(height[-2]-height[-1])
    stat_S[0] = (t0_S[1]-t0_S[0])/(height[1]-height[0])
    stat_S[-1] = (t0_S[-2]-t0_S[-1])/(height[-2]-height[-1])

    return t0_N,t0_S,stat_N,stat_S
    

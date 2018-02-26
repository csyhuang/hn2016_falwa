import numpy as np
from math import pi,exp

def static_stability(height,area,theta,s_et=None,n_et=None):
    """
    The function "static_stability" computes the vertical gradient (z-derivative)
    of hemispheric-averaged potential temperature, i.e. d\tilde{theta}/dz in the def-
    inition of QGPV in eq.(3) of Huang and Nakamura (2016), by central differencing.
    At the boundary, the static stability is estimated by forward/backward differen-
    cing involving two adjacent z-grid points:

        i.e. stat_n[0] = (t0_n[1]-t0_n[0])/(height[1]-height[0])
            stat_n[-1] = (t0_n[-2]-t0_n[-1])/(height[-2]-height[-1])

    Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues


    Parameters
    ----------
    height : sequence or array_like
        Array of z-coordinate [in meters] with dimension = (kmax), equally spaced
    area : ndarray
        Two-dimension numpy array specifying differential areal element of each grid point;
        dimension = (nlat, nlon).
    theta : ndarray
        Matrix of potential temperature [K] with dimension (kmax,nlat,nlon) or (kmax,nlat)
    s_et : int, optional
        Index of the latitude that defines the boundary of the Southern hemispheric domain;
        initialized as nlat/2 if not input
    n_et : int, optional
        Index of the latitude that defines the boundary of the Southern hemispheric domain;
        initialized as nlat/2 if not input


    Returns
    -------
    t0_n : sequence or array_like
        Area-weighted average of potential temperature (\tilde{\theta} in HN16)
        in the Northern hemispheric domain with dimension = (kmax)
    t0_s : sequence or array_like
        Area-weighted average of potential temperature (\tilde{\theta} in HN16)
        in the Southern hemispheric domain with dimension = (kmax)
    stat_n : sequence or array_like
        Static stability (d\tilde{\theta}/dz in HN16) in the Northern hemispheric
        domain with dimension = (kmax)
    stat_s : sequence or array_like
        Static stability (d\tilde{\theta}/dz in HN16) in the Southern hemispheric
        domain with dimension = (kmax)

    """

    nlat = theta.shape[1]
    if s_et==None:
        s_et = nlat//2
    if n_et==None:
        n_et = nlat//2

    stat_n = np.zeros(theta.shape[0])
    stat_s = np.zeros(theta.shape[0])

    if theta.ndim==3:
        zonal_mean = np.mean(theta,axis=-1)
    elif theta.ndim==2:
        zonal_mean = theta

    if area.ndim==2:
        area_zonal_mean = np.mean(area,axis=-1)
    elif area.ndim==1:
        area_zonal_mean = area

    csm_n_et = np.sum(area_zonal_mean[-n_et:])
    csm_s_et = np.sum(area_zonal_mean[:s_et])

    t0_n = np.sum(zonal_mean[:,-n_et:]*area_zonal_mean[np.newaxis,-n_et:],axis=-1)/csm_n_et
    t0_s = np.sum(zonal_mean[:,:s_et]*area_zonal_mean[np.newaxis,:s_et],axis=-1)/csm_s_et

    stat_n[1:-1] = (t0_n[2:]-t0_n[:-2])/(height[2:]-height[:-2])
    stat_s[1:-1] = (t0_s[2:]-t0_s[:-2])/(height[2:]-height[:-2])
    stat_n[0] = (t0_n[1]-t0_n[0])/(height[1]-height[0])
    stat_n[-1] = (t0_n[-2]-t0_n[-1])/(height[-2]-height[-1])
    stat_s[0] = (t0_s[1]-t0_s[0])/(height[1]-height[0])
    stat_s[-1] = (t0_s[-2]-t0_s[-1])/(height[-2]-height[-1])

    return t0_n,t0_s,stat_n,stat_s


def compute_qgpv_givenvort(omega,nlat,nlon,kmax,unih,ylat,avort,potential_temp,
                           t0_cn,t0_cs,stat_cn,stat_cs,nlat_s=None,scale_height=7000.):
    """
    The function "compute_qgpv_givenvort" computes the quasi-geostrophic potential
    vorticity based on the absolute vorticity, potential temperature and static
    stability given.

    Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues


    Parameters
    ----------
    omega : float, optional
        Rotation rate of the planet.
    nlat : int
        Latitudinal dimension of the latitude grid.
    nlon : int
        Longitudinal dimension of the longitude grid.
    kmax : int
        Vertical dimension of the height grid.
    unih : sequence or array_like
        Numpy array of height in [meters]; dimension = (kmax)
    ylat : sequence or array_like
        Numpy array of latitudes in [degrees]; dimension = (nlat)
    avort : ndarray
        Three-dimension numpy array of absolute vorticity (i.e. relative vorticity
        + 2*Omega*sin(lat)) in [1/s]; dimension = (kmax x nlat x nlon)
    potential_temp : ndarray
        Three-dimension numpy array of potential temperature in [K];
        dimension = (kmax x nlat x nlon)
    t0_cn : sequence or array_like
        Area-weighted average of potential temperature (\tilde{\theta} in HN16)
        in the Northern hemispheric domain with dimension = (kmax)
    t0_cs : sequence or array_like
        Area-weighted average of potential temperature (\tilde{\theta} in HN16)
        in the Southern hemispheric domain with dimension = (kmax)
    stat_cn : sequence or array_like
        Static stability (d\tilde{\theta}/dz in HN16) in the Northern hemispheric
        domain with dimension = (kmax)
    stat_cs : sequence or array_like
        Static stability (d\tilde{\theta}/dz in HN16) in the Southern hemispheric
        domain with dimension = (kmax)
    scale_height : float
        Scale height of the atmosphere in [m] with default value 7000.


    Returns
    -------
    QGPV : ndarray
        Three-dimension numpy array of quasi-geostrophic potential vorticity;
        dimension = (kmax x nlat x nlon)
    dzdiv : ndarray
        Three-dimension numpy array of the stretching term in QGPV;
        dimension = (kmax x nlat x nlon)

    """

    if nlat_s==None:
        nlat_s=nlat//2

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
        zdiv[kk,:nlat_s,:] = exp(-unih[kk]/scale_height)*(potential_temp[kk,:nlat_s,:]-t0_cs[kk])/stat_cs[kk]
        zdiv[kk,-nlat_s:,:] = exp(-unih[kk]/scale_height)*(potential_temp[kk,-nlat_s:,:]-t0_cn[kk])/stat_cn[kk]

    dzdiv[1:kmax-1,:,:] = np.exp(unih[1:kmax-1,np.newaxis,np.newaxis]/scale_height)* \
    (zdiv[2:kmax,:,:]-zdiv[0:kmax-2,:,:]) \
    /(unih[2:kmax,np.newaxis,np.newaxis]-unih[0:kmax-2,np.newaxis,np.newaxis])

    dzdiv[0,:,:] = exp(unih[0]/scale_height)*(zdiv[1,:,:]-zdiv[0,:,:])/ \
    (unih[1,np.newaxis,np.newaxis]-unih[0,np.newaxis,np.newaxis])
    dzdiv[kmax-1,:,:] = exp(unih[kmax-1]/scale_height)*(zdiv[kmax-1,:,:]-zdiv[kmax-2,:,:])/ \
    (unih[kmax-1,np.newaxis,np.newaxis]-unih[kmax-2,np.newaxis,np.newaxis])

    qgpv = avort+dzdiv * av1
    return qgpv, dzdiv

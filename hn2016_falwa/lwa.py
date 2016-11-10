from math import *
import numpy as np

def LWA(nlon,nlat,n_points,vort,Q_part,dy):
    ''' At each grid point of vorticity q(x,y) and reference state vorticity Q(y),
    this function calculate the difference between the line integral of [q(x,y+y')-Q(y)]
    over the domain {y+y'>y,q(x,y+y')<Q(y)} and {y+y'<y,q(x,y+y')>Q(y)}. See fig. (1) and
    equation (13) of Huang and Nakamura (2016).
    dy is a vector of length nlat: dy = cos(phi) d(phi) such that phi is the latitude.
    '''    
    LWAct = np.zeros((nlat,nlon))
    for j in np.arange(0,nlat-1):
        vort_e = vort[:,:]-Q_part[j]
        vort_boo = np.zeros((nlat,nlon))
        vort_boo[np.where(vort_e[:,:]<0)] = -1
        vort_boo[:j+1,:] = 0
        vort_boo[np.where(vort_e[:j+1,:]>0)] = 1
        LWAct[j,:] = np.sum(vort_e*vort_boo*dy[:,np.newaxis],axis=0)        
    return LWAct
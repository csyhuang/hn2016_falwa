from math import *
import numpy as np

def LWA(nlon,nlat,vort,Q_part,dy):
    ''' At each grid point of vorticity q(x,y) and reference state vorticity Q(y),
    this function calculate the difference between the line integral of [q(x,y+y')-Q(y)]
    over the domain {y+y'>y,q(x,y+y')<Q(y)} and {y+y'<y,q(x,y+y')>Q(y)}. See fig. (1) and
    equation (13) of Huang and Nakamura (2016).
    dy is a vector of length nlat: dy = cos(phi) d(phi) such that phi is the latitude.
    
    Input variables:
        nlon: integer; longitudinal dimension of vort (i.e. vort.shape[1])
        nlat: integer; latitudinal dimension of vort (i.e. vort.shape[0])
        vort: 2-d numpy array of vorticity values; dimension = [nlat_S x nlon]
        Q_part: 1-d numpy array of Q (vorticity reference state) as a function of
                latitude. Size = nlat.
        dy:   1-d numpy array of latitudinal differential length element 
              (e.g. dy = cos(lat) d(lat)). Size = nlat.
        
    Output variables:
        LWA: 2-d numpy array of local wave activity values (with cosine weighting); 
             dimension = [nlat_S x nlon]        
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
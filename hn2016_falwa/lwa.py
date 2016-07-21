from math import *
import numpy as np
import matplotlib.pyplot as plt

def LWA(nlon,nlat,n_points,vort,Q_part,dy):
    ''' At each grid point of vorticity q(x,y) and reference state vorticity Q(y),
    this function calculate the difference between the line integral of [q(x,y+y')-Q(y)]
    over the domain {y+y'>y,q(x,y+y')<Q(y)} and {y+y'<y,q(x,y+y')>Q(y)}.
    '''    
    LWAct = np.zeros((nlat,nlon))
    for j in np.arange(0,nlat-1):
        vort_e = vort[:,:]-Q_part[j]
        vort_boo = np.zeros((nlat,nlon))
        vort_boo[np.where(vort_e[:,:]<0)] = -1
        vort_boo[:j+1,:] = 0
        vort_boo[np.where(vort_e[:j+1,:]>0)] = 1
        LWAct[j,:] = np.sum(vort_e*vort_boo*dy,axis=0)
    return LWAct
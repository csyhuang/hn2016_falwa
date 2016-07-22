# -*- coding: utf-8 -*-
''' The python package "hn2016_falwa" contains a function "barotropic_Eqlat_LWA" 
that computes the finite-amplitude local wave activity (LWA) in a barotropic model 
with spherical geometry according to the definition in Huang & Nakamura (2016,JAS) 
equation (13). This sample code demonstrates how the function in this package can 
be used to reproduce LWA plots (Fig.4 in HN15) from an absolute vorticity map.

Please email Clare S. Y. Huang if you have any inquiries/suggestions: clare1068@gmail.com
'''

import hn2016_falwa # Module for plotting local wave activity (LWA) plots and 
                      # the corresponding equivalent-latitude profile
from math import *
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype
import datetime as dt

# --- Parameters --- #
a = 6.378e+6 # Earth's radius

# --- Load the zonal wind and QGPV at 240hPa --- #
u_QGPV_File = Dataset('u_QGPV_240hPa_2012_SON.nc', mode='r')

# --- Read in longitude and latitude arrays --- #
xlon = u_QGPV_File.variables['longitude'][:]
ylat = u_QGPV_File.variables['latitude'][:]
nlon = xlon.size
nlat = ylat.size

# --- Parameters needed to use the module HN2015_LWA --- #
dphi = (ylat[2]-ylat[1])*pi/180. # Equal spacing between latitude grid points, in radian
area = 2.*pi*a*a*(np.cos(ylat[:,np.newaxis]*pi/180.)*dphi)/float(nlon) * np.ones((nlat,nlon))
area = np.abs(area) # To make sure area element is always positive (given floating point errors). 

# --- Datestamp ---
Start_date = dt.datetime(2012, 9, 1, 0, 0)
delta_t = dt.timedelta(hours=6)
Datestamp = [Start_date + delta_t*tt for tt in range(364)]

# --- Read in the absolute vorticity field from the netCDF file --- #
u = u_QGPV_File.variables['U'][:]
QGPV = u_QGPV_File.variables['QGPV'][:]

# --- Obtain equivalent-latitude relationship and also the LWA from the absolute vorticity snapshot ---
for tt in range(228,228+13,4):
    Qref, LWA = hn2016_falwa.qgpv_Eqlat_LWA(ylat,QGPV[tt,0,:,:],area,a*dphi)
    
    # --- Plot LWA ---
    fig = plt.subplots(figsize=(6,9))
    plt.subplot(311)
    plt.contourf(xlon,ylat,u[tt,0,:,:],41)
    plt.colorbar()
    plt.title('zonal wind at 240hPa | '+Datestamp[tt].strftime("%y/%m/%d %H:%M"))
    plt.subplot(312)
    plt.contourf(xlon,ylat,LWA,41)
    plt.colorbar()
    plt.title('LWA at 240hPa')
    plt.subplot(313)
    c = plt.contourf(xlon,ylat,QGPV[tt,0,:,:],41)
    cb = plt.colorbar(c)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')                         
    cb.update_ticks()    
    plt.title('QGPV at 240hPa')
    plt.tight_layout()
    #plt.savefig(img_output_dir+'test3_'+str(yr)+'_'+str(sec*3+mths+1).zfill(2)+'_'+str(nccount).zfill(4)+'.png')
    plt.show()

u_QGPV_File.close()
# -*- coding: utf-8 -*-
''' The python package "hn2016_falwa" contains a function "qgpv_Eqlat_LWA" 
that computes the finite-amplitude local wave activity (LWA) based on quasi-
geostrophic potential vorticity (QGPV) field derived from Reanalysis data with 
spherical geometry. It differs from the function "barotropic_Eqlat_LWA" that a 
hemispheric domain (instead of global domain) is used to compute both 
equivalent-latitude relationship and LWA. This is to avoid spurious large values 
of LWA near the equator arising from the small meridional gradient of QGPV there.

This sample code demonstrates how the function in this package can be used to 
reproduce plots of zonal wind, QGPV and LWA plots (Fig.8-9 in HN15) from QGPV
fields.

Please email Clare S. Y. Huang if you have any inquiries/suggestions: clare1068@gmail.com
'''

import hn2016_falwa # Module for plotting local wave activity (LWA) plots and 
                      # the corresponding equivalent-latitude profile
from math import *
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# --- Parameters --- #
a = 6.378e+6 # Earth's radius

# --- Load the zonal wind and QGPV at 240hPa --- #
u_QGPV_File = Dataset('u_QGPV_240hPa_2012Oct28to31.nc', mode='r')

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
Start_date = dt.datetime(2012, 10, 28, 0, 0)
delta_t = dt.timedelta(hours=24)
Datestamp = [Start_date + delta_t*tt for tt in range(4)]

# --- Read in the absolute vorticity field from the netCDF file --- #
u = u_QGPV_File.variables['U'][:]
QGPV = u_QGPV_File.variables['QGPV'][:]

# --- Obtain equivalent-latitude relationship and also the LWA from the absolute vorticity snapshot ---
for tt in range(4):
    Qref, LWA = hn2016_falwa.qgpv_Eqlat_LWA(ylat,QGPV[tt,0,:,:],area,a*dphi)
    
    # --- Plot LWA ---
    fig,ax = plt.subplots(figsize=(6,9))
    plt.subplot(311)
    plt.contourf(xlon,ylat,u[tt,0,:,:],41)
    plt.ylabel('Latitude [deg]')
    plt.colorbar()
    plt.title('zonal wind at 240hPa | '+Datestamp[tt].strftime("%y/%m/%d %H:%M"))
    plt.subplot(312)
    c = plt.contourf(xlon,ylat,QGPV[tt,0,:,:],41)
    plt.ylabel('Latitude [deg]')
    cb = plt.colorbar(c)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')                         
    cb.update_ticks()    
    plt.title('QGPV at 240hPa')
    plt.subplot(313)
    plt.contourf(xlon,ylat,LWA,41)
    plt.ylabel('Latitude [deg]')
    plt.colorbar()
    plt.title('LWA at 240hPa')
    plt.xlabel('Longitude [deg]')
    fig.subplots_adjust(hspace=.3)
    plt.show()
    plt.savefig('u_LWA_QGPV_240hPa_'+Datestamp[tt].strftime("%y_%m_%d_%H")+'.png')

u_QGPV_File.close()
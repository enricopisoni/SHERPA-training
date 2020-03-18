# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:08:34 2017
compute distances of all cells in comparison to the central cell of the 'mask' defined for a specific ir, ic
the routine converts lat lon distances in km distances
@author: pisonen
"""

import numpy as np

AVG_EARTH_RADIUS = 6378.137  # in km
def haversine_vec(lon1,lat1,lon_vec2,lat_vec2):
    # calculate haversine
    dlat = np.radians(lat_vec2) - np.radians(lat1)
    dlon = np.radians(lon_vec2) - np.radians(lon1)
    d = np.sin(dlat * 0.5) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat_vec2)) * np.sin(dlon * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h # in kilometers
    
def distanceComputation(LAT, LON, ir, ic, rad, x, y):
    #define cells subset
    LATsub = LAT[ir:ir+rad+rad+1, ic:ic+rad+rad+1];
    LONsub = LON[ir:ir+rad+rad+1, ic:ic+rad+rad+1];
    #define central cell
    centralLAT = y[ir,ic]
    centralLON = x[ir,ic] 
    #define vector of lat lon to be considered for distances computation
#    LATsubravel = LATsub.flatten()
#    LONsubravel = LONsub.flatten()
    #compute distances in km
    distVec=haversine_vec(centralLON, centralLAT, LONsub, LATsub)
    #reshape to final 'square' dimension
    distfinal = np.reshape(distVec,(LATsub.shape[0],LATsub.shape[1]))
#    distfinal = distfinal + 1
    distfinal[np.where(LATsub==0)]=1
    distfinal[np.where(LONsub==0)]=1     
    distfinal[rad,rad]=1
    return distfinal
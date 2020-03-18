import geopy.distance as gpd
import numpy as np
import matplotlib.pyplot as plt

def computeDistanceRatio(conf):
    if conf.distance == 1:
        #compute step for lat and lon
        lat = np.unique(conf.y)
        lon = np.unique(conf.x)
        step_lat = lat[1] - lat[0]
        step_lon = lon[1] - lon[0]
        #overwrite lon variable to compute how delta lat and lon change
        lon = np.repeat(10, np.size(lat))  # overwtie lonrepeat a given value, as lon does not affect result
        res_delta_lat = [gpd.distance((lat[i],lon[i]),(lat[i]+step_lat,lon[i])).km for i in range(0,np.size(lat))]
        res_delta_lon = [gpd.distance((lat[i],lon[i]),(lat[i],lon[i]+step_lon)).km for i in range(0,np.size(lat))]
        #compute ratio lat lon
        res_delta_lat_array = np.array(res_delta_lat).reshape(-1, 1)
        res_delta_lon_array = np.array(res_delta_lon).reshape(-1, 1)
        ratio_lat_lon = res_delta_lat_array / res_delta_lon_array
        #compute polynomial to approximate how ratio_lat_lon changes with lat, and save it
        ratio_pol = np.polyfit(lat, ratio_lat_lon,5)
        conf.ratioPoly = ratio_pol
    elif conf.distance == 0:
        conf.ratioPoly = np.array([0,0,0,0,0,1])
    return conf

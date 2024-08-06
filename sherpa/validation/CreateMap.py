'''
Created on 25-nov-2016

@author: roncolato
'''
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import scipy.io as sio
import platform
import cartopy.crs as ccrs

def CreateMap(bctarget,target,output,flagRegioMat,x,y,iSc,nomeDir,aqi,absdel,flagReg,domain,conf,thresGraphs):
    print('aqi: '+aqi);
    print('domain: '+domain);
    # creating maps
    target[flagRegioMat==0] = np.nan;
    output[flagRegioMat==0] = np.nan;

    # create graphs for target, output, bias, percentage bias
    nameFile = ['tar-sce-n','out-sce-n','bias-sce-n','biasPerc-sce-n','delta-out-sce-n','delta-tar-sce-n'];

    if absdel==0:
        mapInfo = ['target','output','output-target','(output-target)/target*100','output_DeltaC','target_DeltaC'];
    elif absdel==1:
        mapInfo = ['bctarget-target', 'bctarget-output', '(bctarget-output)-(bctarget-target)', '((bctarget-output)-(bctarget-target))/(bctarget-target)*100', 'output', 'target'];

    colors = ['plt.cm.Reds', 'plt.cm.Reds', 'plt.cm.bwr', 'plt.cm.bwr', 'plt.cm.Reds', 'plt.cm.Reds'];
    # colors=['plt.cm.plasma_r','plt.cm.plasma_r','plt.cm.seismic','plt.cm.seismic','plt.cm.plasma_r','plt.cm.plasma_r'];
    if ('PM10' in aqi):
        levels = [np.arange(0,41,1),np.arange(0,41,1),np.arange(-4,4.2,0.2),np.arange(-10,11,1),np.arange(0,21,1),np.arange(0,21,1)];
        # colors=['plt.cm.Reds','plt.cm.Reds','plt.cm.bwr','plt.cm.bwr','plt.cm.bwr','plt.cm.bwr'];
    elif aqi=='pm10_year_avg':
        levels = [np.arange(0,24,1),np.arange(0,24,1),np.arange(-5,5,0.5),np.arange(-10,10,1)];
    elif (aqi=='o3_year_avg') | (aqi=='SURF_ppb_O3-Sum'):
        levels = [np.arange(0,80,2),np.arange(0,80,2),np.arange(-10,10,1),np.arange(-20,20,1)];
    elif aqi=='SOMO35':
        levels = [np.arange(0,6001,1),np.arange(0,6001,1),np.arange(-4,4.2,0.2),np.arange(-20,21,1),np.arange(-15,16,1),np.arange(-15,16,1)];
        Range = [[0,25],[0,25],[-2,2],[-10,10],[-10,10],[-10,10]];
    elif aqi=='SURF_MAXO3':
        levels = [np.arange(0,61,1),np.arange(0,61,1),np.arange(-10,10.2,0.2),np.arange(-20,21,1),np.arange(-15,16,1),np.arange(-15,16,1)];
    elif ('O3' in aqi):
        levels = [np.arange(0,91,1),np.arange(0,91,1),np.arange(-4,4.2,0.2),np.arange(-20,21,1),np.arange(-15,16,1),np.arange(-15,16,1)];
        Range = [[0,80],[0,80],[-2,2],[-10,10],[-20,20],[-20,20]];
        # colors=['plt.cm.Reds','plt.cm.Reds','plt.cm.bwr','plt.cm.bwr','plt.cm.bwr','plt.cm.bwr'];
    elif ('PM25' in aqi):
        levels = [np.arange(0,26,1),np.arange(0,26,1),np.arange(-4,4.2,0.2),np.arange(-10,11,1),np.arange(0,16,1),np.arange(0,16,1)];
        Range = [[0,25],[0,25],[-2,2],[-10,10],[-10,10],[-10,10]];
        # colors=['plt.cm.plasma_r','plt.cm.plasma_r','plt.cm.seismic','plt.cm.seismic','plt.cm.inferno_r','plt.cm.inferno_r'];
    elif ('DEP' in aqi) :
        levels = [np.arange(0,1800,100),np.arange(0,1800,100),np.arange(-50,2,50),np.arange(-10,11,1),np.arange(0,16,1),np.arange(0,16,1)];
        #Range = [[0,2500],[0,2500],[-2,2],[-10,10],[-10,10],[-10,10]];
    elif ('SIA' in aqi) | (aqi=='SURF_ug_NH4_F'):
        levels = [np.arange(0,26,1),np.arange(0,26,1),np.arange(-4,4.2,0.2),np.arange(-10,11,1),np.arange(0,16,1),np.arange(0,16,1)];
        Range = [[0,25],[0,25],[-2,2],[-10,10],[-10,10],[-10,10]];        
    elif ('NH3' in aqi):
        levels = [np.arange(0,26,1),np.arange(0,26,1),np.arange(-4,4.2,0.2),np.arange(-10,11,1),np.arange(0,16,1),np.arange(0,16,1)];
        Range = [[0,25],[0,25],[-2,2],[-10,10],[-10,10],[-10,10]];  
    elif aqi=='SURF_ug_NO2':
        levels = [np.arange(0,60,1),np.arange(0,60,1),np.arange(-5,5,1),np.arange(-10,11,1),np.arange(0,20,1),np.arange(0,20,1)];
    elif (aqi=='SURF_ug_NOx') | (aqi=='SURF_ug_NO'):
        levels = [np.arange(0,80,1),np.arange(0,80,1),np.arange(-5,5,1),np.arange(-10,11,1),np.arange(0,40,1),np.arange(0,40,1)];
    elif (aqi=='NO2eq') | (aqi=='SURF_ug_NO2-Yea') | ('NOx' in aqi):
        levels = [np.arange(0,91,1),np.arange(0,91,1),np.arange(-4,4.2,0.2),np.arange(-20,21,1),np.arange(0,41,1),np.arange(0,41,1)];
    elif aqi=='no2':
        levels = [np.arange(0,20,1),np.arange(0,20,1),np.arange(-5,5,0.5),np.arange(-30,30,3)];
    elif aqi=='AOT40':
        levels = [np.arange(30000,90000,5000),np.arange(30000,90000,5000),np.arange(-10000,10000,1000),np.arange(-8,8,1)];
    elif aqi=='MAX8H':
        levels = [np.arange(90,130,2),np.arange(90,130,2),np.arange(-5,5,1),np.arange(-4,4,0.5)];
    elif aqi=='SURF_ppb_SO2':
        levels = [np.arange(0,26,1),np.arange(0,26,1),np.arange(-4,4.2,0.2),np.arange(-10,11,1),np.arange(0,16,1),np.arange(0,16,1)];
        Range = [[0,25],[0,25],[-2,2],[-10,10],[-10,10],[-10,10]];
    else:
        levels = [np.arange(0,26,1),np.arange(0,26,1),np.arange(-4,4.2,0.2),np.arange(-10,11,1),np.arange(0,16,1),np.arange(0,16,1)];
        Range = [[0,25],[0,25],[-2,2],[-10,10],[-10,10],[-10,10]];

    for i in range(0, 6):
        # scatter
        h = plt.figure(1);

        matVal = eval(mapInfo[i]);

        #matvmax = Range[i][1];
        #matvmin = Range[i][0];

        if (i>1) & (i<4):
            matVal[bctarget<thresGraphs]=np.nan

        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.contourf(x, y, matVal, transform=ccrs.PlateCarree())
        #ax.coastlines()
        ax.gridlines()
        plt.colorbar()
#        plt.show()
#
#        ax = plt.axes()
#        plt.contourf(lons, lats, sst, 60,
#             transform=ccrs.PlateCarree())
#
#        ax.coastlines()
#
#        map = Basemap(llcrnrlon=-25, llcrnrlat=30, urcrnrlon=56., urcrnrlat=75., resolution='l')
#        map.drawmapboundary()
#        map.drawcountries()
#        map.drawcoastlines()
#
#        if platform.system() == 'Windows':
#            xx, yy = map(x.data, y.data)
#        elif platform.system() == 'Linux':
#            xx, yy = map(x, y)
#
#        # map.contourf(xx, yy, matVal, levels[i], cmap=eval(colors[i]));
#        map.pcolormesh(xx, yy, matVal, cmap=eval(colors[i]), vmin=levels[i][0], vmax=levels[i][-1]);
#        map.colorbar();

        titleString='minval= ' + str(round(np.nanmin(matVal),2)) + ', maxval= ' + str(round(np.nanmax(matVal),2))
        plt.title(titleString)

        nameScatter = nomeDir+nameFile[i]+str(iSc);
        print(nameScatter);

        h.savefig(nameScatter+'_py.png', format="png", dpi=600);
        plt.close(h);

'''
Created on 25-nov-2016

@author: roncolato
'''
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec


def CreateScatter(bctarget,target,output,flagRegioMat,iSc,nx,ny,nomeDir,aqi,absdel,domain,conf,thresGraphs):
    #matplotlib.use("Agg");
    # independent scenario validation scatter
    
    # create scatter for abs and delta values
    nameFile = ('abs-sce-n','del-sce-n');
    if absdel==0:
        scatterTarget = ('target','bctarget-target');
        scatterOutput = ('output','bctarget-output');
    elif absdel==1:
        scatterTarget = ('bctarget-target','target');
        scatterOutput = ('bctarget-output','output');
    
    if ('PM10' in aqi):
        axisBound = ([0,40,0,40],[0,20,0,20]); #for opera
        #axisBound={[0 50 0 50],[-5 40 -5 40]}; %only for riat-lomb
    elif aqi=='pm10_year_avg':
        axisBound = ([0,30,0,30],[-5,15,-5,15]);
    elif aqi == 'SOMO35':
        axisBound = ([0, 6000, 0, 6000], [-1000, 1000, -1000, 1000]);
    elif aqi == 'SURF_MAXO3':
        axisBound = ([0, 60, 0, 60], [-10, 10, -10, 10]);
    elif ('O3' in aqi):
        axisBound = ([0,80,0,80],[-20,20,-20,20]);
    elif (aqi=='o3_year_avg') | (aqi=='SURF_ppb_O3-Sum'):
        axisBound = ([0,80,0,80],[-30,30,-30,30]);
    elif ('PM25' in aqi):
        axisBound = ([0,60,0,60],[0,25,0,25]);
    elif ('SIA' in aqi):
        axisBound = ([0,60,0,60],[0,25,0,25]);
        #axisBound = ([0 20 0 20],[0 10 0 10]); %only for riat-lomb
    elif ('DEP' in aqi) :
        axisBound = ([0,1800,0,1800],[0,400,0,400]);
    elif ('NH3' in aqi) | (aqi=='SURF_ug_NH4_F'):
        axisBound = ([0,60,0,60],[0,25,0,25]);        
    elif aqi=='SURF_ug_NO2':
        axisBound = ([0,60,0,60],[0,20,0,20]);
    elif aqi=='SURF_ug_NOx':
        axisBound = ([0,80,0,80],[0,40,0,40]);
    elif aqi=='NO2eq':
        axisBound = ([-10,200,-10,200],[-5,60,-5,60]);
    elif (aqi=='no2') | (aqi=='SURF_ug_NO2-Yea') | ('NOx' in aqi) | ('NO' in aqi):
        axisBound = ([0,80,0,80],[0,40,0,40]);
    elif aqi=='AOT40':
        axisBound = ([30000,90000,30000,90000],[-20000,20000,-20000,20000]);
    elif aqi=='MAX8H':
        axisBound = ([90,130,90,130],[-10,10,-10,10]);
    elif aqi=='SURF_ppb_SO2':
        axisBound = ([0,40,0,40],[-10,10,-10,10]);
    elif aqi=='SURF_ug_SO4':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    elif aqi=='SURF_ug_NO3_F':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    elif aqi=='SURF_ug_NH4_F':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    elif aqi=='SURF_ug_PM_OM25':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    elif aqi=='SURF_ug_PPM2.5':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    elif aqi=='SURF_ug_PM_OM25':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    elif aqi=='SURF_ug_PPM2.5':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    elif aqi=='SURF_ug_ECFINE':
        axisBound = ([0,40,0,40],[-5,5,-5,5]);
    else:
        axisBound = ([0,60,0,60],[0,25,0,25]);
        
    for i in range(0, 2):
        # scatter
        h = plt.figure(1);
        xgraph = eval(scatterTarget[i]);
        ygraph = eval(scatterOutput[i]);

        xgraph[bctarget<thresGraphs]=np.nan                     
        ygraph[bctarget<thresGraphs]=np.nan
                     
        #xgraph[flagRegioMat==0] = np.nan;
        #ygraph[flagRegioMat==0] = np.nan;
        xgraph = np.ravel(xgraph, order='F')[np.ravel(flagRegioMat, order='F')!=0];
        ygraph = np.ravel(ygraph, order='F')[np.ravel(flagRegioMat, order='F')!=0];
        #xgraph = np.reshape(xgraph,(xgraph.shape[0]*xgraph.shape[1],0));
        #ygraph = np.reshape(ygraph,(ygraph.shape[0]*ygraph.shape[1],0));
        
        plt.plot(xgraph,ygraph,'r*');
        
        plt.axis(axisBound[i]);
        plt.grid(True);
        plt.plot([min(np.nanmin(xgraph),np.nanmin(ygraph)),max(np.nanmax(xgraph),np.nanmax(ygraph))],[min(np.nanmin(xgraph),np.nanmin(ygraph)),max(np.nanmax(xgraph),np.nanmax(ygraph))],'b--');
        maxPlot = axisBound[i][1]
        plt.plot([min(np.nanmin(xgraph),np.nanmin(ygraph)),maxPlot],[min(np.nanmin(xgraph),np.nanmin(ygraph)),maxPlot+0.1*maxPlot],'b--');
        plt.plot([min(np.nanmin(xgraph),np.nanmin(ygraph)),maxPlot],[min(np.nanmin(xgraph),np.nanmin(ygraph)),maxPlot-0.1*maxPlot],'b--');
        #plt.show();
        
        #axis_font = {'fontname':'Arial', 'size':'20'};
        axis_font = {'size':'20'};
        plt.xlabel('Delta AQM model',axis_font);
        plt.ylabel('Delta SR model',axis_font);
        # We change the fontsize of minor ticks label 
        #plt.tick_params(axis='both', which='major', labelsize=10)
        #plt.tick_params(axis='both', which='minor', labelsize=8)
    
        # statistics
        xgraph = xgraph[ygraph!=np.nan];
        ygraph = ygraph[ygraph!=np.nan];
        corr_reg = 0;
        mse_reg = 0;
        matplotlib.rcParams.update({'font.size': 12});
        #set(gca, 'FontSize', 16);
        
        # save
        nameScatter = nomeDir + nameFile[i] + str(iSc);
        #print(h,'-dpng',nameScatter);
        #print('filename: '+nameScatter+'.png');
        #plt.savefig(h, nameScatter+'_py.png', format="png");
        h.savefig(nameScatter+'_py.png', format="png", dpi=300);
        plt.close(h);
        #close all

#         h = plt.figure(2);
#         #h = plt.figure(figsize=(8, 6));
#         gs = gridspec.GridSpec(5, 4);
#         #gs.update(top=0.2, wspace=0.1);
#         matplotlib.rcParams.update({'font.size': 8});
# #        vecChose = [4,5,6,11,12,13,15,16,17,18,20,22,23,24,26,27,32,34,40,41,43,45,48,49,50];
# #        vecChose = [0,1,2,5,6,7,8,9,10,11,12,13,14,15,17,18,19,21,25,26,27,28,29,30,31]
#         #country labels AT	BE BG CH	CY CZ	DE DK	EE EL	ES FI	FR HR	HU IE	IS IT	LI LT	LU LV	ME MK	MT NL	NO PL	PT RO	SE SI	SK UK
#         vecChose = [0,1,2,5,6,7,9,10,11,12,13,14,17,25,26,27,28,29,30,31]
#
#         #vecChose = np.subtract(vecChose,1);
#         # 33vec lux, 38 mlt, 10 cyp
#         icel = 0;
#         for cont in range(0, len(vecChose)):
#             #sub = plt.subplot(5,5,icel);
#             sub = plt.subplot(gs[icel]);
#             xgraph = eval(scatterTarget[i]);
#             ygraph = eval(scatterOutput[i]);
#
#             xgraph[bctarget<thresGraphs]=np.nan
#             ygraph[bctarget<thresGraphs]=np.nan
#
#             scatterRange = [min(np.nanmin(xgraph),np.nanmin(ygraph)), max(np.nanmax(xgraph),np.nanmax(ygraph)), min(np.nanmin(xgraph),np.nanmin(ygraph)), max(np.nanmax(xgraph),np.nanmax(ygraph))];
#             ind = np.where(countryIDmap==vecChose[cont]);
#             xgraph = xgraph[ind];
#             ygraph = ygraph[ind];
#
#             sub.grid(True);
#             #sub.grid(which='minor', alpha=0.2)
#             sub.plot(xgraph,ygraph,'r*');
#             #grid on;hold on;
#             sub.plot([min(np.nanmin(xgraph),np.nanmin(ygraph)), max(np.nanmax(xgraph),np.nanmax(ygraph))],[min(np.nanmin(xgraph),np.nanmin(ygraph)), max(np.nanmax(xgraph),np.nanmax(ygraph))],'b--');
#             # +-10%
#             sub.plot([min(np.nanmin(xgraph),np.nanmin(ygraph)), max(np.nanmax(xgraph),np.nanmax(ygraph))],[min(np.nanmin(xgraph),np.nanmin(ygraph))*1.10, max(np.nanmax(xgraph),np.nanmax(ygraph))*1.05],'b--');
#             sub.plot([min(np.nanmin(xgraph),np.nanmin(ygraph)), max(np.nanmax(xgraph),np.nanmax(ygraph))],[min(np.nanmin(xgraph),np.nanmin(ygraph))*0.90, max(np.nanmax(xgraph),np.nanmax(ygraph))*0.95],'b--');
#
#             #sub.axis(scatterRange);
#             sub.axis(axisBound[i]);
#             sub.set_title(vec[vecChose[cont]]);
#             #plt.title(vec[vecChose[cont]]);
#             icel=icel+1;
#
#         #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9);
#         plt.tight_layout();
#
#         nameScatter = nomeDir+nameFile[i]+'-sp-'+str(iSc);
#         #print(h,'-dpng',nameScatter);
#         #plt.savefig(h,nameScatter+'.png', format="png");
#         h.savefig(nameScatter+'_py.png', format="png");
#         plt.close(h);
#         #plt.close('all');
#         #close all
    
    #print('fine CreateScatter');
    return (corr_reg, mse_reg);


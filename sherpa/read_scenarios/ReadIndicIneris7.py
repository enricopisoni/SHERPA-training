'''
Created on 18-nov-2016
read concentrations
@author: roncolato
'''
import netCDF4 as cdf
import numpy as np
#import sherpa.read_scenarios.correctionPoValley as po

def ReadIndicIneris7(nSc,nPrec,domain,aqiFil,aqiVar,absdel,nx,ny,conf):
    #intialize variables
    IndicTmp = np.zeros((ny,nx,nSc));
    Indic = np.zeros((ny,nx,nSc));
                    
    #read precursor files
    for sce in range(0, nSc):
        fileName = conf.scenConcFileName(sce);
        fh = cdf.Dataset(fileName, mode='r');
        if aqiVar == 'SURF_ug_NOx' :
            tmpMat = np.squeeze(fh.variables['SURF_ug_NO'][:]).transpose(); #when loading, I have 'nx,ny,month';
            tmpMat2 = np.squeeze(fh.variables['SURF_ug_NO2'][:]).transpose(); #when loading, I have 'nx,ny,month';
            tmpMat = tmpMat + tmpMat2
        else:
            tmpMat = np.squeeze(fh.variables[aqiVar][:]).transpose(); #when loading, I have 'nx,ny,month';
            if conf.yearmonth==1: #case for monthly values
                if conf.whichmonth=='DJF':
                    tmpMat = np.mean(tmpMat[:,:,[0,1,11]], axis=2)
                elif conf.whichmonth=='MAM':
                    tmpMat = np.mean(tmpMat[:,:,[2,3,4]], axis=2)    
                elif conf.whichmonth=='JJA':
                    tmpMat = np.mean(tmpMat[:,:,[5,6,7]], axis=2)                    
                elif conf.whichmonth=='SON':
                    tmpMat = np.mean(tmpMat[:,:,[8,9,10]], axis=2)    
            
        fh.close();
        
        if aqiVar == 'O3': #convert from ppb to mg/m3
            tmpMat = tmpMat *2
                
        if aqiFil.find('O3_daymax8hr')>-1:
            tmpMat = tmpMat[:,:,90:273].mean(axis=2);
        IndicTmp[:,:,sce] = np.flipud(tmpMat.transpose());


    IndicBC = IndicTmp[:,:,0];
    
    #store final results
    if absdel == 0:
        Indic = IndicTmp;
    elif absdel == 1: #absolute values
        for j in range(0, nSc):
            Indic[:,:,j] = IndicTmp[:,:,0] - IndicTmp[:,:,j];

    return (Indic, IndicBC)


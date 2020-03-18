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
        fh.close();
        
        if aqiVar == 'O3': #convert from ppb to mg/m3
            tmpMat = tmpMat *2
                
        if aqiFil.find('O3_daymax8hr')>-1:
            tmpMat = tmpMat[:,:,90:273].mean(axis=2);
        IndicTmp[:,:,sce] = np.flipud(tmpMat.transpose());

    #data fusion on Po Valley, to correct PM25 and PM10
    # if conf.domain == 'emep10km':
    #     correctionFlag = 0;
    # elif conf.domain == 'ineris7km':
    #     correctionFlag = 1;
    #
    # if aqiVar == 'PM25':
    #     flagPm25Pm10 = 2;
    # elif aqiVar == 'PM10':
    #     flagPm25Pm10 = 3;
    # elif aqiVar == 'NO2eq':
    #     flagPm25Pm10 = 1;
    # elif (aqiVar == 'O3') | (aqiVar == 'SOMO35'):
    #     flagPm25Pm10 = 1;
    #
    # if correctionFlag == 1:
    #     IndicTmp = po.correctionPoValley(IndicTmp, flagPm25Pm10); # flagPm25Pm10==0 means PM25, 1 means PM10

    IndicBC = IndicTmp[:,:,0];
    
    #store final results
    if absdel == 0:
        Indic = IndicTmp;
    elif absdel == 1: #absolute values
        for j in range(0, nSc):
            Indic[:,:,j] = IndicTmp[:,:,0] - IndicTmp[:,:,j];

    return (Indic, IndicBC)


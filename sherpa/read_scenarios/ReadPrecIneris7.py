'''
Created on 18-nov-2016
read emissions
@author: roncolato
'''
import netCDF4 as cdf
import numpy as np
import platform
#from pprint import pprint

def ReadPrecIneris7(nSc,nPrec,domain,absdel,POLLSEL,emiDenAbs,aqiFil,conf):
    #variable to be read
    
    if conf.domain == 'emep10km':
        precVec = ['Emis_mgm2_nox-Yea','Emis_mgm2_voc-Yea','Emis_mgm2_nh3-Yea','Emis_mgm2_pm25-Yea','Emis_mgm2_sox-Yea'];
    elif conf.domain == 'ineris7km':
        precVec = ['annualNOx','annualNMVOC','annualNH3','annualPM25','annualSOx'];
    elif conf.domain == 'emepV433_camsV221':
        precVec = ['Emis_mgm2_nox','Emis_mgm2_voc','Emis_mgm2_nh3','Emis_mgm2_pm25','Emis_mgm2_sox'];

    flagLL = 0;
    
    #loop over scenarios
    for sce in range(0, nSc):

        #define filename and open netcdf
        fileName = conf.scenEmissionFileName(sce); #'input/'+domain+'/2010Cle_TSAP_Dec_2013_JRC'+sces+'_07b_2009/JRC'+sces+'.nc';
        fh = cdf.Dataset(fileName, mode='r');
                        
        for pre in range(0, nPrec):
            #store latlon
            if flagLL==0:
                if platform.system() == 'Windows':
                    lat = np.squeeze(fh.variables['lat'][:]).transpose();
                    lon = np.squeeze(fh.variables['lon'][:]).transpose();
                elif (platform.system() == 'Linux') and ('emep10km' in fileName):
                    lat = np.squeeze(fh.variables['lat'][:]).transpose();
                    lon = np.squeeze(fh.variables['lon'][:]).transpose();
                elif (platform.system() == 'Linux') and ('ineris' in fileName):
                    lat = np.squeeze(fh.variables['lat'][:]).transpose();
                    lon = np.squeeze(fh.variables['lon'][:]).transpose();

                ny = lat.shape[0];
                nx = lon.shape[0];
                lat = np.kron(np.ones((nx, 1)), np.flipud(lat.transpose())).transpose();
                lon = np.kron(np.ones((ny, 1)), lon.transpose());
                flagLL=1;
                Prec = np.zeros((ny,nx,nSc,nPrec));        
            
            #read variable                       
            tmpMat = np.squeeze(fh.variables[precVec[pre]][:]).transpose();
            
            #convert in case of total emissions     
            if conf.domain == 'ineris7km':
                surfaceValues = fh.variables['surface']; # read surface values
            
            if emiDenAbs==1: # from ton/km2 to ton/cell
                tmpMat = tmpMat * np.tile(surfaceValues,(1,1,10));

            #do this in case of ozone   
            if (conf.domain == 'emep10km') | (conf.domain == 'emepV433_camsV221'):
                tmp = tmpMat
            elif conf.domain == 'ineris7km':
                tmp = np.sum(tmpMat, 2); # APR-SET
                
            if aqiFil.find('O3_daymax8hr')>-1:
                tmp = np.sum(tmpMat, 2);
                tmp = np.sum(tmp[:, :, 1, 4:9], 3);
            
            #store emission in final variable
            Prec[:,:,sce,pre] = np.flipud(tmp.transpose());
            # Prec[:, :, sce, pre] = tmp.transpose();
        fh.close();

    # in case of PM10 replace PPM variable
    if POLLSEL==2:
        if conf.domain == 'emep10km':
            precVec=['Emis_mgm2_pmco-Yea'];
        elif conf.domain == 'emepV433_camsV221':
            precVec = ['Emis_mgm2_pmco'];
        elif conf.domain == 'ineris7km':
            precVec=['annualPMcoarse'];       

        flagLL=0;
        for sce in range(0, nSc):
            fileName = conf.scenEmissionFileName(sce); #'input/'+domain+'/2010Cle_TSAP_Dec_2013_JRC'+sces+'_07b_2009/JRC'+sces+'.nc';
            fh = cdf.Dataset(fileName, mode='r');
            for pre in range(3, 4):
                tmpMat = np.squeeze(fh.variables[precVec[0]][:]).transpose();

                if emiDenAbs==1: # from ton/km2 to ton/cell
                    surfaceValues = fh.variables['surface'];  # read surface values
                    tmpMat = tmpMat * np.tile(surfaceValues,(1,1,10));

                if (conf.domain == 'emep10km') | (conf.domain == 'emepV433_camsV221'):
                    tmp = tmpMat
                elif conf.domain == 'ineris7km':
                    tmp = np.sum(tmpMat, 2); # APR-SET
                                
#                tmp = np.sum(tmpMat, 2); # sum per ms
                Prec[:,:,sce,pre] = Prec[:,:,sce,pre] + np.flipud(tmp.transpose());            
            fh.close();

    # create final matrix - Prec(ny,nx,nSc,nPrec)
    PrecFinal = np.zeros(np.shape(Prec));
    for i in range(0, nPrec):
        for j in range(0, nSc):
            if absdel==0: # absolute values
                PrecFinal[:,:,j,i] = Prec[:,:,j,i];
            elif absdel==1: #delta values, bc-sce
                PrecFinal[:,:,j,i] = Prec[:,:,0,i] - Prec[:,:,j,i];

    return (lon,lat,nx,ny,PrecFinal)

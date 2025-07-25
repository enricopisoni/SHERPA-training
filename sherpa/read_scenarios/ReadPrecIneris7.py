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
    elif (conf.domain == 'emepV433_camsV221') | (conf.domain == 'edgar2015') | (conf.domain == 'emepV434_camsV42') \
        | ('cams' in conf.domain):
        precVec = ['Sec_Emis_mgm2_nox','Sec_Emis_mgm2_voc','Sec_Emis_mgm2_nh3','Sec_Emis_mgm2_pm25','Sec_Emis_mgm2_sox'];
    elif (conf.domain == 'emep4nl_2021'):
        precVec = ['Sec_Emis_mgm2_nox','Sec_Emis_mgm2_voc','Sec_Emis_mgm2_nh3','Sec_Emis_mgm2_pm25','Sec_Emis_mgm2_sox'];
    elif ('wrf' in conf.domain):
        precVec = ['E_NOx_sumsec','E_VOC_sumsec','E_NH3_sumsec','E_PM25_sumsec','E_SO2_sumsec'];
    elif (conf.domain == 'EMEP_45_CAMSv80_01005'):
        precVec = ['Sec_Emis_mgm2_nox','Sec_Emis_mgm2_voc','Sec_Emis_mgm2_nh3','Sec_Emis_mgm2_pm25','Sec_Emis_mgm2_sox'];


    flagLL = 0;
    
    #loop over scenarios
    for sce in range(0, nSc):

        #define filename and open netcdf
        print(sce)
        fileName = conf.scenEmissionFileName(sce); #'input/'+domain+'/2010Cle_TSAP_Dec_2013_JRC'+sces+'_07b_2009/JRC'+sces+'.nc';
        fh = cdf.Dataset(fileName, mode='r');
                        
        for pre in range(0, nPrec):
            #store latlon
            if flagLL==0:
                if (platform.system() == 'Windows') & ('cams' in conf.domain):
                    lat = np.squeeze(fh.variables['lat'][:]).transpose();
                    lon = np.squeeze(fh.variables['lon'][:]).transpose();
                    ny = lat.shape[0];
                    nx = lon.shape[0];
                    lat = np.kron(np.ones((nx, 1)), np.flipud(lat.transpose())).transpose();
                    lon = np.kron(np.ones((ny, 1)), lon.transpose());
                    flagLL=1;
                if (platform.system() == 'Windows') & ('wrf' in conf.domain):
                    
                    lat = np.flipud(fh.variables['lat'][:])
                    lon = np.flipud(fh.variables['lon'][:])
                    
                    ny = lat.shape[0];
                    nx = lon.shape[0];
                    flagLL=1;
                    if ('wrf' in conf.domain):
                        lat = np.tile(lat,(nx,1)).T
                        lon = np.fliplr(np.tile(lon,(ny,1)))
                elif (platform.system() == 'Linux'):
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
            if conf.yearmonth==1: #case for monthly values
                if conf.whichmonth=='DJF':
                    tmpMat = np.sum(tmpMat[:,:,[0,1,11]], axis=2)
                elif conf.whichmonth=='MAM':
                    tmpMat = np.sum(tmpMat[:,:,[2,3,4]], axis=2)    
                elif conf.whichmonth=='JJA':
                    tmpMat = np.sum(tmpMat[:,:,[5,6,7]], axis=2)                    
                elif conf.whichmonth=='SON':
                    tmpMat = np.sum(tmpMat[:,:,[8,9,10]], axis=2)    
                elif conf.whichmonth=='AMJJAS':
                    tmpMat = np.sum(tmpMat[:,:,[3,4,5,6,7,8]], axis=2)    
                elif conf.whichmonth=='ONDJFM':
                    tmpMat = np.sum(tmpMat[:,:,[0,1,2,9,10,11]], axis=2)    
                    

            #convert from mg/km2 to ton/grid - this is the case for CAMS-EMEP, and EDGAR                  
            tmpMat = tmpMat/1000
            
            #convert in case of total emissions     
            if emiDenAbs==1: # from ton/km2 to ton/cell
                surfaceValues = np.squeeze(fh.variables['Area_Grid_km2'][:]).transpose();  # read surface values
                #to kton/km2 dividing by 100, to kton/year multiplying by the area
                tmpMat = tmpMat/1000
                tmpMat = tmpMat * surfaceValues;

            #do this in case of ozone   
            if ('emep' in conf.domain) |  (conf.domain == 'edgar2015') | ('wrf' in conf.domain):
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
        elif (conf.domain == 'emepV433_camsV221') | (conf.domain == 'edgar2015') | ('cams' in conf.domain) :
            precVec = ['Sec_Emis_mgm2_pmco'];
        elif conf.domain == 'ineris7km':
            precVec=['annualPMcoarse'];       
        elif ('wrf' in conf.domain):
            precVec = ['E_PPMco_sumsec'];


        flagLL=0;
        for sce in range(0, nSc):
            fileName = conf.scenEmissionFileName(sce); #'input/'+domain+'/2010Cle_TSAP_Dec_2013_JRC'+sces+'_07b_2009/JRC'+sces+'.nc';
            fh = cdf.Dataset(fileName, mode='r');
            for pre in range(3, 4):
                tmpMat = np.squeeze(fh.variables[precVec[0]][:]).transpose();
                
            if conf.yearmonth==1: #case for monthly values
                if conf.whichmonth=='DJF':
                    tmpMat = np.sum(tmpMat[:,:,[0,1,11]], axis=2)
                elif conf.whichmonth=='MAM':
                    tmpMat = np.sum(tmpMat[:,:,[2,3,4]], axis=2)    
                elif conf.whichmonth=='JJA':
                    tmpMat = np.sum(tmpMat[:,:,[5,6,7]], axis=2)                    
                elif conf.whichmonth=='SON':
                    tmpMat = np.sum(tmpMat[:,:,[8,9,10]], axis=2)    
                elif conf.whichmonth=='AMJJAS':
                    tmpMat = np.sum(tmpMat[:,:,[3,4,5,6,7,8]], axis=2)    
                elif conf.whichmonth=='ONDJFM':
                    tmpMat = np.sum(tmpMat[:,:,[0,1,2,9,10,11]], axis=2)    
 
                        
                #convert from mg/m2 to ton/km2 - this is the case for CAMS-EMEP, and EDGAR                  
                tmpMat = tmpMat/1000
    
                #convert in case of total emissions     
                if emiDenAbs==1: # from ton/km2 to ton/cell
                    surfaceValues = np.squeeze(fh.variables['Area_Grid_km2'][:]).transpose();  # read surface values
                    #to kton/km2 dividing by 100, to kton/year multiplying by the area
                    tmpMat = tmpMat/1000
                    tmpMat = tmpMat * surfaceValues;

                if ('emep' in conf.domain) |  (conf.domain == 'edgar2015') | ('wrf' in conf.domain):
                # if (conf.domain == 'emep10km') | (conf.domain == 'emepV433_camsV221') | (conf.domain == 'edgar2015') | (conf.domain == 'emepV434_camsV42') \
                #     | (conf.domain =='emepV434_camsV42withCond_01005'):
                    tmp = tmpMat
                elif conf.domain == 'ineris7km':
                    tmp = np.sum(tmpMat, 2); # APR-SET
                                
#                tmp = np.sum(tmpMat, 2); # sum per ms
                Prec[:,:,sce,pre] = Prec[:,:,sce,pre] + np.flipud(tmp.transpose());            
            fh.close();

    # create final matrix - Prec(ny,nx,nSc,nPrec)
    PrecFinal = np.zeros(np.shape(Prec));
    
    #EP20210218 - NH3 on London in CAMS is wrong, as they grid all UK NH3 waste management on a single point source in London
    #so I do average of the surrounding points, and replace that point of London with the average
    #if conf.domain == 'emepV433_camsV221' :
    #    for sce in range(0, nSc) :
    #        Prec[199,151,sce,2] = np.mean(Prec[198:201,150:153,sce,2]) #NH3 for London for waste management
    #        Prec[337,301,sce,4] = np.mean(Prec[336:339,300:303,sce,4]) #SOx for Catania for residential sector

    
    for i in range(0, nPrec):
        for j in range(0, nSc):
            if absdel==0: # absolute values
                PrecFinal[:,:,j,i] = Prec[:,:,j,i];
            elif absdel==1: #delta values, bc-sce
                PrecFinal[:,:,j,i] = Prec[:,:,0,i] - Prec[:,:,j,i];

    return (lon,lat,nx,ny,PrecFinal)

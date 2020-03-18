'''
Created on 13-mar-2017

@author: roncolato
'''
import numpy as np
import scipy.io as sio

from sherpa.validation import CreateScatter as cs
from sherpa.validation import CreateMap as cm
from sherpa.validation import saveToNetcdf as sm
import sherpa.training.funcAggreg as fa
import time
from sherpa.training import distanceComputation as dc
from itertools import repeat, starmap
import multiprocessing as mp


def computeOutput(ir,ic,omega,PrecToBeUsed,vecPrecompF_for_map,Ftmp_for_map,Prec2,rad, alpha,poly,lat):
    # print(ir,ic)
    coeff = np.squeeze(omega[ir, ic, [PrecToBeUsed]]);  # could vary from 0.5 to 5
    colToExt = (np.searchsorted(vecPrecompF_for_map, coeff)).tolist()

    dimrad = rad*2+1;

    ratio = np.polyval(poly, lat[ir, ic])
    # print('val')
    # print(lat[ir, ic], ratio)

    Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
    Ftmp = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));
    F = np.zeros((dimrad, dimrad, len(PrecToBeUsed)));
    for poll in range(0, len(PrecToBeUsed)):
        if coeff.size == 1:
            F[:, :, poll] = Ftmp**coeff
        else:
            F[:, :, poll] = Ftmp**coeff[poll]

    # F = Ftmp_for_map[:, :, colToExt]
    PrecDummyQuad = np.squeeze(Prec2[ir:ir + rad + rad + 1, ic:ic + rad + rad + 1, :]);

    if coeff.size == 1:
        PrecPatch = np.sum(PrecDummyQuad * (np.squeeze(F)))
        output = PrecPatch * (np.squeeze(alpha[ir, ic, [PrecToBeUsed]]));
    else:
        PrecPatch = np.sum(np.sum(PrecDummyQuad * F, 0), 0);
        output = PrecPatch.dot(np.squeeze(alpha[ir, ic, [PrecToBeUsed]]));
    return output


def validation(conf):
    Prec = conf.Prec;

    PrecToBeUsed = conf.PrecToBeUsed;
    ny = conf.ny;
    nx = conf.nx;
    rad = conf.radStep2;
    nPrec = conf.nPrec;

    Prec2 = np.zeros((ny+rad*2, nx+rad*2, Prec.shape[2], Prec.shape[3]));
    Prec2[rad:-rad, rad:-rad, :, :] = Prec[:,:,:,:];
    Prec = Prec2;
    
    flagRegioMat = np.copy(conf.flagRegioMat);
    flat = conf.flat;
    Val = conf.Val;
    nameDirOut = conf.nameDirOut;
    nameRegFile = conf.nameRegFile;
    nSc = conf.nSc;
    
    mat = sio.loadmat(nameRegFile);
    alpha = mat.get('alpha');
    omega = mat.get('omega');
    
    if conf.mode=='V':
        conf.omegaFinalStep1 = omega;
        fa.funcAggreg(conf);
        
    if flat:
        flatWeight = mat.get('flatWeight');
    else:
        flatWeight = np.zeros((ny,nx,nPrec));

    #else:
        #sigX = mat.get('sigX');
        #sigY = mat.get('sigY');
        #thet = mat.get('thet');
        #XMin = mat.get('XMin');
        #XMax = mat.get('XMax');
        #yMin = mat.get('yMin');
        #yMax = mat.get('yMax');
    #bInt = mat.get('bInt');

    #if flat:
        # SAVING TO NETCDF
    rad = conf.vw; # only small change to save that the model is with 30 cells with varying weights
    sm.saveToNetcdf(alpha,omega,flatWeight,conf.x,
                                           conf.y,nameDirOut,conf.aqiFil,conf.domain,
                                           conf.radStep2,conf.vw,conf.flat,
                                           conf.rf2, conf.radStep1, conf.Ide, conf.flagRegioMatFile, conf.nametest,
                                           conf.explain_step_1, conf.explain_step_2,
                                           conf.Order_Pollutant, conf.alpha_physical_intepretation,
                                           conf.omega_physical_intepretation, conf.nPrec);

    rad = conf.radStep2; # go back to the 200 cells, as the matrix of weights in reality cover the 200 cells
    
    dimrad = rad*2+1;
    # VALIDATION BEGIN
    outputSherpa = np.zeros((ny, nx, nSc));#max(Val)));
    F = np.zeros((dimrad, dimrad, len(PrecToBeUsed)));

    if conf.distance == 1: #if distance in km, create these variables                
        LAT = conf.y;
        LON = conf.x      
        LAT2 = np.zeros((ny+rad*2, nx+rad*2));
        LAT2[rad:-rad, rad:-rad] = LAT[:,:];
        LAT = LAT2;    
        LON2 = np.zeros((ny+rad*2, nx+rad*2));
        LON2[rad:-rad, rad:-rad] = LON[:,:];
        LON = LON2;                                        
                
    for indVal in Val:#range(0, Val.shape[0]):
        
        # loop on scenarios
        iSc = indVal;#Val[indVal];
        print('Validating on the independent scenario n: _'+str(iSc));
        
        output = np.full([ny,nx],np.nan);
        #create update Prec to speed up calculations
        Prec2=Prec[:,:,iSc,[PrecToBeUsed]][:,:,0,:]

        for ic in range(0, nx):
            print(ic);
            for ir in range(0, ny): #use same routine for training and validation, with 0 for no sliding F
                if flagRegioMat[ir,ic]==1:

                # argslist = itertools.product(ir,ic)
                    vecPrecompF_for_map = conf.vecPrecompF
                    Ftmp_for_map = conf.Ftmp
            
                    coeff = np.squeeze(omega[ir, ic, [PrecToBeUsed]]);  # could vary from 0.5 to 5
                    colToExt = (np.searchsorted(vecPrecompF_for_map, coeff)).tolist()
                
                    dimrad = rad*2+1;
                
                    ratio = np.polyval(conf.ratioPoly, conf.y[ir, ic])
                
                    Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
                    Ftmp = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));
                    F = np.zeros((dimrad, dimrad, len(PrecToBeUsed)));
                    
                    for poll in range(0, len(PrecToBeUsed)):
                        if coeff.size == 1:
                            F[:, :, poll] = Ftmp**coeff
                        else:
                            F[:, :, poll] = Ftmp**coeff[poll]
                
                    # F = Ftmp_for_map[:, :, colToExt]
                    PrecDummyQuad = np.squeeze(Prec2[ir:ir + rad + rad + 1, ic:ic + rad + rad + 1, :]);
                
                    if coeff.size == 1:
                        PrecPatch = np.sum(PrecDummyQuad * (np.squeeze(F)))
                        output[ir,ic] = PrecPatch * (np.squeeze(alpha[ir, ic, [PrecToBeUsed]]));
                    else:
                        PrecPatch = np.sum(np.sum(PrecDummyQuad * F, 0), 0);
                        output[ir,ic]= PrecPatch.dot(np.squeeze(alpha[ir, ic, [PrecToBeUsed]]));

###
#        t1=time.time()
#        argslist = (zip(np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1],
#                        repeat(omega), repeat(PrecToBeUsed), repeat(vecPrecompF_for_map),
#                        repeat(Ftmp_for_map), repeat(Prec2),repeat(rad), repeat(alpha),
#                        repeat(conf.ratioPoly), repeat(conf.y)))
#        #USING MAP FUNCTION
#        # t1 = time.time()
#        # result = list(map(computeOutput, np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1],
#        #                 repeat(omega), repeat(PrecToBeUsed), repeat(vecPrecompF_for_map),
#        #                 repeat(Ftmp_for_map), repeat(Prec2),repeat(rad), repeat(alpha)))
#        # print(str(time.time() - t1))
#
#        pool = mp.Pool() #by default use available corse
#        print('***** Using parallel computing with ' + str(mp.cpu_count()) + ' cores *****')
#        result = pool.starmap_async(computeOutput, argslist)
#        pool.close()
#        pool.join()
#        print(str(time.time()-t1))
#        output[np.where(flagRegioMat)] = result.get()
####
        
        
        thresGraphs=1
        cs.CreateScatter(conf.IndicBC,conf.Indic[:,:,iSc],output,flagRegioMat,iSc,nx,ny,nameDirOut,conf.aqi,conf.absDel,conf.domain,conf,thresGraphs);
        
        # creating maps
        cm.CreateMap(conf.IndicBC,conf.Indic[:,:,iSc],output,flagRegioMat,conf.x,conf.y,iSc,nameDirOut,conf.aqi,conf.absDel,conf.flagReg,conf.domain,conf,thresGraphs);
        #close all
        
        outputSherpa[:,:,iSc] = output;
                    
    savefilename = conf.nameDirOut+'allResults.mat';
    sio.savemat(savefilename, {'IndicBC':conf.IndicBC, 'Indic':conf.Indic, 'outputSherpa':outputSherpa});
    #save allResults  IndicBC Indic outputSherpa

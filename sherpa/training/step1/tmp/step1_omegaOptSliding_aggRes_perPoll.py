'''
Created on 6-feb-2017
Modified the 20170321, by EP

@author: roncolato
'''
import numpy as np
import scipy.interpolate as interpol
import os
from sherpa.training.step1 import from7to28 as f7
from sherpa.training.step1 import quant as q
from sherpa.training.step1 import EquaPrec as ep
from sherpa.training import EquaIndic as ei
from sherpa.training.step1 import nlparci as nlpa
#from sherpa.training.step1 import InvDistN_opt_prec as inv
from sherpa.training.step1 import nlinfit as nlin
import time as t
from scipy.optimize import minimize
import scipy.ndimage as gf
import scipy.io as sio

from sklearn.metrics import mean_squared_error
from math import sqrt

def InvDistN_opt_prec(beta, i0, i1, i2, i3, i4, distanceMat):
    # t1 = t.time();
    F0 = distanceMat ** beta[5];
    F1 = distanceMat ** beta[6];
    F2 = distanceMat ** beta[7];
    F3 = distanceMat ** beta[8];
    F4 = distanceMat ** beta[9];
    # print("1: %s" % (t.time() - t1));
    # t1 = t.time();
    output = beta[0] * np.inner(i0, F0) + beta[1] * np.inner(i1, F1) + beta[2] * np.inner(i2, F2) + beta[3] * np.inner(i3, F3) + beta[4] * np.inner(i4, F4);
    # print("2: %s" % (t.time() - t1));
    return output;

def iop(beta, inp0, inp1, inp2, inp3, inp4, out0, distanceMat):
    # import matplotlib.pyplot as plt
    # t1 = t.time();
    x=InvDistN_opt_prec(beta, inp0, inp1, inp2, inp3, inp4, distanceMat)
    # print("3: %s" % (t.time() - t1));
    y=out0
    # plt.plot(x); plt.plot(y)
    # print(sqrt(mean_squared_error(x, y)))
    return sqrt(mean_squared_error(x, y))


def step1_omegaOptimization(conf):
    
    #convert from 28 to 7 km
#    Prec = conf.Prec;
#    ny = conf.ny;
#    nx = conf.nx;
    Prec = f7.from7to28(conf.Prec);
    ny = int(conf.ny / 4);
    nx = int(conf.nx / 4);

    rad = conf.radStep1;
    nPrec = conf.nPrec;
    rf = conf.rf1;
    flagRegioMat = np.copy(conf.flagRegioMat);
    
    #pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny+rad*2,nx+rad*2,Prec.shape[2],Prec.shape[3]));
    Prec2[rad:-rad,rad:-rad,:,:] = Prec[:,:,:,:];
    Prec=Prec2;
    
    #convert from 28 to 7 km
#    Indic = conf.Indic;
#    flagRegioMat = flagRegioMat;
    Indic = f7.from7to28(conf.Indic);
    flagRegioMat = f7.from7to28(flagRegioMat);
    lat = f7.from7to28(conf.y);
                      
    #initialize variables
    omega = np.full([ny,nx,nPrec],np.nan);
    alpha = np.full([ny,nx,nPrec],np.nan);
                   
    #precomputed vars
    Y, X = np.mgrid[-rad:rad+1:1, -rad:rad+1:1]; 
    distanceMat = 1 / ( (1 + (X**2 + Y**2)**(0.5)))
    distanceMat = distanceMat.flatten()
    
    #define training scenarios; note scenarios number is +1 if checking DoE...as in line 74 it is -1            
    if conf.domain == 'emep10km':
        if conf.aqi == 'SURF_ug_PM25_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 5]),np.array([1, 6]));
        elif conf.aqi == 'SURF_ug_PM10_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 4]),np.array([1, 6]));
    elif conf.domain == 'ineris7km':
        IdeVec = (np.array([1, 2]),np.array([1, 3]),np.array([1, 4]),np.array([1, 5]),np.array([1, 6]));
    elif conf.domain == 'emepV433_camsV221':
        IdeVec = (np.array([1, 1]), np.array([1, 2]), np.array([1, 3]), np.array([1, 4]), np.array([1, 5]));

    for precursor in range(0, nPrec):
        Ide = conf.Ide
        PREC = precursor;
        Ide = IdeVec[precursor];

        for ic in range(0, nx):
            print(ic, nx);      
            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
                    
    #                ratio = np.polyval(conf.ratioPoly, conf.y[ir, ic])
                    ratio = np.polyval(conf.ratioPoly, lat[ir, ic])
                    
                    Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
                    distanceMat = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));
                    distanceMat = distanceMat.flatten()
    
                    
    #                ratio = np.polyval(conf.ratioPoly, lat[ir, ic])
    #                Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
    #                F = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));
    #
    #                Y, X = np.mgrid[-rad:rad+1:1, -rad:rad+1:1]; 
    #                distanceMat = 1 / ( (1 + (X**2 + Y**2)**(0.5)))
    #                distanceMat = distanceMat.flatten()
        
                    x0=[1,1,1,1,1,1.5,1.5,1.5,1.5,1.5]
                    
                    nSc = Ide.shape[0]#-1;# size(Ide,2)-1
                    
    #                t1=t.time(); 
                    inp0 = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide,0],rad); # patches
                    inp1 = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide,1],rad); # patches
                    inp2 = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide,2],rad); # patches
                    inp3 = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide,3],rad); # patches
                    inp4 = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide,4],rad); # patches
    #                print("1: %s" % (t.time() - t1));
    
    #                t1=t.time(); 
                    out0 = ei.EquaIndic(ic,ir,rf,nx,ny,nSc,Indic[:,:,Ide]); # indicator
                    out0 = out0.flatten()
    #                print("2: %s" % (t.time() - t1));
    
                    #remove 0 elements
                    idxToRemove = np.where(out0 == 0)
                    out0 = np.delete(out0, idxToRemove)
                    inp0= np.delete(inp0, idxToRemove, axis=0)
                    inp1 = np.delete(inp1, idxToRemove, axis=0)
                    inp2 = np.delete(inp2, idxToRemove, axis=0)
                    inp3 = np.delete(inp3, idxToRemove, axis=0)
                    inp4 = np.delete(inp4, idxToRemove, axis=0)
    
                    bnds1 = tuple((0,5) for n in range(0,5))
                    bnds2 = tuple((1,3) for n in range(0,5))
                    bnds = bnds1 + bnds2
    
                    #DO NOT FIND OPTIMAL SOLUTION
    #                t1=t.time(); 
    #                mdl = minimize(iop, x0, args=(inp0, inp1, inp2, inp3, inp4, out0, distanceMat), method='Nelder-Mead', options={'maxiter':500})
    #                print (t.time()-t1)
                    
    #                #OK
    #                t1=t.time(); 
    #                mdl = minimize(iop, x0, args=(inp0, inp1, inp2, inp3, inp4, out0, distanceMat), bounds=bnds, method='L-BFGS-B', options={'maxiter':500})
    #                print (t.time()-t1)
    #                
                    
    #                t1=t.time(); 
    #                mdl = minimize(iop, x0, args=(inp0, inp1, inp2, inp3, inp4, out0, distanceMat), method='COBYLA', options={'maxiter':500})
    #                (t.time()-t1)
    #
                    # t1=t.time();
                    mdl = minimize(iop, x0, args=(inp0, inp1, inp2, inp3, inp4, out0, distanceMat),
                                   bounds=bnds, method='SLSQP', options={'maxiter':500})
    #                print(mdl)
                    # print("4: %s" % (t.time() - t1));
    #                print(mdl.x[5:])
                     
                    omega[ir,ic,:] = mdl.x[5:]
                    alpha[ir,ic,:] = mdl.x[0:5]
             
    conf.omegaFinalStep1_notFiltered = omega

    omegaFinal2 = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], 5));

    for i in range(0, nPrec):
        for irAgg in range(0, ny):
            for icAgg in range(0, nx):
                omegaFinal2[irAgg * 4:irAgg * 4 + 4, icAgg * 4:icAgg * 4 + 4, i] = omega[irAgg, icAgg, i]
        print('precursor interpolated: ' + str(i));

    # omegaFinal = omegaFinal2
    omegaFinal = np.zeros_like(omegaFinal2)
    for i in range(0, 5):
        tmp = omegaFinal2[:, :, i]
        tmp = gf.gaussian_filter(tmp, sigma=.5)
        omegaFinal[:, :, i] = tmp

    omegaFinal = np.round(omegaFinal, 1)
    # omegaFinal = q.quant(omegaFinal, 0.2)  # discretize
    
    #keep only results on the mask
    omegaFinal[conf.flagRegioMat==0] =np.nan    
    
    conf.omegaFinalStep1 = omegaFinal;
    conf.ci2Step1 = [];
    conf.CovB2Step1 = [];



#    #store final results
#    omegaFinal=q.quant(omega,0.1) #discretize
#
#    for i in range(0, 5):
#        tmp = omegaFinal[:, :, i]
#        tmp = gf.gaussian_filter(tmp, sigma=1)
#        omegaFinal[:, :, i] = tmp
#
#    conf.omegaFinalStep1 = omegaFinal;
#    
#    conf.alpha = alpha
#    conf.ci2Step1 = [];
#    conf.CovB2Step1 = [];
#    flat = conf.flat               
#    nameRegFile = conf.nameRegFile;
#    nameDirOut = conf.nameDirOut;
#    if not os.path.exists(nameDirOut):
#        os.makedirs(nameDirOut);
#    nameRegFile = conf.nameRegFile;                  
#    
#    if flat:
#        sio.savemat(nameRegFile, {'alpha':alpha, 'omega':omega});
#    else:
#        sio.savemat(nameRegFile, {'alpha':alpha, 'omega':omega});
#    

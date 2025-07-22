'''
Created on 6-feb-2017
alpha calculation
@author: roncolato
'''
import os
import numpy as np
import scipy.io as sio
import statsmodels.formula.api as smf
import time
from sherpa.training import funcAggreg as fa
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sherpa.training import distanceComputation as dc
import numexpr as ne
from itertools import repeat, starmap
import multiprocessing as mp
import statsmodels.api as sm

def computeOutput(ir,ic,rad,Prec,omega,PrecToBeUsed,vecPrecompF,Indic,Ide,poly,lat):
    print('Regression for ' + str(ir) + ', ' +str(ic))
    dimrad = rad*2+1;

    ratio = np.polyval(poly,lat[ir,ic])
    # print('reg')
    # print(lat[ir, ic], ratio)

    Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
    Ftmp = 1/((1+((X/ratio)**2+Y**2)**0.5));

    F = np.zeros((dimrad, dimrad, 1, len(PrecToBeUsed)));
    IndicEq = np.zeros((len(Ide), 1));
    PrecDummyQuad = Prec[ir:ir + rad + rad + 1, ic:ic + rad + rad + 1, :, :];
    coeff = np.squeeze(omega[ir, ic, [PrecToBeUsed]]);
    for poll in range(0, len(PrecToBeUsed)):
        if coeff.size == 1:
            F[:, :, 0, poll] = Ftmp**coeff
        else:
            F[:, :, 0, poll] = Ftmp**coeff[poll]

    PrecPatch = (ne.evaluate('PrecDummyQuad*F')).sum((0, 1));
    IndicEq[:, 0] = Indic[ir, ic, Ide];
    model = smf.OLS(IndicEq, PrecPatch).fit();  # , hasconst=False
    modelCoef = model.params;
    return modelCoef
    # bint = model.conf_int(); #np.tile(np.nan,(5,2)) #
    # alpha[ir, ic, [PrecToBeUsed]] = modelCoef;
    # bInt[ir,ic,[PrecToBeUsed],:] = bint;


def step2(conf):
    #initialize variables
    Prec = conf.Prec;
    Indic = conf.Indic;
    PrecToBeUsed = conf.PrecToBeUsed;
    ny = conf.ny;
    nx = conf.nx;
    rad = conf.radStep2;
    nPrec = conf.nPrec;
    flagRegioMat = np.copy(conf.flagRegioMat);
    flat = conf.flat;    
    Ide = conf.Ide;
    omega = conf.omegaFinalStep1;

    #on nan put average omega per pollutant   
    for poll in range(0, 5):
        tmpMat=omega[:,:,poll]
        uniqueomega = np.unique(tmpMat[np.isfinite(tmpMat)])
        aomega = uniqueomega.mean()
        tmpMat[np.isnan(tmpMat)] = aomega;
        omega[:,:,poll] = tmpMat;
    
    #create output dir
    nameDirOut = conf.nameDirOut;
    if not os.path.exists(nameDirOut):
        os.makedirs(nameDirOut);
    nameRegFile = conf.nameRegFile;
    
    #add zeros to Prec variables, to allow for matrix products
    Prec2 = np.zeros((ny+rad*2, nx+rad*2, Prec.shape[2], Prec.shape[3]));
    Prec2[rad:-rad, rad:-rad, :, :] = Prec[:,:,:,:];
    Prec = Prec2;
    #keep only useful columns
    Prec=Prec[:,:,Ide,:][:,:,:,PrecToBeUsed];
             
    #load weigthing function for all available omega - lookup table of omega
    fa.funcAggreg(conf);
    #flatweight only used in flatweight approximation                 
    flatWeight = np.zeros((ny,nx,nPrec));

    #initialize variables
    alpha = np.full([ny,nx,5],np.nan);
    
    #add alpha with LB_CI, UP_CI
    alpha_lb_ci = np.full([ny,nx,5],np.nan);
    alpha_ub_ci = np.full([ny,nx,5],np.nan);
    
    flatWeight = np.zeros((ny,nx,5));
    XMin = np.zeros((ny,nx,nPrec));
    XMax = np.zeros((ny,nx,nPrec));
    yMin = np.full([ny,nx],np.nan);
    yMax = np.full([ny,nx],np.nan);
    bInt = np.full([ny,nx,5,2],np.nan);
    
    IndicEq = np.zeros((len(Ide),1));

    lat = conf.y
    poly = conf.ratioPoly
    
    #loop over cells to create alpha
    for ic in range(0, nx):
        print('Creating regression on x: _'+str(ic)+' of _'+str(nx));
        for ir in range(0, ny):
            if flagRegioMat[ir,ic]==1:
                dimrad = rad*2+1;
                    
                ratio = np.polyval(poly,lat[ir,ic])
                # print('reg')
                # print(lat[ir, ic], ratio)
            
                Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
                Ftmp = 1/((1+((X/ratio)**2+Y**2)**0.5));
            
                F = np.zeros((dimrad, dimrad, 1, len(PrecToBeUsed)));
                IndicEq = np.zeros((len(Ide), 1));
                PrecDummyQuad = Prec[ir:ir + rad + rad + 1, ic:ic + rad + rad + 1, :, :];
                coeff = np.squeeze(omega[ir, ic, [PrecToBeUsed]]);
                for poll in range(0, len(PrecToBeUsed)):
                    if coeff.size == 1:
                        F[:, :, 0, poll] = Ftmp**coeff
                    else:
                        F[:, :, 0, poll] = Ftmp**coeff[poll]
            
                PrecPatch = (ne.evaluate('PrecDummyQuad*F')).sum((0, 1));
                IndicEq[:, 0] = Indic[ir, ic, Ide];
                # regr = linear_model.LinearRegression()
                regr = linear_model.LinearRegression(fit_intercept=False)
                # regr = linear_model.Ridge(alpha=0.01, fit_intercept=False)
                regr.fit(PrecPatch, IndicEq)                
                alpha[ir,ic,[PrecToBeUsed]] = regr.coef_
                
                lr = sm.OLS(IndicEq, PrecPatch).fit()
                conf_interval = lr.conf_int(0.05)
                
                #save alpha CI
                alpha_lb_ci[ir,ic,[PrecToBeUsed]] = conf_interval[:,0]
                alpha_ub_ci[ir,ic,[PrecToBeUsed]] = conf_interval[:,1]
                
                # print(regr.coef_)

#
#
#                
#                
#    t1 = time.time()
#    argslist = (zip(np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1],
#                    repeat(rad), repeat(Prec), repeat(omega), repeat(conf.PrecToBeUsed),
#                    repeat(conf.vecPrecompF), repeat(conf.Indic),repeat(Ide),
#                    repeat(conf.ratioPoly),repeat(conf.y)))
#
#    pool = mp.Pool()  # by default use available corse
#    print('***** Using parallel computing with ' + str(mp.cpu_count()) + ' cores *****')
#    result = pool.starmap_async(computeOutput, argslist)
#    pool.close()
#    pool.join()
#    res = np.vstack(result.get())  # result as nparray
#
#    if (conf.aqi=='SOMO35') | (conf.aqi=='SURF_MAXO3'):
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 0] = res[:, 0]
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 1] = res[:, 1]
#    elif (conf.aqi=='SURF_ug_NO2'):
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 0] = res[:, 0]
#    else:
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 0] = res[:, 0]
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 1] = res[:, 1]
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 2] = res[:, 2]
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 3] = res[:, 3]
#        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], 4] = res[:, 4]
#    print(str(time.time() - t1))

    # bInt[ir, ic, [PrecToBeUsed], :] = []
    #save results
    if flat:
        sio.savemat(nameRegFile, {'alpha':alpha, 'alpha_lb_ci':alpha_lb_ci, 'alpha_ub_ci':alpha_ub_ci, 'omega':omega, 'flatWeight':flatWeight});
    else:
        sio.savemat(nameRegFile, {'alpha':alpha, 'alpha_lb_ci':alpha_lb_ci, 'alpha_ub_ci':alpha_ub_ci, 'omega':omega, 'XMin':XMin, 'XMax':XMax, 'yMin':yMin, 'yMax':yMax});


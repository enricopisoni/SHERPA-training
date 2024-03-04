'''
Created on 6-feb-2017
Modified the 20170321, by EP

@author: roncolato
'''

import numpy as np
import scipy.interpolate as interpol
from scipy.optimize import minimize

from sherpa.training.step1 import from7to28 as f7
from sherpa.training.step1 import EquaPrec as ep
from sherpa.training import EquaIndic as ei
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.ndimage as gf

def InvDistN_opt_prec(beta,xdata,rad,latVecFilt, poly):
    
    ratio = np.polyval(poly, latVecFilt)
    Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
    F = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));
    F = F**beta[1]
    output = beta[0] * np.inner(xdata, F.flatten());
    
    return output;

def iop(beta,inp1,inp2,rad, latVecFilt, poly):
    x=InvDistN_opt_prec(beta,inp1,rad,latVecFilt, poly)
    y=inp2
    y=y.flatten().T
#    print(x)
#    print(y)
#    print(sqrt(mean_squared_error(y, x)))
    return sqrt(mean_squared_error(y, x))
#    return np.mean(((x - y.T) ** 2))


def step1_omegaOptimization(conf):

    #convert from 28 to 7 km
    Prec = f7.from7to28(conf.Prec);
    ny = int(conf.ny/4);           
    nx = int(conf.nx/4);
    rad = conf.radStep1;
    nPrec = 5#len(conf.vec3[conf.POLLSEL])#conf.nPrec;
    rf = conf.rf1
    flagRegioMat = np.copy(conf.flagRegioMat);

    #pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny+rad*2,nx+rad*2,Prec.shape[2],Prec.shape[3]));
    Prec2[rad:-rad,rad:-rad,:,:] = Prec[:,:,:,:];
    Prec=Prec2;

    #convert from 28 to 7 km
    Indic = f7.from7to28(conf.Indic);
    flagRegioMat = f7.from7to28(flagRegioMat);
    lat = f7.from7to28(conf.y);
    # flagPerNoxPP??m = f7.from7to28(flagPerNoxPPm);

    #initialize variables
    omega = np.full([ny,nx,nPrec],1.5);
    alpha = np.full([ny,nx,nPrec],np.nan);
    ci2 = np.empty((nPrec), dtype=object);
    CovB2 = np.empty((nPrec), dtype=object);
#    alphaTmp = np.zeros((categories.size));
#    omegaTmp = np.zeros((categories.size));

    #define training scenarios; note scenarios number is +1 if checking DoE...as in line 74 it is -1
    if conf.domain == 'emep10km':
        if conf.aqi == 'SURF_ug_PM25_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 5]),np.array([1, 6]));
        elif conf.aqi == 'SURF_ug_PM10_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 4]),np.array([1, 6]));
    elif conf.domain == 'ineris7km':
        IdeVec = (np.array([1, 2]),np.array([1, 3]),np.array([1, 4]),np.array([1, 5]),np.array([1, 6]));
    # elif (conf.domain == 'emepV433_camsV221')  | (conf.domain == 'edgar2015') | (conf.domain == 'emepV434_camsV42'):
    elif ('emep' in conf.domain) |  (conf.domain == 'edgar2015') | ('wrf' in conf.domain):        
        IdeVec = (np.array([1, 1]), np.array([1, 2]), np.array([1, 3]), np.array([1, 4]), np.array([1, 5]));

    #loop over precursors
    for precursor in range(0, 5):
        
        PREC = precursor;
        Ide = IdeVec[precursor];
#        icel = 0;
        
        #20220414, test with decreased bounds
        # bnds = ((0, 1), (1.5, 2.5)) #20220524, used for PM25, PM10, O3
        bnds = ((0, 1), (0.5, 2.5)) #20220524, used for PM25, PM10, O3
        # bnds = ((0, 1), (1.75, 2.5)) #20220524, used for NO2 and NO
        
        #intialize variables
#        numcells = nx*ny
#        numcells = np.sum(flagRegioMat>0) # create empty matrix only for really needed points
#        PrecPatch = np.zeros((numcells,(rad*2+1)**2));
#        IndicEq = np.zeros((numcells,1));
#        latVec =  np.zeros((numcells,1));

        print('precursor: '+str(PREC));

        for ic in range(0, nx):
            print(PREC, ic);

            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
                    #variable to store which group ot be considered
#                    indexUsed[icel] = np.where(val==potency[ir,ic]);

                    #create data for omega calculation
                    nSc = Ide.shape[0]-1;# size(Ide,2)-1
                    tmpPrec = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide[1],PREC],rad); # patches
                    tmpInde = ei.EquaIndic(ic,ir,rf,nx,ny,nSc,Indic[:,:,Ide[1]]); # indicator

                    #store data for omega calculation
#                    PrecPatch[icel,:] = tmpPrec; #np.squeeze(tmpPrec)
#                    IndicEq[icel] = tmpInde;
                    latVec = lat[ir,ic]
#                    icel = icel+1;

#                    remInd = (tmpInde>0).flatten()
                    i=1
                    x0 = [1, 2]; #20220314 - test with different IC
#                    print(remInd)

#                    inp1 = tmpPrec[remInd]#[ind,:];
#                    inp2 = tmpInde[remInd]#[ind];
                    inp1 = tmpPrec#[ind,:];
                    inp2 = tmpInde#[ind];

#                    mdl = minimize(iop, x0, args=(inp1, inp2, rad, latVec, conf.ratioPoly), bounds=bnds, method='BFGS', options=opts)  # L-BFGS-B, TNC
                    opts = {'disp': False}
#                    mdl = minimize(iop, x0, args=(inp1, inp2, rad, latVec, conf.ratioPoly), method='BFGS', options=opts)  # L-BFGS-B, TNC
                    
#                    print(mdl.x)
                    #mdl = minimize(iop, x0, args=(inp1, inp2, rad, latVec, conf.ratioPoly), bounds=bnds, method='L-BFGS-B', options=opts)  # L-BFGS-B, TNC
                    #print('L-BFGS-B')
                    #print(mdl.x[1])
                    mdl = minimize(iop, x0, args=(inp1, inp2, rad, latVec, conf.ratioPoly), bounds=bnds, method='SLSQP', options=opts)  # L-BFGS-B, TNC
                    #print('SLSQP')
                    # print(mdl.x[1])
                    alpha[ir,ic,PREC] = mdl.x[0];
                    omega[ir,ic,PREC] = mdl.x[1];
        
    #rescale to initial spatial resolution, through nearest interpolation
    #initialize variable
    
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
        tmp = gf.gaussian_filter(tmp, sigma=5)
        omegaFinal[:, :, i] = tmp

    omegaFinal = np.round(omegaFinal, 1)
    # omegaFinal = q.quant(omegaFinal, 0.2)  # discretize
    
    #keep only results on the mask
    omegaFinal[conf.flagRegioMat==0] =np.nan    
    
    conf.omegaFinalStep1 = omegaFinal;
    conf.ci2Step1 = [];
    conf.CovB2Step1 = [];
    
    
    
    
#    
#    
#    
#    
#    omegaFinal = np.zeros((conf.Prec.shape[0],conf.Prec.shape[1],5));
#
#    for i in range(0,5):
#        omegaFinal[:,:,i] = np.unique(omega[:,:,i])[0]
#
#    #loop on precursors
#    # for i in range(0, nPrec):
#    #     #define interpolator object
#    #     xgv = np.arange(1., conf.Prec.shape[0]/4+1);
#    #     ygv = np.arange(1., conf.Prec.shape[1]/4+1);
#    #     F=interpol.RegularGridInterpolator((xgv, ygv), omega[:,:,i],method='nearest',bounds_error=False, fill_value=None);
#    #
#    #     #interpolate
#    #     Xq = np.arange(1., conf.Prec.shape[0]/4+1, 1/4);
#    #     Yq = np.arange(1., conf.Prec.shape[1]/4+1, 1/4);
#    #     [Y2,X2] = np.meshgrid(Yq, Xq);
#    #     pts=((X2.flatten(),Y2.flatten()))
#    #     omegaFinal[:,:,i] = F(pts).reshape(conf.Prec.shape[0],conf.Prec.shape[1])
#    #     print('precursor interpolated: '+str(i));
#
#    #store final results
#    # replacingVal = np.unique(omegaFinal[:,:,whichPollToUpdate][~np.isnan(omegaFinal[:,:,whichPollToUpdate])])
#    # conf.omegaFinalStep1[:,:,whichPollToUpdate] = replacingVal#omegaFinal[:,:,whichPollToUpdate];
#    conf.omegaFinalStep1 = omegaFinal
#    # conf.omegaFinalStep1_28km = omegaFinal
#    conf.ci2Step1 = ci2;
#    conf.CovB2Step1 = CovB2;

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

def InvDistN_opt_prec(beta,xdata,rad,latVecFilt, poly):
    output = np.zeros_like(latVecFilt)

    for index,i in enumerate(latVecFilt):
        ratio = np.polyval(poly, i)
        Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
        F = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));
        output[index] = beta[0] * np.inner(xdata[index,:], F.flatten());
    return output;

def iop(beta,inp1,inp2,rad, latVecFilt, poly):
    x=InvDistN_opt_prec(beta,inp1,rad,latVecFilt, poly)
    y=inp2
    #print(np.mean(((x - y.T) ** 2)))
    return np.mean(((x - y.T) ** 2))


def step1_omegaOptimization(conf):

    prctileVec1=np.array([100, 100, 100, 100, 100]);
#    prctileVec2=np.array([70, 70, 70, 70, 70]);
    categories=np.array([1])

    #convert from 28 to 7 km
    Prec = f7.from7to28(conf.Prec);
    ny = int(conf.ny/4);
    nx = int(conf.nx/4);
    rad = conf.radStep1;
    nPrec = len(conf.vec3[conf.POLLSEL])#conf.nPrec;
    rf = 0
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
    omega = np.full([ny,nx,nPrec],np.nan);
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
    elif (conf.domain == 'emepV433_camsV221') | (conf.domain == 'edgar2015') | (conf.domain == 'emepV434_camsV42'):
        IdeVec = (np.array([1, 1]), np.array([1, 2]), np.array([1, 3]), np.array([1, 4]), np.array([1, 5]));

    #loop over precursors
    for precursor in range(0, nPrec):
        PREC = precursor;
        Ide = IdeVec[precursor];
        icel = 0;

        #intialize variables
        numcells = nx*ny
        numcells = np.sum(flagRegioMat>0) # create empty matrix only for really needed points
        PrecPatch = np.zeros((numcells,(rad*2+1)**2));
        IndicEq = np.zeros((numcells,1));
        latVec =  np.zeros((numcells,1));
        indexUsed = np.full((numcells,1),np.nan);#np.zeros((nx*ny,1));
        potency=np.full((numcells),np.nan);#np.zeros((ny,nx));
        print('precursor: '+str(PREC));

        for ic in range(0, nx):
            #print(PREC, ic);
            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
                    #variable to store which group ot be considered
#                    indexUsed[icel] = np.where(val==potency[ir,ic]);

                    #create data for omega calculation
                    nSc = Ide.shape[0]-1;# size(Ide,2)-1
                    tmpPrec = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide[1],PREC],rad); # patches
                    tmpInde = ei.EquaIndic(ic,ir,rf,nx,ny,nSc,Indic[:,:,Ide[1]]); # indicator

                    #store data for omega calculation
                    PrecPatch[icel,:] = tmpPrec; #np.squeeze(tmpPrec)
                    IndicEq[icel] = tmpInde;
                    latVec[icel] = lat[ir,ic]
                    icel = icel+1;

#        indexUsedLin = np.reshape(indexUsed, -1, order='F');

        #compute omega for each group of cells, given precursor p
#        for i in range(val.size):
        remInd = (IndicEq>0).flatten()
        i=1
        x0 = [1, 2];
        
#        ind = np.where(indexUsedLin==i)[0];

        inp1 = PrecPatch[remInd]#[ind,:];
        inp2 = IndicEq[remInd]#[ind];
        latVecFilt = latVec[remInd]

        #rescaling input between min and max
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler.fit(inp1)
        # inp1 = scaler.transform(inp1)

        # iop = lambda inp1,beta1,beta2: inv.InvDistN_opt_prec([beta1,beta2],inp1,rad);

        # [mdl,r,J,CovB] = nlin.nlinfit(iop,inp1,inp2.ravel(),x0);

        bnds = ((0, 2), (0.1, 2.9))
        opts = {'disp': False, 'ftol': 10e-6}

        mdl = minimize(iop, x0, args=(inp1, inp2, rad, latVecFilt, conf.ratioPoly), bounds=bnds, method='L-BFGS-B', options=opts)  # L-BFGS-B, TNC
        # ?import scipy.optimize.dif as bs
        #mdl = bs(iop, x0, args=(inp1, inp2, rad, latVecFilt, conf.ratioPoly), bounds=bnds, method='L-BFGS-B', options=opts)  # L-BFGS-B, TNC

        # print('prec' + str(precursor))
        # print(mdl)


        # ci2[PREC] = nlpa.nlparci(r,J);
#        CovB2[PREC] = CovB;
        alphaTmp = mdl.x[0];
        omegaTmp = mdl.x[1];
        #print(alphaTmp)
        #repeat result for each belonging to a given group
        for ic in range(0, nx):
            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
#                    indexUsed = np.where(val==potency[ir,ic])[0];
                    alpha[ir,ic,PREC] = alphaTmp;
                    omega[ir,ic,PREC] = omegaTmp;

        del(PrecPatch,IndicEq,indexUsed,potency)

    #rescale to initial spatial resolution, through nearest interpolation
    #initialize variable
    omegaFinal = np.zeros((conf.Prec.shape[0],conf.Prec.shape[1],5));

    for i in range(0,5):
        omegaFinal[:,:,i] = np.unique(omega[:,:,i])[0]

    #loop on precursors
    # for i in range(0, nPrec):
    #     #define interpolator object
    #     xgv = np.arange(1., conf.Prec.shape[0]/4+1);
    #     ygv = np.arange(1., conf.Prec.shape[1]/4+1);
    #     F=interpol.RegularGridInterpolator((xgv, ygv), omega[:,:,i],method='nearest',bounds_error=False, fill_value=None);
    #
    #     #interpolate
    #     Xq = np.arange(1., conf.Prec.shape[0]/4+1, 1/4);
    #     Yq = np.arange(1., conf.Prec.shape[1]/4+1, 1/4);
    #     [Y2,X2] = np.meshgrid(Yq, Xq);
    #     pts=((X2.flatten(),Y2.flatten()))
    #     omegaFinal[:,:,i] = F(pts).reshape(conf.Prec.shape[0],conf.Prec.shape[1])
    #     print('precursor interpolated: '+str(i));

    #store final results
    # replacingVal = np.unique(omegaFinal[:,:,whichPollToUpdate][~np.isnan(omegaFinal[:,:,whichPollToUpdate])])
    # conf.omegaFinalStep1[:,:,whichPollToUpdate] = replacingVal#omegaFinal[:,:,whichPollToUpdate];
    conf.omegaFinalStep1 = omegaFinal
    # conf.omegaFinalStep1_28km = omegaFinal
    conf.ci2Step1 = ci2;
    conf.CovB2Step1 = CovB2;

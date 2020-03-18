'''
Created on 6-feb-2017
Modified the 20170321, by EP

@author: roncolato
'''

import numpy as np
import scipy.interpolate as interpol
from scipy.optimize import minimize
from sklearn import preprocessing
from sherpa.training.step1 import from7to28 as f7
from sherpa.training.step1 import quant as q
from sherpa.training.step1 import EquaPrec as ep
from sherpa.training import EquaIndic as ei
from sherpa.training.step1 import nlparci as nlpa
from sherpa.training.step1 import InvDistN_opt_prec as inv
from sherpa.training.step1 import nlinfit as nlin

def InvDistN_opt_prec(beta,xdata,rad):
    Y, X = np.mgrid[-rad:rad+1:1, -rad:rad+1:1]; #  np.meshgrid(r_[-rad:1:rad], r_[-rad:1:rad]);
    F = 1 / ( (1 + (X**2 + Y**2)**(0.5))** beta[1] );
    output = beta[0] * np.inner(xdata, F.flatten());
    return output;

def iop(beta,inp1,inp2,rad):
    x=InvDistN_opt_prec(beta,inp1,rad)
    y=inp2

    return ((np.mean(((x - y.T) ** 2))))


def step1_omegaOptPerPoll_7km(conf, idxX, idxY):


    conf.scaler={}

    prctileVec1=np.array([100, 100, 100, 100, 100]);
#    prctileVec2=np.array([70, 70, 70, 70, 70]);
    categories=np.array([1])

    #convert from 28 to 7 km
    Prec = (conf.Prec);
    ny = int(conf.ny);
    nx = int(conf.nx);
    rad = conf.radStep1;
    nPrec = len(conf.vec3[conf.POLLSEL])#conf.nPrec;
    rf = conf.rf;
    flagRegioMat = np.copy(conf.flagRegioMat_FUL);
    
    #pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny+rad*2,nx+rad*2,Prec.shape[2],Prec.shape[3]));
    Prec2[rad:-rad,rad:-rad,:,:] = Prec[:,:,:,:];
    Prec=Prec2;
    
    #convert from 28 to 7 km
    Indic = (conf.Indic);
    # flagRegioMatFull = (flagRegioMat);
    # flagPerNoxPPm = (flagPerNoxPPm);
                      
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
    
    #loop over precursors
    for precursor in range(0, nPrec):
        maxValLoop = 11
        for numRings in range(3,maxValLoop):
            print('testing ' + str(numRings) + ' rings of cells')
            flagPerNoxVocPpmSo2 = np.zeros_like(conf.flagRegioMat)
            flagPerNoxVocPpmSo2[idxY - numRings:idxY + numRings + 1, idxX - numRings:idxX + numRings + 1] = 1

            PREC = precursor;
            Ide = IdeVec[precursor];
            icel = 0;

    #################################################################
            # PART TO BE MODIFIED FOR TESTING LOCALIZATION
            #flagRegioMat = flagRegioMatFull

            # if (precursor==0) | (precursor==1) | (precursor==3) | (precursor==4):
            flagRegioMat = flagPerNoxVocPpmSo2
            # else:
            #     flagRegioMat = flagPerNH3
    #################################################################

            #intialize variables
            numcells = nx*ny
            numcells = np.sum(flagRegioMat>0) # create empty matrix only for really needed points
            PrecPatch = np.zeros((numcells,(rad*2+1)**2));
            IndicEq = np.zeros((numcells,1));
            indexUsed = np.full((numcells,1),np.nan);#np.zeros((nx*ny,1));
            potency=np.full((numcells),np.nan);#np.zeros((ny,nx));

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

            # iop = lambda inp1,beta1,beta2: inv.InvDistN_opt_prec([beta1,beta2],inp1,rad);
            #
            # [mdl,r,J,CovB] = nlin.nlinfit(iop,inp1,inp2.ravel(),x0);

            bnds = ((-2, 2), (0, 4))
            # opt = {'ftol':1e-12, 'gtol':1e-12}
            opt = {'disp':False, 'xtol': 0.0001, 'ftol': 0.0001, 'maxiter': 2000, 'maxfev': 2000}
            # mdl = minimize(iop, x0, args=(inp1, inp2, rad), bounds=bnds, method='L-BFGS-B')#, options = opt)  # L-BFGS-B, TNC
            mdl = minimize(iop, x0, args=(inp1, inp2, rad), bounds=bnds, method='L-BFGS-B', options=opt)  # L-BFGS-B, TNC

            if (mdl.x[1]<0.2) | (mdl.x[1]>3.8):
                if numRings == maxValLoop - 1:
                    mdl.x[1] = np.unique(conf.omegaFinalStep1_28km[idxY - 1:idxY + 1, idxX - 1:idxX + 1, precursor])[0]
                else:
                    continue

            # scaler = preprocessing.MinMaxScaler().fit(inp1)
            # inp1 = scaler.transform(inp1)
            # conf.scaler[precursor] = scaler

            # mdl = minimize(iop, x0, args=(inp1, inp2, rad), method='Powell', options=opt)  # L-BFGS-B, TNC
            # print('prec' + str(precursor))
            print('Precursor: '+str(PREC)+'. Optimization success: ' + str(mdl.success) + '. Omega value: ' + str(mdl.x[1]))
            print('success with ' + str(numRings) + ' rings!')
            break

        # ci2[PREC] = nlpa.nlparci(r,J);
#        CovB2[PREC] = CovB;
        alphaTmp = mdl.x[0];
        omegaTmp = mdl.x[1];

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
    omegaFinal = omega#np.zeros((conf.Prec.shape[0],conf.Prec.shape[1],5));
    # #loop on precursors
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
    #
    #store final results
    conf.omegaFinalStep1 = omegaFinal;
    conf.ci2Step1 = ci2;
    conf.CovB2Step1 = CovB2;
    

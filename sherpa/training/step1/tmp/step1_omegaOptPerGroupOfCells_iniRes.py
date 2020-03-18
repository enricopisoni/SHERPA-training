'''
Created on 6-feb-2017
Modified the 20170321, by EP

@author: roncolato
'''
import numpy as np
import scipy.interpolate as interpol

from sherpa.training.step1 import from7to28 as f7
from sherpa.training.step1 import quant as q
from sherpa.training.step1 import EquaPrec as ep
from sherpa.training import EquaIndic as ei
from sherpa.training.step1 import nlparci as nlpa
#from sherpa.training.step1 import InvDistN_opt_prec as inv
from sherpa.training.step1 import nlinfit as nlin
import time
from scipy.optimize import minimize
import scipy.ndimage as gf
from sklearn.preprocessing import MinMaxScaler


def InvDistN_opt_prec(beta,xdata,rad):
    Y, X = np.mgrid[-rad:rad+1:1, -rad:rad+1:1]; #  np.meshgrid(r_[-rad:1:rad], r_[-rad:1:rad]);
    F = 1 / ( (1 + (X**2 + Y**2)**(0.5))** beta[1] );
    output = beta[0] * np.inner(xdata, F.flatten());
    return output;

def iop(beta,inp1,inp2,rad):
    x=InvDistN_opt_prec(beta,inp1,rad)
    y=inp2
    return np.mean(((x - y.T) ** 2))

def step1_omegaOptPerGroupOfCells_iniRes(conf):
    
    #convert from 28 to 7 km
    Prec = conf.Prec;
    ny = conf.ny;
    nx = conf.nx;
    rad = conf.radStep1;
    nPrec = conf.nPrec;
    rf = conf.rf;
    flagRegioMat = np.copy(conf.flagRegioMat);
    step = conf.stepOptPerGroupCells
    #pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny+rad*2,nx+rad*2,Prec.shape[2],Prec.shape[3]));
    Prec2[rad:-rad,rad:-rad,:,:] = Prec[:,:,:,:];
    Prec=Prec2;
    
    #convert from 28 to 7 km
    Indic = conf.Indic;
    flagRegioMat = flagRegioMat;
                      
    #initialize variables
    omega = np.full([ny,nx,nPrec],np.nan);
#    alpha = np.full([ny,nx,nPrec],np.nan);
                   
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
        PREC = precursor;
        Ide = IdeVec[precursor];        

        #intialize variables        
        print('precursor: '+str(PREC));
#        icel=1     
#        flagIC = 0


        for ic in range(0, nx, step):
            for ir in range(0, ny, step):
                # print(PREC, ic);
                icel = 0
                PrecPatch = np.zeros((step * step, (rad * 2 + 1) ** 2));
                IndicEq = np.zeros((step * step, 1));

                print(PREC, ir, ic)

                for icc in range(ic, ic+step):
                    for irr in range(ir, ir+step):
                        if (icc<nx) and (irr<ny) and (flagRegioMat[ir,ic])>0:
                            nSc = Ide.shape[0]-1;# size(Ide,2)-1
                            tmpPrec = ep.EquaPrec(icc,irr,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide[1],PREC],rad); # patches
                            tmpInde = ei.EquaIndic(icc,irr,rf,nx,ny,nSc,Indic[:,:,Ide[1]]); # indicator
        #                    print(inp1.shape)
                            PrecPatch[icel,:] = tmpPrec; #np.squeeze(tmpPrec)
                            IndicEq[icel] = tmpInde;
                            icel = icel+1;

                remInd = (IndicEq > 0).flatten()

                x0 = [1, 1.5];
                bnds = ((0, 2), (0.1, 2.9))

                inp1 = PrecPatch[remInd]  # [ind,:];
                inp2 = IndicEq[remInd]  # [ind];

                if inp2.size==0:
                    omega[ir:ir+step, ic:ic+step, PREC] = conf.omegaFinalStep1_28km[irr-step,icc-step,PREC]
                else:
                    # scaler = MinMaxScaler(feature_range=(0, 1))
                    # scaler.fit(inp1)
                    # inp1 = scaler.transform(inp1)
                    opts = {'disp': False, 'ftol': 10e-9}
                    mdl = minimize(iop, x0, args=(inp1, inp2, rad), bounds=bnds, method='L-BFGS-B', options=opts) #L-BFGS-B, COBYLA SLSQP
                    if (mdl.success == True) and (mdl.x[1]>0.1) and (mdl.x[1]<2.9):
                        omega[ir:ir+step, ic:ic+step, PREC] = mdl.x[1];#mdl[1];
                    else:
                        omega[ir:ir+step, ic:ic+step, PREC] = conf.omegaFinalStep1_28km[irr-step,icc-step,PREC]
    
###    if results < 1, interpolate         
#    for i in range(0,5):
#        val=omega[:,:,i]
#        val[val<=0.1]=np.nan
#        x = np.arange(0, val.shape[1])
#        y = np.arange(0, val.shape[0])   
#        val = np.ma.masked_invalid(val)
#        xx, yy = np.meshgrid(x, y)
#        #get only the valid values
#        x1 = xx[~val.mask]
#        y1 = yy[~val.mask]
#        newarr = val[~val.mask]
#        GD1 = interpol.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')   
##        GD1 = gf.filters.gaussian_filter(val, 1, order=0)
#        omega[:,:,i]=GD1        
             

    #store final results

    # for i in range(0,5):
    #     tmp = omega[:,:,i]
    #     tmp = gf.gaussian_filter(tmp, sigma=10)
    #     omega[:, :, i] = tmp

    omegaFinal=q.quant(omega,0.1) #discretize

    conf.omegaFinalStep1 = omegaFinal;
    conf.ci2Step1 = [];
    conf.CovB2Step1 = [];
    

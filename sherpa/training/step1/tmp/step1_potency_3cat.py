a'''
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
from sherpa.training.step1 import InvDistN_opt_prec as inv
from sherpa.training.step1 import nlinfit as nlin

def step1_potency(conf):
    
    prctileVec1=np.array([35, 35, 35, 35, 35]);
    prctileVec2=np.array([70, 70, 70, 70, 70]);
    categories=np.array([1,2,3])

    #convert from 28 to 7 km
    Prec = f7.from7to28(conf.Prec);
    ny = int(conf.ny/4);
    nx = int(conf.nx/4);
    rad = conf.radStep1;
    nPrec = conf.nPrec;
    rf = conf.rf;
    flagRegioMat = np.copy(conf.flagRegioMat);
    
    #pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny+rad*2,nx+rad*2,Prec.shape[2],Prec.shape[3]));
    Prec2[rad:-rad,rad:-rad,:,:] = Prec[:,:,:,:];
    Prec=Prec2;
    
    #convert from 28 to 7 km
    Indic = f7.from7to28(conf.Indic);
    flagRegioMat = f7.from7to28(flagRegioMat);
                      
    #initialize variables
    omega = np.full([ny,nx,nPrec],np.nan);
    alpha = np.full([ny,nx,nPrec],np.nan);
    ci2 = np.empty((categories.size,nPrec), dtype=object);
    CovB2 = np.empty((categories.size,nPrec), dtype=object);
    alphaTmp = np.zeros((categories.size));
    omegaTmp = np.zeros((categories.size));
                   
       #define training scenarios; note scenarios number is +1 if checking DoE...as in line 74 it is -1            
    if conf.domain == 'emep10km':
        if conf.aqi == 'SURF_ug_PM25_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 5]),np.array([1, 6]));
        elif conf.aqi == 'SURF_ug_PM10_rh50-Yea':
            IdeVec = (np.array([1, 1]),np.array([1, 2]),np.array([1, 3]),np.array([1, 4]),np.array([1, 6]));
    elif conf.domain == 'ineris7km':
        IdeVec = (np.array([1, 8]),np.array([1, 9]),np.array([1, 10]),np.array([1, 11]),np.array([1, 12]));
    
    #loop over precursors
    for precursor in range(0, nPrec):
        PREC = precursor;
        Ide = IdeVec[precursor];        
        icel = 0;

        #intialize variables        
        PrecPatch = np.zeros((nx*ny,(rad*2+1)**2));
        IndicEq = np.zeros((nx*ny,1));
        indexUsed = np.full((nx*ny,1),np.nan);#np.zeros((nx*ny,1));
        potency=np.full((ny,nx),np.nan);#np.zeros((ny,nx));
        print('precursor: '+str(PREC));
             
        #loop over cells to create groups
        for ic in range(0, nx):
            #print(PREC, ic);
            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
                    #create data for omega calculation         
                    nSc = Ide.shape[0]-1;# size(Ide,2)-1
                    tmpPrec = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide[1],PREC],rad); # patches
                    tmpInde = ei.EquaIndic(ic,ir,rf,nx,ny,nSc,Indic[:,:,Ide[1]]); # indicator
                    
                    x0=np.array([1, 2]);
                    [inp2_aggemi]= inv.InvDistN_opt_prec(x0,tmpPrec,rad);                                              
                    #store data for omega calculation
                    potency[ir,ic]=tmpInde/inp2_aggemi;
                           
        prc1=np.percentile(potency[np.isfinite(potency)],prctileVec1[precursor]);
        prc9=np.percentile(potency[np.isfinite(potency)],prctileVec2[precursor]);        
        speed=potency.copy();
        speed[np.isnan(speed)]=0
        potency[speed<prc1]=1;
        potency[(speed>=prc1) & (speed<prc9)]=2;
        potency[speed>=prc9]=3;        
        val=categories;

        for ic in range(0, nx):
            #print(PREC, ic);
            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
                    #variable to store which group ot be considered
                    indexUsed[icel] = np.where(val==potency[ir,ic]);

                    #create data for omega calculation         
                    nSc = Ide.shape[0]-1;# size(Ide,2)-1
                    tmpPrec = ep.EquaPrec(ic,ir,rf,nx,ny,nSc,Prec.shape[3],Prec[:,:,Ide[1],PREC],rad); # patches
                    tmpInde = ei.EquaIndic(ic,ir,rf,nx,ny,nSc,Indic[:,:,Ide[1]]); # indicator

                    #store data for omega calculation
                    PrecPatch[icel,:] = tmpPrec; #np.squeeze(tmpPrec)
                    IndicEq[icel] = tmpInde;
                    icel = icel+1;
        
        indexUsedLin = np.reshape(indexUsed, -1, order='F');
        
        #compute omega for each group of cells, given precursor p                      
        for i in range(val.size):
            x0 = [1, 2];
            ind = np.where(indexUsedLin==i)[0];
            
            inp1 = PrecPatch[ind,:];
            inp2 = IndicEq[ind];
            
            iop = lambda inp1,beta1,beta2: inv.InvDistN_opt_prec([beta1,beta2],inp1,rad);

            [mdl,r,J,CovB] = nlin.nlinfit(iop,inp1,inp2.ravel(),x0);

            ci2[i,PREC] = nlpa.nlparci(r,J);
            CovB2[i,PREC] = CovB;
            alphaTmp[i] = mdl[0];
            omegaTmp[i] = mdl[1];

        #repeat result for each belonging to a given group
        for ic in range(0, nx):
            for ir in range(0, ny):
                if flagRegioMat[ir,ic]>0:
                    indexUsed = np.where(val==potency[ir,ic])[0];
                    alpha[ir,ic,PREC] = alphaTmp[indexUsed];
                    omega[ir,ic,PREC] = omegaTmp[indexUsed];
                         
        del(PrecPatch,IndicEq,indexUsed,potency,speed)                         
    
    #rescale to initial spatial resolution, through nearest interpolation
    #initialize variable
    omegaFinal = np.zeros((conf.Prec.shape[0],conf.Prec.shape[1],5));                         
    #loop on precursors                     
    for i in range(0, nPrec):
        #define interpolator object
        xgv = np.arange(1., conf.Prec.shape[0]/4+1);
        ygv = np.arange(1., conf.Prec.shape[1]/4+1);
        F=interpol.RegularGridInterpolator((xgv, ygv), omega[:,:,i],method='nearest',bounds_error=False, fill_value=None);

        #interpolate
        Xq = np.arange(1., conf.Prec.shape[0]/4+1, 1/4);
        Yq = np.arange(1., conf.Prec.shape[1]/4+1, 1/4);
        [Y2,X2] = np.meshgrid(Yq, Xq);
        pts=((X2.flatten(),Y2.flatten()))
        omegaFinal[:,:,i] = F(pts).reshape(conf.Prec.shape[0],conf.Prec.shape[1])
        print('precursor interpolated: '+str(i));

    #store final results
    conf.omegaFinalStep1 = omegaFinal;
    conf.ci2Step1 = ci2;
    conf.CovB2Step1 = CovB2;
    

'''
Created on 13-mar-2017
used in step2 to compute the weighting function
@author: roncolato
'''
import numpy as np

def funcAggreg(conf):
    
    PrecToBeUsed = conf.PrecToBeUsed;
    rad = conf.radStep2;
    dimrad = rad*2+1;
    flat = conf.flat;
    omega = conf.omegaFinalStep1;

    # DEFINITION OF THE FUNCTION USED FOR EMISSION AGGREGATIONS
    Y, X = np.mgrid[-rad:rad+1:1, -rad:rad+1:1];
    omega=omega[:,:,PrecToBeUsed]
    #vecPrecompF = np.unique(omega[np.isfinite(omega[:,:,PrecToBeUsed])]);#np.unique(omega[:,:,PrecToBeUsed]);
    vecPrecompF = np.unique(omega[np.isfinite(omega)])
    #vecPrecompF = vecPrecompF[np.logical_not(np.isnan(vecPrecompF))];
    Ftmp = np.zeros((dimrad, dimrad, len(vecPrecompF)));
    tmp1 = np.zeros((dimrad, dimrad));
    F_TBP = np.zeros((len(vecPrecompF)));
    coeff_TBP = np.zeros((len(vecPrecompF)));
    for i in range(0, len(vecPrecompF)):
        coeff = vecPrecompF[i]; # could vary from 0.5 to 5_gaus and sigma_x_gaus
        Ftmp[:,:,i] = 1/((1+(X**2+Y**2)**0.5)**coeff);
            
        #this part is not used anymore
        if flat:
            vw = conf.vw;
            tmp1[:] = Ftmp[:,:,i];
            minVal = tmp1[rad+0-vw,rad+0];
            tmp1[tmp1>=minVal] = np.NaN;
            aveVal = np.nanmean(np.nanmean(tmp1, axis=0));    
            tmp1[:] = Ftmp[:,:,i];
            tmp1[tmp1<minVal] = aveVal;
            Ftmp[:,:,i] = tmp1;        
            F_TBP[i] = aveVal;
            coeff_TBP[i] = coeff;

    conf.vecPrecompF = vecPrecompF;
    conf.Ftmp = Ftmp;
    conf.F_TBP = F_TBP;
    conf.coeff_TBP = coeff_TBP;
    
'''
Created on 25-nov-2016
create data for omega calculation: target
@author: roncolato
'''
import numpy as np

def EquaIndic(ic,ir,rf,nx,ny,nSc,Indic):
    # IndicEq: Indicator values for constant F
    
    # data (Indic) written as:
    # x1,yn ... xn,yn
    # x1,y1 ... xn,y1
    iEq = 0;
    IndicEq = np.zeros((((rf*2)+1)*((rf*2)+1)*nSc,1));
    indicNumDim = len(Indic.shape);
    
    for iSc in range(0, nSc): # loop on scenarios
        # look for data around your reference cell.
        for jr in range(-rf, rf+1): # move vertically around reference cell
            for jc in range(-rf, rf+1): # move horizontally on the matrix
                if (ic+jc > 0 and ic+jc < nx and ir+jr > 0 and ir+jr < ny):  # save infos only if you are in the matrix bounds
                    if (indicNumDim==2):
                        IndicEq[iEq,0] = Indic[ir+jr, ic+jc];
                    if (indicNumDim==3):
                        IndicEq[iEq,0] = Indic[ir+jr, ic+jc, iSc];
                    iEq = iEq+1;

    return IndicEq;


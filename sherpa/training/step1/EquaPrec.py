'''
Created on 25-nov-2016
create data for omega calculation: input
@author: roncolato
'''
import numpy as np

def EquaPrec(ic,ir,rf,nx,ny,nSc,xx,Prec,rad):
    iEq = 0;
    finPrecPatch = np.zeros((((rf*2)+1)*((rf*2)+1)*nSc,(rad*2+1)*(rad*2+1)));
    percNumDim = len(Prec.shape);

    for iSc in range(0, nSc): # loop on scenarios
        for jr in range(-rf, rf+1): # move vertically around reference cell
            for jc in range(-rf, rf+1): # move horizontally on the matrix
                if (ic+jc > 0 and ic+jc < nx and ir+jr > 0 and ir+jr < ny):  # only if you are in the matrix bounds
                    if (percNumDim==2):
                        PrecDummyQuad = Prec[ir+jr:ir+jr+rad+rad+1, ic+jc:ic+jc+rad+rad+1];
                    elif (percNumDim==3):
                        PrecDummyQuad = Prec[ir+jr:ir+jr+rad+rad+1, ic+jc:ic+jc+rad+rad+1, iSc];
                    finPrecPatch[iEq,:] = PrecDummyQuad.flatten('F').T;
#                    print(len(PrecDummyQuad.flatten(1).T))
                    iEq = iEq+1;
    finPrecPatch[iEq:,:] = np.zeros(finPrecPatch[iEq:,:].shape);
    inputReg = finPrecPatch;
    return inputReg;


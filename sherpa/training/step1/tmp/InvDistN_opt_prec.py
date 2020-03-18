'''
Created on 2-dic-2016
function to be optimized
@author: roncolato
'''
import numpy as np

def InvDistN_opt_prec(beta,xdata,rad):

    Y, X = np.mgrid[-rad:rad+1:1, -rad:rad+1:1]; #  np.meshgrid(r_[-rad:1:rad], r_[-rad:1:rad]);
    F = 1 / ( (1 + (X**2 + Y**2)**(0.5))** beta[1] );
    output = beta[0] * np.inner(xdata, F.flatten());

    return output;


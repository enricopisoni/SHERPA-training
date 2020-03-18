'''
Created on 13-gen-2017
read emissions, concentrations and information for grouping cells for omega computation
@author: roncolato
'''
import scipy.io as sio
import sherpa.read_scenarios.ReadPrecIneris7 as rp
import sherpa.read_scenarios.ReadIndicIneris7 as ri
import numpy as np
import netCDF4 as cdf

def ReadScenarios(conf):
    
    #define variables
    conf.aqiFil = conf.vec1[conf.POLLSEL];
    conf.aqi = conf.vec2[conf.POLLSEL];
    conf.PrecToBeUsed = conf.vec3[conf.POLLSEL];
    conf.nameOptOmega = conf.vec4[conf.POLLSEL];
        
    #read emissions and concentrations                                 
    [conf.x, conf.y, conf.nx, conf.ny, conf.Prec] = rp.ReadPrecIneris7(conf.nSc,conf.nPrec,conf.domain,conf.absDel,conf.POLLSEL,conf.emiDenAbs,conf.aqiFil,conf); #0=abs, 1=delta
    [conf.Indic, conf.IndicBC] = ri.ReadIndicIneris7(conf.nSc,conf.nPrec,conf.domain,conf.aqiFil,conf.aqi,conf.absDel,conf.nx,conf.ny,conf); # 0=abs, 1=delta
    
    #read land mask
    # conf.flagRegioMat = sio.loadmat(conf.flagRegioMatFile).get('flagRegioMat');
    fh = cdf.Dataset(conf.flagRegioMatFile, mode='r');
    conf.flagRegioMat = fh.variables['flagRegioMat'][:];  # when loading, I have 'nx,ny,month';
#    conf.flagRegioMat = np.ones_like(conf.IndicBC)
#    conf.flagRegioMat = np.zeros_like(conf.IndicBC)
 #   conf.flagRegioMat[220:260, 220:260] = 1
    
    #read info for grouping cells for omega calculation
    # fh = cdf.Dataset(conf.ncFileStep1, mode='r');
    # windU = np.squeeze(fh.variables[conf.ncFileStep1Var1][:]).transpose();
    # windV = np.squeeze(fh.variables[conf.ncFileStep1Var2][:]).transpose();
    # fh.close();
    # #process info (in this case wind, to get wind speed and direction)
    # windUmeanF = np.fliplr(windU).transpose();
    # windVmeanF = np.fliplr(windV).transpose();
    # conf.dir_, conf.speed = c.cart2compass(windUmeanF, windVmeanF);
    
    return conf;


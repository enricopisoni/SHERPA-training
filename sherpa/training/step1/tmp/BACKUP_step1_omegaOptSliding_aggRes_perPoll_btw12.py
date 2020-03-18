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
# from sherpa.training.step1 import InvDistN_opt_prec as inv
from sherpa.training.step1 import nlinfit as nlin
import time
from scipy.optimize import minimize
import scipy.ndimage as gf
from itertools import repeat, starmap
import multiprocessing as mp

def computeOutput(ir,ic,Ide,rf, nx, ny, Prec, PREC, Indic, ICvecalpha, ICvecomega, rad,poly,lat):
    print(ir,ic)
    ratio = np.polyval(poly,lat[ir,ic])
    # print('step1')
    # print(lat[ir, ic], ratio)
    Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];
    F = 1 / ((1 + ((X / ratio) ** 2 + Y ** 2) ** 0.5));

    nSc = Ide.shape[0] - 1;  # size(Ide,2)-1
    inp1 = ep.EquaPrec(ic, ir, rf, nx, ny, nSc, Prec.shape[3], Prec[:, :, Ide[1], PREC],rad);  # patches
    inp2 = ei.EquaIndic(ic, ir, rf, nx, ny, nSc, Indic[:, :, Ide[1]]);  # indicator
    #                    print(inp1.shape)
    remInd = (inp2 > 0).flatten()
    inp1 = inp1[remInd]  # [ind,:];
    inp2 = inp2[remInd]  # [ind];

    x0tmp_alpha = ICvecalpha[PREC]
    x0tmp_omega = ICvecomega[PREC]
    x0 = [x0tmp_alpha, x0tmp_omega]

    bndstmp_alpha = (0, 10)  # x0tmp_alpha - 0.5, x0tmp_alpha + 0.5)
    bndstmp_omega = (x0tmp_omega - 0.5, x0tmp_omega + 0.5)
    # bnds = ((0, 2), bndstmp)
    bnds = (bndstmp_alpha, bndstmp_omega)

    # a=time.time()
    opts = {'disp': False, 'ftol': 10e-6}
    mdl = minimize(iop2, x0, args=(inp1, inp2, rad, F), bounds=bnds, method='L-BFGS-B',
                   options=opts)  # L-BFGS-B, COBYLA SLSQP
    # print(mdl.x)

    return mdl.x


def iop2(beta, inp1, inp2, rad, F):
    F2 = F ** beta[1];
    x = beta[0] * np.inner(inp1, F2.flatten());
    # x=InvDistN_opt_prec(beta,inp1,rad, F)
    y = inp2
    # print(np.mean(((x - y.T) ** 2)))
    return np.mean(((x - y.T) ** 2))

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
    rf = 0#conf.rf1;
    flagRegioMat = np.copy(conf.flagRegioMat);
    
    #pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny+rad*2,nx+rad*2,Prec.shape[2],Prec.shape[3]));
    Prec2[rad:-rad,rad:-rad,:,:] = Prec[:,:,:,:];
    Prec=Prec2;
    
    #convert from 28 to 7 km
    Indic = f7.from7to28(conf.Indic);
    flagRegioMat = f7.from7to28(flagRegioMat);
    lat = f7.from7to28(conf.y);
                      
    #initialize variables
    omega = np.full([ny,nx,nPrec],np.nan);
    alpha = np.full([ny,nx,nPrec],np.nan);
    ci2 = np.empty((nPrec), dtype=object);
    CovB2 = np.empty((nPrec), dtype=object);
#    alphaTmp = np.zeros((categories.size));
#    omegaTmp = np.zeros((categories.size));
    # define training scenarios; note scenarios number is +1 if checking DoE...as in line 74 it is -1

    if conf.domain == 'emep10km':
        # if conf.aqi == 'SURF_ug_PM25_rh50-Yea':
        if 'SURF_ug_PM25_rh50' in conf.aqi:
            IdeVec = (np.array([1, 1]), np.array([1, 2]), np.array([1, 3]), np.array([1, 5]), np.array([1, 6]));
        # elif conf.aqi == 'SURF_ug_PM10_rh50-Yea':
        elif 'SURF_ug_PM10_rh50' in conf.aqi:
            IdeVec = (np.array([1, 1]), np.array([1, 2]), np.array([1, 3]), np.array([1, 4]), np.array([1, 6]));
        elif 'SURF_ppb_O3' in conf.aqi:
            IdeVec = (np.array([1, 1]), np.array([1, 2]));
        elif 'SURF_ug_NOx' in conf.aqi:
            IdeVec = (np.array([1, 1]), np.array([1, 2]));
    elif conf.domain == 'ineris7km':
        IdeVec = (np.array([1, 2]), np.array([1, 3]), np.array([1, 4]), np.array([1, 5]), np.array([1, 6]));

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
        x0 = [1, 1.5];
#        ind = np.where(indexUsedLin==i)[0];
        
        inp1 = PrecPatch[remInd]#[ind,:];
        inp2 = IndicEq[remInd]#[ind];
        latVecFilt = latVec[remInd]

        bnds = ((0, 2), (0.1, 2.9))
        opts = {'disp': False, 'ftol': 10e-9}

        mdl = minimize(iop, x0, args=(inp1, inp2, rad, latVecFilt, conf.ratioPoly), bounds=bnds, method='L-BFGS-B', options=opts)  # L-BFGS-B, TNC

        # iop = lambda inp1,beta1,beta2: inv.InvDistN_opt_prec([beta1,beta2],inp1,rad);
        #
        # [mdl,r,J,CovB] = nlin.nlinfit(iop,inp1,inp2.ravel(),x0);

#        ci2[PREC] = nlpa.nlparci(r,J);
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
    omegaFinal = np.zeros((conf.Prec.shape[0],conf.Prec.shape[1],5));
    alphaFinal = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], 5));

    [i, j] = np.where(np.isfinite(omega[:, :, 0]))

    for p in range(0,nPrec):
        omegaFinal[:, :, p] = omega[i[0], j[0], p]
        alphaFinal[:, :, p] = alpha[i[0], j[0], p]

     #store final results
    conf.omegaFinalStep1_alldom = omegaFinal;
    conf.alphaFinalStep1_alldom = alphaFinal;
    conf.ci2Step1 = ci2;
    conf.CovB2Step1 = CovB2;

    #######################################################
    #######################################################
    #START NOW THE SECOND STEP OF THE COMPUTATION OF OMEGA
    #######################################################
    #######################################################
    # convert from 28 to 7 km
    Prec = f7.from7to28(conf.Prec);
    ny = int(conf.ny / 4);
    nx = int(conf.nx / 4);
    rad = conf.radStep2;
    nPrec = conf.nPrec;
    rf = conf.rf1;
    flagRegioMat = np.copy(conf.flagRegioMat);

    # Y, X = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1];  # np.meshgrid(r_[-rad:1:rad], r_[-rad:1:rad]);
    # F = 1 / ((1 + (X ** 2 + Y ** 2) ** (0.5)));

    # pad Prec with zeros around initial matrix, to perform matrix products later on
    Prec2 = np.zeros((ny + rad * 2, nx + rad * 2, Prec.shape[2], Prec.shape[3]));
    Prec2[rad:-rad, rad:-rad, :, :] = Prec[:, :, :, :];
    Prec = Prec2;

    # convert from 28 to 7 km
    Indic = f7.from7to28(conf.Indic);
    flagRegioMat = f7.from7to28(flagRegioMat);

    af = conf.alphaFinalStep1_alldom
    of = conf.omegaFinalStep1_alldom

    ICvecalpha = [af[np.isfinite(af)][0], af[np.isfinite(af)][1], af[np.isfinite(af)][2],
                  af[np.isfinite(af)][3], af[np.isfinite(af)][4]]

    ICvecomega = [of[np.isfinite(of)][0], of[np.isfinite(of)][1], of[np.isfinite(of)][2],
                  of[np.isfinite(of)][3], of[np.isfinite(of)][4]]

    # initialize variables
    omega = np.full([ny, nx, nPrec], 1.5);
    # omega = np.full([ny, nx, nPrec], 2);
    # for i in range(0,5):
    #     omega[:,:,i] = of[np.isfinite(of)][i]

    ICvecalpha = [1, 1, 1, 1, 1]
    ICvecomega = [1.5, 1.5, 1.5, 1.5, 1.5]

    # loop over precursors
    for precursor in range(0, nPrec):
        PREC = precursor;
        Ide = IdeVec[precursor];

        # intialize variables
        print('precursor: ' + str(PREC));
        #        icel=1
        #        flagIC = 0

        t1 = time.time()
#Ide or conf.Ide ???
        argslist = (zip(np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1],
                        repeat(Ide), repeat(rf), repeat(nx), repeat(ny),
                        repeat(Prec), repeat(PREC), repeat(Indic), repeat(ICvecalpha), repeat(ICvecomega),
                        repeat(rad), repeat(conf.ratioPoly),repeat(lat)))

        pool = mp.Pool()  # by default use available corse
        print('***** Using parallel computing with ' + str(mp.cpu_count()) + ' cores *****')
        result = pool.starmap_async(computeOutput, argslist)
        pool.close()
        pool.join()
        print(str(time.time() - t1))
        res = np.vstack(result.get())  # result as nparray
        print(res)
        alpha[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], PREC] = res[:, 0]
        omega[np.where(flagRegioMat > 0)[0], np.where(flagRegioMat > 0)[1], PREC] = res[:, 1]

    conf.omegaFinalStep1_notFiltered = omega

    omegaFinal2 = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], 5));

    #from aggregated to initial resolution
    for i in range(0, nPrec):
        for irAgg in range(0, ny):
            for icAgg in range(0, nx):
                omegaFinal2[irAgg * 4:irAgg * 4 + 4, icAgg * 4:icAgg * 4 + 4, i] = omega[irAgg, icAgg, i]
        print('precursor interpolated: ' + str(i));

    #initialize matrix and if needed apply gaussian filter
    omegaFinal = np.zeros_like(omegaFinal2)
    for i in range(0, 5):
        tmp = omegaFinal2[:, :, i]
        if conf.gf == 1:  # if gaussian filter has to be applied
            tmp = gf.gaussian_filter(tmp, sigma=10)
        omegaFinal[:, :, i] = tmp

    #rount results and save it to final matrix
    omegaFinal = np.round(omegaFinal, 1)
    conf.omegaFinalStep1 = omegaFinal;
    conf.ci2Step1 = [];
    conf.CovB2Step1 = [];
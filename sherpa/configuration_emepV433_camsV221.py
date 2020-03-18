'''
Created on 13-mar-2017
define configuration of the training and validation run
@author: roncolato
'''
import numpy as np
import platform
import datetime

def configuration(chooseOpt):

    #class configuration defines methods and attributes
    #methods: used to create names of scenarios to be loaded
    class Config:
        def scenEmissionFileName(self, sce):
            sces = '%01i'%(sce);
            root = 'input/'+self.domain+'/sce'+sces+'/';
            return root+'sce'+sces+'.nc';
        def scenConcFileName(self, sce):
            sces = '%01i'%(sce);
            root = 'input/'+self.domain+'/sce'+sces+'/';
            fileName = root+'sce'+sces+'.nc';
            return fileName;
        pass;

    #attributes: to define features of the training and validation run
    conf = Config();

    ###########################################################################
    #modify for testing
    conf.domain = 'emepV433_camsV221';
    conf.flagReg = 'emepV433_camsV221';
    conf.distance = 1 # 0=cells, 1=distance in km
    conf.gf = 0
    conf.rf1 = 3 # window of cells of training varying F (1=one ring of cells used for training, surrounding the target cell0
    conf.rf2 = 0
    conf.radStep1 = 25; # number of cells to be considered in step1
    conf.radStep2 = 100; # number of cells to be considered in step1


    conf.POLLSEL = 1; # 0=SURF_ug_NO2, 1=SURF_ug_PM25_rh50, 2=SURF_ug_PM10_rh50, 3=SOMO35, 4=SURF_MAX03, 5=SURF_ug_NOx
    #NB: in case of 5=SURF_ug_NOx, NO and NO2 are summed up to produce NOx
    conf.nPrec = 5; # 5 for PM, 2 for O3 (nox, voc), 1 for NO2 (nox)

    conf.season = 'Yea'
    # conf.nPrec = 2; # 5 for PM, 2 for O3 (nox, voc), 1 for NO2 (nox)
    conf.nSc = 28; #total number of scenarios
    conf.Ide = np.array([0,1,2,3,4,5,6]) #np.arange(0, 8);  #training scenarios
    conf.Val = np.arange(1,28)#np.arange(1, 33); #validation scenarios
    #conf.flagRegioMatFile = 'input/'+conf.domain+'/createFlagRegioMat/flagRegioMat.nc'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'; #fixed problem on west coast cells, and small islands
    conf.flagRegioMatFile = 'input/'+conf.domain+'/createFlagRegioMat/flagRegioMat_noSea.nc'#all but #ATL	32	Remaining North-East Atlantic Ocean
    
    #conf.flagRegioMatFile = 'input/' + conf.domain + '/output/flagRegioMat_poValley_red.mat'

    ###
    date = datetime.datetime.now().strftime("%Y%m%d")
    # date = (datetime.datetime.now() - datetime.timedelta(2)).strftime("%Y%m%d")
    conf.nametest = date + '_' + chooseOpt + '_rad' + str(conf.radStep1) + '-' + str(conf.radStep2) + \
                    '_rf_' + str(conf.rf1) + '-' + str(conf.rf2) + '-distCelKm-' + str(conf.distance);

    # conf.nametest = '20190703_omegaSli07km_btw12_rf3_rad120';

    # print(conf.nametest)
    # conf.stepOptPerGroupCells_INI = 25
    # conf.stepOptPerGroupCells_REF = 5
    conf.filter = 1 #0 means do not filter omega results REF, 1 means apply gaussian filter
    ###
    if chooseOpt=='step1_omegaPerPoll_aggRes':
        conf.explain_step_1 = 'omega fixed per pollutant, computed at 28km, same for all the cells'
        conf.explain_step_2 = 'alpha optimized per cell, all scenarios together, original resolution'
    elif chooseOpt == 'step1_omegaPerPoll_aggRes_perPoll':
        conf.explain_step_1 = 'omega slidind per pollutant, computed at 28km, same for all the cells'
        conf.explain_step_2 = 'alpha optimized per cell, all scenarios together, original resolution'

    conf.Order_Pollutant = 'NOx, NMVOC, NH3, PPM, SOx'
    conf.alpha_physical_intepretation = 'alpha specifies the precursor relative importance';
    conf.omega_physical_intepretation = 'omega specifies the slope of the bell-shape';

    ###########################################################################

    conf.modelVariability = 1; # 1=a different model for each cell
    conf.typeOfModel = 2; # 2=regression
    conf.pcaFlag = 0; # 0 means no PCA - no Norm
    conf.absDel = 1; # absolute(0) or delta(1) values
    conf.arealPoint = 0; # 0 means areal and point summed up
    conf.flat = False; # do not use flat weight
    conf.vw = 30; #not used anymore
    conf.emiDenAbs = 0; # 0=emission density, 1=emission in absolute values

    conf.vec1 = ('SURF_ug_NO2','SURF_ug_PM25_rh50','SURF_ug_PM10_rh50','SOMO35', 'SURF_MAXO3', 'SURF_ug_NOx');
    # n1 = 'SURF_ug_NOx-' + conf.season
    # n2 = 'SURF_ug_PM25_rh50-' + conf.season
    # n3 = 'SURF_ug_PM10_rh50-' + conf.season
    # n4 = 'SURF_ppb_O3-' + conf.season
    conf.vec2 = conf.vec1 #(n1, n2, n3, n4)
    conf.vec3 = [[0],[0,1,2,3,4],[0,1,2,3,4],[0,1],[0,1],[0]]; # no2 2voc 3nh3 4pm25 5so2 5nox
    conf.vec4 = ('1step_SURF_ug_NO2','1step_SURF_ug_PM25_rh50','1step_SURF_ug_PM10_rh50','1SURF_ppb_O3','1SURF_ppb_MAXO3','1SURF_ppb_NOx'); #not used anymore
    aqiFil = conf.vec1[conf.POLLSEL];

    conf.ncFileStep1 = 'input/'+conf.domain+'/output/EMEP01_MetData_2014_yearly.nc'; #information used for omega calculation
    conf.ncFileStep1Var1 = 'met2d_u10'; #wind information used for omega calculation
    conf.ncFileStep1Var2 = 'met2d_u10'; #wind information used for omega calculation

    conf.nameDirOut = 'output/'+conf.domain+'/'+aqiFil+'/absDel_'+str(conf.absDel)+'/arealPoint_'+str(conf.arealPoint)+'/'+conf.nametest+'/rf_'+str(conf.rf2)+'-modTyp'+str(conf.typeOfModel)+'-modVar'+str(conf.modelVariability)+'-pca'+str(conf.pcaFlag)+'/';
    conf.nameRegFile = conf.nameDirOut+'regression.mat';

    conf.mode = 'T'; # T or V
    if platform.system()=='Windows':
        conf.datapath = 'D:\\WORK\\projects\\1_urbIam\\1_CODE_MATLAB\\SHERPA';
    elif platform.system()=='Linux':
        conf.datapath = '/home/pisonen/sherpa-training-validation';

    conf.shapeFile = 'input/'+conf.domain+'/Cntry02_emep-extent/cntry02_4km_emep_extent.shp';
    conf.filenameCellPerMs = 'input/'+conf.domain+'/grid_int_emep/grid_int_emep_perc_noTurkey.csv';

    return conf;

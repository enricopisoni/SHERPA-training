'''
Created on 13-mar-2017
define configuration of the training and validation run
@author: roncolato
'''
import numpy as np
import platform
import datetime

def configuration(chooseModel, chooseOpt, time_resol, time_loop, aqi_selected, source_split_instance):

    #class configuration defines methods and attributes
    #methods: used to create names of scenarios to be loaded
    class Config:
        def scenEmissionFileName(self, sce):
            sces = '%01i'%(sce);
            root = 'input/'+self.domain+'/sce'+sces+'/' + time_resol + '/';         
            return root+'sce'+sces+source_split_instance+'.nc';
        def scenConcFileName(self, sce):
            sces = '%01i'%(sce);
            root = 'input/'+self.domain+'/sce'+sces+'/' + time_resol + '/';
            fileName = root+'sce'+sces+source_split_instance+'.nc';
            return fileName;
        pass;

    #attributes: to define features of the training and validation run
    conf = Config();
    
    #ep 20200610
    if time_loop=='YEA':
        conf.yearmonth = 0 #0=year, 1=month
    else :
        conf.yearmonth = 1 #0=year, 1=month
        conf.whichmonth = time_loop #DJF, MAM, JJA, SON
    
    ###########################################################################
    #modify for testing
    conf.domain = chooseModel + source_split_instance;
    conf.flagReg = chooseModel + source_split_instance;
    
    # conf.flagReg = 'emepV434_camsV42withCond_01005';
    conf.distance = 0 # 0=cells, 1=distance in km
    conf.gf = 0
    conf.rf1 = 3 # window of cells of training varying F (1=one ring of cells used for training, surrounding the target cell0
    conf.rf2 = 0
    conf.radStep1 = 25; # number of cells to be considered in step1
    conf.radStep2 = 100; # number of cells to be considered in step1


    conf.POLLSEL = aqi_selected # 0=SURF_ug_NO2, 1=SURF_ug_PM25_rh50, 2=SURF_ug_PM10_rh50, 3=SOMO35, 
                      # 4=SURF_MAX03, 5=SURF_ug_NOx, 6=SURF_ppb_O3, 7=SURF_ppb_SO2,
                      # 8=SURF_ug_SO4, 9=SURF_ug_NO3_F, 10=SURF_ug_NH4_F, 
                      # 11=SURF_ug_PM_OM25, 12=SURF_ug_PPM25, 13='SURF_ug_ECFINE', 14='SURF_ug_NO', 15='SURF_ug_SIA');
                      #16=DDEP_OXN_m2Grid, 17=DDEP_RDN_m2Grid, 18=WDEP_OXN, 19=WDEP_RDN
    #NB: in case of 5=SURF_ug_NOx, NO and NO2 are summed up to produce NOx
    conf.nPrec = 5; # 5 for PM, 2 for O3 (nox, voc), 1 for NO2 (nox)

    #conf.season = 'Yea'
    # conf.nPrec = 2; # 5 for PM, 2 for O3 (nox, voc), 1 for NO2 (nox)
    
    conf.Ide = np.array([0,1,2,3,4,5,6]) #np.arange(0, 8);  #training scenarios
    
    ######################################################################
    #EP 20230511
    # MODIFY THIS TO WORK WITH LH TOGETHER, or LH SEPARATED
    if len(source_split_instance)==0 : #case of LH TOGETHER
        # conf.nSc = 18;              # LH TOGETHER, total number of scenarios
        # conf.Val = np.arange(1,18); # LH TOGETHER, validation scenarios
        conf.nSc = 7;              # LH TOGETHER, total number of scenarios
        conf.Val = np.arange(1,7); # LH TOGETHER, validation scenarios
    elif len(source_split_instance)!=0 : #case of LH SEPARATED    
        conf.nSc = 7;                 # LH SEPARATED, total number of scenarios in case of splitting sources
        conf.Val = conf.Ide;          # LH SEPARATED, validation if splitting low and high ... final validation is done externally to the code
    ######################################################################
    
    #conf.flagRegioMatFile = 'input/'+conf.domain+'/createFlagRegioMat/flagRegioMat.nc'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'; #fixed problem on west coast cells, and small islands
    conf.flagRegioMatFile = 'input/'+conf.domain+'/createFlagRegioMat/flagRegioMat_noSea_v7.nc'#all but #ATL	32	Remaining North-East Atlantic Ocean
    
    #conf.flagRegioMatFile = 'input/' + conf.domain + '/output/flagRegioMat_poValley_red.mat'

    ###
    date = datetime.datetime.now().strftime("%Y%m%d")
    # date = (datetime.datetime.now() - datetime.timedelta(2)).strftime("%Y%m%d")
    # conf.nametest = date + '_rad' + str(conf.radStep1) + '-' + str(conf.radStep2) + \
    #                 '_rf_' + str(conf.rf1) + '-' + str(conf.rf2) + 'Sec_Emi_Vars_' + time_loop;
    conf.nametest = chooseModel + '_' + date + '_' + time_loop;

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

    #changing this so that now PPM25 and PPM10 are differentiated (not only PPM always)
    conf.Order_Pollutant = 'NOx, NMVOC, NH3, PPM25, SOx'
    if conf.POLLSEL == 2: #only of PM10 modelling, use primary PPM10 emissions as input
        conf.Order_Pollutant = 'NOx, NMVOC, NH3, PPM10, SOx'
    #changing this so that now PPM25 and PPM10 are differentiated (not only PPM always)
    
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
    conf.emiDenAbs = 0; # 0=emission density [ton/km2], 1=emission in absolute values [kton/year]

    conf.vec1 = ('SURF_ug_NO2','SURF_ug_PM25_rh50','SURF_ug_PM10_rh50','SOMO35', 'SURF_MAXO3',
                 'SURF_ppb_O3', 'SURF_ppb_SO2','SURF_ug_SO4', 'SURF_ug_NO3_F','SURF_ug_NH4_F',
                 'SURF_ug_PM_OM25', 'SURF_ug_PPM25', 'SURF_ug_ECFINE', 'SURF_ug_NO', 'SURF_ug_SIA',
                 'DDEP_OXN_m2Grid', 'DDEP_RDN_m2Grid', 'WDEP_OXN', 'WDEP_RDN');

    # n1 = 'SURF_ug_NOx-' + conf.season
    # n2 = 'SURF_ug_PM25_rh50-' + conf.season
    # n3 = 'SURF_ug_PM10_rh50-' + conf.season
    # n4 = 'SURF_ppb_O3-' + conf.season
    conf.vec2 = conf.vec1 #(n1, n2, n3, n4)
    conf.vec3 = [[0,1],[0,1,2,3,4],[0,1,2,3,4],[0,1],[0,1],[0,1],[0,1,2,3,4],[0,1,2,3,4], 
                 [0], [2], [0,1,2,3,4], [3], [0,1,2,3,4], [0,1], [0,1,2,4],
                 [0,2], [0,2], [0,2], [0,2]]; # no2 2voc 3nh3 4pm25 5so2 5nox
    #conf.vec4 = ('1step_SURF_ug_NO2','1step_SURF_ug_PM25_rh50','1step_SURF_ug_PM10_rh50','1SURF_ppb_O3','1SURF_ppb_MAXO3','1SURF_ppb_NOx'); #not used anymore
    aqiFil = conf.vec1[conf.POLLSEL];

    conf.ncFileStep1 = 'input/'+conf.domain+'/output/EMEP01_MetData_2014_yearly.nc'; #information used for omega calculation
    conf.ncFileStep1Var1 = 'met2d_u10'; #wind information used for omega calculation
    conf.ncFileStep1Var2 = 'met2d_u10'; #wind information used for omega calculation

    conf.nameDirOut = 'output/'+conf.domain+'/'+aqiFil+'/absDel_'+str(conf.absDel)+'/'+conf.nametest+'/';
    conf.nameRegFile = conf.nameDirOut+'regression.mat';

    conf.mode = 'T'; # T or V
    if platform.system()=='Windows':
        conf.datapath = 'X:\\Integrated_assessment\\pisonen\\WORK\\projects\\1_urbIam\\1_CODE_MATLAB\\SHERPA';
    elif platform.system()=='Linux':
        conf.datapath = '/eos/jeodpp/data/projects/IAM-SUL/transfer/SHERPA/';

    conf.shapeFile = 'input/'+conf.domain+'/Cntry02_emep-extent/cntry02_4km_emep_extent.shp';
    conf.filenameCellPerMs = 'input/'+conf.domain+'/grid_int_emep/grid_int_emep_perc_noTurkey.csv';

    return conf;

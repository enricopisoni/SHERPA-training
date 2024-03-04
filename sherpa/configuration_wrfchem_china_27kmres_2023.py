'''
Created on 13-mar-2017
define configuration of the training and validation run
@author: roncolato
'''
import numpy as np
import platform
import datetime

def configuration(chooseOpt, time_resol, time_loop, aqi_selected, source_split_instance):

    #class configuration defines methods and attributes
    #methods: used to create names of scenarios to be loaded
    class Config:
        def scenEmissionFileName(self, sce):
            sces = '%01i'%(sce);
            root = 'input/'+self.domain+'/sce'+sces+'/' + time_resol + '/';         
            # return root+'sce'+sces+source_split_instance+'_updated.nc';
            return root+'new_0.2x0.2_mod_sce'+sces+source_split_instance+'_withEmiSum.nc';
        def scenConcFileName(self, sce):
            sces = '%01i'%(sce);
            root = 'input/'+self.domain+'/sce'+sces+'/' + time_resol + '/';
            # fileName = root+'sce'+sces+source_split_instance+'_updated.nc';
            fileName = root+'new_0.2x0.2_mod_sce'+sces+source_split_instance+'_withEmiSum.nc';
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
    conf.domain = 'wrfchem_china_27kmres_2023'+source_split_instance;
    conf.flagReg = 'wrfchem_china_27kmres_2023'+source_split_instance;
    
    # conf.flagReg = 'emepV434_camsV42withCond_01005';
    conf.distance = 0 # 0=cells, 1=distance in km
    conf.gf = 0
    conf.rf1 = 5#3 # window of cells of training varying F (1=one ring of cells used for training, surrounding the target cell0
    conf.rf2 = 0
    conf.radStep1 = 10#5; # number of cells to be considered in step1
    conf.radStep2 = 50#100; # number of cells to be considered in step1


    conf.POLLSEL = aqi_selected 
    conf.nPrec = 5; # 5 for PM, 2 for O3 (nox, voc), 1 for NO2 (nox)

    #conf.season = 'Yea'
    # conf.nPrec = 2; # 5 for PM, 2 for O3 (nox, voc), 1 for NO2 (nox)
    
    conf.Ide = np.array([0,1,2,3,4,5,6]) #np.arange(0, 8);  #training scenarios
    
    ######################################################################
    #EP 20230511
    # MODIFY THIS TO WORK WITH LH TOGETHER, or LH SEPARATED
    if len(source_split_instance)==0 : #case of LH TOGETHER
        conf.nSc = 10;              # LH TOGETHER, total number of scenarios
        conf.Val = np.arange(1,10); # LH TOGETHER, validation scenarios
    elif len(source_split_instance)!=0 : #case of LH SEPARATED    
        conf.nSc = 7;                 # LH SEPARATED, total number of scenarios in case of splitting sources
        conf.Val = conf.Ide;          # LH SEPARATED, validation if splitting low and high ... final validation is done externally to the code
    ######################################################################
    
    #conf.flagRegioMatFile = 'input/'+conf.domain+'/createFlagRegioMat/flagRegioMat.nc'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'#flagRegioMat-allEmepDomain.mat'#flagRegioMat_onlyLandEu28_noTurkey_noIceland.mat'; #fixed problem on west coast cells, and small islands
    #conf.flagRegioMatFile = 'input/'+conf.domain+'/createFlagRegioMat/flagRegioMat_noSea_v3.nc'#all but #ATL	32	Remaining North-East Atlantic Ocean
    
    conf.flagRegioMatFile = 'input/' + conf.domain + '/flagRegioMat/new_0.2x0.2_mod_flagRegioMat_China.nc'

    ###
    date = datetime.datetime.now().strftime("%Y%m%d")
    # date = (datetime.datetime.now() - datetime.timedelta(2)).strftime("%Y%m%d")
    conf.nametest = date + '_rad' + str(conf.radStep1) + '-' + str(conf.radStep2) + \
                    '_rf_' + str(conf.rf1) + '-' + str(conf.rf2) + 'Sec_Emi_Vars_' + time_loop;

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

    conf.vec1 = ('C_H2O2','C_HNO3','C_NH3','C_NO',
                 'C_NO2','C_O3','C_PM10','C_PM25','C_PM25_BC',
                 'C_PM25_NACL','C_PM25_NH4','C_PM25_NO3',
                 'C_PM25_OC','C_PM25_OIN','C_PM25_SO4','C_SO2');

    # n1 = 'SURF_ug_NOx-' + conf.season
    # n2 = 'SURF_ug_PM25_rh50-' + conf.season
    # n3 = 'SURF_ug_PM10_rh50-' + conf.season
    # n4 = 'SURF_ppb_O3-' + conf.season
    conf.vec2 = conf.vec1 #(n1, n2, n3, n4)
    conf.vec3 = [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],
                 [0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],
                 [0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],
                 [0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]]; # no2 2voc 3nh3 4pm25 5so2 5nox
    #conf.vec4 = ('1step_SURF_ug_NO2','1step_SURF_ug_PM25_rh50','1step_SURF_ug_PM10_rh50','1SURF_ppb_O3','1SURF_ppb_MAXO3','1SURF_ppb_NOx'); #not used anymore
    aqiFil = conf.vec1[conf.POLLSEL];

    #conf.ncFileStep1 = 'input/'+conf.domain+'/output/EMEP01_MetData_2014_yearly.nc'; #information used for omega calculation
    #conf.ncFileStep1Var1 = 'met2d_u10'; #wind information used for omega calculation
    #conf.ncFileStep1Var2 = 'met2d_u10'; #wind information used for omega calculation

    conf.nameDirOut = 'output/'+conf.domain+'/'+aqiFil+'/absDel_'+str(conf.absDel)+'/'+conf.nametest+'/';
    conf.nameRegFile = conf.nameDirOut+'regression.mat';

    conf.mode = 'T'; # T or V
    if platform.system()=='Windows':
        # conf.datapath = 'D:\\WORK\\projects\\1_urbIam\\1_CODE_MATLAB\\SHERPA';
        conf.datapath = 'X:\\Integrated_assessment\\pisonen\\WORK\\projects\\1_urbIam\\1_CODE_MATLAB\\SHERPA'
    elif platform.system()=='Linux':
        conf.datapath = '/eos/jeodpp/home/users/pisonen/SHERPA/';

    conf.shapeFile = 'input/'+conf.domain+'/Cntry02_emep-extent/cntry02_4km_emep_extent.shp';
    conf.filenameCellPerMs = 'input/'+conf.domain+'/grid_int_emep/grid_int_emep_perc_noTurkey.csv';

    return conf;

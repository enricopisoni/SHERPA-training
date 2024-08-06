#!/usr/bin/env python
# encoding: utf-8 

#20230206 
# I am now using this general function, that allows to work with
# - yearly values, low and high sources together
# - seasonal values (DJF, MAM, JJA, SON), low and high sources together
# - yearly values, low and high sources split

# chooseModel =  'wrfchem_china_27kmres_2023'
#'wrfchem_china_27kmres_2023'  # this in the case you use low and high sources summed up
# chooseModel = 'emepV434_camsV42withCond_01005_month'
chooseModel = 'emepV4_45_cams61_withCond_01005_2019'#

#20230206 define if to split low and high level sources
split_low_high_sources = False
if split_low_high_sources :
    source_split=['', '_low', '_high'] 
else :
    source_split=[''] 
    
#20230206 define if to consider only yearly, or also seasonal indicators
time_agg_period = ['yearly', 'monthly', 'monthly', 'monthly', 'monthly']
time_agg_tag = ['YEA', 'DJF', 'MAM', 'JJA', 'SON']    
start_time_loop = 0; end_time_loop = 1 #0,1 means you run only yearly values - 0,3 is 6 month average, 0,7  means YEA, SEA + 4 seasons

#20230206 list of SRR to be tested
# aqi_to_be_tested = list([0,1,2,6])
aqi_to_be_tested = list([1])

#20230206 standard optimization to be performed
chooseOpt = 'step1_omegaPerPoll_aggRes_perPoll'        

import os
import sys
import time
import numpy as np
import sherpa.read_scenarios.ReadScenarios as rs
import sherpa.read_scenarios.computeDistanceRatio as cr
import sherpa.training.step2.step2 as s2
import sherpa.validation.validation as v
from optparse import OptionParser

#20230206 only emepV434_camsV42withCond_01005_month is currently used
if chooseModel == 'emep10km':
    import sherpa.configuration_emep as c
elif chooseModel == 'emepV434_camsV42withCond_01005_month':
    import sherpa.configuration_emepV434_camsV42withCond_01005_month as c
elif chooseModel == 'emepV4_45_cams61_withCond_01005_2019':
    import sherpa.configuration_emepV4_45_cams61_withCond_01005_2019 as c
elif chooseModel == 'wrfchem_china_27kmres_2023':
    import sherpa.configuration_wrfchem_china_27kmres_2023 as c
   
#20230206 only 'step1_omegaPerPoll_aggRes_perPoll' is currently used    
if chooseOpt == 'step1_omegaPerPoll_aggRes':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes as s1
elif chooseOpt == 'step1_omegaPerPoll_aggRes_perPoll':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes_perPoll as s1
elif chooseOpt == 'step1_omegaPerPoll_aggRes_perPoll_ch':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes_perPoll_CH as s1

__all__ = []
__version__ = 0.1
__date__ = '2017-01-13'
__updated__ = '2017-01-25'

def main(argv=None):
    #loop on air quality indicators
    
    #20230206 train 2 SRR in case of low and high level split, only 1 SRR of no split
    for source_split_instance in source_split:
        
        #20230206 loop to compute only YEA, or also seasonal (DJF, MAM, JJA, SON) indicators
        for aqi_selected in aqi_to_be_tested:
            
            #loop on time aggregation
            for iter_loop in range(start_time_loop, end_time_loop):
                print('processing ' + str(time_agg_period[iter_loop]), ', indicator ' + str(aqi_selected))
                
                '''Command line options.'''
            
                program_name = os.path.basename(sys.argv[0])
                program_version = "v%s" % __version__
                program_build_date = "%s" % __updated__
                program_version_string = '%%prog %s (%s)' % (program_version, program_build_date)
                program_longdesc = '''''' # optional - give further explanation about what the program does
                program_license = "Copyright 2017 ISPRA                                                    \
                Licensed under the Apache License 2.0 \
                http://www.apache.org/licenses/LICENSE-2.0"
            
                if argv is None:
                    argv = sys.argv[1:]
                try:    
                    # setup option parser
                    parser = OptionParser(version=program_version_string, epilog=program_longdesc, description=program_license)
                    parser.add_option("-p", "--path", dest="datapath", help="set data path [default: %default]", metavar="DIR")
                    parser.add_option("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %default]")
                    parser.add_option("-m", "--mode", dest="mode", help="set mode (T:training and validation, V:validation) [default: %default]")
            
                    ############################
                    conf = c.configuration(chooseOpt, time_agg_period[iter_loop], time_agg_tag[iter_loop], aqi_selected, source_split_instance);
            
                    parser.set_defaults(datapath=conf.datapath, mode=conf.mode)
            
                    (opts, args) = parser.parse_args(argv)
                    conf.mode = opts.mode;
            
                    if opts.verbose and opts.verbose > 0:
                        print("verbosity level = %d" % opts.verbose)
                    if opts.datapath:
                        print("datapath = %s" % opts.datapath)
                    if opts.mode:
                        print("mode = %s" % opts.mode)
            
                    start = time.time();
            
                    os.chdir(opts.datapath)
            
                    print('read_scenarios');
                    rs.ReadScenarios(conf);
            
                    #compute ratio useful to correct weighting factor matrix, in the case of lat lon
                    cr.computeDistanceRatio(conf)
            
                    if opts.mode=='T':
                        print('step1');
            
                        #this uses varying omega
                        s1.step1_omegaOptimization(conf)
            
            #           # this is the test done during December 2019, using fixed omega
                        # conf.omegaFinalStep1_alldom = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], 5));
                        # conf.omegaFinalStep1 = np.zeros_like(conf.omegaFinalStep1_alldom)
                        # conf.omegaFinalStep1[:] = 2.5 #if you want to consider 1.5 fix
            
                        print('step2');
                        s2.step2(conf);
            
                    print('validation');
                    v.validation(conf);
                    ############################
            
                    print('end');
                    print("Execution time: %s" % (time.time() - start));
            
                except Exception as e:
                    indent = len(program_name) * " "
                    # print(traceback.format_exc())
                    sys.stderr.write(program_name + ": " + repr(e) + "\n")
                    sys.stderr.write(indent + "  for help use --help")
                    #print(sys.exc_info()[0])
                    return 2

    

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# encoding: utf-8 

################################################################################################################
#ep 20200610, do you want yearly or monthly results?
#YEARLY VALUES
#emepV434_camsV42withCond_01005 works with CAMS4.2, with condensables, at 0.1x0.05 spatial resolution
chooseModel = 'emepV434_camsV42withCond_01005' #'emepV434_camsV42' #'ineris7km' # 'emep10km' #'china5km' #emepV433_camsV221 "edgar2015#
#emep4nl_2021
#test for NL, 20210729
#chooseModel = 'emep4nl_2021' #'ineris7km' # 'emep10km' #'china5km' #emepV433_camsV221 "edgar2015#
#MONTHLY VALUES
#chooseModel = 'emepV433_camsV221_monthly' 
################################################################################################################

chooseOpt = 'step1_omegaPerPoll_aggRes_perPoll' #'step1_omegaPerPoll_aggRes VS 

import os
import sys
import time
import numpy as np

import sherpa.read_scenarios.ReadScenarios as rs
import sherpa.read_scenarios.computeDistanceRatio as cr
import sherpa.training.step2.step2 as s2

if chooseModel == 'emep10km':
    import sherpa.configuration_emep as c
elif chooseModel == 'ineris7km':
    import sherpa.configuration_ineris as c
elif chooseModel == 'china5km':
    import sherpa.configuration_china as c
elif chooseModel == 'emepV433_camsV221':
    import sherpa.configuration_emepV433_camsV221 as c
elif chooseModel == 'emepV433_camsV221_monthly':
    import sherpa.configuration_emepV433_camsV221_monthly as c
elif chooseModel == 'edgar2015':
    import sherpa.configuration_edgar2015 as c
elif chooseModel == 'emepV434_camsV42':
    import sherpa.configuration_emepV434_camsV42 as c
elif chooseModel == 'emep4nl_2021':
    import sherpa.configuration_emep4nl_2021 as c
elif chooseModel == 'emepV434_camsV42withCond_01005':
    import sherpa.configuration_emepV434_camsV42withCond_01005 as c
    
if chooseOpt ==    'step1_omegaPerPoll_aggRes':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes as s1
elif chooseOpt == 'step1_omegaPerPoll_aggRes_perPoll':
    import sherpa.training.step1.step1_omegaPerPoll_aggRes_perPoll as s1

import sherpa.validation.validation as v
from optparse import OptionParser

__all__ = []
__version__ = 0.1
__date__ = '2017-01-13'
__updated__ = '2017-01-25'

def main(argv=None):
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
        conf = c.configuration(chooseOpt);

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

#            # this is the test done during December 2019, using fixed omega
            # conf.alphaFinalStep1_alldom = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], 5));
            # conf.omegaFinalStep1_alldom = np.zeros((conf.Prec.shape[0], conf.Prec.shape[1], 5));
            # conf.alphaFinalStep1_alldom[:] = 1
            # conf.omegaFinalStep1_alldom[:] = 1.75 #THIS IS NOT USED AFTER
            # conf.omegaFinalStep1 = np.zeros_like(conf.omegaFinalStep1_alldom)
            # conf.omegaFinalStep1[:] = 1.75 #if you want to consider 1.5 fix
            # conf.omegaFinalStep1[:, :, 3] = 1.75

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

    '''
    if DEBUG:
        print(1);
        #sys.argv.append("-h")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'sherpa.main_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    '''
    sys.exit(main())

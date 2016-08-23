#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems, itervalues

import argparse
import os
import shutil
import subprocess
import prms_cfg as prms_cfg
import prms_lib as prms
from prms_calib_helpers import get_sim_obs_stat, related_params

__version__ = 0.5


def link_data(src, dst):
    """Symbolically link a file into a directory. This adds code to handle
       doing this over an NFS mount. The src parameter should be a path/filename
       and the dst parameter should be a directory"""

    dst_file = '%s/%s' % (dst, os.path.basename(src))

    if os.path.isfile(dst_file):
        # Remove any pre-existing file/symlink

        # These operations over NFS sometimes generate spurious ENOENT and
        # EEXIST errors. To get around this just rename the link/file first
        # and then delete the renamed version. This works because rename is
        # atomic whereas unlink and remove are not.
        tmpfile = '%s/tossme' % dst
        os.rename(dst_file, tmpfile)
        try:
            if os.path.islink(tmpfile):
                os.unlink(tmpfile)
            else:
                os.remove(tmpfile)
        except OSError as (errno, strerror):
            print("I/O error({0}): {1}".format(errno, strerror))

            if not os.path.exists(tmpfile):
                print("\tHmmm... file must be gone already.")

    # Now create the symbolic link
    os.symlink(src, dst_file)


# Command line arguments
parser = argparse.ArgumentParser(description='Post MOCOM processing')
parser.add_argument('mocomrun', help='MOCOM run id')
parser.add_argument('modelrun', help='PRMS model run id')
parser.add_argument('parameters', help='Parameters calculated by MOCOM',
                    nargs=argparse.REMAINDER, type=float)

args = parser.parse_args()

# This is set as a hardcoded default for now
configfile = 'basin.cfg'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read the config file
cfg = prms_cfg.cfg(configfile)

cdir = os.getcwd()  # Save current dir in case we need it later

# basin_dir = '%s/%s' % (cfg.get_value('base_calib_dir'), cfg.get_value('basin'))
# run_dir = '%s/%s/%s/%s' % (basin_dir, cfg.get_value('runs_sub'), args.mocomrun, args.modelrun)

base_calib_dir = cfg.get_value('base_calib_dir')
basin = cfg.get_value('basin')
runid = cfg.get_value('runid')

calib_job_dir = '{0}/{1}/{2}'.format(base_calib_dir, runid, basin)
model_src = '{0}/{1}'.format(base_calib_dir, cfg.get_value('basin'))

# control_file = '%s/%s' % (model_src, cfg.get_value('prms_control_file'))
# paramfile = '%s/%s' % (model_src, cfg.get_value('prms_input_file'))
param_range_file = '{0}/{1}'.format(calib_job_dir, cfg.get_value('param_range_file'))

run_dir = '{0}/{1}'.format(calib_job_dir, args.modelrun)

# Make the run directory and change into it
try:
    os.makedirs(run_dir)
except OSError:
    # The args.modelrun directory already exists. Delete it and re-create it.
    shutil.rmtree(run_dir)
    os.makedirs(run_dir)
finally:
    os.chdir(run_dir)

# Copy the model files from model_src into the calibration directory
cmd_opts = ' {0}/* .'.format(model_src)
subprocess.call(cfg.get_value('cmd_cp') + cmd_opts, shell=True)

# Copy the control and input parameter files into calibration directory
# for dd in cfg.get_value('source_config'):
#     cmd_opts = " %s ." % dd
#     subprocess.call(cfg.get_value('cmd_cp') + cmd_opts, shell=True)

# Link in the data files
# for dd in cfg.get_value('source_data'):
#     link_data(dd, os.getcwd())

# Read the param_range_file
# Check that the number of parameters in the file matches the parameters
# passed in the command line arguments.
# The param_range_file is located in the root basin directory
pfile = open('{0}/{1}'.format(cdir, cfg.get_value('param_range_file')), 'r')

params = []
for rr in pfile:
    params.append(rr.split(' ')[0])
pfile.close()

if len(params) != len(args.parameters):
    print("ERROR: mismatch between number of parameters ({0:d} vs. {1:d})".format(len(params), len(args.parameters)))
    exit(1)

# ************************************
# Check related parameters
# ************************************
for ii, pp in enumerate(params):
    if pp in related_params:
        for kk, vv in iteritems(related_params[pp]):
            if kk in params:
                # Test if the parameter meets the condition for the related parameter
                if not vv(args.parameters[ii], args.parameters[params.index(kk)]):
                    # Write the stats.txt file with -1 for each objective function and
                    # skip running PRMS, return stats.txt with -1.0 for each objective function
                    print('{0:s} = {1:s} failed relationship with {2:s} = {3:s}'
                          .format(pp, args.parameters[ii], kk, args.parameters[params.index(kk)]))
                    tmpfile = open('tmpstats', 'w')
                    objfcn_link = cfg.get_value('of_link')

                    # must use version of MOCOM that has been modified to detect
                    # -9999.0 as a bad set.
                    for mm in objfcn_link:
                        tmpfile.write('-9999.0\n')

                    tmpfile.close()

                    # Move the stats file to its final place - MOCOM looks for this file
                    os.rename('tmpstats', 'stats.txt')

                    # Return to the starting directory
                    os.chdir(cdir)
                    exit(0)     # We'll skip the model run and return success to MOCOM

# For each parameter, either write individual values or redistribute the mean to the original parameters
# and update the input parameter file
pobj = prms.parameters(cfg.get_value('prms_input_file'))

# Check if all params are identical
# This would be the case if calibrating a parameter using individual values
# e.g. tmax_allsnow when calibrating each month
if params.count(params[0]) == len(params):
    # check if number of params entries equals a dimension for that input parameter
    # params are assumed to be in sequential order (e.g. 1..12)
    pobj.replace_values(params[0], args.parameters)
else:
    # We're redistributing a mean value
    for ii, pp in enumerate(params):
        pobj.distribute_mean_value(pp, args.parameters[ii])

# Write the new values back to the input parameter file
pobj.write_param_file(cfg.get_value('prms_input_file'))

# Convert the config file start/end date strings to datetime
st_date_model = prms.to_datetime(cfg.get_value('start_date_model'))
st_date_calib = prms.to_datetime(cfg.get_value('start_date'))
en_date = prms.to_datetime(cfg.get_value('end_date'))

# Run PRMS
cmd_opts = " -C{0} -set param_file {1} -set start_time {2} -set end_time {3}".format(cfg.get_value('prms_control_file'),
                                                                                     cfg.get_value('prms_input_file'),
                                                                                     prms.to_prms_datetime(st_date_model),
                                                                                     prms.to_prms_datetime(en_date))

try:
    retcode = subprocess.call(cfg.get_value('cmd_prms') + cmd_opts, shell=True)
except OSError:
    print('ERROR: PRMS executable is missing.')
    exit(2)
else:
    if retcode != 0:
        raise RuntimeError('Error from PRMS execution.')
        print('\nCMD {}'.format(cmd))
        exit(2)


# ***************************************************************************
# ***************************************************************************
# Generate the statistics for the run
tmpfile = open("tmpstats", 'w')
objfcn_link = cfg.get_value('of_link')

for vv in itervalues(objfcn_link):
    tmpfile.write('{0:0.6f} '.format(get_sim_obs_stat(cfg, vv)))
tmpfile.write('\n')
tmpfile.close()


# Open the control file from the run and get the names of the data files used
ctl = prms.control('{0:s}/{1:s}'.format(run_dir, cfg.get_value('prms_control_file')))
for cf in ['data_file', 'precip_day', 'tmax_day', 'tmin_day']:
    # Remove unneeded data files from run directory
    cfile = ctl.get_values(cf)
    try:
        print('Removing {0}/{1}'.format(run_dir, cfile))
        os.remove('{0}/{1}'.format(run_dir, cfile))
    except Exception as inst:
        print("ERROR: {}".format(inst))

# Get the files used by the objective functions and delete them from the run
# filelist = []

for vv in itervalues(cfg.get_value('objfcn')):
    if vv['obs_file']:
        try:
            print('Removing {0}/{1}'.format(run_dir, vv['obs_file']))
            os.remove('{0:s}/{1:s}'.format(run_dir, vv['obs_file']))
        except OSError as err:
            print('WARNING: Removing file: {}\n\t{}'.format(vv['obs_file'], err))
    if vv['sd_file']:
        try:
            print('Removing {0:s}/{1:s}'.format(run_dir, vv['sd_file']))
            os.remove('{0:s}/{1:s}'.format(run_dir, vv['sd_file']))
        except OSError as err:
            print('WARNING: Removing file: {}\n\t{}'.format(vv['sd_file'], err))


# Move the stats file to its final place - MOCOM looks for this file
os.rename('tmpstats', 'stats.txt')

# Return to the starting directory
os.chdir(cdir)

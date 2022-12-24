#!/usr/bin/env python
"""
Optimize parameters of any external program to any target property that can be computed with the programs.

Usage:
  {0:s} [options]

Options:
  -h, --help  Show this message and exit.
  --nproc NPROC
              Number of processes to be used. If it's less than 1, use as many processes as possible. [default: 0]
  --subdir-prefix PREFIX_DIR
              Prefix for subjob directory. [default: subdir_]
  --subjob-script SCRIPT
              Name of script that performs MD and post-processing. [default: subjob.sh]
  --subjob-prefix PREFIX_JOB
              Prefix for performing subjob. [default: ]
  --subjob-timeout TIMEOUT
              Timeout for a subjob in sec. [default: 3600]
  --random-seed SEED
              Random seed for reproducibility, if negative current time in second is applied. [default: -1]
"""

__all__ = ['io','get_best','opt2md','opt2prms']

import os
import sys
import shutil
import glob
from docopt import docopt
import numpy as np
from numpy import sin,cos,sqrt
import subprocess
import time
from datetime import datetime

from .opt2prms import vars2params
from .io import read_in_optzer, write_info, write_vars_optzer, read_vars_optzer, \
    read_data
from .de import DE
from .cs import CS
from .tpe import TPE

__author__ = "RYO KOBAYASHI"
__version__ = "0.1.0"

_infname = 'in.optzer'

def get_data(basedir,prefix=None,**kwargs):
    """
    New implementation of get_data, which loads data to be used to fit parameters.
    If the prefix ref, some special treatment will be done.
    """

    targets = kwargs['target']

    if prefix == 'ref':
        print(' Reference data and weights:')

    data = {}
    for t in targets:
        if prefix == 'ref':
            fname = basedir+'/data.{0:s}.{1:s}'.format(prefix,t)
        else:  # not known about prefix
            files = glob.glob(basedir+f'/data.*.{t:s}')
            fname = files[0]  # assume 1st one as the generated data, which could be wrong
        # print('m,fname=',m,fname)
        try:
            data[t] = read_data(fname,)
        except:
            data[t] = None
            pass
        if prefix == 'ref':
            print('   {0:s}  {1:.3f}'.format(t,data[t]['wdat']))
    return data

def loss_func(tdata,eps=1.0e-8,**kwargs):
    """
    Compute loss function value using general get_data func.
    """
    refdata = kwargs['refdata']
    losses = {}
    L = 0.0
    misval = kwargs['missing_value']
    luplim = kwargs['fval_upper_limit']
    for name in refdata.keys():
        ref = refdata[name]
        wgt = ref['wdat']
        dtype = ref['datatype']
        eps = ref['eps']
        trial = tdata[name]
        if trial == None:
            losses[name] = misval
            L += losses[name] *wgt
            continue
        num = ref['ndat']
        refd = ref['data']
        td = trial['data']
        z2 = 0.0
        sumdiff2 = 0.0
        if dtype[:5] == 'indep':  # independent data
            epss = ref['epss']
            for n in range(num):
                diff = td[n] - refd[n]
                sumdiff2 += diff*diff /epss[n]**2
            if num > 0:
                sumdiff2 /= num
            losses[name] = min(sumdiff2, luplim)
        elif dtype[:3] == 'sep':  # data treated separately
            for n in range(num):
                # print('n=',n)
                diff = td[n] -refd[n]
                sumdiff2 += diff*diff /(refd[n]**2+eps)
            losses[name] = min(sumdiff2, luplim)
        else:  # data treated all together (default)
            for n in range(num):
                # print('n=',n)
                diff = td[n] -refd[n]
                sumdiff2 += diff*diff
                z2 += refd[n]*refd[n]
            losses[name] = min(sumdiff2 /(z2+eps), luplim)

        if 'subject_to' in ref.keys():
            import re
            for prange in ref['subject_to']:
                pid = prange['pid']  # int
                lower0 = prange['lower']  # str
                upper0 = prange['upper']  # str
                penalty = prange['penalty']  # float
                lower1 = re.sub(r'\{([0-9]+)\}',r'{p[\1]}',lower0)
                upper1 = re.sub(r'\{([0-9]+)\}',r'{p[\1]}',upper0)
                lower2 = lower1.format(p=td)
                upper2 = upper1.format(p=td)
                lower = eval(lower2)
                upper = eval(upper2)
                if lower > upper:
                    raise ValueError('lower is greater than upper in subject_to ',pid)
                if not (lower < td[pid] < upper):
                    losses[name] += penalty

        L += losses[name] *wgt
        
    if kwargs['print_level'] > 0:
        print('   iid,losses= {0:8d}'.format(kwargs['iid']),end='')
        for k in losses.keys():
            loss = losses[k]
            print(' {0:10.4f}'.format(loss),end='')
        print(' {0:11.5f}'.format(L),flush=True)
    return L

def func_wrapper(variables, **kwargs):
    """
    Wrapper function for the above loss_func().
    This converts variables to be optimized to parameters for the external program,
    perform it, then get the target properties.
    Then pass them to the the loss_func().
    """
    refdata = kwargs['refdata']
    wgts = kwargs['weights']
    refdata = kwargs['refdata']
    subjobscript = kwargs['subjob-script']
    subdir = kwargs['subdir-prefix'] +'{0:03d}'.format(kwargs['index'])
    print_level = kwargs['print_level']

    #...Create param_files in each subdir
    varsfp = {}
    varsfp['variables'] = variables
    cwd = os.getcwd()
    if not os.path.exists(subdir):
        os.mkdir(subdir)
        shutil.copy(subjobscript,subdir+'/')
    os.chdir(subdir)

    if len(kwargs['param_files']) == 0:
        raise ValueError('param_files not given correctly in in.optzer')
    else:
        for pfile in kwargs['param_files']:
            if os.path.exists(pfile):
                os.remove(pfile)
        vars2params(varsfp['variables'], **kwargs)

    #...Do sub-jobs in the subdir_###
    L_up_lim = kwargs['fval_upper_limit']
    if print_level > 1:
        print('Performing subjobs at '+subdir, flush=True)
    try:
        prefix = kwargs['subjob-prefix']
        timeout= kwargs['subjob-timeout']
        cmd = prefix +" ./{0:s} > log.iid_{1:d}".format(subjobscript,kwargs['iid'])
        subprocess.run(cmd,shell=True,check=True,timeout=timeout)
        optdata = get_data('.',**kwargs)
        L = loss_func(optdata,**kwargs)
    except Exception as e:
        if print_level > 0:
            print('  Since subjobs failed at {0:s}, '.format(subdir)
                  +'the upper limit value is applied to its loss function.',
                  flush=True)
        os.chdir(cwd)
        L = L_up_lim

    try:
        #...Store data in iid_### directories
        os.mkdir("iid_{0:d}".format(kwargs['iid']))
        txt = ''
        for t in kwargs['target']:
            files = glob.glob(f'data*{t}')
            if f'data.ref.{t}' in files:
                files.remove(f'data.ref.{t}')
            txt += ' {0:s}'.format(files[0])
        for f in kwargs['param_files']:
            txt += f' {f}'
        for f in glob.glob('out.*'):
            txt += f' {f}'
        os.system("cp {0} iid_{1:d}/".format(txt,kwargs['iid']))
        os.chdir(cwd)
    except Exception as e:
        if print_level > 0:
            print(' Something wrong with storing data in iid_{0:d}'.format(kwargs['iid']))
            print(e)
        
        
    return L
    
def main():

    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    headline()
    start = time.time()

    nproc = int(args['--nproc'])
    seed = int(args['--random-seed'])
    if seed < 0:
        seed = int(start)
        print(f' Random seed was set from the current time: {seed:d}')
    else:
        print(f' Random seed was given: {seed:d}')
    
    infp = read_in_optzer(_infname)
    write_info(infp,args)

    vs,vrs,vrsh,options,vopts = read_vars_optzer(infp['vars_file'])

    kwargs = infp
    kwargs['options'] = options
    kwargs['hardlim'] = vrsh
    # kwargs['infp'] = infp
    kwargs['subdir-prefix'] = args['--subdir-prefix']
    kwargs['subjob-script'] = args['--subjob-script']
    kwargs['subjob-prefix'] = args['--subjob-prefix']
    kwargs['subjob-timeout'] = int(args['--subjob-timeout'])
    kwargs['start'] = start
    kwargs['vopts'] = vopts
    
    if len(kwargs['target']) != 0:
        refdata = get_data('.',prefix='ref',**kwargs)
    else:
        raise ValueError(f'target may not given in {_infname}.')
    kwargs['refdata'] = refdata

    if len(kwargs['param_files']) == 0:
        raise ValueError(f'param_files should be given in {_infname}.')
    else:
        for fname in kwargs['param_files']:
            with open(fname,'r') as f:
                kwargs[fname] = f.read()

    print('\n # iid,losses=      iid',end='')
    if len(kwargs['target']) > 0:
        for t in kwargs['target']:
            print('  {0:>9s}'.format(t),end='')
    print('      total')

    maxiter = kwargs['num_iteration']
    if kwargs['opt_method'] in ('de','DE'):
        N = infp['de_num_individuals']
        F = infp['de_fraction']
        T = infp['de_temperature']
        CR = infp['de_crossover_rate']
        opt = DE(N,F,CR,T, vs,vrs,vrsh, func_wrapper, write_vars_optzer,
                 nproc=nproc, seed=seed, **kwargs)
    elif kwargs['opt_method'] in ('cs','CS','cuckoo','Cuckoo'):
        N = infp['cs_num_individuals']
        F = infp['cs_fraction']
        opt = CS(N,F, vs,vrs,vrsh, func_wrapper, write_vars_optzer,
                 nproc=nproc, seed=seed, **kwargs)
    elif kwargs['opt_method'] in ('tpe','TPE','wpe','WPE'):
        nbatch = nproc
        opt = TPE(nbatch, vs, vrs, vrsh, func_wrapper, write_vars_optzer,
                  seed=seed, **kwargs)
    
    opt.run(maxiter)

    print('\n optzer finished since it exceeds the max interation.')
    print(' Elapsed time = {0:.1f} sec.'.format(time.time()-start))
    
    return None

def headline():
    print('')
    print(' optzer --- optimize parameters of any external program to any target property ---')
    print('')
    cmd = ' '.join(s for s in sys.argv)
    print('   Executed as {0:s}'.format(cmd))
    hostname = subprocess.run(['hostname',], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print('            at {0:s} on {1:s}'.format(os.getcwd(),hostname.strip()))
    print('            at {0:s}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print()
    print('   Please cite:')
    print('     1) R. Kobayashi, J. Open Source Software, 6(57), 2768 (2021)')
    print()
    return None

if __name__ == "__main__":

    main()

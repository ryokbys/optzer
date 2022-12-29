import os
import numpy as np
import pandas as pd

__author__ = "RYO KOBAYASHI"
__version__ = "221227"

def read_in_optzer(fname='in.optzer'):
    #...initialize
    infp = {}
    infp['fval_upper_limit'] = 100.0
    infp['missing_value'] = 1.0
    infp['print_level'] = 1
    infp['weights'] = {'rdf':1.0, 'adf':1.0, 'vol':1.0, 'lat':1.0}
    infp['update_vrange'] = -1
    infp['vars_file'] = 'in.vars.optzer'

    mode = None
    infp['target'] = []
    infp['param_files'] = []
    
    with open(fname,'r') as f:
        lines = f.readlines()

    for line in lines:
        if line[0] in ('!','#'):
            mode = None
            continue
        data = line.split()
        if len(data) == 0:
            mode = None
            continue
        if data[0] == 'num_iteration':
            maxiter = int(data[1])
            infp['num_iteration'] = maxiter
            mode = None
        elif data[0] == 'print_level':
            print_level = int(data[1])
            infp['print_level'] = print_level
            mode = None
        elif data[0] == 'opt_method':
            opt_method = data[1]
            infp['opt_method'] = opt_method
            mode = None
        elif data[0] == 'param_files':
            infp[data[0]] = [ name for name in data[1:] ]
            mode = None
        elif data[0] == 'fval_upper_limit':
            fup_limit = float(data[1])
            infp['fval_upper_limit'] = fup_limit
            mode = None
        elif data[0] == 'missing_value':
            misval = float(data[1])
            infp['missing_value'] = misval
            mode = None
        elif data[0] in ('target'):
            if len(data) < 2:
                raise RuntimeError('target entry requires at least one keyword.')
            for i in range(1,len(data)):
                infp['target'].append(data[i])
            mode = None
        elif data[0] == 'num_individuals' or data[0] == 'num_trials':
            nind = int(data[1])
            infp['num_individuals'] = nind
            mode = None
        elif data[0] == 'cs_fraction':
            frac = float(data[1])
            infp['cs_fraction'] = frac
            mode = None
        elif data[0] == 'tpe_gamma':
            infp['tpe_gamma'] = float(data[1])
            mode = None
        elif 'tpe_nsmpl_prior' in data[0]:
            infp['tpe_nsmpl_prior'] = int(data[1])
            mode = None
        elif 'tpe_ntrial' in data[0]:
            infp['tpe_ntrial'] = int(data[1])
            mode = None
        elif data[0] == 'update_vrange':
            infp['update_vrange'] = int(data[1])
            mode = None
        else:
            mode = None
            pass
    
    return infp

def read_out_optzer(fname='out.optzer'):
    with open(fname,'r') as f:
        lines = f.readlines()
    bestiid = -1
    for line in reversed(lines):
        if 'step,time' in line:
            data = line.split()
            bestiid = int(data[3])
            bestloss = float(data[4])
            break
    if bestiid < 0:
        raise ValueError(f'Failed to get best_iid from {fname}!!!')

    targets = []
    weights = []
    losses = []
    mode = ''
    for line in lines:
        if 'weights:' in line:
            mode = 'weights'
            continue
        if '# iid,losses' in line:
            data = line.split()
            targets = [ t for t in data[3:] ]
            continue
        elif 'iid,losses' in line and ' {0:d} '.format(bestiid) in line:
            if len(targets) < 1:
                raise ValueError('len(targets) < 1 !!!')
            data = line.split()
            losses = [ float(l) for l in data[2:] ]
            break
        if mode == 'weights':
            data = line.split()
            if len(data) != 2:
                mode = ''
                continue
            weights.append(float(data[1]))

    return bestiid,targets,weights,losses

def write_info(infp,args):
    """
    Write out information on input parameters for fp.
    """

    print('\n Input')
    print(' ----------')
    print('   num of processes (given by --nproc option)  ',int(args['--nproc']))
    try:
        if len(infp['param_files']) == 0:
            print('   potential       {0:s}'.format(infp['potential']))
        else:
            print('   param_files  ',end='')
            for fname in infp['param_files']:
                print(f'  {fname}', end='')
            print('')
    except:
        raise

    fmethod = infp['opt_method']
    print('   opt_method  {0:s}'.format(fmethod))
    if fmethod in ('cs','CS'):
        print('   num_individuals   {0:d}'.format(infp['num_individuals']))
        print('   fraction          {0:7.4f}'.format(infp['cs_fraction']))
    elif fmethod in ('tpe','TPE','wpe','WPE'):
        pass
    else:
        print('   There is no such opt_method...')
    print('   num_iteration   {0:d}'.format(infp['num_iteration']))
    print('   missing_value   {0:.1f}'.format(infp['missing_value']))
    print(' ----------')
    return None

def write_vars_optzer(vnames,vs,slims,hlims,fname='in.vars.optzer',**kwargs):
    """Write in.vars.optzer.
    
    Each line (except 1st line) has the following values.
      1: initial guess of the parameter
      2: soft lower limit
      3: soft upper limit
      4: hard lower limit
      5: hard upper limit
      6: name of the parameter
    """
    voptions = kwargs['voptions']
    nv = len(vs)
    with open(fname,'w') as f:
        f.write(' {0:5d} \n'.format(nv))
        # for i in range(len(vs)):
        for k in vnames:
            f.write(' {0:15.7f}  {1:15.7f}  {2:15.7f}'.format(vs[k],*slims[k])
                    +'  {0:10.4f}  {1:10.4f}'.format(*hlims[k])
                    +f'  {k}\n')
    return None

def read_vars_optzer(fname='in.vars.optzer'):
    """Read in.vars.optzer.
    
    Each line (except 1st line) should have the following values.
      1: initial guess of the parameter
      2: soft lower limit
      3: soft upper limit
      4: hard lower limit
      5: hard upper limit
      6: name of the parameter

    The name of the parameter is required since version 221227.
    """
    with open(fname,'r') as f:
        lines = f.readlines()
    iv = 0
    nv = -1
    vs = {}
    slims = {}
    hlims = {}
    vnames = []
    voptions = {}
    for line in lines:
        if line[0] in ('!','#'):
            k,v = parse_option(line)
            if k is not None:
                voptions[k] = v[0]
                #print(' option: ',k,v)
            continue
        data = line.split()
        if len(data) == 0:
            continue
        if nv < 0:
            nv = int(data[0])
            continue
        else:
            iv += 1
            if iv > nv:
                break
            vname = data[5]
            if vname in vnames:
                ValueError('Parameter name should be unique: '+vname)
            vnames.append(vname)
            vs[vname] = float(data[0])
            slims[vname] = [ float(data[1]), float(data[2]) ]
            hlims[vname] = [ float(data[3]), float(data[4]) ]
            # vs.append(float(data[0]))
            # vrs.append([ float(data[1]), float(data[2])])
            # vrsh.append([float(data[3]), float(data[4])])

    # vs = np.array(vs)
    # vrs = np.array(vrs)
    # vrsh = np.array(vrsh)
    return vnames,vs,slims,hlims,voptions
    

def read_data(fname,):
    """
    General routine of reading data.
    
    Input file format
    -----------------
    ```
    #  Comment lines begins with '#' or '!'
    #  Options start with "option-name: "
    10    1.0
    0.1234  0.2345  0.3456  0.4567  0.5678  0.6789
    0.7890  0.8901  0.9012  0.0123
    ```
    - 1st line:  num of data (NDAT),  weight of the data (WDAT)
    - 2nd line-: data values (number of data should be equal to NDAT)
    ----------------------------------------
    In case that "datatype: independent" is specified in the option,
    the input file format is changed as the following,
    ```
      10     1.0
      0.1234    0.1234
     -0.2345    0.2345
      0.3456    0.1
      ...
    ```
    - Below the 1st line-:  (value, error eps) pair
    """
    if not os.path.exists(fname):
        raise RuntimeError('File not exsits: ',fname)
    
    with open(fname,'r') as f:
        lines = f.readlines()

    ndat = 0
    wdat = 0.0
    data = None
    epss = None
    idat = 0
    done = False
    options = {'datatype': 'continuous', 'eps':1.0e-3}
    for line in lines:
        if line[0] in ('#','!'):
            try: 
                k,v = parse_option(line)
            except:
                k = None
                v = None
            if k != None and v != None: # valid option
                if len(v) == 1: # options that take only one argument
                    options[k] = v[0]
                else: # options that take more than one arguments
                    if k == 'subject_to':
                        if len(v) != 4:
                            raise ValueError('Num of arguments for subject_to option is wrong, len(v)= ',len(v))
                        if 'subject_to' not in options.keys():
                            options[k] = []
                        options[k].append({'pid':int(v[0]),
                                           'lower':v[1],
                                           'upper':v[2],
                                           'penalty':float(v[3])})
                    else:
                        options[k] = v
            continue
        ldat = line.split()
        if ndat < 1:
            ndat = int(ldat[0])
            wdat = float(ldat[1])
            data = np.zeros(ndat)
            if 'indep' in options['datatype']:
                epss = np.zeros(ndat)
        else:
            if data is None:
                raise RuntimeError('data is None, which should not happen.')
            if 'indep' in options['datatype']:
                # In case of datatype==independent, each line has (value,eps) pair information
                data[idat] = float(ldat[0])
                epss[idat] = float(ldat[1])
                idat += 1
                if idat == ndat:
                    done = True
                    break
            else:
                for i,d in enumerate(ldat):
                    data[idat] = float(d)
                    idat += 1
                    if idat == ndat:
                        done = True
                        break
        if done:
            break
    options['eps'] = float(options['eps'])
    if 'indep' in options['datatype']:
        options['epss'] = epss
    return {'ndat':ndat, 'wdat':wdat, 'data':data, **options}

def parse_option(line):
    """
    Parse option from a comment line.
    """
    words = line.split()
    if len(words) < 2 or words[1][-1] != ':':
        return None,None
    optname = words[1]
    k = words[1][:-1]
    v = words[2:]
    return k,v

def read_out_cs_gen(fname='out.cs.generations'):
    """Read out.cs.generations file.
    Return generations, iids, and losses."""
    with open(fname,'r') as f:
        lines = f.readlines()
    gens = []
    iids = []
    losses = []
    for l in lines:
        if l[0] == '#':
            continue
        data = l.split()
        gens.append(int(data[0]))
        iids.append(int(data[1]))
        losses.append(float(data[2]))
    return gens,iids,losses

def read_out_cs_ind(fname='out.cs.individuals'):
    """Read out.cs.individuals file."""
    with open(fname,'r') as f:
        lines = f.readlines()
    iids = []
    losses = []
    prms = []
    for l in lines:
        if l[0] == '#':
            continue
        data = l.split()
        iids.append(int(data[0]))
        losses.append(float(data[1]))
        prms.append([ float(d) for d in data[2:]])
    return iids,losses,prms

def write_db_optzer(db,fname='db.optzer.json'):
    """Write db in JSON format."""
    with open(fname,'w') as f:
        f.write(db.to_json(orient='records',lines=True))
    return None

def read_db_optzer(fname='db.optzer.json'):
    """Read db_optzer in JSON format."""
    try:
        db = pd.read_json(fname, orient='records', lines=True)
    except:
        raise
    return db

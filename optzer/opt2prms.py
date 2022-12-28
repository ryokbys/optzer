#!/usr/bin/env python
"""
Convert optzer parameters `in.vars.optzer` to parameter files.

Usage:
  opt2prms.py [options] <v-file> <p-files> [<p-files>...]

Options:
  -h, --help  Show this message and exit.
"""
import os
from docopt import docopt

from .io import read_vars_optzer

__author__ = "RYO KOBAYASHI"
__version__ = "221223"

def read_in_fitpot2(infname='in.fitpot'):
    """
    Get specorder, pairs, triplets from in.fitpot.
    """
    if not os.path.exists(infname):
        raise FileNotFoundError(infname)
    
    with open(infname,'r') as f:
        lines = f.readlines()

    mode = None
    specorder = []
    interact = []
    param_files = []
    for line in lines:
        data = line.split()
        if len(data) == 0:
            mode = None
            continue
        if data[0] in ('#','!'):
            mode = None
            continue
        elif data[0] == 'specorder':
            specorder = [ x for x in data[1:] ]
            continue
        elif data[0] == 'interactions':
            num_interact = int(data[1])
            mode = 'interactions'
            continue
        elif data[0] == 'param_files':
            param_files = [ name for name in data[1:] ]
            mode = None
        else:
            if mode == 'interactions':
                if len(data) not in (2,3):
                    raise Exception('len(data) is not 2 nor 3.')
                interact.append(data)
                if len(interact) == num_interact:
                    mode = None
            else:
                mode = None

    return specorder, interact, param_files
    

def read_vars_fitpot(fname='in.vars.fitpot'):
    """
    Read in.vars.fitpot and return data.
    """
    with open(fname,'r') as f:
        lines = f.readlines()

    fpvars = []
    vranges = []
    il = -1
    nv = -1
    while True:
        il += 1
        line = lines[il]
        if line[0] in ('!','#'):  # skip comment line
            il += 1
            continue
        data = line.split()
        if nv < 0:
            nv = int(data[0])
            rc = float(data[1])
            rc3= float(data[2])
        else:
            fpvars.append(float(data[0]))
            vranges.append([ float(x) for x in data[1:3]])
            if len(fpvars) == nv:
                break
    varsfp = {}
    varsfp['rc2'] = rc
    varsfp['rc3'] = rc3
    varsfp['variables'] = fpvars
    varsfp['vranges'] = vranges
    return varsfp
    
def vars2params(vs,**kwargs):
    """
    Conversion from optzer-vars to files specified in param_files in in.optzer.
    The param_files should contain key-phrases such as '{p[0]}' that are converted from fp-vars,
    and the indices in the key-phrases must correspond to those in optzer-vars..
    """
    import re
    try:
        param_fnames = kwargs['param_files']
    except:
        raise
    if type(param_fnames) != list or len(param_fnames) < 1:
        raise ValueError('param_files may not be specified correctly...')
        
    for fname in param_fnames:
        try:
            fcontents = kwargs[fname]
            #...If the format is '{0:.1f}'-style, replace them to '{p[0]:.2f}'-style
            res = re.search(r'\{[0-9]+:',fcontents)
            if res != None:
                fcontents = re.sub(r'\{([0-9]+):',r'{p[\1]:',fcontents)
            new_contents = fcontents.format(p=vs)
        except:
            print('ERROR: Failed to replace the parameters in param_files !!!')
            print(fcontents)
            raise
        newfname = os.path.basename(fname)
        if os.path.exists(newfname):
            raise ValueError(f'{newfname} already exists here !!!\n'
                             +f'Remove {newfname} and try again.')
        with open(newfname,'w') as f:
            f.write(new_contents)
    return None


def main():
    import os,sys
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    # print(args)
    vfile = args['<v-file>']
    pfiles = args['<p-files>']

    kwargs = {}
    kwargs['param_files'] = pfiles
    
    # vs,vrs,vrsh,options,vopts = read_vars_optzer(vfile)
    vnames,vs,slims,hlims,options = read_vars_optzer(vfile)

    # for pfile in pfiles:
    #     with open(pfile,'r') as f:
    #         kwargs[pfile] = f.read()
    #     os.system(f'cp -f {pfile} {pfile}.bak')
    # vars2params(vs, **kwargs)
    print(' Convert the following files by replacing with optimized parameters:')
    for pfile in pfiles:
        newpfile = os.path.basename(pfile)
        if os.path.exists(newpfile):
            Exception(f'{newpfile} already exists and cannot overwrite it!')
        print(f'   - {pfile} ==> {newpfile}')
        with open(pfile,'r') as f:
            fcontent = f.read()
        newfcontent = fcontent.format(**vs)
        with open(newpfile,'w') as f:
            f.write(newfcontent)
    print('')
    return None

if __name__ == "__main__":
    main()

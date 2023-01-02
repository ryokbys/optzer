#!/usr/bin/env python
"""
Convert optzer parameters `in.vars.optzer` to parameter files.

Usage:
  {0:s} [options] <db-file> <p-files> [<p-files>...]

Options:
  -h, --help  Show this message and exit.
"""
import os
from docopt import docopt

from optzer.io import read_db_optzer

__author__ = "RYO KOBAYASHI"
__version__ = "230102"

def main():
    import os,sys
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    # print(args)
    dbfile = args['<db-file>']
    pfiles = args['<p-files>']

    kwargs = {}
    kwargs['param_files'] = pfiles
    
    db = read_db_optzer(dbfile)

    bestidx = db.loss.argmin()
    best = db.iloc[bestidx]
    cols = db.columns.to_list()
    cols.remove('iid')
    cols.remove('loss')
    cols.remove('gen')
    vs = { k:v for k,v in best.items() if k in cols }

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

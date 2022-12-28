#!/usr/bin/env python
"""
Get best data from the directory that optzer was performed.

Usage:
  {0:s} [options]

Options:
  -h, --help  Show this message and exit.
  --bestdir-name BESTDIR
              Best data dir name. [default: best_data]
"""
import os,sys
import shutil
from docopt import docopt
import pandas as pd

# from optzer.io import read_out_optzer

__author__ = "RYO KOBAYASHI"
__version__ = "221227"

_fname_db = 'db.optzer.json'

def read_output(fname='out.optzer'):
    """Read output from optzer and return some information."""
    with open(fname,'r') as f:
        lines = f.readlines()
    bestiid = -1
    for line in lines:
        if 'step,time' in line:
            data = line.split()
            bestiid = int(data[3])
            bestloss = float(data[4])
    return bestiid,bestloss

def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    bestdname = args['--bestdir-name']

    db = pd.read_json(_fname_db)
    bestidx = db.loss.argmin()
    bestiid = db.iid[bestidx]

    if os.path.exists(bestdname):
        shutil.rmtree(bestdname)

    # Look for the bestiid directory
    print(f' copying iid_{bestiid} ==> ./{bestdname}')
    cwd = os.getcwd()
    for path,dirs,files in os.walk(cwd):
        dname = os.path.basename(path)
        if dname == f'iid_{bestiid:d}':
            shutil.copytree(path, f'./{bestdname}')

    return None

if __name__ == "__main__":

    main()

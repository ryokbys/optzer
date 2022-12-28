#!/usr/bin/env python
"""
Convert db.optzer.json to csv with white-space separation.

Usage:
  {0:s} [options] <db-file> <csv-file>

Options:
  -h, --help  Show this message and exit.
  --sep SEP   Separator in csv format. [default: None]
"""
import os, sys
from docopt import docopt
import pandas as pd

# from optzer.io import read_out_optzer

__author__ = "RYO KOBAYASHI"
__version__ = "221227"

_fname_db = 'db.optzer.json'

def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    sep = args['--sep']
    if sep == 'None':
        sep = ' '
    dbfname = args['<db-file>']
    csvfname = args['<csv-file>']

    db = pd.read_json(dbfname)
    db.to_csv(csvfname, sep=sep)

    return None

if __name__ == "__main__":

    main()

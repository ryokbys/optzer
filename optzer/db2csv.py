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
from optzer.io import read_db_optzer

# from optzer.io import read_out_optzer

__author__ = "RYO KOBAYASHI"
__version__ = "221227"

_fname_db = 'db.optzer.json'


def write_db_csv(db,fname='out.db.csv'):
    cols = db.columns.to_list()
    cols.remove('iid')
    cols.remove('loss')
    cols.remove('gen')
    with open(fname,'w') as f:
        f.write(' {0:>6s} {1:>5s} {2:>11s}'.format('iid','gen','loss'))
        for c in cols:
            f.write(f' {c:>12s}')
        f.write('\n')
        for i in range(len(db)):
            row = db.iloc[i].to_dict()
            f.write(' {0:6d} {1:5d} {2:11.3e}'.format(int(row['iid']),
                                                      int(row['gen']),
                                                      row['loss']))
            for c in cols:
                f.write(f' {row[c]:11.3e}')
            f.write('\n')
    return None
    
def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    sep = args['--sep']
    if sep == 'None':
        sep = ' '
    dbfname = args['<db-file>']
    csvfname = args['<csv-file>']

    db = read_db_optzer(dbfname)
    write_db_csv(db,fname=csvfname)

    return None

if __name__ == "__main__":

    main()

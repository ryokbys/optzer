#!/usr/bin/env python
"""
Extract the target losses from optzer output and output as a markdown table.

Usage:
  {0:s} [options] <out-file>

Options:
  -h, --help     Show this message and exit.
  -i,--iid IID   Specify IID to be extracted. If not specified, the best one is chosen. [default: -1]
"""
import os,sys

from docopt import docopt
from optzer.io import read_out_optzer

__author__ = "Ryo KOBAYASHI"
__version__ = "230109"

def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    ofname = args['<out-file>']
    iid0 = int(args['--iid'])

    bestiid,targets,weights,losses = read_out_optzer(ofname)

    if iid0 > 0:
        iid = iid0
        if not iid in losses.keys():
            raise ValueError('No such iid in the losses.')
    else:
        iid = bestiid

    #...Markdown table
    maxlen = 0
    wxls = []
    for i,t in enumerate(targets):
        maxlen = max(maxlen,len(t))
        l = losses[iid][i]
        if t != 'total':
            w = weights[i]
            wxls.append(w*l)
        else:
            wxls.append(l)
    txt = '\n'
    if iid0 > 0:
        txt += f'Extracted iid = {iid:d}\n\n'
    else:
        txt += f'Best iid = {bestiid:d}\n\n'
    txt += f'|  Target  |  Weight |  Loss  | Weight x Loss |\n'
    txt +=  '|----------|---------|--------|---------------|\n'
    for i in range(len(targets)):
        t = targets[i]
        if t != 'total':
            w = weights[i]
            l = losses[iid][i]
            wxl = wxls[i]
            txt += f'|  {t:s}  |  {w:5.3f}  |  {l:.4f}  |  {wxl:.4f}  |\n'
        else:
            l = losses[iid][i]
            txt += f'|  {t:s}  |  -----  |  -----  |  {l:.4f}  |\n'
    print(txt)
    
    return None

if __name__ == "__main__":

    main()

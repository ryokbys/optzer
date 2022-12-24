#!/usr/bin/env python
"""
Output a markdown table of the summary of optzer results.

Usage:
  {0:s} [options] <out-file>

Options:
  -h, --help  Show this message and exit.
"""
import os,sys

from docopt import docopt
from .io import read_out_optzer

__author__ = "Ryo KOBAYASHI"
__version__ = "221224"

def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    ofname = args['<out-file>']

    bestiid,targets,weights,losses = read_out_optzer(ofname)

    #...Markdown table
    maxlen = 0
    wxls = []
    for i,t in enumerate(targets):
        maxlen = max(maxlen,len(t))
        l = losses[i]
        if t != 'total':
            w = weights[i]
            wxls.append(w*l)
        else:
            wxls.append(l)
    txt = '\n'
    txt += f'Best iid = {bestiid:d}\n\n'
    txt += f'|  Target  |  Weight |  Loss  | Weight x Loss |\n'
    txt +=  '|----------|---------|--------|---------------|\n'
    for i in range(len(targets)):
        t = targets[i]
        if t != 'total':
            w = weights[i]
            l = losses[i]
            wxl = wxls[i]
            txt += f'|  {t:s}  |  {w:5.3f}  |  {l:.4f}  |  {wxl:.4f}  |\n'
        else:
            l = losses[i]
            txt += f'|  {t:s}  |  -----  |  -----  |  {l:.4f}  |\n'
    print(txt)
    
    return None

if __name__ == "__main__":

    main()

import numpy as np

__author__ = "RYO KOBAYASHI"
__version__ = "221228"

def testfunc(var, **kwargs):
    x= var['x']
    y= var['y']
    res= x**2 +y**2 +100.0*exp(-x**2 -y**2)*sin(2.0*(x+y))*cos(2*(x-y)) \
         +80.0*exp(-(x-1)**2 -(y-1)**2)*cos(x+4*y)*sin(2*x-y) \
         +200.0*sin(x+y)*exp(-(x-3)**2-(y-1)**2)
    return res

def write_vars_for_testfunc(vs,slims,hlims,
                            fname='in.vars.optzer',**kwargs):
    vnames = vs.keys()
    with open(fname,'w') as f:
        f.write('  {0:d}\n'.format(len(vnames)))
        for k in vnames:
            f.write(' {0:10.3f}'.format(vs[k])
                    +'  {0:10.3f}  {1:10.3f}'.format(*slims[k])
                    +'  {0:10.3f}  {1:10.3f}'.format(*hlims[k])
                    +'  {0:s}\n'.format(k))
    return None


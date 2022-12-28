import numpy as np
import copy
import pandas as pd

__author__ = "RYO KOBAYASHI"
__version__ = "221227"

def ind_from_db(db,idx,vnames,slims):
    """Extract an individual from an index of the DB.
    Since the DB does not have information of variable names and limits,
    one should specify them."""
    di = db.iloc[idx].to_dict()
    iid = int(di['iid'])
    ind = Individual(iid, vnames)
    try:
        vs = { k:di[k] for k in vnames }
    except:
        raise
    ind.set_variables(vs, slims)
    ind.loss = di['loss']
    if 'gen' in di.keys():
        ind.gen = int(di['gen'])
    return ind

class Individual:
    """
    Individual class that consists of variables, soft and hard limits,
    and loss function.
    """
    def __init__(self, iid, vnames, gen=None):
        self.iid = iid
        self.vnames = vnames
        self.vs = { k:0.0 for k in vnames }
        self.gen = gen
        self.loss = None
        return None

    def set_variables(self,variables,slims):
        if len(variables) != len(self.vs):
            raise ValueError('len(variables) != len(self.vs)')

        self.vs = variables
        self.wrap(slims)
        self.loss = None
        return None

    def init_random(self, slims):
        for key in self.vnames:
            vmin, vmax = slims[key]
            v = np.random.random()*(vmax -vmin) +vmin
            self.vs[key] = v
        self.wrap(slims)
        self.loss = None
        return None

    def wrap(self, slims):
        """Wrap variables inside the given soft limits (slims)."""
        newvs = {}
        for k in self.vs.keys():
            vmin, vmax = slims[k]
            newvs[k] = min(max(self.vs[k],vmin),vmax)
        self.vs = newvs
        return None

    def calc(self,loss_func,kwargs):
        """
        Compute loss function value using self.loss_func function given in the constructor.
        """
        vs = self.vs
        if 'vlogs' in kwargs.keys():
            vlogs = kwargs['vlogs']
            for k in vs.keys():
                if vlogs[k]:
                    vs[k] = np.exp(vs[k])
        self.loss = loss_func(vs, **kwargs)
        return self.loss, kwargs['index']

    def to_DataFrame(self):
        """Return a pandas DataFrame of this individual."""
        data = {}
        data['iid'] = [self.iid]
        data['loss'] = [self.loss]
        data['gen'] = [self.gen]
        for k in self.vnames:
            data[k] = [self.vs[k]]
        return pd.DataFrame(data)

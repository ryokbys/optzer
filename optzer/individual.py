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

def update_slims(slims,hlims,history_db,ntops=100):
    """
    Update soft limits of variables adaptively using all the individuals
    information.
    """
    #...Extract top NTOPS individuals from the history DB
    losses = history_db['loss']
    tops = []
    for idx in history_db.index:
        if len(tops) < ntops:  # just add it
            for itop,idxtop in enumerate(tops):
                if losses[idx] < losses[idxtop]:
                    tops.insert(itop,idx)
                    break
            if not idx in tops:
                tops.append(idx)
        else: # insert the individual and pop out the worst one in the tops
            for itop,idxtop in enumerate(tops):
                if losses[idx] < losses[idxtop]:
                    tops.insert(itop,idx)
                    break
            if len(tops) > ntops:
                del tops[ntops:len(tops)]

    #...Get new ranges
    new_slims = {}
    vss = []
    for idx in tops:
        di = history_db.iloc[idx].to_dict()
        vsi = { k:di[k] for k in slims.keys() }
        vss.append(vsi)
    # for i,ind in enumerate(tops):
    #     vss.append(ind.vs)
    for k in slims.keys():
        vmin =  1e+30
        vmax = -1e+30
        for i in range(len(tops)):
            vi = vss[i][k]
            vmin = min(vi,vmin)
            vmax = max(vi,vmax)
        new_slims[k] = [vmin,vmax]

    #...Set best variables center in the ranges
    fbest = losses[tops[0]]
    d0 = history_db.iloc[tops[0]].to_dict()
    vbest = { k:d0[k] for k in slims.keys() }
    for k in slims.keys():
        vmin = new_slims[k][0]
        vmax = new_slims[k][1]
        wmax = max(abs(vmin-vbest[k]),abs(vmax-vbest[k]))
        new_slims[k][0] = max(min(vmin,vbest[k]-wmax),hlims[k][0])
        new_slims[k][1] = min(max(vmax,vbest[k]+wmax),hlims[k][1])
    
    return new_slims

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

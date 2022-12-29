#!/usr/bin/env python
"""
Cuckoo search.

Usage:
  cs.py [options]

Options:
  -h, --help  Show this message and exit.
  -n N        Number of generations in CS. [default: 20]
  --print-level LEVEL
              Print verbose level. [default: 1]
  --update-vrange NUPDATE
              Update variable range per N. [default: -1]
"""
import os
import sys
from docopt import docopt
import numpy as np
from numpy import exp, sin, cos
import random
import copy
#from multiprocessing import Process, Pool
from multiprocess import Pool
from time import time
from scipy.special import gamma
import pandas as pd

from optzer.individual import Individual, ind_from_db, update_slims
from optzer.testfunc import testfunc, write_vars_for_testfunc
from optzer.io import write_db_optzer, read_db_optzer

__author__ = "RYO KOBAYASHI"
__version__ = "221227"

_fname_gen = 'out.cs.generations'
_fname_ind = 'out.cs.individuals'
_fname_db = 'db.optzer.json'

def test_DB_vs_vnames(db, vnames):
    """Test the consistency between given DB and vnames."""
    colnames = db.columns
    for k in vnames:
        if not k in colnames:
            return False
    return True

class CS:
    """
    Cuckoo search class.
    """

    def __init__(self, nind, frac, vnames, vs0, slims,
                 hlims, loss_func, write_func=None,
                 nproc=0, seed=42, **kwargs):
        """
        Conctructor of CS class.

        nind:  Number of individuals.
        frac:  Fraction of worse individuals to be abondoned.
        loss_func:  function
            Loss function to be minimized with variables and **kwargs.
        nproc:  int
            Number of processes used to run N individuals.
        vnames:  list
            Variable names.
        vs0: dict
            Initial guess of variables.
        slims,hlims: dict
            Set of variables with names.
        """
        if nind < 2:
            raise ValueError('nind must be greater than 1 in CS!')
        np.random.seed(seed)
        random.seed(seed)
        self.nind = nind   # Number of individuals in a generation
        self.frac = frac   # Fraction of worse individuals to be abondoned
        self.nproc = nproc
        self.vnames = vnames
        self.vs0 = vs0
        self.slims = slims
        self.hlims = hlims
        self.vws = {}
        for k in vnames:
            self.vws[k] = max(self.slims[k][1] -self.slims[k][0], 0.0)
        self.loss_func = loss_func
        self.write_func = write_func
        self.kwargs = kwargs
        self.bestind = None
        self.print_level = 0
        if 'print_level' in kwargs.keys():
            self.print_level = int(kwargs['print_level'])
        
        if 'update_vrange' in kwargs.keys():
            self.update_slims_per = kwargs['update_vrange']
        else:
            self.update_slims_per = -1

        self.beta = 1.5
        self.betai = 1.0 /self.beta
        self.usgm = (gamma(1+self.beta)*np.sin(np.pi*self.beta/2)/ \
                     gamma((1+self.beta)/2)*self.beta*2.0**((self.beta-1)/2))**self.betai
        self.vsgm = 1.0

        cols = ['iid','loss','gen'] +self.vnames
        self.history_db = pd.DataFrame(columns=cols)
        #...Try to restart by loading db.optzer.json if exists
        if os.path.exists(_fname_db):
            try:
                self.history_db = read_db_optzer(_fname_db)
                print(f'\n Restarting with existing DB, {_fname_db}.')
            except Exception as e:
                print('\n !!!!!'
                      +f'\n Failed to load {_fname_db} for restart because of {e},'
                      +' even if it exists...'
                      +'\n So starting with the given initial guess.'
                      +'\n !!!!!\n')
                self.history_db = self.history_db[0:0]  # Remove all records

        #...Check the consistency between DB and given vnames
        if not test_DB_vs_vnames(self.history_db, self.vnames):
            raise ValueError('DB and vnames are not consistent, which is not allowed.\n'
                             +f'Please delete {_fname_db} and try again.')
                
        #...initialize population
        self.population = []
        if len(self.history_db) == 0:
            self.igen0 = 0
            self.iidinc = 0
            for i in range(self.nind):
                self.iidinc += 1
                ind = Individual(self.iidinc, self.vnames)
                if i == 0:
                    ind.set_variables(self.vs0, self.slims)
                else:
                    ind.init_random(self.slims)
                self.population.append(ind)
        else:  # DB loaded
            #...Use DB to narrow the soft limits
            self.slims = update_slims(self.slims,self.hlims,
                                      self.history_db)
            print(' Update variable ranges')
            for k in self.vnames:
                print(' {0:>15s}:  {1:7.3f}  {2:7.3f}'.format(k,
                                                              self.slims[k][0],
                                                              self.slims[k][1]))
            #...Get best one from the DB
            bestidx = self.history_db.loss.argmin()
            self.bestind = ind_from_db(self.history_db,
                                       bestidx,
                                       self.vnames, self.slims)
            #...Generate random population with the updated slims
            self.iidinc = int(self.history_db.iid.max()) +1
            self.igen0 = int(self.history_db.gen.max()) +1
            for i in range(self.nind):
                ind = Individual(self.iidinc, self.vnames)
                ind.init_random(self.slims)
                self.population.append(ind)
                self.iidinc += 1

        return None

    def keep_best(self):
        losses = []
        for i,pi in enumerate(self.population):
            if pi.loss == None:
                raise ValueError('Something went wrong.')
            losses.append(pi.loss)

        minloss = min(losses)
        if self.bestind == None or minloss < self.bestind.loss:
            idx = losses.index(minloss)
            self.bestind = copy.deepcopy(self.population[idx])
        return None

    def sort_individuals(self):
        """
        Sort individuals in the population in the ascending order.
        """
        jtop = self.nind
        for i in range(self.nind):
            jtop -= 1
            for j in range(jtop):
                pj = self.population[j]
                pjp = self.population[j+1]
                if pj.loss > pjp.loss:
                    self.population[j] = pjp
                    self.population[j+1] = pj
        return None
        
    def run(self,max_gen=100):
        """
        Perfom CS.
        """

        if 'start' in self.kwargs.keys():
            starttime = self.kwargs['start']
        else:
            starttime = time()

        #...Create pool before going into max_gen-loop,
        #...since creating pool inside could cause "Too many files" error.
        prcs = []
        if self.nproc > 0 :  # use specified number of cores by nproc
            pool = Pool(processes=self.nproc)
        else:
            pool = Pool()
            
        #...Evaluate loss function values
        for ip,ind in enumerate(self.population):
            kwtmp = copy.copy(self.kwargs)
            kwtmp['index'] = ip
            kwtmp['iid'] = ind.iid
            prcs.append(pool.apply_async(ind.calc, (self.loss_func,kwtmp,)))
        results = [ res.get() for res in prcs ]
        for res in results:
            loss,ip = res
            self.population[ip].loss = loss
            self.population[ip].gen = self.igen0

        self.keep_best()
        #...Create history DB if not exists
        pop2dfs = [ pi.to_DataFrame() for pi in self.population ]
        self.history_db = pd.concat([self.history_db] +pop2dfs,
                                    ignore_index=True)
            
        #...Once the history is updated, dump it to the file
        write_db_optzer(self.history_db,fname=_fname_db)
        
        if self.print_level > 0:
            self._write_step_info( self.igen0, starttime)

        for igen in range(self.igen0, self.igen0+max_gen):
            self.sort_individuals()
            #...Create candidates from current population using Levy flight
            candidates = []
            vbest = self.bestind.vs
            for ip,pi in enumerate(self.population):
                vi = pi.vs
                vnew = copy.copy(vi)
                for k in self.vnames:
                    u = np.random.normal()*self.usgm
                    v = abs(np.random.normal()*self.vsgm)
                    v = max(v,1.0e-8)
                    w = u/v**self.betai
                    zeta = self.vws[k] *0.01 *w
                    # zeta = self.vws[iv]*0.01 *w *(vi[iv] -vbest[iv])
                    # if ip == 0:
                    #     zeta = self.vws[iv] *0.001 *w
                    # else:
                    #     zeta = 0.01 *w *(vi[iv] -vbest[iv])
                    vnew[k] = vnew[k] +zeta*np.random.normal()
                #...create new individual for trial
                # print('ip,vi,vnew=',ip,vi,vnew)
                self.iidinc += 1
                newind = Individual(self.iidinc, self.vnames)
                newind.set_variables(vnew, self.slims)
                candidates.append(newind)

            #...Create new completely random candidates
            iab = int((1.0 -self.frac)*self.nind)
            rnd_candidates = []
            for iv in range(iab,self.nind):
                self.iidinc += 1
                newind = Individual(self.iidinc, self.vnames)
                newind.init_random(self.slims)
                rnd_candidates.append(newind)

            #...Evaluate loss function values of updated candidates and new random ones
            prcs = []
            for ic,ci in enumerate(candidates):
                kwtmp = copy.copy(self.kwargs)
                kwtmp['index'] = ic
                kwtmp['iid'] = ci.iid
                prcs.append(pool.apply_async(ci.calc, (self.loss_func,kwtmp,)))
            rnd_prcs = []
            for ic,ci in enumerate(rnd_candidates):
                kwtmp = copy.copy(self.kwargs)
                kwtmp['index'] = len(candidates) +ic
                kwtmp['iid'] = ci.iid
                rnd_prcs.append(pool.apply_async(ci.calc, (self.loss_func,kwtmp,)))
            
            results = [ res.get() for res in prcs ]
            rnd_results = [ res.get() for res in rnd_prcs ]

            for res in results:
                loss,ic = res
                candidates[ic].loss = loss
                candidates[ic].gen = igen+1
            db_add = [ c.to_DataFrame() for c in candidates ] 

            for res in rnd_results:
                loss,ic_rnd = res
                ic = ic_rnd -len(candidates)
                rnd_candidates[ic].loss = loss
                rnd_candidates[ic].gen = igen+1
            db_add.extend([ c.to_DataFrame() for c in rnd_candidates ])
            self.history_db = pd.concat([self.history_db] +db_add)
            self.history_db.drop_duplicates(subset='iid',
                                            inplace=True,
                                            ignore_index=True)
            #...Dump the DB once it is updated.
            write_db_optzer(self.history_db,fname=_fname_db)

            #...Pick j that is to be compared with i
            js = random.sample(range(self.nind),k=self.nind)
            #...Decide whether or not to adopt new one
            for jc,jv in enumerate(js):
                pj = self.population[jv]
                cj = candidates[jc]
                dloss = cj.loss -pj.loss
                if dloss < 0.0:  # replace with new individual
                    self.population[jv] = cj
                else:
                    pass

            #...Rank individuals
            self.sort_individuals()
            
            #...Replace to-be-abandoned ones with new random ones
            ic = 0
            for iv in range(iab,self.nind):
                ci = rnd_candidates[ic]
                ic += 1
                self.population[iv] = ci

            #...Check best
            best_updated = False
            for ic,ci in enumerate(self.population):
                if ci.loss < self.bestind.loss:
                    self.bestind = ci
                    best_updated = True
            if best_updated:
                self.write_variables(self.bestind,
                                     fname='in.vars.optzer.best',
                                     **self.kwargs)

            #...Update variable ranges if needed
            if self.update_slims_per > 0 and (igen+1) % self.update_slims_per == 0:
                self.slims = update_slims(self.slims,self.hlims,
                                          self.history_db)
                print(' Update variable ranges')
                for k in self.vnames:
                    print(' {0:>10s}:  {1:7.3f}  {2:7.3f}'.format(k,self.slims[k][0],self.slims[k][1]))
            
            if self.print_level > 0:
                self._write_step_info(igen, starttime)

        pool.close()
        #...Finaly write out the best one
        self.write_variables(self.bestind,
                             fname='in.vars.optzer.best',
                             **self.kwargs)
        return None

    def write_variables(self,ind,fname='in.vars.optzer',**kwargs):
        if self.write_func != None:
            self.write_func(ind.vnames, ind.vs, self.slims, self.hlims,
                            fname, **kwargs)
        return None

    def _write_step_info(self, istp, starttime):
        print(' step,time,best_iid,best_loss,vars='
              +' {0:6d} {1:8.1f} {2:5d} {3:8.4f}'.format(istp,
                                                         time()-starttime,
                                                         self.bestind.iid,
                                                         self.bestind.loss),
              end="")
        inc = 0
        for k in self.vnames:
            if inc < 16:
                print(' {0:6.3f}'.format(self.bestind.vs[k]),end="")
            else:
                break
            inc += 1
        print('', flush=True)
        return None
        

def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    ngen = int(args['-n'])
    kwargs = {}
    kwargs['print_level'] = int(args['--print-level'])
    kwargs['update_vrange'] = int(args['--update-vrange'])

    vnames = ['x','y']
    vs = { 'x':1.0, 'y':-0.5 }
    slims = { 'x':[-1.0, 2.0],'y':[-1.0, 1.0] }
    hlims = { 'x':[-2.0, 2.0],'y':[-2.0, 2.0] }
    
    cs = CS(10, 0.25, vnames, vs, slims, hlims,
            testfunc, write_vars_for_testfunc, **kwargs)
    cs.run(ngen)
    return None
    

if __name__ == "__main__":

    main()

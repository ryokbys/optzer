#!/usr/bin/env python
"""
Implicit Natural Gradient Optimization (INGO).
See,  Y. Lyu, and I.W. Tsang, “Black-box optimizer with implicit natural gradient,” arXiv [Cs.LG], (2019).

Usage:
  {0:s} [options]

Options:
  -h, --help  Show this message and exit.
  -n N        Number of samples in INGO. [default: 20]
  --print-level LEVEL
              Print verbose level. [default: 1]
  --algorithm ALGO
              Which variant of INGO algorithm to use.
              normal, step, or fast. [default: normal]
"""
import os, sys
from docopt import docopt
import numpy as np
import scipy
import random
import copy
#from multiprocessing import Process, Pool
from multiprocess import Pool
from time import time
import pandas as pd

from optzer.individual import Individual, ind_from_db, update_slims
from optzer.io import write_db_optzer, read_db_optzer

__author__ = "RYO KOBAYASHI"
__revision__ = "240831"

_fname_db = 'db.optzer.json'

def gauss_kernel(x):
    return np.exp(-0.5*x*x)/np.sqrt(2.0*np.pi)

class INGO:
    """
    Class for Implicit Natural Gradient Optimization (INGO).
    
    Refs:
      1. Y. Lyu, and I.W. Tsang, “Black-box optimizer with implicit natural gradient,” arXiv [Cs.LG], (2019).
    """

    def __init__(self, nbatch, vnames, vs0, slims, hlims, loss_func,
                 write_func=None, seed=42, nproc=1,
                 loss_criteria=-1.0,
                 stuck_criteria=-1,
                 **kwargs):
        """
        Conctructor of TPE class.

        Parameters:
          nbatch : int
              Number of samples in a batch which is the same as num of processes.
          vs0 : dict
              Initial guess of variables.
          slims : dict
              Current lower and upper limit of variables, which are used only at random sampling processes.
          hlims : dict
              Hard limits of variables, which are fixed during TPE iterations. 
          loss_func : function
              Loss function to be minimized with variables and **kwargs.
          nproc : int
              Number of processes.
          write_func : function
              Function for outputing some info.
          loss_criteria: float
              Convergence criterion for the loss.
              If negtive (default), not to set criterion.
          stuck_criteria: int
              Convergence criteria for number of generation within which the loss is not improved. 
              If negtive (default), not to set the criteria.
        """
        np.random.seed(seed)
        random.seed(seed)
        self.nbatch = nbatch
        self.nproc = nproc
        self.ndim = len(vs0)
        if self.nbatch <= self.ndim:
            raise ValueError(f'nbatch should be >= {self.ndim+1:d}.')
        self.vnames = vnames
        self.vs0 = vs0
        self.hlims = hlims
        self.slims = slims
        self.hlims = hlims
        self.vlogs = { k:False for k in vnames }
        if 'vlogs' in kwargs.keys():
            self.vlogs = kwargs['vlogs']
        self.loss_func = loss_func
        self.write_func = write_func
        self.loss_criteria = loss_criteria
        self.stuck_criteria = stuck_criteria
        self.num_loss_stuck = 0
        self.kwargs = kwargs
        self.best_pnt = None
        self.print_level = 0
        self.method = kwargs['opt_method']

        self.beta = 1.0/self.ndim
        #...Change default values if specified
        if 'print_level' in kwargs.keys():
            self.print_level = int(kwargs['print_level'])

        # #...Write info
        # print('')
        # if self.method in ('ingo','INGO'):
        #     print(' {0:s} information:'.format(self.method))

        # elif self.method in ('fastingo','fastINGO'):
        #     print(' {0:s} information:'.format(self.method))
        #     self.beta = 1.0 / np.sqrt(self.ndim)

        # print('')
    
        #...Change vrange if log domain
        for k,v in self.vlogs.items():
            if not v: continue
            self.vs0[k] = np.log(self.vs0[k])
            for l in range(2):
                self.slims[k][l] = np.log(self.slims[k][l])
                self.hlims[k][l] = np.log(self.hlims[k][l])
        # for i in range(self.ndim):
        #     if self.vlogs[i]:
        #         self.vs0[i] = np.log(self.vs0[i])
        #         for l in range(2):
        #             self.slims[i,l] = np.log(self.slims[i,l])
        #             self.hlims[i,l] = np.log(self.hlims[i,l])

        cols = ['iid','loss','gen'] +self.vnames
        self.history_db = pd.DataFrame(columns=cols)  # History of all samples
        #...Try to restart by loading db.optzer.json if exists
        if os.path.exists(_fname_db):
            try:
                self.history_db = read_db_optzer(_fname_db)
            except Exception as e:
                print(e)
                print(f'\n !!! Failed to load {_fname_db} for restart. !!!'
                      +'\n !!! So start with the given initial guess.     !!!')
            print(f'\n Restarting with existing DB, {_fname_db}.\n')

        #...Initialize sample history
        self.candidates = []
        self.igen0 = 0
        self.bestsmpl = None
        if len(self.history_db) == 0:
            self.iidinc = 0
            for i in range(self.nbatch):
                self.iidinc += 1
                smpl = Individual(self.iidinc, self.vnames )
                if i == 0:
                    smpl.set_variables(self.vs0, self.slims)
                else:
                    #...Random pick in soft bounds
                    smpl.init_random(self.slims)
                self.candidates.append(smpl)
        else:  # DB loaded
            #...Get best one from the DB
            bestidx = self.history_db.loss.argmin()
            self.bestsmpl = ind_from_db(self.history_db,
                                        bestidx,
                                        self.vnames, self.slims)
            # self.slims = update_slims(self.slims, self.hlims,
            #                           self.history_db,
            #                           ntops=max(100,int(len(self.history_db)*self.gamma)))
            #...Generate random population with the updated slims
            self.iidinc = self.history_db.iid.max() +1
            self.igen0 = self.history_db.gen.max() +1
            for i in range(self.nbatch):
                ind = Individual(self.iidinc, self.vnames)
                ind.init_random(self.slims)
                self.candidates.append(ind)
                self.iidinc += 1
            
        #...Compute mean and covariant matrix of the samples
        Xmat = np.zeros((self.nbatch, self.ndim))
        for i in range(self.nbatch):
            ci = self.candidates[i]
            vi = ci.get_variables()
            Xmat[i,:] = vi[:]
        self.mean = np.mean(Xmat, axis=0)
        self.cov = np.cov(Xmat, rowvar=False)
        if self.method in ('fastingo', 'fastINGO'):
            self.cov = diag_only(self.cov)
        import json
        return None

    def keep_best(self):
        losses = []
        for i,si in enumerate(self.candidates):
            if si.loss == None:
                raise ValueError('Something went wrong.')
            losses.append(si.loss)

        minloss = min(losses)
        if self.bestsmpl == None or minloss < self.bestsmpl.loss:
            idx = losses.index(minloss)
            self.bestsmpl = copy.deepcopy(self.candidates[idx])
        return None

    def run(self,maxstp=0):
        """
        Perfom TPE.
        """

        starttime = time()

        #...Create pool before going into maxstp-loop,
        #...since creating pool inside could cause "Too many files" error.
        prcs = []
        if self.nproc > 1:
            pool = Pool(processes=self.nproc)
        else:
            pool = Pool()

        #...Sanity check before calc loss
        self._sanity_check()
            
        #...Evaluate sample losses of initial sets
        for i,ci in enumerate(self.candidates):
            kwtmp = copy.copy(self.kwargs)
            kwtmp['index'] = i
            kwtmp['iid'] = ci.iid
            prcs.append(pool.apply_async(ci.calc, (self.loss_func,kwtmp,)))

        results = [ res.get() for res in prcs ]
        for res in results:
            loss, losses, i = res
            self.candidates[i].loss = loss
            self.candidates[i].losses = losses
            self.candidates[i].gen = self.igen0
        db_add = [ ci.to_DataFrame() for ci in self.candidates ]
        self.history_db = pd.concat([self.history_db] +db_add)
        self.history_db.drop_duplicates(subset='iid',
                                        inplace=True,
                                        ignore_index=True)
        #...Dump the DB once it is updated.
        write_db_optzer(self.history_db,fname=_fname_db)

        #...report after calc loss
        self._report()

        # self.slims = update_slims(self.slims, self.hlims,
        #                           self.history_db,
        #                           ntops=max(100,int(len(self.history_db)*self.gamma)))
        
        #...Check best
        self.keep_best()
        self.write_variables(self.bestsmpl,
                             fname='in.vars.optzer.best',
                             **self.kwargs)
        
        if self.print_level > 0:
            self._write_step_info(self.igen0,starttime)

        if self.loss_criteria > 0.0:
            if self.bestsmpl.loss < self.loss_criteria:
                print(' Convergence achieved since the best loss < loss_criteria.\n'
                      +'   Best loss and criterion = '
                      +'{0:.3f}  {1:.3f}'.format(self.bestsmpl.loss,
                                                 self.loss_criteria))
                return None

        # #...Set var mean as the best one
        # self.mean[:] = self.bestsmpl.get_variables()[:]
            
        self.num_loss_stuck = 0
        #...INGO loop starts
        for igen in range(self.igen0+1, self.igen0+1+maxstp):
            #...Create candidates
            if self.method in ('fastingo','fastINGO'):
                #...Create candidates by fastINGO
                self.candidates = self._candidates_by_fastINGO()
            else:
                #...Create candidates by normal INGO
                self.candidates = self._candidates_by_INGO()

            #...Sanity check before calc loss
            self._sanity_check()
            
            #...Evaluate sample losses
            prcs = []
            for i,ci in enumerate(self.candidates):
                kwtmp = copy.copy(self.kwargs)
                kwtmp['index'] = i
                kwtmp['iid'] = ci.iid
                prcs.append(pool.apply_async(ci.calc, (self.loss_func,kwtmp,)))

            results = [ res.get() for res in prcs ]
            for res in results:
                loss, losses, i = res
                self.candidates[i].loss = loss
                self.candidates[i].losses = losses
                self.candidates[i].gen = igen
            db_add = [ c.to_DataFrame() for c in self.candidates ]
            self.history_db = pd.concat([self.history_db] +db_add)
            self.history_db.drop_duplicates(subset='iid',
                                            inplace=True,
                                            ignore_index=True)
            #...Dump the DB once it is updated.
            write_db_optzer(self.history_db,fname=_fname_db)

            #...report after calc loss
            self._report()

            # self.slims = update_slims(self.slims, self.hlims,
            #                           self.history_db,
            #                           ntops=max(100,int(len(self.history_db)*self.gamma)))

            #...Check best
            best_updated = False
            for ip,ci in enumerate(self.candidates):
                if ci.loss < self.bestsmpl.loss:
                    self.bestsmpl = ci
                    best_updated = True
            if best_updated:
                self.write_variables(self.bestsmpl,
                                     fname='in.vars.optzer.best',
                                     **self.kwargs)
                self.num_loss_stuck = 0
            else:
                self.num_loss_stuck += 1

            #...Write info
            if self.print_level > 0:
                self._write_step_info(igen,starttime)
            
            if self.loss_criteria > 0.0:
                if self.bestsmpl.loss < self.loss_criteria:
                    print(' Convergence achieved since the best loss < loss_criteria.\n'
                          +'   Best loss and criterion = '
                          +'{0:.3f}  {1:.3f}'.format(self.bestsmpl.loss,
                                                     self.loss_criteria))
                    return None
            if self.stuck_criteria > 0 and \
               self.num_loss_stuck > self.stuck_criteria:
                print(' Convergence achieved since the num of consecutive generations with no improvement'
                      +' exceeds <stuck_criteria> = {0:d}.\n'.format(self.stuck_criteria))
                return None

        pool.close()
        print(' Finished since it exceeds the max iteration')
        return None

    def _write_step_info(self,istp,starttime):
        print(' step,time,best_iid,best_loss='
              +' {0:6d} {1:8.1f} {2:5d} {3:8.4f}'.format(istp,
                                                         time()-starttime,
                                                         self.bestsmpl.iid,
                                                         self.bestsmpl.loss),end="")
        # inc = 0
        # for k in self.vnames:
        #     if inc < 16:
        #         print(' {0:6.3f}'.format(self.bestsmpl.vs[k]),end="")
        #     else:
        #         break
        #     inc += 1
        print('', flush=True)
        return None

    def _candidates_by_INGO(self, ):
        """
        Create candidates by the INGO normal algorithm.
        """
        N = self.nbatch
        losses = np.zeros(len(self.candidates))
        xs = np.zeros((len(losses),self.ndim))
        for i, ci in enumerate(self.candidates):
            losses[i] = ci.loss
            for j,vn in enumerate(ci.vnames):
                xs[i,j] = ci.vs[vn]

        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        icov = np.linalg.inv(self.cov)

        #...New inverse covariance matrix
        new_icov = np.zeros(icov.shape)
        new_icov += icov
        for i in range(len(losses)):
            xi = xs[i] -self.mean
            coeff = (losses[i] -loss_mean)/(loss_std*N)
            x_outer = np.outer(xi,xi)
            new_icov += self.beta *coeff \
                *np.dot(icov, np.dot(x_outer, icov))
        #...New mean
        new_mean = np.zeros(self.mean.shape)
        new_mean += self.mean
        new_cov = np.linalg.inv(new_icov)
        for i in range(len(losses)):
            xi = xs[i] -self.mean
            coeff = (losses[i] -loss_mean)/(loss_std*N)
            new_mean -= self.beta *coeff \
                *np.dot(np.dot(new_cov, icov), xi)

        #...Create candidates with variables sampled from new mean and cov
        self.cov = new_cov
        self.mean = new_mean
        zero_mean = np.zeros(self.ndim)
        sqcov = scipy.linalg.sqrtm(self.cov)
        candidates = []
        for ib in range(self.nbatch):
            #...Create 
            zi = np.random.multivariate_normal(zero_mean, np.eye(self.ndim))
            xi = self.mean + np.dot(sqcov, zi)
            self.iidinc += 1
            smpl = Individual(self.iidinc, self.vnames)
            vsnew = {}
            for j,k in enumerate(self.vnames):
                vsnew[k] = xi[j]
            smpl.set_variables(vsnew, self.slims)
            candidates.append(smpl)
        return candidates

    def _candidates_by_fastINGO(self, ):
        """
        Create candidates by the fastINGO algorithm.
        """
        N = self.nbatch
        losses = np.zeros(len(self.candidates))
        xs = np.zeros((len(losses),self.ndim))
        for i, ci in enumerate(self.candidates):
            losses[i] = ci.loss
            for j,vn in enumerate(ci.vnames):
                xs[i,j] = ci.vs[vn]

        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        icov = np.linalg.inv(self.cov)

        #...New inverse covariance matrix
        new_icov = np.zeros(icov.shape)
        new_icov += icov
        for i in range(len(losses)):
            xi = xs[i] -self.mean
            coeff = (losses[i] -loss_mean)/(loss_std*N)
            x_outer = np.outer(xi,xi)
            new_icov += self.beta *coeff \
                *np.dot(icov, np.dot(x_outer, icov))
        #...New mean
        new_mean = np.zeros(self.mean.shape)
        new_mean += self.mean
        new_cov = np.linalg.inv(new_icov)
        new_cov = diag_only(new_cov)
        for i in range(len(losses)):
            xi = xs[i] -self.mean
            coeff = (losses[i] -loss_mean)/(loss_std*N)
            new_mean -= self.beta *coeff \
                *np.dot(np.dot(new_cov, icov), xi)

        #...Create candidates with variables sampled from new mean and cov
        self.cov = new_cov
        self.mean = new_mean
        zero_mean = np.zeros(self.ndim)
        sqcov = scipy.linalg.sqrtm(self.cov)
        candidates = []
        for ib in range(self.nbatch):
            #...Create 
            zi = np.random.multivariate_normal(zero_mean, np.eye(self.ndim))
            xi = self.mean + np.dot(sqcov, zi)
            self.iidinc += 1
            smpl = Individual(self.iidinc, self.vnames)
            vsnew = {}
            for j,k in enumerate(self.vnames):
                vsnew[k] = xi[j]
            smpl.set_variables(vsnew, self.slims)
            candidates.append(smpl)
        return candidates


    def _report(self,):
        for i,ci in enumerate(self.candidates):
            vi = ci.get_variables()
            li = ci.loss
            print(f' {vi} --> {li:8.4f}')
        sys.stdout.flush()
        return None

        
    def _sanity_check(self,):
        eigvals, eigvecs = np.linalg.eig(self.cov)
        rvals = np.real(eigvals)
        print(' mean:  ',end='')
        for i in range(len(self.mean)):
            print(f' {self.mean[i]:6.3f}',end='')
        print('')
        print(' std:   ',end='')
        for i in range(len(self.mean)):
            print(f' {np.sqrt(self.cov[i,i]):6.3f}',end='')
        print('')
        print(' evals: ',end='')
        for i in range(len(self.mean)):
            print(f' {rvals[i]:6.3f}',end='')
        print('')
        if np.any(np.abs(rvals) < np.finfo(float).eps):
            raise ValueError(' Some eigen values of Cov[X] are zero...\n'
                             +' --> Inversion of Cov[X] will fail.')
        elif np.any(rvals < 0.0):
            # print(' Real(eigen values):')
            # print('   ',rvals)
            raise ValueError(' Some eigen values of Cov[X] are negative...')
        sys.stdout.flush()
        return None

    
    def write_variables(self,smpl,fname='in.vars.optzer',**kwargs):
        if self.write_func != None:
            self.write_func(smpl.vnames, smpl.vs, self.slims, self.hlims,
                            fname, **kwargs)
        return None

def diag_only(mat):
    N = len(mat)
    for i in range(N):
        for j in range(N):
            if j == i: continue
            mat[i,j] = 0.0
    return mat
    
def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    ngen = int(args['-n'])
    kwargs = {}
    kwargs['print_level'] = int(args['--print-level'])
    kwargs['algorithm'] = int(args['--algorithm'])
    # kwargs['update_vrange'] = int(args['--update-vrange'])

    vnames = ['x','y']
    vs = { 'x':1.0, 'y':-0.5 }
    slims = { 'x':[-2.0, 2.0],'y':[-2.0, 2.0] }
    hlims = { 'x':[-2.0, 2.0],'y':[-2.0, 2.0] }
    
    ingo = INGO(10, 0.25, vnames, vs, slims, hlims,
                testfunc, write_vars_for_testfunc, **kwargs)
    ingo.run(ngen)
    return None

if __name__ == "__main__":

    main()

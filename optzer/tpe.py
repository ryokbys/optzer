#!/usr/bin/env python
"""
Tree-based Parzen Estimator (TPE).

Usage:
  {0:s} [options]

Options:
  -h, --help  Show this message and exit.
"""
import os, sys
from docopt import docopt
import numpy as np
import random
import copy
#from multiprocessing import Process, Pool
from multiprocess import Pool
from time import time
import pandas as pd

from optzer.individual import Individual, ind_from_db, update_slims
from optzer.io import write_db_optzer, read_db_optzer

__author__ = "RYO KOBAYASHI"
__version__ = "221228"

_fname_db = 'db.optzer.json'

def gauss_kernel(x):
    return np.exp(-0.5*x*x)/np.sqrt(2.0*np.pi)

class TPE:
    """
    Class for Tree-based Parzen Estimator (TPE) or Weighted Parzen Estimator (WPE).
    
    Refs:
      1. Bergstra, J., Bardenet, R., Bengio, Y. & Kégl, B.  in Proc. NIPS-24th, 2546–2554
    """

    def __init__(self, nbatch, vnames, vs0, slims, hlims, loss_func,
                 write_func=None, seed=42, criterion=-1.0,
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
          write_func : function
              Function for outputing some info.
          criterion: float
              Convergence criterion for the loss.
              If negtive (default), not to set criterion.
        """
        if nbatch < 1:
            raise ValueError('nbatch must be > 0.')
        np.random.seed(seed)
        random.seed(seed)
        self.nbatch = nbatch
        self.ndim = len(vs0)
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
        self.criterion = criterion
        self.kwargs = kwargs
        self.best_pnt = None
        self.print_level = 0
        self.nsmpl_prior = 100
        self.ntrial = 100
        self.method = kwargs['opt_method']

        self.gamma = 0.15
        #...Change default values if specified
        if 'print_level' in kwargs.keys():
            self.print_level = int(kwargs['print_level'])
        if 'tpe_nsmpl_prior' in kwargs.keys():
            self.nsmpl_prior = int(kwargs['tpe_nsmpl_prior'])
        if 'tpe_ntrial' in kwargs.keys():
            self.ntrial = int(kwargs['tpe_ntrial'])
        if 'tpe_gamma' in kwargs.keys():
            self.gamma = float(kwargs['tpe_gamma'])

        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError('gamma must be within 0. and 1., whereas gamma = ',self.gamma)

        #...Write info
        print('')
        if self.method in ('wpe','WPE'):
            print(' {0:s} infomation:'.format(self.method))
            print(f'   Num of prior samples = {self.nsmpl_prior:d}')
            print(f'   Num of top samples used for density estimation = {self.ntrial:d}')
        elif self.method in ('tpe','TPE'):
            print(' {0:s} infomation:'.format(self.method))
            print(f'   Num of prior samples = {self.nsmpl_prior:d}')
            print(f'   Num of trials for sampling = {self.ntrial:d}')
            print(f'   Gamma for dividing high and low = {self.gamma:4.2f}')
        print('')
    
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
            self.slims = update_slims(self.slims, self.hlims,
                                      self.history_db,
                                      ntops=max(100,int(len(self.history_db)*self.gamma)))
            #...Generate random population with the updated slims
            self.iidinc = self.history_db.iid.max() +1
            self.igen0 = self.history_db.gen.max() +1
            for i in range(self.nbatch):
                ind = Individual(self.iidinc, self.vnames)
                ind.init_random(self.slims)
                self.candidates.append(ind)
                self.iidinc += 1
            
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

        #...Whether of not kwargs contains 'temperature'...
        if 'initial_temperature' in self.kwargs.keys():
            tini = self.kwargs['initial_temperature']
            tfin = self.kwargs['final_temperature']
            self.kwargs['temperature'] = tini
            dtemp = (tfin -tini) /max(maxstp,1)
            print('   Initial,final,delta temperatures = '
                  + f'{tini:.1f}  {tfin:.1f}  {dtemp:.3f}')
            print('',flush=True)
        elif 'temperature' in self.kwargs.keys():
            print('   Temperature = {0:.1f}'.format(self.kwargs['temperature']))
            print('',flush=True)
            
        #...Create pool before going into maxstp-loop,
        #...since creating pool inside could cause "Too many files" error.
        pool = Pool(processes=self.nbatch)

        #...Evaluate sample losses of initial sets
        prcs = []
        for i,ci in enumerate(self.candidates):
            kwtmp = copy.copy(self.kwargs)
            kwtmp['index'] = i
            kwtmp['iid'] = ci.iid
            prcs.append(pool.apply_async(ci.calc, (self.loss_func,kwtmp,)))

        results = [ res.get() for res in prcs ]
        for res in results:
            loss, i = res
            self.candidates[i].loss = loss
            self.candidates[i].gen = self.igen0
        db_add = [ ci.to_DataFrame() for ci in self.candidates ]
        self.history_db = pd.concat([self.history_db] +db_add)
        self.history_db.drop_duplicates(subset='iid',
                                        inplace=True,
                                        ignore_index=True)
        #...Dump the DB once it is updated.
        write_db_optzer(self.history_db,fname=_fname_db)

        self.slims = update_slims(self.slims, self.hlims,
                                  self.history_db,
                                  ntops=max(100,int(len(self.history_db)*self.gamma)))
        
        #...Check best
        self.keep_best()
        self.write_variables(self.bestsmpl,
                             fname='in.vars.optzer.best',
                             **self.kwargs)
        
        if self.print_level > 0:
            self._write_step_info(self.igen0,starttime)

        if self.criterion > 0.0:
            if self.bestsmpl.loss < self.criterion:
                print(' Convergence achieved since the best loss < criterion.\n'
                      +'   Best loss and criterion = '
                      +'{0:.3f}  {1:.3f}'.format(self.bestsmpl.loss,
                                                 self.criterion))
                return None

        #...TPE loop starts
        for igen in range(self.igen0+1, self.igen0+1+maxstp):
            #...Create candidates by either random or TPE
            if len(self.history_db) <= self.nsmpl_prior:
                #...Create random candidates
                self.candidates = []
                for i in range(self.nbatch):
                    self.iidinc += 1
                    newsmpl = Individual(self.iidinc, self.vnames)
                    newsmpl.init_random(self.slims)
                    self.candidates.append(newsmpl)
            else:
                if self.method in ('wpe,''WPE'):
                    temp = 1.0
                    if 'temperature' in self.kwargs.keys():
                        temp = self.kwargs['temperature']
                    #...Create candidates by WPE
                    self.candidates = self._candidates_by_WPE(temperature=temp)
                    if 'initial_temperature' in self.kwargs.keys():
                        self.kwargs['temperature'] += dtemp
                else:
                    #...Create candidates by TPE
                    self.candidates = self._candidates_by_TPE()

            #...Evaluate sample losses
            prcs = []
            for i,ci in enumerate(self.candidates):
                kwtmp = copy.copy(self.kwargs)
                kwtmp['index'] = i
                kwtmp['iid'] = ci.iid
                prcs.append(pool.apply_async(ci.calc, (self.loss_func,kwtmp,)))

            results = [ res.get() for res in prcs ]
            for res in results:
                loss, i = res
                self.candidates[i].loss = loss
                self.candidates[i].gen = igen
            db_add = [ c.to_DataFrame() for c in self.candidates ]
            self.history_db = pd.concat([self.history_db] +db_add)
            self.history_db.drop_duplicates(subset='iid',
                                            inplace=True,
                                            ignore_index=True)
            #...Dump the DB once it is updated.
            write_db_optzer(self.history_db,fname=_fname_db)

            self.slims = update_slims(self.slims, self.hlims,
                                      self.history_db,
                                      ntops=max(100,int(len(self.history_db)*self.gamma)))

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

            #...Write info
            if self.print_level > 0:
                self._write_step_info(igen,starttime)
            
            if self.criterion > 0.0:
                if self.bestsmpl.loss < self.criterion:
                    print(' Convergence achieved since the best loss < criterion.\n'
                          +'   Best loss and criterion = '
                          +'{0:.3f}  {1:.3f}'.format(self.bestsmpl.loss,
                                                     self.criterion))
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

    def _candidates_by_TPE(self,):
        """
        Create candidates by using TPE.
        """
        losses = self.history_db.loss
        nlow = int(self.gamma *len(losses))
        nhigh = len(losses) -nlow
        argpart = np.argpartition(losses,nlow)
        argsrt = np.argsort(losses)
        Xlow = np.zeros((nlow,self.ndim))
        Xhigh = np.zeros((nhigh,self.ndim))
        for i in range(nlow):
            #idx = argpart[i]
            idx = argsrt[i]
            vs = ind_from_db(self.history_db,idx,self.vnames,self.slims).vs
            for j,k in enumerate(self.vnames):
                Xlow[i,j] = vs[k]
        for i in range(nhigh):
            #idx = argpart[nlow+i]
            idx = argsrt[nlow+i]
            vs = ind_from_db(self.history_db,idx,self.vnames,self.slims).vs
            for j,k in enumerate(self.vnames):
                Xhigh[i,j] = vs[k]
        

        #...Sampling variable candidates
        xcandidates = np.empty((self.nbatch,self.ndim))
        ntrial = max(self.ntrial,self.nbatch)
        # print('')
        for idim in range(self.ndim):
            # print('idim = ',idim)
            key = self.vnames[idim]
            # xhmin = self.hlims[key][0]
            # xhmax = self.hlims[key][1]
            xhmin = self.slims[key][0]
            xhmax = self.slims[key][1]
            xlowsrt = np.sort(Xlow[:,idim])
            npnt = len(xlowsrt)
            #...Determine smoothness parameter, h, by Silverman's method
            q75, q25 = np.percentile(xlowsrt, [75,25])
            std = np.std(xlowsrt)
            sgm = min(std, (q75-q25)/1.34)
            h = 1.06 *sgm /np.power(npnt, 1.0/5)
            # print(f'  q25,q75,std,sgm,h= {q25:.3f}, {q75:3f},'
            #       +f' {std:.3f}, {sgm:.3f}, {h:.3f}')
            # print('  xlowsrt=',xlowsrt)
            #...Prepare for g(x)
            xhighsrt = Xhigh[:,idim]
            q75, q25 = np.percentile(xhighsrt, [75,25])
            sgmh = min(np.std(xhighsrt), (q75-q25)/1.34)
            hh = 1.06 *sgmh /np.power(len(xhighsrt),1.0/5)
            # print(f'  q25,q75,sgmh,hh= {q25:.3f}, {q75:3f},'
            #       +f' {sgmh:.3f}, {hh:.3f}')
            #...Several trials for selection
            aquisition = np.zeros(ntrial)
            xs = np.empty(ntrial)
            for itry in range(ntrial):
                ipnt = int(np.random.random()*npnt)
                xi = xlowsrt[ipnt]
                r = h *np.sqrt(-2.0*np.log(np.random.random()))
                th = 2.0 *np.pi *np.random.random()
                x = xi + r*np.cos(th)
                #...Wrap by vranges
                x = min(max(x,xhmin),xhmax)
                xs[itry] = x
                #...Compute l(x) and g(x)
                lx = 0.0
                for j in range(npnt):
                    z = (x-xlowsrt[j])/h
                    lx += np.exp(-0.5*z*z)
                lx /= npnt*h *np.sqrt(2.0*np.pi)
                gx = 0.0
                for j in range(len(xhighsrt)):
                    z = (x-xhighsrt[j])/hh
                    gx += np.exp(-0.5*z*z)
                gx /= len(xhighsrt)*hh *np.sqrt(2.0*np.pi)
                aquisition[itry] = gx/lx
            #...Pick nbatch of minimum aquisition points
            idxsort = np.argsort(aquisition)
            xcandidates[:,idim] = xs[idxsort[0:self.nbatch]]
            # print('  aquisition=',aquisition[idxsort[0:self.nbatch]])
            # print('  xcandidates=',xcandidates[:,idim])
        #...Create sample with xcandidate as variables
        candidates = []
        for ib in range(self.nbatch):
            self.iidinc += 1
            smpl = Individual(self.iidinc, self.vnames)
            vsnew = {}
            for j,k in enumerate(self.vnames):
                vsnew[k] = xcandidates[ib,j]
            smpl.set_variables(vsnew, self.slims)
            candidates.append(smpl)
        return candidates

    def _candidates_by_WPE(self,veps=1e-8,temperature=1.0):
        """
        Create candidates by using WPE.
        """
        losses = self.history_db.loss
        if len(self.history_db) > self.ntrial:
            iargs = np.argpartition(losses, self.ntrial)
            #tmpsmpls = [ self.history[i] for i in iargs[:self.ntrial] ]
            tmpsmpls = [ ind_from_db(self.history_db,idx,self.vnames,self.slims)
                         for idx in iargs[:self.ntrial] ]
            losses = losses[ iargs[:self.ntrial] ]
        else:
            #tmpsmpls = copy.copy(self.history)
            tmpsmpls = [ ind_from_db(self.history_db,idx,self.vnames,self.slims)
                         for idx in self.history_db.index ]

        vmin = losses.min()
        wgts = np.array([ np.exp(-(v-vmin)/(vmin+veps)/temperature)
                          for v in losses ])
        xtmps = np.zeros((len(tmpsmpls),self.ndim))
        for i in range(len(tmpsmpls)):
            for j,k in enumerate(self.vnames):
                xtmps[i,j] = tmpsmpls[i].vs[k]
        #...Sampling variable candidates
        xcandidates = np.empty((self.nbatch,self.ndim))
        for idim in range(self.ndim):
            key = self.vnames[idim]
            # xhmin = self.hlims[key][0]
            # xhmax = self.hlims[key][1]
            xhmin = self.slims[key][0]
            xhmax = self.slims[key][1]
            xsrt = np.sort(xtmps[:,idim])
            #...Determine smoothness parameter, h, by Silverman's method
            q75, q25 = np.percentile(xsrt, [75,25])
            sgm = min(np.std(xsrt), (q75-q25)/1.34)
            h = 1.06 *sgm /np.power(len(xsrt), 1.0/5)
            xs = np.empty(self.nbatch)
            for ib in range(self.nbatch):
                ipnt = random.choices([j for j in range(len(wgts))],weights=wgts)[0]
                xi = xtmps[ipnt,idim]
                r = h *np.sqrt(-2.0*np.log(np.random.random()))
                th = 2.0 *np.pi *np.random.random()
                x = xi + r*np.cos(th)
                #...Wrap by slims
                x = min(max(x,xhmin),xhmax)
                xs[ib] = x
            xcandidates[:,idim] = xs[:]
        
        #...Create sample with xcandidate as variables
        candidates = []
        for ib in range(self.nbatch):
            self.iidinc += 1
            smpl = Individual(self.iidinc, self.vnames)
            vsnew = {}
            for j,k in enumerate(self.vnames):
                vsnew[k] = xcandidates[ib,j]
            smpl.set_variables(vsnew, self.slims)
            candidates.append(smpl)
        
        return candidates

    def write_variables(self,smpl,fname='in.vars.optzer',**kwargs):
        if self.write_func != None:
            self.write_func(smpl.vnames, smpl.vs, self.slims, self.hlims,
                            fname, **kwargs)
        return None

def main():
    args = docopt(__doc__.format(os.path.basename(sys.argv[0])))
    return None

if __name__ == "__main__":

    main()

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
from multiprocessing import Process, Pool
from time import time
from scipy.special import gamma

from optzer.individual import Individual

__author__ = "RYO KOBAYASHI"
__version__ = "221227"

_fname_gen = 'out.cs.generations'
_fname_ind = 'out.cs.individuals'

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

def update_vrange(slims,hlims,all_indivisuals,ntops=100):
    """
    Update variable ranges adaptively using all the individuals information.
    """
    #...Extract top NTOPS individuals from all
    tops = []
    # print('len(all_indivisuals)=',len(all_indivisuals))
    for i,ind in enumerate(all_indivisuals):
        if len(tops) < ntops:  # add the individual
            # print(' i (< ntops)=',i)
            for it,t in enumerate(tops):
                if ind.loss < t.loss:
                    tops.insert(it,ind)
                    break
            if not ind in tops:
                tops.append(ind)
        else: # insert the individual and pop out the worst one
            # print(' i (>=ntops)=',i)
            for it,t in enumerate(tops):
                if ind.loss < t.loss:
                    tops.insert(it,ind)
                    break
            if len(tops) > ntops:
                del tops[ntops:len(tops)]

    #...Get new ranges
    new_slims = {}
    vss = []
    for i,ind in enumerate(tops):
        vss.append(ind.vs)
    for k in slims.keys():
        vmin =  1e+30
        vmax = -1e+30
        for i in range(len(tops)):
            vi = vss[i][k]
            vmin = min(vi,vmin)
            vmax = max(vi,vmax)
        new_slims[k] = [vmin,vmax]

    #...Set best variables center in the ranges
    fbest = tops[0].loss
    vbest = tops[0].vs
    for k in slims.keys():
        vmin = new_slims[k][0]
        vmax = new_slims[k][1]
        wmax = max(abs(vmin-vbest[k]),abs(vmax-vbest[k]))
        new_slims[k][0] = max(min(vmin,vbest[k]-wmax),hlims[k][0])
        new_slims[k][1] = min(max(vmax,vbest[k]+wmax),hlims[k][1])
    
    return new_slims

class CS:
    """
    Cuckoo search class.
    """

    def __init__(self, nind, frac, vnames, vs, slims,
                 hlims, loss_func, write_func,
                 nproc=0,seed=42,**kwargs):
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
        vs: dict
            Variables with names.
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
        self.vs = vs
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

        #...initialize population
        self.population = []
        self.all_indivisuals = []
        self.iidinc = 0
        for i in range(self.nind):
            self.iidinc += 1
            ind = Individual(self.iidinc, self.vnames, self.slims,
                             self.hlims, self.loss_func)
            if i == 0:
                ind.set_variable(self.vs)
            else:
                ind.init_random()
            self.population.append(ind)

        #...Evaluate loss function values
        prcs = []
        if self.nproc > 0 :  # use specified number of cores by nproc
            pool = Pool(processes=self.nproc)
        else:
            pool = Pool()
            
        for ip,ind in enumerate(self.population):
            kwtmp = copy.copy(self.kwargs)
            kwtmp['index'] = ip
            kwtmp['iid'] = ind.iid
            #prcs.append(pool.apply_async(pi.calc_loss_func, (kwtmp,qs[ip])))
            prcs.append(pool.apply_async(ind.calc_loss_func, (kwtmp,)))
        results = [ res.get() for res in prcs ]
        for res in results:
            loss,ip = res
            self.population[ip].loss = loss

        pool.close()
        
        self.keep_best()
        self.all_indivisuals.extend(self.population)
        if self.print_level > 2:
            for ind in self.population:
                fname = 'in.vars.optzer.{0:d}'.format(ind.iid)
                self.write_variables(ind,
                                     fname=fname,
                                     **self.kwargs)
        else:
            fname = 'in.vars.optzer.{0:d}'.format(self.bestind.iid)
            self.write_variables(self.bestind,
                                 fname=fname,
                                 **self.kwargs)

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
            start = self.kwargs['start']
        else:
            start = time()
        fgen = open(_fname_gen,'w')
        find = open(_fname_ind,'w')
        #...Headers
        fgen.write('# {0:>4s}  {1:>8s}  {2:12s}\n'.format('gen','iid','loss'))
        find.write('# {0:>7s}  {1:>12s}'.format('iid', 'loss'))
        for i in range(len(self.population[0].vs)):
            find.write(' {0:>8d}-th'.format(i+1))
        find.write('\n')
        
        for i,ind in enumerate(self.population):
            fgen.write('     0  {0:8d}  {1:12.4e}\n'.format(ind.iid, ind.loss))
            find.write(' {0:8d}  {1:12.4e}'.format(ind.iid, ind.loss))
            for k in self.vnames:
                find.write(' {0:11.4e}'.format(ind.vs[k]))
            # for j,vj in enumerate(ind.vs):
            #     find.write(' {0:11.4e}'.format(vj))
            find.write('\n')

        if self.print_level > 0:
            print(' step,time,best_iid,best_loss,vars='
                  +' {0:6d} {1:8.1f} {2:5d} {3:8.4f}'.format(0,time()-start,
                                                             self.bestind.iid,
                                                             self.bestind.loss),end="")
            inc = 0
            for k in self.vnames:
                if inc < 16:
                    print(' {0:6.3f}'.format(self.bestind.vs[k]),end="")
                else:
                    break
                inc += 1
            print('', flush=True)

        #...Create pool before going into max_gen-loop,
        #...since creating pool inside could cause "Too many files" error.
        if self.nproc > 0 :  # use specified number of cores by nproc
            pool = Pool(processes=self.nproc)
        else:
            pool = Pool()
            
        for it in range(max_gen):
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
                newind = Individual(self.iidinc, self.vnames, self.slims,
                                    self.hlims, self.loss_func)
                newind.set_variable(vnew)
                candidates.append(newind)

            #...Create new completely random candidates
            iab = int((1.0 -self.frac)*self.nind)
            rnd_candidates = []
            for iv in range(iab,self.nind):
                self.iidinc += 1
                newind = Individual(self.iidinc, self.vnames, self.slims,
                                    self.hlims, self.loss_func)
                newind.init_random()
                rnd_candidates.append(newind)

            #...Evaluate loss function values of updated candidates and new random ones
            prcs = []
            for ic,ci in enumerate(candidates):
                kwtmp = copy.copy(self.kwargs)
                kwtmp['index'] = ic
                kwtmp['iid'] = ci.iid
                # prcs.append(Process(target=ci.calc_loss_func, args=(kwtmp,qs[ic])))
                prcs.append(pool.apply_async(ci.calc_loss_func, (kwtmp,)))
            rnd_prcs = []
            for ic,ci in enumerate(rnd_candidates):
                kwtmp = copy.copy(self.kwargs)
                kwtmp['index'] = len(candidates) +ic
                kwtmp['iid'] = ci.iid
                # prcs.append(Process(target=ci.calc_loss_func, args=(kwtmp,qs[ic])))
                rnd_prcs.append(pool.apply_async(ci.calc_loss_func, (kwtmp,)))
            
            results = [ res.get() for res in prcs ]
            rnd_results = [ res.get() for res in rnd_prcs ]

            for res in results:
                loss,ic = res
                candidates[ic].loss = loss
            self.all_indivisuals.extend(candidates)

            for res in rnd_results:
                loss,ic_rnd = res
                ic = ic_rnd -len(candidates)
                rnd_candidates[ic].loss = loss
            self.all_indivisuals.extend(rnd_candidates)

            #...Pick j that is to be compared with i
            js = random.sample(range(self.nind),k=self.nind)
            #...Decide whether or not to adopt new one
            for jc,jv in enumerate(js):
                pj = self.population[jv]
                cj = candidates[jc]
                dloss = cj.loss -pj.loss
                if dloss < 0.0:  # replace with new individual
                    self.population[jv] = cj
                    find.write(' {0:8d}  {1:12.4e}'.format(cj.iid, cj.loss))
                    # for k,vk in enumerate(cj.vs):
                    for k in self.vnames:
                        find.write(' {0:11.4e}'.format(cj.vs[k]))
                    find.write('\n')
                    find.flush()
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
                fname = 'in.vars.optzer.{0:d}'.format(self.bestind.iid)
                self.write_variables(self.bestind,
                                     fname=fname,
                                     **self.kwargs)
                os.system('cp -f {0:s} in.vars.optzer.best'.format(fname))

            #...Update variable ranges if needed
            if self.update_slims_per > 0 and (it+1) % self.update_slims_per == 0:
                self.slims = update_vrange(self.slims,self.hlims,self.all_indivisuals)
                print(' Update variable ranges')
                # for i in range(len(self.slims)):
                for k in self.vnames:
                    print(' {0:>10s}:  {1:7.3f}  {2:7.3f}'.format(k,self.slims[k][0],self.slims[k][1]))
                #...Set variable ranges of all individuals in the population
                for iv in range(len(self.population)):
                    self.population[iv].slims = self.slims
            
            if self.print_level > 0:
                print(' step,time,best_iid,best_loss,vars='
                      +' {0:6d} {1:8.1f} {2:5d} {3:8.4f}'.format(it+1,time()-start,
                                                                 self.bestind.iid,
                                                                 self.bestind.loss),end="")

                inc = 0
                for k in self.vnames:
                    if inc < 16:
                        print(' {0:6.3f}'.format(self.bestind.vs[k]),end="")
                    else:
                        break
                    inc += 1
                print('', flush=True)

            for i,ind in enumerate(self.population):
                fgen.write(' {0:5d}  {1:8d}  {2:12.4e}\n'.format(it+1,
                                                                 ind.iid,
                                                                 ind.loss))
                fgen.flush()
        fgen.close()
        find.close()
        pool.close()
        #...Finaly write out the best one
        self.write_variables(self.bestind,fname='in.vars.optzer.best',
                             **self.kwargs)
        return None

    def write_variables(self,ind,fname='in.vars.optzer',**kwargs):
        self.write_func(ind.vnames, ind.vs, ind.slims, ind.hlims,
                        fname, **kwargs)
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

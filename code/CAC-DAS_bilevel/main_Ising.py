import numpy as np

from tqdm import tqdm
import time

import itertools

import os
import sys
import misc

import model_wrap

module_dir = os.path.abspath("./../code")
#module_dir = os.path.abspath("./../optimize")
sys.path.append(module_dir)

import model_wrap
from model_wrap import model_airbus
from model_wrap import instance

from utils import visualize_solution
from utils import solution_to_report
from tools import try_to_fix_a_broken_solution
from tools import evaluate_solution
from tools import generate_feasible_solution    

from qubovert.sim import anneal_qubo
        
####################################################
#SOLVER

def run_tuner_with_T(name_instance,instance_Ising,H0,T,R,delta,nsamp_max,order
                     ,is_sparse=True,online_tuning=False,rho=0.5,use_external_obj=True,run_BP=False,run_QAOA=False,use_model=False):

    solvertype = 'Bilevel_CACm'
    solvertype += '_BP' if run_BP else ''
    solvertype += '_QAOA' if run_QAOA else ''
    solvertype += '_online' if online_tuning else ''
    solvertype += '_obj' if use_external_obj else ''
    
    name = solvertype + '_' + name_instance
    output_dir = f"Tune_T={T}_R={R}"

    # Check if the folder already exists
    if not os.path.exists(output_dir):
        # Create the folder if it doesn't exist
        os.makedirs(output_dir)
        print(f"Folder created: {output_dir}")
    else:
        print(f"Folder already exists: {output_dir}")

    if online_tuning==False:
        PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a", 'T']
        x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 6.0])
    else: #online tuning has constant T
        PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a"]
        x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0])

    #generate problem instance
    tunerparams = {'nsamp_max': nsamp_max, 'R': R, 'T': T, 'delta': delta, 'rho': rho}
  
    import tune_wrap
  
    obj_best, sol_best, objv_p, solv_p = tune_wrap.tune_instance_with_T(name, instance_Ising, H0, output_dir, PARAM_NAMES, x, tunerparams, order,
                                   is_sparse=is_sparse, online_tuning=online_tuning,use_external_obj=use_external_obj, run_BP=run_BP, run_QAOA=run_QAOA, use_model=use_model)

    return obj_best, sol_best, objv_p, solv_p

def LoadInstance(path,i):
    filepath = f'{path}_{i+1}'
    file = open ( filepath , 'r')
    wmat = np.array([[float(num) for num in line.split(' ')] for line in file ])
    w = np.zeros((N,N))
    for l in range(len(wmat[:,0])):
        w[int(wmat[l,0]-1),int(wmat[l,1]-1)] = wmat[l,2]
    w = w + w.T
    return w

def LoadOptimalC(path,w,i):
    filepath = f'{path}_SOL'
    file = open ( filepath , 'r')
    Cv = np.array([[float(num) for num in line.split(' ')] for line in file ])
    C0 = float(Cv[i])
    H0 = -4*C0 - np.sum(w)
    H0 = H0/2
    return H0

# This code is mainly for tests
if __name__ == '__main__':
        
    R = 100
    
    online_tuning = True
    run_BP = True
    run_QAOA = False
    
    use_model = False
    use_external_obj = False
   
    data = 'SK' # 'GSET' #'SK'
            
    #GSET online tuning
    if data=='GSET' :
        
        # parameters
        #nsamp_max = 200000
        nsamp_max = 4000
        reps = 2
        delta = 0.1 #annealing approx ratio
        rho = 0.5 #proportion of solution to reset at each DAS step
        T = 500

        # load instance
        N = 800
        i = 0
        J = misc.load_GSET_J(N,i)
        #H0 = misc.load_GSET_H0(N,i,J)
        H0 = 0
        offset = 0
        h = np.zeros(N)
        name_instance = f'GSET_{N}'
        Q, g, offset = misc.ising_to_qubo(J, h)        
        instance_Ising = (N,J,h,offset,Q,g)
        is_sparse = True
        
    #online tuning SK
    if data=='SK':
        
        # parameters
        #nsamp_max = 200000
        nsamp_max = 2000
        reps = 2
        delta = 0.1 #annealing approx ratio
        rho = 0.5 #proportion of solution to reset at each DAS step
        T = 20
        
        # load instance
        N = 100
        name_instance = f'SK_{N}'
        path = './../Data/BENCHSKL_100_100/BENCHSKL_100_100'
        i = 0
        J = LoadInstance(path,i)
        H0 = LoadOptimalC(path,J,i)
        H0 = - 0.763 * N * np.sqrt(N) * 0.878 # energy per site SK * SDP approx ratio
        h = np.zeros(N)
        Q, g, offset = misc.ising_to_qubo(J, h)
        instance_Ising = (N,J,h,offset,Q,g)
        is_sparse = False
        
    #tune
    start_time = time.time()
    obj_best, sol_best, objv_p, solv_p = run_tuner_with_T(name_instance,instance_Ising,H0,T,R,delta,nsamp_max,reps,
                     is_sparse=is_sparse,online_tuning=online_tuning,rho=rho,use_external_obj=use_external_obj,run_BP=run_BP,use_model=use_model)
    run_time = time.time()-start_time
    
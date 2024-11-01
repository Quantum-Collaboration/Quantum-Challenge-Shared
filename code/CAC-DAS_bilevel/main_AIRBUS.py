import numpy as np

from tqdm import tqdm
import time

import itertools

import os
import sys
import misc

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

import argparse

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

    if use_external_obj==False:
        LM0 = 3
        PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a", 'T']
        x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 6.0])
    else: #online tuning has constant T
        #PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a"]
        #x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0])
        if online_tuning==True:
            if use_model==True:
                PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a", 'l1','l2', 'l3', 'l4']
                LM0 = 3
                x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, LM0, LM0, LM0, LM0])
            else:
                PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a"]
                LM0 = 3
                x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0])
        else:
            if use_model==True:
                PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a", "T", 'l1','l2', 'l3', 'l4']
                LM0 = 3
                x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, T, LM0, LM0, LM0, LM0])
            else:
                PARAM_NAMES = ["beta_BP","beta", "lamb1", "lamb2", "xi", "gamma", "a", "T"]
                LM0 = 3
                x = np.log([1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, T])
    #generate problem instance
    tunerparams = {'nsamp_max': nsamp_max, 'R': R, 'T': T, 'delta': delta, 'rho': rho, 'LM0': LM0}
            
    import tune_wrap
    
    #starting state

    obj_best, sol_best, objv_p, solv_p = tune_wrap.tune_instance_with_T(name, instance_Ising, H0, output_dir, PARAM_NAMES, x, tunerparams, order,
                                   is_sparse=is_sparse, online_tuning=online_tuning,use_external_obj=use_external_obj, run_BP=run_BP, run_QAOA=run_QAOA, use_model=use_model)

    return obj_best, sol_best, objv_p, solv_p

# TODO: Wrap this function to run on weights and alpha_pq
def run_benchmark(weights,alpha_pq,reloadModel=True,online_tuning=True,use_model=False,run_BP=False,run_QAOA=False):
#if __name__ == '__main__':
    
    data = 'AIRBUS'
    
    if data=='AIRBUS':
        
        #TODO: set reloadModel=True if this is the first time executing
        #reloadModel = False
        #online_tuning = True
        is_sparse = True
        use_external_obj = True
        #run_BP = False
        #run_QAOA = False
        proprocessing = False
        #use_model = True
        
        # set problem parameters
        
        v0 = 1e-15
        LM0 = 3
        lambda_values = [LM0,LM0,LM0,LM0,v0,v0]      # Langrage multipliers for constraints 1 to 6 (constraint 1 is currently unused!)

        # set solver hyperparameters
        
        # R=6, nsamp_max = 200 takes about 20 min
        
        R = 100                                      # number of samples per DAS iteration
        #nsamp_max = 10000  #12 hours                 # total number of samples DAS
        nsamp_max = 25000  #30 hours                 # total number of samples DAS
        T = 500                                      # number of time steps Ising solver
        #R = 6                                         # number of samples per DAS iteration
        #nsamp_max = 20                                # total number of samples DAS
        #T = 30                                        # number of time steps Ising solver
        
        delta = 0.25                                 # annealing approx ratio
        rho = 0.5                                    # proportion of solution to reset at each DAS step
        reps = 2
    
        # load model
        
        name_instance = f'AIRBUS_w_'
        for W in weights:
            name_instance += f'{W}_'
        name_instance += 'a_'
        for alp in alpha_pq:
            name_instance += f'{alp}_'
        
        dataIsing = model_wrap.instantiate_model_data(weights,alpha_pq,lambda_values)
        
        if reloadModel:
            Q,g,J,h = model_wrap.decompose_model(name_instance,weights,alpha_pq)
            dataIsing = model_wrap.instantiate_model_data(weights,alpha_pq,lambda_values)
            
        else:
            Q,g,J,h = misc.reload_model(name_instance)
        
        offset = 0
        N = len(h[0])
        H0 = 0
            
        instance_Ising = (N,J,h,offset,Q,g)
        
    #tune
    start_time = time.time()
    obj_best, sol_best, objv_p, solv_p = run_tuner_with_T(name_instance,instance_Ising,H0,T,R,delta,nsamp_max,reps,
                     is_sparse=is_sparse,online_tuning=online_tuning,rho=rho,use_external_obj=use_external_obj,run_BP=run_BP,run_QAOA=run_QAOA,use_model=use_model)
    run_time = time.time()-start_time
    
    # postprocessing
    if proprocessing and data=='AIRBUS':
        #load solution
        path_solution = f'./Tune_T={T}_R={R}/CACm_{name_instance}_P_0.txt'
        P = np.loadtxt(path_solution)
        S = (np.sign(P)+1)/2
        x_sol = dict(zip(list(instance.model.variables), [val for val in S.tolist()]))
        
        is_valid0=instance.model.is_solution_valid(x_sol)
        print(f"CAC-BP: Before fixing feasibility: {is_valid0}")
        
        #try to fix
        x_solution = x_sol
        if is_valid0==False:
            x_fixed = try_to_fix_a_broken_solution(model_airbus, instance, x_sol, max_iterations=25, seed=None, ignore_ancillas=False) # set ignore_ancillas=True to speed up significantly
            if x_fixed is not None: # method can sometimes fail and return None, then increase max_iterations and hope for the best    print(instance.model.is_solution_valid(x_fixed))
                x_solution = x_fixed
                
        #check validity
        #x_solution = x_sol
        
        is_valid1 = instance.model.is_solution_valid(x_solution)
        print(f'CAC-BP: Before fixing feasibility: {is_valid1}')
        
        
        (co2, eur, time_, ws), obj = evaluate_solution(model_airbus, instance, x_solution)
        print("objective 1         ", co2)
        print("objective 2         ", eur)
        print("objective 3         ", time_)
        print("objective 4         ", ws)
        print("scalarized objective", obj)
        print("compare QUBO obj    ", instance.model.value(x_solution)) # just for demonstration, small deviations are rounding errors
    
        #visualize
        visualize_solution(model_airbus, instance, x_solution)
    
        #report
        report = solution_to_report(model_airbus, instance, x_solution)
        
    print(f'Finished. Running time of the algorithm was {run_time} s')
    
    
    
if __name__ == '__main__':
    
    #python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel
    #python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel -online_tuning
    #python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel -online_tuning -use_model
    #python3 main_AIRBUS.py 0.25 0.25 0.25 0.25 4 5 -reloadModel -online_tuning -use_model -run_BP -run_QAOA
    
    parser = argparse.ArgumentParser(description="Run main_AIRBUS with parameters w1, w2, w3, w4, a1, a2 and optional flags.")
   
    # Positional arguments for the main parameters
    parser.add_argument("w1", type=float, help="Parameter w1")
    parser.add_argument("w2", type=float, help="Parameter w2")
    parser.add_argument("w3", type=float, help="Parameter w3")
    parser.add_argument("w4", type=float, help="Parameter w4")
    parser.add_argument("a1", type=float, help="Parameter a1")
    parser.add_argument("a2", type=float, help="Parameter a2")
   
    # Optional flag for online tuning
    parser.add_argument("-reloadModel", action="store_true", help="Enable reload Model")
    parser.add_argument("-online_tuning", action="store_true", help="Enable online tuning")
    parser.add_argument("-use_model", action="store_true", help="Enable use model")
    parser.add_argument("-run_BP", action="store_true", help="Enable run BP")
    parser.add_argument("-run_QAOA", action="store_true", help="Enable run QAOA")

    # Parse the arguments
    args = parser.parse_args()

    weights = [args.w1,args.w2,args.w3,args.w4]
    alpha_pq = [args.a1,args.a2]  
        
    run_benchmark(weights,alpha_pq,
                  reloadModel=args.reloadModel,online_tuning=args.online_tuning,
                  use_model=args.use_model,run_BP=args.run_BP,run_QAOA=args.run_QAOA)

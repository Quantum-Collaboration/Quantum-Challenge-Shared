import os
import sys
import numpy as np
import random

from tqdm import tqdm

module_dir = os.path.abspath("./../code")
sys.path.append(module_dir)

from model import Model

from quantum_qaoa import run_qaoa

from tools import generate_feasible_solution    
    
import model_wrap
from model_wrap import model_airbus
from model_wrap import instance

def QAOA_wrap_(seed,P_in):

    b = (P_in+1)/2
    var_names = list(instance.variable_index_to_name_dict.values())
    N = len(var_names)
    
    x_sol = {}
    for key, bi in zip(np.sort(var_names),b):
         x_sol[key] = int(bi)
    
    p = 3                                              # depth
    use_custom_mixer = True                            # use dicke states and XY mixer according to constraints
    dev_kwargs = dict(name="default.qubit", shots=32)  # local pennylane simulator (important: specify shots!)
    N_qaoa = 8                                         # maximum number of qubits
    seed = seed                                        # rng seed
    #select_indices = [0,1,2,4,6,8,10]                  # indices to select from instance.qubo_matrix (set to None for all)
    select_indices = np.random.choice(range(N), N_qaoa, replace=False)
    
    x, samp_dict = run_qaoa(instance, p, use_custom_mixer, dev_kwargs, N_qaoa, seed, select_indices)
    
    # replace within this solution
    
    for key, value in x.items():
        #print(key,value)
        x_sol[key] = value
    
    var_names = list(x_sol.keys())
    sorted_indices = sorted(range(len(var_names)), key=lambda i: var_names[i])
    b = list(x_sol.values())
    b = np.array(b)[sorted_indices]
    P = 2*b-1
    
    return P

def ex_QAOA(J, h, P_in):
    if P_in.ndim == 3:
        reps,N,R = np.shape(P_in)
        S = np.zeros((reps,N,R))
        H = np.zeros((reps,R))
        for r in range(reps):
            for j in tqdm(range(R)):
                seed = random.randint(0, 2**32 - 1)
                P_out_ = QAOA_wrap_(seed,P_in[r,:,j])
                S_ = np.sign(P_out_)
                H_ = -0.5 * np.sum(S_ * (J @ S_)) - np.sum(h*S_)
                H[r,j] = H_                 
                S[r,:,j] = S_
        
    elif P_in.ndim == 2:
        N,R = np.shape(P_in)
        S = np.zeros((N,R))
        H = np.zeros(R)
        for j in tqdm(range(R)):
            seed = random.randint(0, 2**32 - 1)
            P_out_ = QAOA_wrap_(seed,P_in[:,j])
            S_ = np.sign(P_out_)
            H_ = -0.5 * np.sum(S_ * (J @ S_)) - np.sum(h*S_)
            H[j] = H_                 
            S[:,j] = S_

    return H, S
    
if __name__ == '__main__':
       
    
    # create a model (a model constains all structural information about the problem)
    model = Model(data_path='./../code/data', cache_path="./../code/cache")  # optionally set explicit paths
    
    # create instances from the model (an instance contains all QUBO information about the problem)
    weights = [.25,.25,.25,.25]                   # objective scalarization weights: co2, eur, time, ws
    alpha_pq = [4,5]                              # specify alpha as a rational number alpha = p / q with [p, q], alpha in [.5, .8]
    lambda_values = [1,1,1,1,1,1]                 # Langrage multipliers for constraints 1 to 6 (constraint 1 is currently unused!)
    enable_box_constraint_lambda_rescaling = True # rescale box constraints based on ancilla factors
    variable_assignments = {}                     # enforce these variable assignments
    verbose = True                                # control output
    instance = model.spawn_instance(weights, alpha_pq, lambda_values, enable_box_constraint_lambda_rescaling, variable_assignments, verbose)
    
    # generate a random feasible solution from scratch

    x_sol = generate_feasible_solution(model, instance, seed=None, ignore_ancillas=False)
    instance.model.is_solution_valid(x_sol)
    
    # qaoa on an instance of parts of an instance
    
    p = 3                                              # depth
    use_custom_mixer = True                            # use dicke states and XY mixer according to constraints
    dev_kwargs = dict(name="default.qubit", shots=32)  # local pennylane simulator (important: specify shots!)
    N_qaoa = 8                                         # maximum number of qubits
    seed = None                                        # rng seed
    select_indices = [0,1,2,4,6,8,10]                  # indices to select from instance.qubo_matrix (set to None for all)
    
    x, samp_dict = run_qaoa(instance, p, use_custom_mixer, dev_kwargs, N_qaoa, seed, select_indices)
    
    # replace within this solution
    
    for key, value in x.items():
        x_sol[key] = value
    
    
    var_names = list(instance.variable_index_to_name_dict.values())
    N = len(var_names)
    sorted_indices = sorted(range(len(var_names)), key=lambda i: var_names[i])
    P = list(x_sol.values())
    P = np.array(P)[sorted_indices]
    P = 2*P-1

    print(P)
    
    #test function
    P_in = np.sign(np.random.uniform(low=-1, high=1, size=N))
    P_out = QAOA_wrap_(0,P_in)
    
    J = np.ones((N,N))
    h = np.ones(N)
    
    P_in = np.sign(np.random.uniform(low=-1, high=1, size=(2,N,10)))
    H_out, P_out = ex_QAOA(J, h, P_in)
    
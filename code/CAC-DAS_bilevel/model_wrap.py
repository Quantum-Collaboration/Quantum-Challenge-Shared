import sys
import os
import numpy as np
import misc

module_dir = os.path.abspath("./../code")
#module_dir = os.path.abspath("./../optimize")
sys.path.append(module_dir)

from model import Model

from qubovert import QUBO

import numpy as np
from tools import try_to_fix_a_broken_solution
    
from qubovert.sim import anneal_qubo

from tools import evaluate_solution

from tqdm import tqdm

model_airbus = None
instance = None

# generate model
def generate_model_data(weights,alpha_pq,lambda_values,do_rescale=True):

    global model_airbus, instance
    
    # create a model (a model constains all structural information about the problem)
    model_airbus = Model(data_path='./../code/data', cache_path="./../code/cache")  # optionally set explicit paths

    # create instances from the model (an instance contains all QUBO information about the problem)
    enable_box_constraint_lambda_rescaling = do_rescale # rescale box constraints based on ancilla factors
    variable_assignments = {}                     # enforce these variable assignments
    verbose = True                                # control output
    instance = model_airbus.spawn_instance(weights, alpha_pq, lambda_values, enable_box_constraint_lambda_rescaling, variable_assignments, verbose)
    num_variables = len(instance.model.variables)
    
    print(f"Number of variables: {num_variables}")

# convert PCBO model to QUBO and Ising
def instantiate_model_data(weights,alpha_pq,lambda_values,do_rescale=True):
    
    generate_model_data(weights,alpha_pq,lambda_values,do_rescale=do_rescale)
    
    global model_airbus, instance
    
    N = len(instance.model.variables)

    model_PCBO = instance.model
    
    #convert to qubo
    qubo_matrix = model_PCBO.to_qubo()
    qubo = QUBO(qubo_matrix)
    
    #save qubo
    Q = np.zeros((N, N))
    g = np.zeros(N)
    C = 0
    for key, value in qubo_matrix.items():
        if len(key)==2:
            (i,j) = key
            Q[i, j] = value/2
            Q[j, i] = value/2
        elif len(key)==1:
            i = key
            g[i] = value
        elif len(key)==0:
            C = value
      
    # convert to Ising
    J, h, offset = misc.qubo_to_ising(Q, g)

    data = (J,h,offset,Q,g)

    return data

# brute force decomposition of QUBO into 5 linear combination
# TODO: retrieve decomposition directly from the model
def decompose_model(name_instance,weights_in,alpha_pq_in):
    
    global model_airbus, instance
    
    dataQ = {}
    datag = {}
    dataJ = {}
    datah = {}
    
    v0 = 1e-15
    
    W_list = [1,v0,v0,v0,v0,1]
    LM1_list = [v0,1,v0,v0,v0,1]
    LM2_list = [v0,v0,1,v0,v0,1]
    LM3_list = [v0,v0,v0,1,v0,1]
    LM4_list = [v0,v0,v0,v0,1,1]
    LMb = v0
    
    count = 0
    dictvar = []
    for W, LM1, LM2, LM3, LM4 in zip(W_list,LM1_list,LM2_list,LM3_list,LM4_list):
    
        weights = (W*np.array(weights_in)).tolist()                 # objective weights: co2, eur, time, ws (set to 0 to disable)
        alpha_pq = alpha_pq_in
        lambda_values = [LM1,LM2,LM3,LM4,LMb,LMb]                   # Langrage multipliers for constraints 1 to 6 (constraint 1 is currently unused!)
        data = instantiate_model_data(weights,alpha_pq,lambda_values,do_rescale=False)
        
        J,h,offset,Q,g = data
        
        #variables are ordered alphabetically for the QUBO indices
        var_names = list(instance.variable_index_to_name_dict.values())
        sorted_indices = sorted(range(len(var_names)), key=lambda i: var_names[i])
        dictvar.append(instance.variable_index_to_name_dict)
        
        h_new = h[sorted_indices]
        g_new = g[sorted_indices]
        
        J_new = J[np.ix_(sorted_indices, sorted_indices)]
        Q_new = Q[np.ix_(sorted_indices, sorted_indices)]
        
        dataQ[count] = Q_new
        datag[count] = g_new
        dataJ[count] = J_new
        datah[count] = h_new
        
        count+=1
        
    # save model
    
    folder_path = f"model/full_qubo/{name_instance}/"
    os.makedirs(folder_path, exist_ok=True)
    for i in range(5):
        misc.save_matrix(dataQ[i], folder_path, f'Q_{i}',1e-7)
        np.savetxt(os.path.join(folder_path, f'g_{i}.txt'),datag[i])
        
    folder_path = f"model/full_Ising/{name_instance}/"
    os.makedirs(folder_path, exist_ok=True)
    for i in range(5):
        misc.save_matrix(dataJ[i], folder_path, f'J_{i}',1e-7)
        np.savetxt(os.path.join(folder_path, f'h_{i}.txt'),datah[i])
        
        
    # test
    print("Testing the decomposition")
    
    Q_comb = dataQ[0] + dataQ[1] + dataQ[2] + dataQ[3] + dataQ[4]
    g_comb = datag[0] + datag[1] + datag[2] + datag[3] + datag[4]
    J_comb = dataJ[0] + dataJ[1] + dataJ[2] + dataJ[3] + dataJ[4]
    h_comb = datah[0] + datah[1] + datah[2] + datah[3] + datah[4]

    Q_ref = dataQ[5]
    g_ref = datag[5]
    J_ref = dataJ[5]
    h_ref = datah[5]
    
    Q = []
    g = []
    J = []
    h = []
    for i in range(5):
        Q.append(dataQ[i].tolist())
        g.append(datag[i].tolist())
        J.append(dataJ[i].tolist())
        h.append(datah[i].tolist())

    print(np.sum(np.abs(Q_comb-Q_ref)))
    print(np.sum(np.abs(g_comb-g_ref)))
    print(np.sum(np.abs(J_comb-J_ref)))
    print(np.sum(np.abs(h_comb-h_ref)))

    print("Finished")
    
    return Q,g,J,h
    
        
if __name__ == '__main__':
    
    # load model

    v0 = 1e-15
    LM0 = 10
    weights = [1,1,1,1]                           # objective weights: co2, eur, time, ws (set to 0 to disable)
    alpha_pq = [4,5]                             # specify alpha as a rational number alpha = p / q with [p, q], alpha in [.5, .8]
    lambda_values = [LM0,LM0,LM0,LM0,v0,v0]                 # Langrage multipliers for constraints 1 to 6 (constraint 1 is currently unused!)
    
    dataIsing = instantiate_model_data(weights,alpha_pq,lambda_values)
    
    # solve SA
    
    print("Solving with SA")
    
    res = anneal_qubo(instance.model, num_anneals=100)
    x_sol = res.best.state
    
    # fix a broken solution

    print("Fixing solutions")

    count=0
    all_solutions = []
    for r in tqdm(range(25)):
        
        # fix it
        x_fixed = try_to_fix_a_broken_solution(model_airbus, instance, x_sol, max_iterations=10, seed=None, ignore_ancillas=False)
        if x_fixed is not None: # method can sometimes fail and return None, then increase max_iterations and hope for the best
            print(instance.model.is_solution_valid(x_fixed))
            x_solution = x_fixed
            all_solutions.append(x_solution)
            count+=1
        else:
            print(False)
            x_solution = x_sol
   
    # calculate objective
    
    all_obj = []
    for x_solution in all_solutions:
        
        (co2, eur, time, ws), obj = evaluate_solution(model_airbus, instance, x_solution)
        print("objective 1         ", co2)
        print("objective 2         ", eur)
        print("objective 3         ", time)
        print("objective 4         ", ws)
        print("scalarized objective", obj)
        print("compare QUBO obj    ", instance.model.value(x_solution)) # just for demonstration, small deviations are rounding errors
        
        all_obj.append(obj)
        
        
    
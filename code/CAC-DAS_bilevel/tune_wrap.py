import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import DASTuneADAM as DASTuner
from CAC import *

module_BP = os.path.abspath("./../IBP")
sys.path.append(module_BP)
from bp_random_subtree import IBP_iterate

import QAOA_wrap
from QAOA_wrap import ex_QAOA

module_dir = os.path.abspath("./../code")
sys.path.append(module_dir)

from tools import try_to_fix_a_broken_solution
from tools import generate_feasible_solution    
from tools import evaluate_solution
from tools import save_result

from tqdm import tqdm

import model_wrap
from model_wrap import model_airbus
from model_wrap import instance

#from concurrent.futures import ThreadPoolExecutor, as_completed

#import dask
#from dask import delayed, compute

coeffs = {
    1: np.array([0, 1]),
    2: np.array([0, 1/2, 3/2]),
    3: np.array([0, 1/3, 5/6, 11/6])
}

OBJMAX = 1000000

datapath = 'Data'

#calculate using external objective after fixing solution (for AIRBUS model only)
def calculate_external_objective_(b,obj_p,b_v,instance_Ising,name,seed=None):
 
    if True:
        N,J,h,offset,Q,g = instance_Ising
        
        keys = list(instance.model.variables)
        keys = sorted(keys)
        
        x_sol = dict(zip(keys, [val for val in b.tolist()]))
        
        has_changed = False
        for r in range(3):
            try:
                x_fixed = try_to_fix_a_broken_solution(model_airbus, instance, x_sol, max_iterations=50, seed=None, ignore_ancillas=False, verbose=False)
            except:
                x_fixed = None
                
            if x_fixed is not None: # method can sometimes fail and return None, then increase max_iterations and hope for the best
                #print(instance.model.is_solution_valid(x_fixed))
                x_solution = x_fixed
                for key, value in x_fixed.items():
                    x_solution[key] = value
                try:
                    #objv = (co2, eur, time, ws)
                    objv, obj = evaluate_solution(model_airbus, instance, x_solution)
                    print('Valid')
                    has_changed = True
                    break
                except:
                    has_changed = False
                
        if has_changed:
            #sort alphabetically
            keys = list(x_solution.keys())
            sorted_indices = sorted(range(len(keys)), key=lambda i: keys[i])
            b = list(x_solution.values())
            b = np.array(b)[sorted_indices]
            return obj, objv, b
        else:
            #TODO: if cannot fix, return best ?
            #return obj_p, b_v
            return OBJMAX, OBJMAX*np.ones(4), b_v

    #keep for tests
    else:
        N,J,h,offset,Q,g = instance_Ising
        
        obj = np.sum(b*(Q@b)) + np.sum(b*g)# - offset
        return obj, np.zeros(4), b
    
# wrapper for calculate_external_objective_
def calculate_external_objective(b,obj,b_p,instance_Ising,name):
    # If b is 2D, apply along axis 0 (columns); if 3D, apply along axis 1 (N dimension)
    print('Fixing solution to evaluate external objective...')
    if b.ndim == 3:
        reps,N,R = np.shape(b)
        solv = np.zeros((reps,N,R))
        obj = np.zeros((reps,R))
        objv = np.zeros((reps,R,4))
        for i in range(reps):  # Loop over the second axis (K)
            
            for r in tqdm(range(R)):  # Loop over the second axis (K)
                seed = np.random.seed(i+r)
                obj_, objv_, sol = calculate_external_objective_(b[i,:,r],obj[i,r],b_p[i,:,r],instance_Ising,name,seed=seed)
                solv[i,:,r] = sol
                obj[i,r] = obj_
                objv[i,r,:] = objv_
                
    elif b.ndim == 2:
        N,R = np.shape(b)
        solv = np.zeros((N,K))
        obj = np.zeros(K)
        objv = np.zeros((R,4))
        for r in tqdm(range(R)):  # Loop over the second axis (K)
            seed = np.random.seed(r)
            obj_,objv_, sol = calculate_external_objective_(b[:,r],obj[r],b_p[:,r],instance_Ising,name,seed=seed)
            solv[:,r] = sol.values()
            obj[r] = obj_
            objv[r,:] = objv_
        
    return obj, objv, solv

# wrapper for IBP code
def run_IBP_iterate(b, Q_IDX, Q_cost, g, J, h, beta, T):
    if b.ndim == 3:
        reps,N,R = np.shape(b)
        S_BP = np.zeros((reps,N,R))
        H_BP = np.zeros((reps,R))
        for r in range(reps):
            x_BP_ = IBP_iterate(b[r,:,:], Q_IDX, Q_cost, g, N, R, beta, num_step = T)
            S_BP_ = 2*np.array(x_BP_)-1
            H_BP_ = -0.5 * np.sum(S_BP_ * (J @ S_BP_),0) - np.sum(h[:,None]*S_BP_,0)
            H_BP[r,:] = H_BP_                 
            S_BP[r,:,:] = S_BP_
    
    elif b.ndim == 2:
        N,R = np.shape(b)
        S_BP = np.zeros((N,R))
        H_BP = np.zeros(R)
        x_BP = IBP_iterate(np.array(b), Q_IDX, Q_cost, g, N, R, beta, num_step = T)
        S_BP = 2*np.array(x_BP)-1
        H_BP = -0.5 * np.sum(S_BP * (J @ S_BP),0) - np.sum(h[:,None]*S_BP,0)

    return H_BP, S_BP
    
# convert QUBO model to IBP format
def convert_format_BP(N,Q,g):
    m=0
    Q_IDX = []
    Q_cost = []

    for i in range(N):
        for j in range(i+1,N):
            if np.abs(Q[i,j])>0:
                Q_IDX.append([i,j])
                Q_cost.append(Q[i,j])
                m+=1
                
    Q_IDX = np.transpose(Q_IDX).astype(np.int64)
    Q_cost = np.array(Q_cost)
    
    return Q_IDX,Q_cost,g

def tune_instance_with_T(name, instance_Ising, H0, output_dir, PARAM_NAMES, x, tunerparams, reps,
                         is_sparse=True, online_tuning=False,use_external_obj=True,run_BP=False, run_QAOA=False, use_model=False):    

    nsamp_max = tunerparams['nsamp_max']
    delta = tunerparams['delta']
    rho = tunerparams['rho']
    R = tunerparams['R']

    #load instance
    N,J,h,offset,Q,g = instance_Ising
    
    #global variables to save after DAS step
    global count
    global P, E, E_best, P_best
    global sol, obj, obj_best, objv_best, sol_best
    global prop_BPCAC_list, prop_QAOACAC_list

    #initialize DAS
    x_init = np.array(x)
    L_init = np.diag(np.ones(len(x))*0.5)

    #initial step of DAS iterations
    if False:
    #if use_external_obj:
        P0 = []
        for r in range(reps):
            P0_ = []
            for i in tqdm(range(R)):
                seed = np.random.seed(i+r)
                x_temp = generate_feasible_solution(model_airbus, instance, seed=None, ignore_ancillas=False)
                var_names = list(instance.variable_index_to_name_dict.values())
                sorted_indices = sorted(range(len(var_names)), key=lambda i: var_names[i])
                P = list(x_temp.values())
                P = np.array(P)[sorted_indices]
                P = 2*P-1
                P0_.append(P.tolist())
            P0.append(P0_)
        P0 = np.array(P0)
        P0 = np.transpose(P0,(0,2,1))
    else:
        P0 = np.sign(np.random.uniform(low=-1, high=1, size=(reps,N,R))) #bag of solution, not used for now

    #intialization of saved states    
    P = P0
    E = np.ones((reps,R))*H0
    P_best = P[0,:,0]
    E_best = H0
    sol = (np.sign(P)+1)/2
    obj = np.ones((reps,R))*OBJMAX
    sol_best = ((np.sign(P_best)+1)/2).tolist()
    obj_best = OBJMAX
    objv_best = np.ones(4)*OBJMAX
        
    prop_BPCAC_list = []
    prop_QAOACAC_list = []
    count = 0
    
    # create file to keep track of best solutions visited
    file_name = f"{output_dir}/{name}_tuning_history.txt"
    with open(file_name, 'w') as file:
        pass  # Do nothing, which leaves the file empty
    
    #DAS sampling function
    def sample(x, prop_samp, seed):

        # E_opt, P_opt: Ising solver results
        # P, E, P_best, E_best: saved Ising solver results
        
        # obj, objv, sol : external cost
        # obj_best, sol_best, obj, sol: saved external cost results

        global count, P, P_best, E, E_best, sol, sol_best, obj, objv, obj_best, objv_best

        # setup functions
        def intialize_state(P,E,sol,obj):
            R = list(hyperparams.values())[0].shape[0]
            new_R = 100 * int(np.ceil(R / 100))
            reps,N,R_ = np.shape(P)
             
            if online_tuning and R_==R: #use as starting point previous optimal
                Pin = P.astype(float)
                rint = np.random.randint(0, R, size=new_R)
                ran_R = int(R*rho)+1
                if Pin.ndim==3:
                    br = np.random.randint(0, 2, size=1)[0]
                    if ran_R>0:
                        Prand = np.sign(np.random.uniform(low=-1, high=1, size=(reps,N,ran_R)))
                        if use_external_obj:
                            idran = np.argsort(obj[br,:])[-ran_R:]
                            Pin[:,:,idran] = Prand
                            Pin = np.transpose(Pin[br,:,rint])
                    
                        else:
                            idran = np.argsort(E[br,:])[-ran_R:]
                            Pin[:,:,idran] = Prand
                            Pin = np.transpose(Pin[br,:,rint])
                        
                            
                if P.ndim==2:
                    Pin = np.random.uniform(low=-1, high=1, size=(N, new_R))
                        
            else:
                Pin = np.random.uniform(low=-1, high=1, size=(N, new_R))
                
            return Pin
    
        #setup solver parameters

        R = x.shape[1]
        D = x.shape[0]

        hyperparams = {}
        for idx, param_name in enumerate(PARAM_NAMES):
            hyperparams[param_name] = np.exp(x[idx, :])

        eps0 = np.mean(np.abs(J)) #scaling factor for dynamics
    
        if online_tuning==False:
            T = hyperparams['T']
            del hyperparams['T']
        else:
            T = tunerparams['T']


        # starting state for solver
        Pin = intialize_state(P,E,sol,obj)

        if ('l1' not in hyperparams) and use_external_obj:
            hyperparams['l1'] = tunerparams['LM0']*np.ones(R)
            hyperparams['l2'] = tunerparams['LM0']*np.ones(R)
            hyperparams['l3'] = tunerparams['LM0']*np.ones(R)
            hyperparams['l4'] = tunerparams['LM0']*np.ones(R)
            
            
        #setup other solvers
        if (run_BP or run_QAOA) and use_external_obj:
            #prepare for other solvers
            
            l_list = [1]
            for i in range(1,5):
                l_list.append(np.mean(hyperparams[f'l{i}']))
    
            J_eff = J[0]
            h_eff = h[0]
            Q_eff = Q[0]
            g_eff = g[0]
            for i in range(1,5):
                J_eff += l_list[i]*np.array(J[i])
                h_eff += l_list[i]*np.array(h[i])
                Q_eff += l_list[i]*np.array(Q[i])
                g_eff += l_list[i]*np.array(g[i])
        
        else:
            J_eff = J
            h_eff = h
            Q_eff = Q
            g_eff = g

        #---- RUN ONE ITERATION (return E_opt, P_opt)
        
        # run CAC
        
        E_opt, P_opt = run_cac_with_T(
            J, h, E_best, eps0, T, hyperparams, reps, Pin, np.random.randint(100000),
            is_sparse=is_sparse,online_tuning=online_tuning,rho=rho
        )
        
        #---- OTHER SOLVERS
        
        # run IBP
        if run_BP:
        
            print('Running IBP')
            
            Q_IDX,Q_cost,g_cost = convert_format_BP(N,Q_eff,g_eff)
            
            b_BP = np.copy((np.sign(Pin[:,:R])+1)/2)
            
            beta_BP = hyperparams['beta_BP']
            
            H_BP, S_BP = run_IBP_iterate(b_BP, Q_IDX, 2*Q_cost, g_cost, J_eff, h_eff, beta_BP, np.mean(T).astype(int))
            
            H_BP = np.tile(H_BP[None,:],[reps,1])
            S_BP = np.tile(S_BP[None,:,:],[reps,1,1])

            prop_BPCAC = np.mean(H_BP<E_opt)
            #print(f'Proportion smaller with BP {prop_BPCAC}')
      
            if P_opt.ndim == 3:
                E_opt_comb = np.concatenate((E_opt,H_BP),axis=1)
                P_opt_comb = np.concatenate((P_opt,S_BP),axis=2)
                
            else:
                E_opt_comb = np.concatenate((E_opt,H_BP),axis=1)
                P_opt_comb = np.concatenate((P_opt,S_BP),axis=1)
         
                
        else:
            prop_BPCAC = np.nan
            
            
        if run_QAOA:
            
            if run_BP:
                E_opt2 = E_opt_comb
                P_opt2 = P_opt_comb
            else:
                E_opt2 = E_opt
                P_opt2 = P_opt
                    
            print('Running QAOA')
            
            H_QAOA, P_QAOA = ex_QAOA(J_eff, h_eff, np.sign(Pin[:,:R]))
            
            H_QAOA = np.tile(H_QAOA[None,:],[reps,1])
            P_QAOA = np.tile(P_QAOA[None,:,:],[reps,1,1])
            
            prop_QAOACAC = np.mean(H_QAOA<E_opt)
            
            if P_opt2.ndim == 3:
                E_opt_comb = np.concatenate((E_opt2,H_QAOA),axis=1)
                P_opt_comb = np.concatenate((P_opt2,P_QAOA),axis=2)
                
            else:
                E_opt_comb = np.concatenate((E_opt2,H_QAOA),axis=1)
                P_opt_comb = np.concatenate((P_opt2,P_QAOA),axis=1)
                
        else:
            prop_QAOACAC = np.nan
            
        #combine solutions
        if run_BP or run_QAOA:
            
            if P_opt.ndim == 3:
                
                iopt = np.argsort(E_opt_comb,1)[:,:R]
                row, col = np.unravel_index(iopt, E_opt_comb.shape)
                E_opt = E_opt_comb[row,col]
                P_opt = np.transpose(P_opt_comb[row,:,col],[0,2,1])
                
            else:
             
                iopt = np.argsort(E_opt_comb)[:R]
                E_opt = E_opt_comb[iopt]
                P_opt = P_opt_comb[:,iopt]
        
        
        #---- POST-PROCESSING
        
        #evaluate external objective
        if use_external_obj:
            if np.mod(count,10)==0:
                b = (np.sign(P_opt)+1)/2
                obj_opt,objv_opt,sol_opt = calculate_external_objective(b,obj,sol,instance_Ising,name)
                print(f"Step {count}, Vector of objective value found:",obj_opt)
            else:
                obj_opt = obj
                sol_opt = sol
                objv_opt = objv
        
        else:
            obj_opt = E_opt
            sol_opt = (np.sign(P_opt)+1)/2
            objv_opt = np.zeros((reps,R,4))
        
        #update best
        if np.min(E_opt)<E_best:
            E_best = np.min(E_opt)
            iopt = np.argmin(E_opt)
            row, col = np.unravel_index(iopt, E_opt.shape)
            P_best = P_opt[row,:,col]
                
        if np.min(obj)<obj_best:
            obj_best = np.min(obj)
            iopt = np.argmin(obj)
            row, col = np.unravel_index(iopt, obj.shape)
            sol_best = sol[row,:,col]
            objv_best = objv[row,col,:]
            
            #save new best in file
            sol_best_str = ''.join([str(int(round(x))) for x in sol_best])
            with open(file_name, 'a') as file:
                #file.write(f"{count} {sol_best_str} {obj_best}\n")
                file.write(f"{count} {sol_best_str} {obj_best} {objv_best[0]} {objv_best[1]} {objv_best[2]} {objv_best[3]}\n")

        #compute fitness for DAS
        if use_model == False:
            success = np.abs((E_opt-E_best)/E_opt) <= delta*(1-prop_samp)
        else:
            #success = np.abs((E_opt-E_best)/E_opt) <= delta*(1-prop_samp)
            success = np.abs((obj-obj_best)/obj_best) <= delta*(1-prop_samp)
            #success = objv - obj_best <= 0
            
        num_success = np.sum(success, axis=0)

        fitness = coeffs[reps][num_success]
        fitness = 100 * fitness / T
  
        #update global variables
        P = P_opt
        E = E_opt
        obj = obj_opt
        objv = objv_opt
        sol = sol_opt
        
        #print for progress
        if use_external_obj==False:
            print(f'Progress: {prop_samp}, Fitness={np.mean(fitness)}, E_best/E_avg/E_min={E_best}/{np.mean(E)}/{np.min(E)}, BP/CAC:{prop_BPCAC}/{1-prop_BPCAC}, QAOA/CAC:{prop_QAOACAC}/{1-prop_QAOACAC}')
        else:
            print(f'Progress: {prop_samp}, Fitness={np.mean(fitness)}, obj_best/obj_avg/obj_min={obj_best}/{np.mean(obj)}/{np.min(obj)}, E_best/E_avg/E_min={E_best}/{np.mean(E)}/{np.min(E)}, BP/CAC:{prop_BPCAC}/{1-prop_BPCAC}, QAOA/CAC:{prop_QAOACAC}/{1-prop_QAOACAC}')
        
        prop_BPCAC_list.append(prop_BPCAC)
        prop_QAOACAC_list.append(prop_QAOACAC)
  
        count+=1
                  
        return fitness, E_best, E, P_best, obj_best, objv_best, sol_best


    D = len(PARAM_NAMES)


    #####################################################

    #use DAS tuner

    fit_est_beta = 0.01

    tuner = DASTuner.Sampler(sample, D, R)
    tuner.fit_est_beta = fit_est_beta
    tuner.curv_est_beta = fit_est_beta
    tuner.grad_est_beta = fit_est_beta/D

    tuner.init_window(x_init, L_init)

    # tuner.dt_log = np.log(dt0/f_est)
    tuner.dt0 = 0.5

    tot_samp_rec, x_rec, L_rec, fit_rec, Hbest_rec, obj_best_rec, objv_best_rec, Hmin_rec, Havg_rec = tuner.optimize(tot_samp_max = nsamp_max, R_end = 10.0)

    param_out = x_rec[len(x_rec)-1]

    #####################################################

    # final evaluation

    print("opt found ", param_out)
    print("evaluating...")
    if use_external_obj:
        R_eval = R
    else:
        R_eval = 2000
    f_eval = 0
    N_inst = 1
    evalist = []
    P_bestlist = []
    sol_bestlist = []
    E_best_list = []
    obj_best_list = []
    objv_best_list = []
    for i in range(N_inst):

        fitness,E_best,E,P_best,obj_best,objv_best,sol_best = sample(np.outer(param_out, np.ones(R_eval)),0,range(R_eval))
        eva = np.average(fitness)
        f_eval += eva
        evalist.append(eva)
        P_bestlist.append(P_best)
        sol_bestlist.append(sol_best)
        E_best_list.append(E_best)
        obj_best_list.append(obj_best)
        objv_best_list.append(objv_best)
        
    f_eval = f_eval/N_inst

    print("f_eval", f_eval)
    print("L", tuner.L)

    #####################################################

    # save output

    file_name = f"{output_dir}/{name}.txt"
    with open(file_name, 'w') as f:
        f.write(' '.join(map(str, [f_eval])) + '\n')
        f.write(' '.join(map(str, evalist)) + '\n')
        f.write(' '.join(map(str, np.exp(param_out))) + '\n')
        f.write(' '.join(map(str, np.exp(param_out))) + '\n')

    for i in range(N_inst):
        P_best = P_bestlist[i]
        file_name = f"{output_dir}/{name}_P_best_{i}.txt"
        np.savetxt(file_name,P_best)
        
        E_best = E_best_list[i]
        file_name = f"{output_dir}/{name}_E_best_{i}.txt"
        np.savetxt(file_name,[E_best])
    
    if use_external_obj:
        for i in range(N_inst):
            sol_best = sol_bestlist[i]
            file_name = f"{output_dir}/{name}_solbest_{i}.txt"
            np.savetxt(file_name,sol_best)
            
            # save best solution found using saver function
            keys = list(instance.model.variables)
            keys = sorted(keys)
            x_sol = dict(zip(keys, [val for val in sol_best.tolist()]))
            file_name = f"{output_dir}/{name}_result_{i}.json"
            suppl_data = dict() # add any json compatible supplementary data to the solution file
            result_dict = save_result(file_name, model_airbus, instance, x_sol, suppl_data)
        
            obj_best = obj_best_list[i]
            file_name = f"{output_dir}/{name}_obj_best_{i}.txt"
            np.savetxt(file_name,[obj_best])
            
            objv_best = objv_best_list[i]
            file_name = f"{output_dir}/{name}_objv_best_{i}.txt"
            np.savetxt(file_name,objv_best)
        
    #####################################################


    def save_to_file_and_plot(plot_file_name, PARAM_NAMES, x_rec, fit_rec, Hbest_rec, obj_best_rec, objv_best_rec, Hmin_rec, Havg_rec):
        # Construct the file name for data

        # Construct the file name for the plot
        plot_file_path = os.path.join(output_dir, plot_file_name + '_x.png')
        
        # Plot the figure
        plt.figure()
        for idx, PARAM in enumerate(PARAM_NAMES):
            plt.plot(np.exp(x_rec)[:,idx],label=PARAM)

        plt.xlabel('steps')
        plt.ylabel('parameters')

        plt.legend()

        plt.yscale('log')

        ax = plt.gca()
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(True)

        # Save the figure to a file
        plt.savefig(plot_file_path)

        # Close the figure
        plt.close()
        
        # Save to text file
        for idx, PARAM in enumerate(PARAM_NAMES):
            val = np.exp(x_rec)[:,idx]
            plot_file_path = os.path.join(output_dir, plot_file_name + f'_{PARAM}.txt')
            np.savetxt(plot_file_path,val)
        
        # Construct the file name for the plot
        plot_file_path = os.path.join(output_dir, plot_file_name + '_H.png')
        
        # Plot the figure
        plt.figure()

        plt.subplot(3,1,1)
        
        plt.plot(fit_rec, label='fitness')
        
        plt.xlabel('steps')
        plt.ylabel('fitness')
        plt.yscale('log')
        
        # Save to text file
        plot_file_path_txt = os.path.join(output_dir, plot_file_name + f'_fitness.txt')
        np.savetxt(plot_file_path_txt,fit_rec)


        plt.subplot(3,1,2)
        
        plt.plot(np.abs(Hbest_rec), label='|Hbest|')
        plt.plot(np.abs(Havg_rec), label='|Havg|')
        plt.plot(np.abs(Hmin_rec), label='|Hmin|')

        plt.xlabel('steps')
        plt.ylabel('H')

        plt.yscale('log')

        plt.legend()
        
        # Save to text file
        plot_file_path_txt = os.path.join(output_dir, plot_file_name + f'_Hbest.txt')
        np.savetxt(plot_file_path_txt,Hbest_rec)
        plot_file_path_txt = os.path.join(output_dir, plot_file_name + f'_Havg.txt')
        np.savetxt(plot_file_path_txt,Havg_rec)
        plot_file_path_txt = os.path.join(output_dir, plot_file_name + f'_Hmin.txt')
        np.savetxt(plot_file_path_txt,Hmin_rec)

        plt.subplot(3,1,3)
        
        plt.plot(np.abs(obj_best_rec), label='|objective|')
        
        plot_file_path_txt = os.path.join(output_dir, plot_file_name + f'_obj_best.txt')
        np.savetxt(plot_file_path_txt,obj_best_rec)

        objv_best_rec_reshaped = np.array(objv_best_rec).reshape(np.array(objv_best_rec).shape[0], -1)
        plot_file_path_txt = os.path.join(output_dir, plot_file_name + f'_objv_best.txt')
        np.savetxt(plot_file_path_txt,objv_best_rec_reshaped)

        plt.xlabel('steps')
        plt.ylabel('objective')

        plt.yscale('log')

        plt.legend()

        ax = plt.gca()
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(True)

        # Save the figure to a file
        plt.savefig(plot_file_path)

        # Close the figure
        plt.close()

        print(f"Figure saved to file: {plot_file_path}")

    file_name = f"{name}"
    save_to_file_and_plot(file_name, PARAM_NAMES, x_rec, fit_rec, Hbest_rec, obj_best_rec, objv_best_rec, Hmin_rec, Havg_rec)

    plot_file_path_txt = os.path.join(output_dir, file_name + f'prop_BPCAC.txt')
    np.savetxt(plot_file_path_txt,prop_BPCAC_list)
    
    plot_file_path_txt = os.path.join(output_dir, file_name + f'prop_QAOACAC.txt')
    np.savetxt(plot_file_path_txt,prop_QAOACAC_list)
    

    return obj_best, sol_best, obj, sol
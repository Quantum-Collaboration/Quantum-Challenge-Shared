from datetime import datetime

from qubovert.sim import anneal_qubo

from quantum_qaoa import run_qaoa
from quantum_qaoalogn import run_qaoalogn


def sub_problem_solver_anneal(n, sub_instance, num_anneals, include_suppl_data):
    res = anneal_qubo(sub_instance.model, num_anneals=num_anneals)
    x_sol = res.best.state
    return x_sol, dict(timestamp=datetime.now().timestamp()) if include_suppl_data else {}


def sub_problem_solver_qaoa(n, sub_instance, p, use_custom_mixer, dev_kwargs, N_qaoa, include_suppl_data, seed):
    x_sol, samp_dict = run_qaoa(sub_instance, p, use_custom_mixer, dev_kwargs, N_qaoa, seed, select_indices=None)
    return x_sol, dict(timestamp=datetime.now().timestamp(), samp_dict=samp_dict) if include_suppl_data else {}


def sub_problem_solver_qaoalogn(n, sub_instance, p, t, da_maxiter, use_custom_mixer, cutoff, use_logn_sim, dev_kwargs,
                                total_shots, N_qaoa, include_suppl_data, seed):
    if len(sub_instance.variables) > N_qaoa:
        x_sol, (best_sol, history, res) = run_qaoalogn(sub_instance, p, t, da_maxiter, use_custom_mixer, cutoff,
                                                       use_logn_sim, dev_kwargs, total_shots, callback_fun=None,
                                                       N_qaoa=N_qaoa, seed=seed, verbose=False)
        optimizer_result = dict(x=res.x.tolist(), success=res.success, status=res.status, message=res.message,
                                fun=res.fun)
        return x_sol, dict(timestamp=datetime.now().timestamp(), history=history,
                           optimizer_result=optimizer_result) if include_suppl_data else {}
    else:
        return sub_problem_solver_qaoa(n, sub_instance, p, use_custom_mixer, dev_kwargs, N_qaoa, include_suppl_data,
                                       seed)

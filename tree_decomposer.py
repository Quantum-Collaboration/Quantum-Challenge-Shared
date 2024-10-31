import json
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools import generate_feasible_solution
from tools import try_to_fix_a_broken_solution, evaluate_solution, informed_local_search


def extract_random_subgraph(G, rng, desired_subgraph_size, start_node, assignable_products):
    subgraph_nodes = {start_node}
    potential_nodes = list(G.neighbors(start_node))
    assignable_subgraph_node_counter = 1
    while assignable_subgraph_node_counter < desired_subgraph_size and potential_nodes:
        next_node = rng.choice(potential_nodes)
        subgraph_nodes.add(next_node)
        potential_nodes.extend([n for n in G.neighbors(next_node) if n not in subgraph_nodes])
        potential_nodes.remove(next_node)
        if next_node in assignable_products:
            assignable_subgraph_node_counter += 1
    return G.subgraph(subgraph_nodes)


def create_PBS_graph(df_products, exclude_products=[]):
    G = nx.Graph()
    include_products = set(df_products["name"].tolist()) - set(exclude_products)
    for product_name in include_products:
        G.add_node(product_name)
    for product_name in include_products:
        parts = df_products.query(f"name=='{product_name}'").iloc[0]["parts"]
        if not pd.isna(parts):
            for part_name in parts.split(";"):
                G.add_edge(product_name, part_name)
    return G


def warm_start(model, instance, num_warmstart_iters, seed, verbose):
    x_best, obj_best = None, np.inf
    for _ in tqdm(range(num_warmstart_iters), 'warmstart', disable=not verbose):
        try:
            x_init = generate_feasible_solution(model, instance, seed=seed, ignore_ancillas=False, verbose=False)
            res = evaluate_solution(model, instance, x_init)
        except:
            continue
        _, obj = res
        if obj < obj_best:
            x_best = x_init.copy()
            obj_best = obj
    return x_best


def tree_decomposition_solver(model, instance, sub_problem_solver, sub_problem_solver_kwargs, num_warmstart_iters,
                              desired_subgraph_size, num_iterations, max_total_options, local_search_freq,
                              always_keep_best, collection_method, max_repair_iterations, local_search_max_iterations,
                              local_search_impatience, callback_fun=None, seed=None, interim_save_file_path=None, verbose=True):
    def save_sate(file_path, x, n, history, timing_data, supplementary_solver_data, x_best, out_best,
                  reset_tree_iterations_counter, processed_products):
        state_dict = dict(x=x, n=n, history=history, timing_data=timing_data,
                          supplementary_solver_data=supplementary_solver_data,
                          x_best=x_best, out_best=out_best, reset_tree_iterations_counter=reset_tree_iterations_counter,
                          processed_products=list(processed_products))
        output_path = os.path.split(file_path)[0]
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        with open(file_path, 'w') as fh:
            json.dump(state_dict, fh, default=str)

    def load_sate(file_path):
        with open(file_path, 'r') as fh:
            state_dict = json.load(fh)
        x = state_dict['x']
        n = state_dict['n']
        history = state_dict['history']
        timing_data = state_dict['timing_data']
        supplementary_solver_data = state_dict['supplementary_solver_data']
        x_best = state_dict['x_best']
        out_best = state_dict['out_best']
        reset_tree_iterations_counter = state_dict['reset_tree_iterations_counter']
        processed_products = set(state_dict['processed_products'])
        return x, n, history, timing_data, supplementary_solver_data, x_best, out_best, reset_tree_iterations_counter, processed_products

    # data
    products = model.data_processor.df_products["name"].tolist()
    product_name_to_index = {product: model.data_processor.df_products.query(f"name=='{product}'")["index"].iloc[0] for
                             product in products}
    product_index_to_name = {value: key for key, value in product_name_to_index.items()}
    levels = sorted(list(set(model.data_processor.df_products["level"])))
    products_on_level = {level: model.data_processor.df_products.query(f"level=={level}")["name"].tolist() for level in
                         levels}
    product_level_map = {product: model.data_processor.df_products.query(f"name=='{product}'")["level"].iloc[0] for
                         product in products}
    product_name_to_option_cont = {product: len(instance.df_solution_space.query(f"product=='{product}'")) for product
                                   in products}
    assignable_products = [product for product, option_count in product_name_to_option_cont.items() if option_count > 1]

    # seed generator
    rng = np.random.RandomState(seed)

    def new_seed():
        return rng.randint(0, 2 ** 31)

    if interim_save_file_path is not None and os.path.exists(interim_save_file_path):
        # restore
        x, n, history, timing_data, supplementary_solver_data, x_best, out_best, reset_tree_iterations_counter, processed_products = load_sate(
            interim_save_file_path)
        n_start = n + 1
        if verbose:
            print(f"recovered state from {interim_save_file_path}")

    else:
        # start fresh

        # warmstart
        x_init = warm_start(model, instance, num_warmstart_iters, new_seed(), verbose)
        out_init = evaluate_solution(model, instance, x_init)

        # setup
        history = {"init": (x_init.copy(), out_init)}
        timing_data = {'solve': {}, 'repair': {}, 'boost': {}, 'iter': {}}
        supplementary_solver_data = {}
        x_best, out_best = x_init.copy(), out_init
        processed_products = set()
        reset_tree_iterations_counter = 0
        x = x_init.copy()
        n_start = 0

    # run iterative decomposition
    pbar = tqdm(range(n_start, num_iterations), total=num_iterations - n_start, desc='optimize', disable=not verbose)
    for n in pbar:
        # print({product: product_level_map[product] for product in products if product not in processed_products})
        pbar.set_postfix_str("decompose")
        start_iter_time = time.time()

        # select product on deepest level
        max_level_of_unprocessed_products = max(
            [product_level_map[product] for product in products if product not in processed_products])
        start_node = rng.choice(
            [product_name for product_name in products_on_level[max_level_of_unprocessed_products] if
             product_name not in processed_products])

        # select neighboring nodes
        G = create_PBS_graph(model.data_processor.df_products, exclude_products=[])
        G_sub = extract_random_subgraph(G, rng, desired_subgraph_size, start_node, assignable_products)
        selected_products = list(G_sub.nodes())

        # get products
        processed_products = set.union(processed_products, selected_products)
        selected_product_indices = [product_name_to_index[product_name] for product_name in selected_products]
        selected_product_indices = [product_index for product_index in selected_product_indices if
                                    product_index_to_name[product_index] in assignable_products]  # filter assignable

        # collect options
        if collection_method == "random":
            options_per_product = {
                product_index: instance.df_solution_space.query(f"product_index=={product_index}")["index"].tolist() for
                product_index in selected_product_indices}
            rng.shuffle(selected_product_indices)  # iterate in random order
            collected_options = []
            idx = 0
            while len(collected_options) < max_total_options // 2 and any(
                    [len(options) > 0 for options in options_per_product.values()]):
                product_index = selected_product_indices[idx]
                if product_index_to_name[
                    product_index] in assignable_products:  # only collect options of assignable products
                    number_of_options = len(options_per_product[product_index])
                    if number_of_options > 0:
                        selected_option_idx = rng.randint(0, number_of_options)
                        selected_option_index = options_per_product[product_index].pop(selected_option_idx)
                        collected_options.append(selected_option_index)
                idx = (idx + 1) % len(selected_product_indices)
        else:
            raise ValueError(f"collection method '{collection_method}' not implemented")

        # get variable assignments for sub-problem
        free_variable_names = []
        for option_index in collected_options:
            for double_source_index in range(2):
                option = instance.df_solution_space.query(f"index=={option_index}").iloc[0]
                product_index = option["product_index"]
                site_index = option["site_index"]
                supplier_index = option["supplier_index"]
                variable_name = instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                  supplier_index)
                free_variable_names.append(variable_name)

        # variable assignmed for non-free variables
        variable_assignments = {}
        for variable_name, value in x.items():  ##################
            if '__a' in variable_name:
                continue
            if variable_name not in free_variable_names:
                variable_assignments[variable_name] = value

        # variable assignmed for non-considered free variables
        for product_index in selected_product_indices:
            if product_index_to_name[product_index] in assignable_products:
                for _, row in instance.df_solution_space.query(f"product_index=={product_index}").iterrows():
                    site_index = int(row['site_index'])
                    supplier_index = int(row['supplier_index'])
                    for double_source_index in range(2):
                        variable_name = instance.variable_indices_to_name(product_index, double_source_index,
                                                                          site_index,
                                                                          supplier_index)
                        if variable_name not in free_variable_names:
                            variable_assignments[
                                variable_name] = 0  # set all non-considered variables of the sub-problem to 0

        # crate new instance for sub-problem
        sub_instance = model.spawn_instance(instance.weights, instance.alpha_pq, instance.lambda_values,
                                            instance.enable_box_constraint_lambda_rescaling,
                                            variable_assignments=variable_assignments, verbose=False)

        # solve sub-problem
        # print(f"run sub-solver on QUBO of size {sub_instance.model.num_binary_variables}") # DEBUG
        pbar.set_postfix_str("solve")
        start_time = time.time()
        x_sub, suppl_data = sub_problem_solver(n, sub_instance, **sub_problem_solver_kwargs)
        timing_data['solve'][n] = time.time() - start_time
        supplementary_solver_data[n] = suppl_data

        # integrate solution into current solution
        x.update(x_sub)  # set the solved sub-problem variables to the given value

        # fix it, if necessary
        if not instance.model.is_solution_valid(x):
            # print("updated solution is broken") # DEBUG
            try:
                pbar.set_postfix_str("repair")
                start_time = time.time()
                x_corrected = try_to_fix_a_broken_solution(model, instance, x, max_iterations=max_repair_iterations,
                                                           seed=new_seed(), verbose=False)
                timing_data['repair'][n] = time.time() - start_time
                if x_corrected is not None:
                    x = x_corrected
            except:
                pass  # should not happen, but lets be careful

        # perform local search
        if n % local_search_freq == 0:
            try:
                pbar.set_postfix_str("boost")
                start_time = time.time()
                x_improved = informed_local_search(model, instance, x, num_iterations=local_search_max_iterations,
                                                   impatience=local_search_impatience, seed=new_seed(), verbose=False)
                timing_data['boost'][n] = time.time() - start_time
                x = x_improved
            except:
                pass  # should not happen, but lets be careful

        # process results
        pbar.set_postfix_str("evaluate")

        # check for reset
        if len(set(products) - processed_products) < desired_subgraph_size:
            # start with a fresh tree traversal if not enough products are left
            reset_tree_iterations_counter += 1
            # print(f"reset tree iteration: {reset_tree_iterations_counter}") # DEBUG
            processed_products.clear()

        # evaluate
        if instance.model.is_solution_valid(x):
            out = evaluate_solution(model, instance, x, allow_infasible_solutions=True)
        else:
            out = None

        # get new best
        if out is not None:
            _, obj = out
            _, obj_best = out_best
            if obj < obj_best:
                x_best = x.copy()
                out_best = out
        if always_keep_best:
            x = x_best.copy()
            out = out_best

        # progress
        history[n] = (x.copy(), out)
        timing_data['iter'][n] = time.time() - start_iter_time
        if interim_save_file_path is not None:
            save_sate(interim_save_file_path, x, n, history, timing_data, supplementary_solver_data, x_best, out_best,
                      reset_tree_iterations_counter, processed_products)
        if callback_fun is not None:
            callback_fun(n, x, out)

        # DEBUG
        # print(out)

    # return result
    return (x_best, out_best), history, timing_data, supplementary_solver_data

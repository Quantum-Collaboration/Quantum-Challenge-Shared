import json
import os
import warnings

import numpy as np
import pandas as pd
from qubovert import PCBO
from qubovert import boolean_var
from tqdm import tqdm

from process import DataProcessor
from tools import get_rel_value, A

VAR_TYPE = type(PCBO())
WS_SCALE = 1 / 100
MODEL_VERSION = "FINAL"


class Instance():

    def __init__(self, df_solution_space, weights, alpha_p, alpha_q, lambda_values,
                 enable_box_constraint_lambda_rescaling, objective_normalization_limits_dict):
        self.df_solution_space = df_solution_space
        self.weights = weights
        self.alpha_p = alpha_p
        self.alpha_q = alpha_q
        self.lambda_values = lambda_values
        self.enable_box_constraint_lambda_rescaling = enable_box_constraint_lambda_rescaling
        self.objective_normalization_limits_dict = objective_normalization_limits_dict  # n : (loqer, upper)
        self.model = None  # PCBO
        self.ancilla_con_map = {}  # constraint index -> sorted list of ancilla variable names
        self.expression_con_map = {}  # constraint index -> constraint expression
        self.constants = {}  # variable name -> value (just to check, cannot be modified to change model)
        self.non_connectivity_cache = {}  # dictionary of non-existing connections in the format: i -> i_site_index -> [ list of j_site_index ]
        self.con_info = {}  # constraint index -> constraint count

    @property
    def qubo_matrix(self):
        return self.model.to_qubo().Q

    @property
    def variables(self):
        return self.model.variables

    @property
    def alpha(self):
        return self.alpha_p / self.alpha_q

    @property
    def alpha_pq(self):
        return [self.alpha_p, self.alpha_q]

    @property
    def variable_index_to_name_dict(self):
        return {index: name for name, index in self.model.mapping.items()}

    @property
    def variable_name_to_index_dict(self):
        return self.model.mapping

    @property
    def normalize_objectives(self):
        return self.objective_normalization_limits_dict is not None

    @property
    def is_fully_connected(self):  # if False, some connections do not exist
        return len(self.non_connectivity_cache) == 0

    @classmethod
    def variable_indices_to_name(cls, product_index, double_source_index, site_index, supplier_index):
        name = f'x_p{product_index}_a{double_source_index}_s{site_index}_u{supplier_index}'
        return name

    @classmethod
    def variable_name_to_indices(cls, name):
        product_index, double_source_index, site_index, supplier_index = [int(s[1:]) for s in name[2:].split('_')]
        return product_index, double_source_index, site_index, supplier_index

    def get_site_supplier_list(self, product_index):
        site_supplier_list = []
        for _, row in self.df_solution_space.query(f'product_index=={product_index}').iterrows():
            site_index = row['site_index']
            supplier_index = row['supplier_index']
            site_supplier_list.append((site_index, supplier_index))
        return site_supplier_list

    def variable_is_in_model(self, product_index, double_source_index, site_index, supplier_index):
        return self.variable_indices_to_name(product_index, double_source_index, site_index,
                                             supplier_index) in self.model.variables

    def spawn_instance(self, selected_variable_name_set=None, variable_assignments=None):
        if selected_variable_name_set is None:
            selected_variable_name_set = self.variables
        if variable_assignments is None:
            variable_assignments = {}
        new_model = self.model.subgraph(set(selected_variable_name_set) - set(list(variable_assignments.keys())),
                                        variable_assignments)
        new_instance = Instance(self.df_solution_space, self.weights, self.alpha_p, self.alpha_q, self.lambda_values,
                                self.enable_box_constraint_lambda_rescaling, self.objective_normalization_limits_dict)
        new_instance.model = new_model
        new_instance.constants = self.constants.copy()
        new_instance.ancilla_con_map = self.ancilla_con_map.copy()
        new_instance.expression_con_map = {idx: con.copy() for idx, con in self.expression_con_map.items()}
        new_instance.constants.update(variable_assignments)
        new_instance.non_connectivity_cache = self.non_connectivity_cache.copy()
        new_instance.con_info = self.con_info.copy()
        return new_instance


class Model():

    def __init__(self, allow_in_memory_cache=True, data_path=None, cache_path=None, verbose_data_processor=True):
        self.data_processor = DataProcessor(data_path, cache_path, verbose_data_processor)
        self._initialize()
        if allow_in_memory_cache:
            self._CACHED_OCV = {}
        self.ws_obj_factor = WS_SCALE
        self.model_version = MODEL_VERSION

    @classmethod
    def get_costs(self, df_cost_matrix, i, i_site_index, j_site_index):
        return self._get_costs(df_cost_matrix, i, i_site_index, j_site_index)

    def rescale_total_objective(self, weights, objective_normalization_limits_dict, total_objective):
        if objective_normalization_limits_dict is not None:
            min_value, max_value = 0, 0
            for n in range(4):
                min_value_n, max_value_n = objective_normalization_limits_dict[n]
                min_value += min_value_n * weights[n]
                max_value += max_value_n * weights[n]
            assert max_value > min_value
            rescaled_objective = (total_objective - min_value) / (max_value - min_value)
            return rescaled_objective
        return total_objective

    def _initialize(self):

        # lists
        self.product_indices = self.data_processor.df_products["index"].tolist()

        # PBS: phi
        self.phi = []  # product, parent
        for _, row in self.data_processor.df_products.iterrows():
            product_index = row['index']
            parent_name = row['parent']
            if not pd.isna(parent_name):
                parent_index = int(self.data_processor.df_products.query(f'name=="{parent_name}"')["index"].iloc[0])
                self.phi.append((product_index, parent_index))

        # PBS: psi
        self.psi = {}  # product -> parts
        for _, row in self.data_processor.df_products.iterrows():
            product_index = row['index']
            self.psi[product_index] = []
            parts = row['parts']
            if not pd.isna(parts):
                for part_name in parts.split(";"):
                    part_index = int(self.data_processor.df_products.query(f'name=="{part_name}"')["index"].iloc[0])
                    self.psi[product_index].append(part_index)

        # PBS: psi inverse
        self.phi_inv = {}  # product -> parent
        for _, row in self.data_processor.df_products.iterrows():
            product_index = row['index']
            parent_name = row['parent']
            if not pd.isna(parent_name):
                parent_index = int(self.data_processor.df_products.query(f'name=="{parent_name}"')["index"].iloc[0])
                self.phi_inv[product_index] = parent_index

        # rel_values
        self.product_rel_values_by_index = {
            product_index: get_rel_value(self.data_processor.df_products, product_index) for product_index in
            self.data_processor.df_products["index"].tolist()}
        self.product_rel_values_p_by_index = {product_index: int(
            self.data_processor.df_products.query(f"index=={product_index}").iloc[0]["rel_value_p"]) for
            product_index in self.data_processor.df_products["index"].tolist()}
        self.product_rel_values_q_by_index = {
            product_index: int(
                self.data_processor.df_products.query(f"index=={product_index}").iloc[0]["rel_value_q"]) for
            product_index in
            self.data_processor.df_products["index"].tolist()}

    def _get_costs(self, df_cost_matrix, i, i_site_index, j_site_index):
        if i_site_index == j_site_index:
            return [0.] * self.data_processor.number_of_kpis  # no same-site costs
        df_costs = df_cost_matrix.query(
            f'product_index=={i}&start_site_index=={i_site_index}&end_site_index=={j_site_index}')
        if len(df_costs) == 0:
            return None  # is_connected() == False
        assert len(df_costs) == 1
        costs = df_costs.iloc[0][self.data_processor.kpi_names].tolist()
        assert min(costs) >= 0
        return [float(cost) for cost in costs]

    def _calculate_objective_normalization(self, df_cost_matrix, D, instance):
        weights = instance.weights
        normalization_file_path = os.path.join(self.data_processor.cache_path,
                                               f'objective_normalization-{";".join([f"{weight:.9f}" for weight in weights])}.json')
        if os.path.exists(normalization_file_path):
            with open(normalization_file_path, 'r') as fh:
                json_dict = json.load(fh)
                objective_normalization_limits_dict = {int(key): value for key, value in json_dict.items()}
        else:
            objective_normalization_limits_dict = {idx: None for idx in range(4)}
            # objectives 1 to 3
            for n in range(3):
                lower, upper = 0, 0
                for i, j in tqdm(self.phi, desc=f"normalize objective {n + 1}"):
                    cost_list = []
                    for i_site_index, i_supplier_index in instance.get_site_supplier_list(i):
                        for j_site_index, j_supplier_index in instance.get_site_supplier_list(j):
                            costs = self._get_costs(df_cost_matrix, i, i_site_index, j_site_index)
                            if costs is not None:
                                cost = costs[n]
                                cost_list.append(cost)
                    if len(cost_list) > 0:
                        upper += D[i] * D[j] * max(cost_list)
                        lower += D[i] * D[j] * min(cost_list)
                assert lower < upper
                objective_normalization_limits_dict[n] = (lower, upper)
            # objective 4
            lower, upper = 0, 0
            for supplier_index in tqdm(sorted(list(set(instance.df_solution_space["supplier_index"].tolist()))),
                                       desc="normalize objective 4"):
                targetWorkshare_supplier = float(
                    self.data_processor.df_suppliers.query(f'index=={supplier_index}').iloc[0][
                        'targetWorkshare_supplier'])
                upper1 = -targetWorkshare_supplier
                for product_index in self.data_processor.df_products["index"].tolist():
                    for site_index, product_supplier_index in instance.get_site_supplier_list(product_index):
                        if supplier_index == product_supplier_index:
                            rel_value = self.product_rel_values_by_index[product_index]
                            upper1 += rel_value
                upper1 = upper1 ** 2
                upper2 = targetWorkshare_supplier ** 2
                upper += max(upper1, upper2)
            assert lower < upper
            objective_normalization_limits_dict[3] = (lower, upper)
            with open(normalization_file_path, 'w') as fh:
                json.dump(objective_normalization_limits_dict, fh)
        return objective_normalization_limits_dict

    def _generated_or_load_objectives_and_constraints(self, instance, df_cost_matrix, variable_assignments, verbose):
        ocv_key = tuple(instance.weights[:3] + [instance.alpha_p, instance.alpha_q])
        if hasattr(self, '_CACHED_OCV') and ocv_key in self._CACHED_OCV and len(variable_assignments) == 0:
            (OBJS, CONS, VARS) = self._CACHED_OCV[ocv_key]
            if verbose:
                print("loaded data from OCV cache")
        else:
            OBJS, CONS, VARS = self._generate_objectives_and_constraints(instance, df_cost_matrix, variable_assignments,
                                                                         verbose)
            if hasattr(self, '_CACHED_OCV') and len(variable_assignments) == 0:
                self._CACHED_OCV[ocv_key] = (OBJS, CONS, VARS)
        return (OBJS, CONS, VARS)

    def _generate_objectives_and_constraints(self, instance, df_cost_matrix, variable_assignments, verbose):

        # lists
        feasible_site_indices = sorted(list(set(instance.df_solution_space["site_index"].tolist())))
        feasible_supplier_indices = sorted(list(set(instance.df_solution_space["supplier_index"].tolist())))

        # regions
        R = dict()  # site index -> region index
        for site_index in feasible_site_indices:
            region_indices = list(set(instance.df_solution_space.query(f"site_index=={site_index}")["region_index"]))
            assert len(region_indices) == 1
            region_index = region_indices[0]
            R[site_index] = region_index

        # product_index -> number of regions
        N = {}  # product index -> number of feasible regions
        for product_index in self.data_processor.df_products["index"].tolist():
            regions = set(instance.df_solution_space.query(f"product_index=={product_index}")["region"].tolist())
            N[product_index] = len(regions)

        # caching
        caching_file_path = os.path.join(self.data_processor.cache_path,
                                         f'model_cache-{";".join([f"{weight:.9f}" for weight in instance.weights])}.json')
        if os.path.exists(caching_file_path):
            with open(caching_file_path, 'r') as fh:
                json_dict = json.load(fh)
                site_supplier_list_dict = {int(key): value for key, value in
                                           json_dict["site_supplier_list_dict"].items()}
                cost_dict = {tuple([int(k) for k in key[1:-1].split(",")]): value for key, value in
                             json_dict["cost_dict"].items()}
        else:
            site_supplier_list_dict = {}
            for product_index in self.product_indices:
                site_supplier_list_dict[product_index] = instance.get_site_supplier_list(product_index)
            cost_dict = {}
            for i, j in self.phi:
                for i_site_index, i_supplier_index in site_supplier_list_dict[i]:
                    for j_site_index, j_supplier_index in site_supplier_list_dict[j]:
                        costs = self._get_costs(df_cost_matrix, i, i_site_index, j_site_index)
                        cost_dict[(i, i_site_index, j_site_index)] = costs
            caching_data = dict(site_supplier_list_dict=site_supplier_list_dict,
                                cost_dict={str(key): value for key, value in cost_dict.items()})
            with open(caching_file_path, 'w') as fh:
                json.dump(caching_data, fh, default=str)

        # variables
        x = {}  # key -> boolean variable or constant
        D = {}  # product index -> 1: single sourcing, 2: double sourcing
        constants = {}
        for _, row in tqdm(instance.df_solution_space.iterrows(), desc="variables",
                           total=len(instance.df_solution_space), disable=not verbose):
            product_index = row['product_index']
            site_index = row['site_index']
            supplier_index = row['supplier_index']
            num_options_for_product = len(instance.df_solution_space.query(f'product_index=={product_index}'))
            d = 1 if num_options_for_product == 1 else 2  # 1: single sourcing, 2: double sourcing
            if product_index not in D:
                D[product_index] = d
            else:
                assert D[product_index] == d
            for double_source_index in range(2):
                variable_name = instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                  supplier_index)
                if variable_name in variable_assignments:
                    # variable is already fixed
                    constant_value = variable_assignments[variable_name]
                    x[variable_name] = constant_value
                    constants[variable_name] = constant_value
                elif d == 1:
                    # every single source product leads to two constants, both sources are assigned to the only available site (there is no other option)
                    constant_value = 1
                    x[variable_name] = constant_value
                    constants[variable_name] = constant_value
                else:
                    # every double source product leads to two variables, one for each source
                    x[variable_name] = boolean_var(variable_name)

        # objective: co2, eur, time (presume: production time = 0)
        OBJ_co2, OBJ_eur, OBJ_time = 0, 0, 0
        non_connectivity_cache = {}
        for i, j in tqdm(self.phi, desc="objectives 1, 2, 3", disable=not verbose):
            for i_site_index, i_supplier_index in site_supplier_list_dict[i]:
                for j_site_index, j_supplier_index in site_supplier_list_dict[j]:
                    costs = cost_dict[(i, i_site_index, j_site_index)]
                    if costs is None:
                        if i not in non_connectivity_cache:
                            non_connectivity_cache[i] = {}
                        if i_site_index not in non_connectivity_cache[i]:
                            non_connectivity_cache[i][i_site_index] = []
                        non_connectivity_cache[i][i_site_index].append(j_site_index)
                        continue  # non-existing connection
                    cost_co2 = costs[0]
                    cost_eur = costs[1]
                    cost_time = costs[2]
                    i_double_source_index = 0
                    for j_double_source_index in range(2):
                        x_i = x[
                            instance.variable_indices_to_name(i, i_double_source_index, i_site_index, i_supplier_index)]
                        x_j = x[
                            instance.variable_indices_to_name(j, j_double_source_index, j_site_index, j_supplier_index)]
                        OBJ_co2 += A(0, instance) * cost_co2 * x_i * x_j
                        OBJ_eur += A(0, instance) * cost_eur * x_i * x_j
                        OBJ_time += A(0, instance) * cost_time * x_i * x_j
                    i_double_source_index = 1
                    for j_double_source_index in range(2):
                        x_i = x[
                            instance.variable_indices_to_name(i, i_double_source_index, i_site_index, i_supplier_index)]
                        x_j = x[
                            instance.variable_indices_to_name(j, j_double_source_index, j_site_index, j_supplier_index)]
                        OBJ_co2 += A(1, instance) * cost_co2 * x_i * x_j
                        OBJ_eur += A(1, instance) * cost_eur * x_i * x_j
                        OBJ_time += A(1, instance) * cost_time * x_i * x_j

        # objective: supplier target workshare
        OBJ_ws = 0
        for supplier_index in tqdm(feasible_supplier_indices,
                                   desc="objective 4", disable=not verbose):
            targetWorkshare_supplier = float(
                self.data_processor.df_suppliers.query(f'index=={supplier_index}').iloc[0]['targetWorkshare_supplier'])
            con = -targetWorkshare_supplier
            for product_index in self.data_processor.df_products["index"].tolist():
                rel_value = self.product_rel_values_by_index[product_index]
                for site_index, product_supplier_index in site_supplier_list_dict[product_index]:
                    if supplier_index == product_supplier_index:
                        # first source
                        double_source_index = 0
                        x_1 = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                  supplier_index)]
                        if x_1 is not None:
                            con += A(0, instance) * rel_value * x_1
                        # second source
                        double_source_index = 1
                        x_2 = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                  supplier_index)]
                        if x_2 is not None:
                            con += A(1, instance) * rel_value * x_2
            OBJ_ws += con ** 2
        OBJ_ws = OBJ_ws * self.ws_obj_factor  # rescale with a fixed constant

        # con 1: avoid forbidden connections
        CON_1_list = []
        con_dict = {}
        for i, j in tqdm(self.phi, desc="constraint 1", disable=not verbose):
            if len(non_connectivity_cache) == 0:
                continue
            is_immovable = self.data_processor.df_products.query(f"index=={i}").iloc[0]["is_immovable"]
            for i_site_index, i_supplier_index in site_supplier_list_dict[i]:
                for j_site_index, j_supplier_index in site_supplier_list_dict[j]:
                    if cost_dict[(i, i_site_index, j_site_index)] is None:  # not connected
                        if is_immovable:
                            for i_double_source_index in range(2):
                                j_double_source_index = i_double_source_index
                                x_i = x[instance.variable_indices_to_name(i, i_double_source_index, i_site_index,
                                                                          i_supplier_index)]
                                x_j = x[instance.variable_indices_to_name(j, j_double_source_index, j_site_index,
                                                                          j_supplier_index)]
                                con_key = (i, i_site_index, j_site_index)
                                if con_key not in con_dict:
                                    con_dict[con_key] = 0
                                con_dict[con_key] += x_i * x_j
                        else:
                            for i_double_source_index in range(2):
                                for j_double_source_index in range(2):
                                    x_i = x[instance.variable_indices_to_name(i, i_double_source_index, i_site_index,
                                                                              i_supplier_index)]
                                    x_j = x[instance.variable_indices_to_name(j, j_double_source_index, j_site_index,
                                                                              j_supplier_index)]
                                    con_key = (i, i_site_index, j_site_index)
                                    if con_key not in con_dict:
                                        con_dict[con_key] = 0
                                    con_dict[con_key] += x_i * x_j
        for con in con_dict.values():
            if type(con) is VAR_TYPE:
                CON_1_list.append(con)  # eq_zero
            else:
                assert con == 0

        # con 2: double sourcing: one assignment per source
        CON_2_list = []
        for product_index, d in tqdm(D.items(), desc='constraint 2', disable=not verbose):
            if d == 2:
                for double_source_index in range(2):
                    con = -1
                    for site_index, supplier_index in site_supplier_list_dict[product_index]:
                        x_con = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                    supplier_index)]
                        con += x_con

                    if type(con) is VAR_TYPE:
                        CON_2_list.append(con)  # eq_zero
                    else:
                        assert con == 0, (product_index, double_source_index, con)

        # con 3: double sourcing: different sites
        CON_3_list = []
        for product_index, d in tqdm(D.items(), desc='constraint 3', disable=not verbose):
            if d == 2:
                con_dict1, con_dict2 = dict(), dict()
                for site_index, supplier_index in site_supplier_list_dict[product_index]:
                    # first source
                    double_source_index = 0
                    if site_index not in con_dict1:
                        con_dict1[site_index] = 0
                    x_con = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                supplier_index)]
                    con_dict1[site_index] += x_con
                    # second source
                    double_source_index = 1
                    if site_index not in con_dict2:
                        con_dict2[site_index] = 0
                    x_con = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                supplier_index)]
                    con_dict2[site_index] += x_con

                for site_index in con_dict1.keys():
                    con1 = con_dict1[site_index]
                    con2 = con_dict2[site_index]
                    con = con1 * con2
                    if type(con) is VAR_TYPE:
                        CON_3_list.append(con)  # eq_zero
                    else:
                        assert con == 0

        # con 4: double sourcing: different regions
        CON_4_list = []
        for product_index, n in tqdm(N.items(), desc='constraint 4', disable=not verbose):
            if n >= 2:
                for r in R.keys():
                    con1, con2 = 0, 0
                    for site_index, supplier_index in site_supplier_list_dict[product_index]:
                        # first source
                        double_source_index = 0
                        if R[site_index] == r:
                            x_con = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                        supplier_index)]
                            con1 += x_con
                        # second source
                        double_source_index = 1
                        if R[site_index] == r:
                            x_con = x[
                                instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                  supplier_index)]
                            con2 += x_con
                    con = con1 * con2
                    if type(con) is VAR_TYPE:
                        CON_4_list.append(con)  # eq_zero
                    else:
                        assert con == 0

        # con 5: site work share (box constraint)
        CON_5_list = []
        q_list = set(list(self.product_rel_values_q_by_index.values()))
        assert len(q_list) == 1
        q = int(list(q_list)[0])
        for site_index in tqdm(feasible_site_indices,
                               desc='constraint 5', disable=not verbose):
            con = 0
            con_min_val = 0
            con_max_val = 0
            for product_index in D.keys():
                rel_value_p = self.product_rel_values_p_by_index[product_index]
                for product_site_index, supplier_index in site_supplier_list_dict[product_index]:
                    if site_index == product_site_index:
                        # first source
                        double_source_index = 0
                        x_1 = x[
                            instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                              supplier_index)]
                        coefficient = instance.alpha_p * rel_value_p
                        con += coefficient * x_1  # alpha
                        con_max_val += coefficient  # is positive
                        if type(x_1) is not VAR_TYPE:
                            con_min_val += coefficient

                        # second source
                        double_source_index = 1
                        x_2 = x[
                            instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                              supplier_index)]
                        coefficient = (instance.alpha_q - instance.alpha_p) * rel_value_p
                        con += coefficient * x_2  # 1-alpha
                        con_max_val += coefficient  # is positive
                        if type(x_2) is not VAR_TYPE:
                            con_min_val += coefficient

            minimumWorkshare_site = int(
                self.data_processor.df_production_sites.query(f'index=={site_index}').iloc[0]['minimumWorkshare_site'])
            maximumWorkshare_site = int(
                self.data_processor.df_production_sites.query(f'index=={site_index}').iloc[0]['maximumWorkshare_site'])
            m = minimumWorkshare_site * q * instance.alpha_q
            M = maximumWorkshare_site * q * instance.alpha_q
            if m <= M:  # equality could be handled better, but it's not in the data
                if type(con) is VAR_TYPE:
                    if con_max_val < m:  # violation
                        # print(f"site {site_index} (ws:{minimumWorkshare_site}..{maximumWorkshare_site}): {con_min_val} < {m} < {con_max_val}: min cannot be fulfilled")
                        pass
                    elif con_min_val >= m:  # trivial
                        # print(f"site {site_index} (ws:{minimumWorkshare_site}..{maximumWorkshare_site}): {con_min_val} < {m} < {con_max_val}: min is always fulfilled")
                        pass
                    else:  # m <= con
                        CON_5_list.append(m - con)  # le_zero
                    if con_min_val > M:  # violation
                        # print(f"site {site_index} (ws:{minimumWorkshare_site}..{maximumWorkshare_site}): {con_min_val} < {M} < {con_max_val}: max cannot be fulfilled")
                        pass
                    elif con_max_val <= M:  # trivial
                        # print(f"site {site_index} (ws:{minimumWorkshare_site}..{maximumWorkshare_site}): {con_min_val} < {M} < {con_max_val}: max is always fulfilled")
                        pass
                    else:  # M >= c
                        CON_5_list.append(con - M)  # le_zero
                else:
                    # con is number
                    if m > con or con > M:
                        if verbose:
                            print(
                                f"invalid constant work share constraint for site {site_index}: {m} < {con} < {M} is violated")
            else:
                if verbose:
                    print(f"invalid workshare constraint for site {site_index}: {m} < {M}")

        # con 6: supplier work share (box constraint)
        CON_6_list = []
        for supplier_index in tqdm(feasible_supplier_indices,
                                   desc='constraint 6', disable=not verbose):
            con = 0
            con_min_val = 0
            con_max_val = 0
            for product_index in D.keys():
                rel_value_p = self.product_rel_values_p_by_index[product_index]
                for site_index, product_supplier_index in site_supplier_list_dict[product_index]:
                    if supplier_index == product_supplier_index:

                        # first source
                        double_source_index = 0
                        x_1 = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                  supplier_index)]
                        coefficient = instance.alpha_p * rel_value_p
                        con += coefficient * x_1  # alpha
                        con_max_val += coefficient  # is positive
                        if type(x_1) is not VAR_TYPE:
                            con_min_val += coefficient

                        # second source
                        double_source_index = 1
                        x_2 = x[instance.variable_indices_to_name(product_index, double_source_index, site_index,
                                                                  supplier_index)]
                        coefficient = (instance.alpha_q - instance.alpha_p) * rel_value_p
                        con += coefficient * x_2  # 1-alpha
                        con_max_val += coefficient  # is positive
                        if type(x_2) is not VAR_TYPE:
                            con_min_val += coefficient

            minimumWorkshare_supplier = int(
                self.data_processor.df_suppliers.query(f'index=={supplier_index}').iloc[0]['minimumWorkshare_supplier'])
            maximumWorkshare_supplier = int(
                self.data_processor.df_suppliers.query(f'index=={supplier_index}').iloc[0]['maximumWorkshare_supplier'])
            m = minimumWorkshare_supplier * q * instance.alpha_q
            M = maximumWorkshare_supplier * q * instance.alpha_q
            if m <= M:  # equality could be handled better, but it's not in the data
                if type(con) is VAR_TYPE:
                    if con_max_val < m:  # violation
                        # print(f"supplier {supplier_index} (ws:{minimumWorkshare_supplier}..{maximumWorkshare_supplier}): {con_min_val} < {m} < {con_max_val}: min cannot be fulfilled")
                        pass
                    elif con_min_val >= m:  # trivial
                        # print(f"supplier {supplier_index} (ws:{minimumWorkshare_supplier}..{maximumWorkshare_supplier}): {con_min_val} < {m} < {con_max_val}: min is always fulfilled")
                        pass
                    else:  # m <= con
                        CON_6_list.append(m - con)  # le_zero
                    if con_min_val > M:  # violation
                        # print(f"supplier {supplier_index} (ws:{minimumWorkshare_supplier}..{maximumWorkshare_supplier}): {con_min_val} < {M} < {con_max_val}: max cannot be fulfilled")
                        pass
                    elif con_max_val <= M:  # trivial
                        # print(f"supplier {supplier_index} (ws:{minimumWorkshare_supplier}..{maximumWorkshare_supplier}): {con_min_val} < {M} < {con_max_val}: max is always fulfilled")
                        pass
                    else:  # M >= c
                        CON_6_list.append(con - M)  # le_zero
                else:
                    # con is number
                    if m > con or con > M:
                        if verbose:
                            print(
                                f"invalid constant work share constraint for supplier {supplier_index}: {m} < {con} < {M} is violated")
            else:
                if verbose:
                    print(f"invalid workshare constraint for site {site_index}: {m} < {M}")

        # return
        OBJS = (OBJ_co2, OBJ_eur, OBJ_time, OBJ_ws)
        CONS = (CON_1_list, CON_2_list, CON_3_list, CON_4_list, CON_5_list, CON_6_list)
        VARS = (x, D, constants, non_connectivity_cache)
        return OBJS, CONS, VARS

    def _compose_model(self, instance, x, enable_box_constraint_lambda_rescaling, OBJ_co2, OBJ_eur, OBJ_time, OBJ_ws,
                       CON_1_list, CON_2_list, CON_3_list, CON_4_list, CON_5_list, CON_6_list, verbose):
        # build scalarized model
        # for the implementation of le constraints, see https://github.com/jtiosue/qubovert/blob/16f21d29d26b74a349bc9fee43dcc3128013be22/qubovert/_pcbo.py#L1016
        assigned_ancillas = []
        instance.model = PCBO()
        for variable in x.values():
            if type(variable) is type(instance.model):
                instance.model.create_var(variable.name)

        total_objective = OBJ_co2 * instance.weights[0] + OBJ_eur * instance.weights[1] + OBJ_time * instance.weights[
            2] + OBJ_ws * instance.weights[3]
        rescaled_objective = self.rescale_total_objective(instance.weights,
                                                          instance.objective_normalization_limits_dict,
                                                          total_objective)
        instance.model += rescaled_objective
        if verbose:
            print(
                f"constraint 1 [forbidden routes]        count: {len(CON_1_list):3d}, lambda: {instance.lambda_values[0]}")
        for con in CON_1_list:
            instance.model.add_constraint_eq_zero(con, instance.lambda_values[0], suppress_warnings=not verbose)
        if verbose:
            print(
                f"constraint 2 [one assignment/product]  count: {len(CON_2_list):3d}, lambda: {instance.lambda_values[1]}")
        for con in CON_2_list:
            instance.model.add_constraint_eq_zero(con, instance.lambda_values[1], suppress_warnings=not verbose)
        if verbose:
            print(
                f"constraint 3 [double sourcing sites]   count: {len(CON_3_list):3d}, lambda: {instance.lambda_values[2]}")
        for con in CON_3_list:
            instance.model.add_constraint_eq_zero(con, instance.lambda_values[2], suppress_warnings=not verbose)
        if verbose:
            print(
                f"constraint 4 [double sourcing regions] count: {len(CON_4_list):3d}, lambda: {instance.lambda_values[3]}")
        for con in CON_4_list:
            instance.model.add_constraint_eq_zero(con, instance.lambda_values[3], suppress_warnings=not verbose)
        if verbose:
            print(
                f"constraint 5 [workshare site box]      count: {len(CON_5_list):3d}, lambda: {instance.lambda_values[4]}")
        for con in CON_5_list:
            if enable_box_constraint_lambda_rescaling:
                neg_val = sum([value for value in con.values() if value < 0])
                num_ancillas = np.ceil(np.log2(-neg_val))
                lambda_factor = 1 / 2 ** num_ancillas
            else:
                lambda_factor = 1
            effective_lambda = instance.lambda_values[4] * lambda_factor

            # add constraint
            instance.model.add_constraint_le_zero(con, effective_lambda, suppress_warnings=not verbose)

            # process ancillas
            new_ancilla_variables = [variable_name for variable_name in instance.model.variables if
                                     '__a' in variable_name and variable_name not in assigned_ancillas]
            # assert num_ancillas == len(new_ancilla_variables)
            if len(new_ancilla_variables) > 0:
                assigned_ancillas += new_ancilla_variables
                instance.ancilla_con_map[len(instance.model.constraints['le']) - 1] = sorted(
                    [ancilla_name for ancilla_name in set(new_ancilla_variables)],
                    key=lambda ancilla_name: int(ancilla_name[3:]))
                instance.expression_con_map[len(instance.model.constraints['le']) - 1] = con.copy()
        if verbose:
            print(
                f"constraint 6 [workshare supplier box]  count: {len(CON_6_list):3d}, lambda: {instance.lambda_values[5]}")
        for con in CON_6_list:
            if enable_box_constraint_lambda_rescaling:
                neg_val = sum([value for value in con.values() if value < 0])
                num_ancillas = np.ceil(np.log2(-neg_val))
                lambda_factor = 1 / 2 ** num_ancillas
            else:
                lambda_factor = 1
            effective_lambda = instance.lambda_values[4] * lambda_factor

            # add constraint
            instance.model.add_constraint_le_zero(con, effective_lambda, suppress_warnings=not verbose)

            # process ancillas
            new_ancilla_variables = [variable_name for variable_name in instance.model.variables if
                                     '__a' in variable_name and variable_name not in assigned_ancillas]
            # assert num_ancillas == len(new_ancilla_variables)
            if len(new_ancilla_variables) > 0:
                assigned_ancillas += new_ancilla_variables
                instance.ancilla_con_map[len(instance.model.constraints['le']) - 1] = sorted(
                    [ancilla_name for ancilla_name in set(new_ancilla_variables)],
                    key=lambda ancilla_name: int(ancilla_name[3:]))
                instance.expression_con_map[len(instance.model.constraints['le']) - 1] = con.copy()
        assert instance.model.degree <= 2
        if verbose:
            print(
                f"instance generated: {instance.model.num_binary_variables} variables (of which {instance.model.num_ancillas} are ancillas)")

        # store constraint info: type of constraint -> number of constraints
        instance.con_info = {1: len(CON_1_list), 2: len(CON_2_list), 3: len(CON_3_list), 4: len(CON_4_list),
                             5: len(CON_5_list), 6: len(CON_6_list)}

    def spawn_instance(self, weights, alpha_pq, lambda_values, enable_box_constraint_lambda_rescaling,
                       variable_assignments=None, verbose=True, normalize_objectives=False):
        # check settings
        assert len(weights) == 4
        assert len(lambda_values) == 6
        if normalize_objectives:
            warnings.warn(
                "model settings: use normalize_objectives=True only for testing and if you know what you are doing")
        if abs(sum(weights) - 1) > 1e-9:
            warnings.warn("model settings: non-unit weights")
        if not enable_box_constraint_lambda_rescaling:
            warnings.warn("model settings: enable_box_constraint_lambda_rescaling not enabled")
        if variable_assignments is None:
            variable_assignments = {}  # variable name -> variable value
        elif len(variable_assignments) > 0:
            if verbose:
                print(f"variable assignments: {len(variable_assignments)}")
        assert len(alpha_pq) == 2
        alpha_p, alpha_q = alpha_pq
        assert alpha_p - int(alpha_p) == 0 and alpha_p > 0
        assert alpha_q - int(alpha_q) == 0 and alpha_q > 0
        alpha_p = int(alpha_p)
        alpha_q = int(alpha_q)
        alpha = alpha_p / alpha_q
        assert alpha <= 0.8 and alpha >= .5

        # get data (is cached)
        df_cost_matrix, df_solution_space, _ = self.data_processor.evaluate(weights[:3])

        # create instance
        instance = Instance(df_solution_space, weights, alpha_p, alpha_q, lambda_values,
                            enable_box_constraint_lambda_rescaling, None)

        # create objectives
        (OBJS, CONS, VARS) = self._generated_or_load_objectives_and_constraints(instance, df_cost_matrix,
                                                                                variable_assignments, verbose)
        (OBJ_co2, OBJ_eur, OBJ_time, OBJ_ws) = OBJS
        (CON_1_list, CON_2_list, CON_3_list, CON_4_list, CON_5_list, CON_6_list) = CONS
        (x, D, constants, non_connectivity_cache) = VARS

        # assign to instance
        instance.constants = constants.copy()
        instance.non_connectivity_cache = non_connectivity_cache.copy()

        # get normalization (is cached)
        instance.objective_normalization_limits_dict = self._calculate_objective_normalization(df_cost_matrix, D,
                                                                                               instance) if normalize_objectives else None

        # compose model
        self._compose_model(instance, x, enable_box_constraint_lambda_rescaling, OBJ_co2, OBJ_eur, OBJ_time, OBJ_ws,
                            CON_1_list, CON_2_list, CON_3_list, CON_4_list, CON_5_list, CON_6_list, verbose)

        return instance

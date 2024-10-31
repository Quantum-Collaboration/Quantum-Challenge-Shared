import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


def A(double_source_index, instance):
    if double_source_index == 0:
        return float(instance.alpha)
    if double_source_index == 1:
        return float(1 - instance.alpha)


def get_rel_value(df_products, product_id):
    if type(product_id) is int:
        product_index = product_id
        rel_value_p = df_products.query(f"index=={product_index}").iloc[0]["rel_value_p"]
        rel_value_q = df_products.query(f"index=={product_index}").iloc[0]["rel_value_q"]
    else:
        product_name = str(product_id)
        rel_value_p = df_products.query(f"name=='{product_name}'").iloc[0]["rel_value_p"]
        rel_value_q = df_products.query(f"name=='{product_name}'").iloc[0]["rel_value_q"]
    rel_value = rel_value_p / rel_value_q
    return float(rel_value)


def find_correct_ancillas_for_solution(instance, x_solution):
    ancilla_solution = {}
    for con_idx, ancilla_group in instance.ancilla_con_map.items():
        ancilla_group_solution = {ancilla: 0 for ancilla in ancilla_group}
        con_expression = instance.expression_con_map[con_idx]
        # requirement is that: con_expression + sum_i 2^i a_i = 0
        constant = con_expression.value(x_solution)
        assert constant <= 0
        for idx, b in enumerate(bin(-constant)[2:][::-1]):
            ancilla_group_solution[ancilla_group[idx]] = int(b)
        ancilla_solution.update(ancilla_group_solution)
    return ancilla_solution


def generate_feasible_solution(model, instance, ws_based_selection_method=True, seed=None, ignore_ancillas=False,
                               verbose=True):
    # guess a new feasible solution
    # NOTE: ONLY WORKS FOR FULL INSTANCE, NOT SUB INSTANCES

    # random number generator
    rng = np.random.RandomState(seed)

    # gather
    products = model.data_processor.df_products["name"].tolist()
    product_name_to_index = {row["name"]: row["index"] for _, row in model.data_processor.df_products.iterrows()}
    product_rel_values = {product_name: get_rel_value(model.data_processor.df_products, product_name) for product_name
                          in products}
    product_parents = {row["name"]: None if pd.isna(row["parent"]) else row["parent"] for _, row in
                       model.data_processor.df_products.iterrows()}
    product_is_immovable = {row["name"]: row["is_immovable"] for _, row in
                            model.data_processor.df_products.iterrows()}
    levels = sorted(list(set(model.data_processor.df_products["level"])))
    suppliers = list(set(instance.df_solution_space["supplier"].tolist()))
    supplier_ws_limits = {supplier: model.data_processor.df_suppliers.query(f"name=='{supplier}'").iloc[0][
        ["minimumWorkshare_supplier", "maximumWorkshare_supplier"]].tolist() for supplier in suppliers}
    sites = list(set(instance.df_solution_space["site"].tolist()))
    site_ws_limits = {site: model.data_processor.df_production_sites.query(f"name=='{site}'").iloc[0][
        ["minimumWorkshare_site", "maximumWorkshare_site"]].tolist() for site in sites}
    site_regions = {site: model.data_processor.df_production_sites.query(f"name=='{site}'").iloc[0]["country"] for site
                    in sites}
    site_index_to_name = {row["index"]: row["name"] for _, row in
                          model.data_processor.df_production_sites.iterrows()}

    # counter
    supplier_ws = {supplier: 0 for supplier in suppliers}
    site_ws = {site: 0 for site in sites}

    # iterate levels
    assignments = []  # (product_index, double_source_index, site_index, supplier_index)
    for level in tqdm(levels, desc="generate", disable=not verbose):
        products_on_level = model.data_processor.df_products.query(f"level=={level}")["name"].tolist()
        rng.shuffle(products_on_level)  # <- random iteration of products within a level

        # iterate products
        for product_name in products_on_level:
            df_solution_space_product = instance.df_solution_space.query(f"product=='{product_name}'")
            rel_value = product_rel_values[product_name]

            # iterate sources
            exclusive_double_sourcing = len(df_solution_space_product) > 1
            exclusive_regions = exclusive_double_sourcing and len(
                set(df_solution_space_product["region"].tolist())) >= 2
            primary_source_assignment = None
            for double_source_index in range(2):
                a = A(double_source_index, instance)

                # exclusion criterion
                if double_source_index == 1 and exclusive_double_sourcing:
                    _, _, assigned_site_index, _ = primary_source_assignment
                    exclude_site = \
                        model.data_processor.df_production_sites.query(f"index=={assigned_site_index}").iloc[0]["name"]
                    if exclusive_regions:
                        exclude_region = \
                            model.data_processor.df_production_sites.query(f"index=={assigned_site_index}").iloc[0][
                                "country"]
                    else:
                        exclude_region = None
                else:
                    exclude_site = None
                    exclude_region = None

                # filter feasible sites and suppliers
                feasible_sites = set([site for site in df_solution_space_product["site"] if
                                      site_ws[site] + a * rel_value <= site_ws_limits[site][
                                          1] and site != exclude_site and site_regions[site] != exclude_region])
                feasible_suppliers = set([supplier for supplier in df_solution_space_product["supplier"] if
                                          supplier_ws[supplier] + a * rel_value <= supplier_ws_limits[supplier][1]])

                # connectivity
                product_index = product_name_to_index[product_name]
                if not instance.is_fully_connected and product_index in instance.non_connectivity_cache:
                    parent_name = product_parents[product_name]
                    forbidden_sites = set()
                    if parent_name is not None:
                        parent_index = product_name_to_index[parent_name]
                        if product_is_immovable[product_name]:
                            parent_sites = [assigned_site_index for
                                            assigned_product_index, assigned_double_source_index, assigned_site_index, _
                                            in assignments if
                                            assigned_product_index == parent_index and assigned_double_source_index == double_source_index]
                            assert len(parent_sites) == 1
                            parent_site = parent_sites[0]
                            for i_site_index, j_site_index_list in instance.non_connectivity_cache[
                                product_index].items():
                                if site_index_to_name[i_site_index] in feasible_sites and parent_site in j_site_index_list:
                                    # no connection between product site and parent site
                                    forbidden_sites.add(site_index_to_name[i_site_index])

                        else:
                            parent_sites = [assigned_site_index for
                                            assigned_product_index, assigned_double_source_index, assigned_site_index, _
                                            in assignments if
                                            assigned_product_index == parent_index]
                            assert len(parent_sites) == 2
                            for i_site_index, j_site_index_list in instance.non_connectivity_cache[
                                product_index].items():
                                if site_index_to_name[i_site_index] in feasible_sites and (
                                        parent_sites[0] in j_site_index_list or parent_sites[1] in j_site_index_list):
                                    # no connection between product site and parent site
                                    forbidden_sites.add(site_index_to_name[i_site_index])
                    feasible_sites -= forbidden_sites

                # perform filtering
                site_query = ('(' + '|'.join([f'site=="{name}"' for name in feasible_sites]) + ')') if len(
                    feasible_sites) > 0 else "index<0"  # invalid condition
                supplier_query = ('(' + '|'.join([f'supplier=="{name}"' for name in feasible_suppliers]) + ')') if len(
                    feasible_suppliers) > 0 else "index<0"  # invalid condition
                df_feasible_solution_space_product = df_solution_space_product.query(site_query + "&" + supplier_query)
                if len(df_feasible_solution_space_product) == 0:
                    raise RuntimeError(f"product {product_name} cannot be assigned on level {level}")

                # randomly choose feasible site
                feasible_option_indices = df_feasible_solution_space_product["index"].tolist()

                # choose index
                if ws_based_selection_method:
                    # choose indices with highest score
                    scores = {}
                    for _, row in df_feasible_solution_space_product.iterrows():
                        site_name = row["site"]
                        supplier_name = row["supplier"]
                        score = 0
                        min_ws, _ = site_ws_limits[site_name]
                        ws = site_ws[site_name]
                        if ws < min_ws:
                            score += min_ws - ws
                        min_ws, _ = supplier_ws_limits[supplier_name]
                        ws = supplier_ws[supplier_name]
                        if ws < min_ws:
                            score += min_ws - ws
                        scores[row["index"]] = score
                    max_cost = max(scores.values())
                    max_indices = [index for index, value in scores.items() if value == max_cost]
                    index = rng.choice(max_indices)
                else:
                    # random choice
                    index = rng.choice(feasible_option_indices)

                # process chosen option
                chosen_option = df_feasible_solution_space_product.query(f"index=={index}").iloc[0]
                key = (chosen_option["product_index"], double_source_index, chosen_option["site_index"],
                       chosen_option["supplier_index"])
                assignments.append(key)
                site_ws[chosen_option["site"]] += a * rel_value
                supplier_ws[chosen_option["supplier"]] += a * rel_value
                if double_source_index == 0:
                    primary_source_assignment = key

                # DEBUG
                # print(level, product_name, double_source_index, chosen_option["site"], chosen_option["supplier"])

    # check workshares
    for site, ws in site_ws.items():
        ws_min, ws_max = site_ws_limits[site]
        if ws < ws_min or ws > ws_max:
            raise RuntimeError(f"site {site} has invalid workshare: {ws_min} <= {ws} <= {ws_max}")
    for supplier, ws in supplier_ws.items():
        ws_min, ws_max = supplier_ws_limits[supplier]
        if ws < ws_min or ws > ws_max:
            raise RuntimeError(f"site {supplier} has invalid workshare: {ws_min} <= {ws} <= {ws_max}")

    # convert to dict
    x_init = {variable_name: 0 for variable_name in instance.variables if '__a' not in variable_name}
    for key in assignments:
        name = instance.variable_indices_to_name(*key)
        if name in instance.variables:
            x_init[name] = 1
        else:
            assert name in instance.constants and instance.constants[name] == 1
    if any(['__a' in variable_name for variable_name in instance.model.variables]):
        # instance contains ancilla variables that need to be found
        if not ignore_ancillas:
            ancilla_solution = find_correct_ancillas_for_solution(instance, x_init)
            x_init.update(ancilla_solution)
        else:
            if verbose:
                print("ancillas ignored")

    # final check
    assert instance.model.is_solution_valid(x_init), "generated solution is not feasible!"
    return x_init


def try_to_fix_a_broken_solution(model, instance, x, max_iterations=25, seed=None, ignore_ancillas=False, verbose=True):
    # repair a broken solution without changing it too much
    # NOTE: ONLY WORKS FOR FULL INSTANCE, NOT SUB INSTANCES

    # random number generator
    rng = np.random.RandomState(seed)

    def shuffle_dict(d):
        d = [item for item in d.items()]
        rng.shuffle(d)
        return dict(d)

    def shuffle_list(l):
        indices = np.arange(len(l))
        rng.shuffle(indices)
        return [l[idx] for idx in indices]

    # gather
    product_levels = {row["name"]: row["level"] for _, row in model.data_processor.df_products.iterrows()}
    products = sorted(model.data_processor.df_products["name"].tolist(), key=lambda name: product_levels[name])
    product_index_to_name = {row["index"]: row["name"] for _, row in model.data_processor.df_products.iterrows()}
    product_name_to_index = {row["name"]: row["index"] for _, row in model.data_processor.df_products.iterrows()}
    solution_space_per_product = {product_name: instance.df_solution_space.query(f"product=='{product_name}'") for
                                  product_name in products}
    product_exclusive_double_sourcing = {
        product_name: len(solution_space_per_product[product_name]) > 1 for product_name in
        products}
    product_exclusive_regions = {
        product_name: len(set(solution_space_per_product[product_name]["region"].tolist())) >= 2
        for
        product_name in products}
    product_rel_values = {product_name: get_rel_value(model.data_processor.df_products, product_name) for product_name
                          in products}
    product_parents = {row["name"]: None if pd.isna(row["parent"]) else row["parent"] for _, row in
                       model.data_processor.df_products.iterrows()}
    product_is_immovable = {row["name"]: row["is_immovable"] for _, row in
                            model.data_processor.df_products.iterrows()}
    feasible_suppliers_per_product = {
        product_name: solution_space_per_product[product_name]["supplier"].tolist() for
        product_name in products}
    feasible_sites_per_product = {
        product_name: solution_space_per_product[product_name]["site"].tolist() for product_name
        in products}
    suppliers = list(set(instance.df_solution_space["supplier"].tolist()))
    supplier_index_to_name = {row["index"]: row["name"] for _, row in model.data_processor.df_suppliers.iterrows()}
    supplier_name_to_index = {row["name"]: row["index"] for _, row in model.data_processor.df_suppliers.iterrows()}
    supplier_ws_limits = {supplier: model.data_processor.df_suppliers.query(f"name=='{supplier}'").iloc[0][
        ["minimumWorkshare_supplier", "maximumWorkshare_supplier"]].tolist() for supplier in suppliers}
    sites = list(set(instance.df_solution_space["site"].tolist()))
    site_name_to_index = {row["name"]: row["index"] for _, row in
                          model.data_processor.df_production_sites.iterrows()}
    site_index_to_name = {row["index"]: row["name"] for _, row in
                          model.data_processor.df_production_sites.iterrows()}
    site_ws_limits = {site: model.data_processor.df_production_sites.query(f"name=='{site}'").iloc[0][
        ["minimumWorkshare_site", "maximumWorkshare_site"]].tolist() for site in sites}
    site_regions = {site: model.data_processor.df_production_sites.query(f"name=='{site}'").iloc[0]["country"] for site
                    in
                    sites}

    # collect assignments
    product_source_assigned_site_supplier_region = {(product_name, double_source_index): [] for product_name in products
                                                    for
                                                    double_source_index in range(2)}
    for name, value in list(x.items()) + list(instance.constants.items()):
        if '__a' in name:
            # skip ancilla variables
            continue
        if value == 1:
            product_index, double_source_index, site_index, supplier_index = instance.variable_name_to_indices(name)
            product_name = product_index_to_name[product_index]
            site_name = site_index_to_name[site_index]
            supplier_name = supplier_index_to_name[supplier_index]
            region_name = site_regions[site_name]
            product_source_assigned_site_supplier_region[(product_name, double_source_index)].append(
                (site_name, supplier_name, region_name))

    def has_valid_connectivity(product_name, i_site, i_double_source_index, j_site, j_double_source_index):
        i = product_name_to_index[product_name]
        if instance.is_fully_connected or i not in instance.non_connectivity_cache or i_site not in \
                instance.non_connectivity_cache[i]:
            return True
        if product_is_immovable[product_name] and i_double_source_index != j_double_source_index:
            return True
        if j_site in instance.non_connectivity_cache[i][i_site]:
            return False
        return True

    # check infeasible assignments
    def check_assignments(product_source_assigned_site_supplier_region):
        product_source_infeasibility = {}
        supplier_ws = {supplier: 0 for supplier in suppliers}
        site_ws = {site: 0 for site in sites}
        for product_name in products:  # sorted by levels from top to bottom
            # check formal assignment requirements
            for double_source_index in range(2):
                if len(product_source_assigned_site_supplier_region[(product_name, double_source_index)]) > 1:
                    product_source_infeasibility[(product_name, double_source_index)] = "overassigned"
                elif len(product_source_assigned_site_supplier_region[(product_name, double_source_index)]) == 0:
                    product_source_infeasibility[(product_name, double_source_index)] = "unassigned"
            if (product_name, 0) not in product_source_infeasibility and (
                    product_name, 1) not in product_source_infeasibility:
                assert len(product_source_assigned_site_supplier_region[(product_name, 0)]) == 1
                assert len(product_source_assigned_site_supplier_region[(product_name, 1)]) == 1
                # product has a formally correct assignment, check if this assignment is also feasible
                site_name1, supplier_name1, region_name1 = \
                    product_source_assigned_site_supplier_region[(product_name, 0)][
                        0]
                site_name2, supplier_name2, region_name2 = \
                    product_source_assigned_site_supplier_region[(product_name, 1)][
                        0]
                if product_exclusive_double_sourcing[product_name] and site_name1 == site_name2:
                    product_source_infeasibility[(product_name, 0)] = "sites"
                    product_source_infeasibility[(product_name, 1)] = "sites"
                    continue
                elif product_exclusive_regions[product_name] and region_name1 == region_name2:
                    product_source_infeasibility[(product_name, 0)] = "regions"
                    product_source_infeasibility[(product_name, 1)] = "regions"
                    continue
                # check connectivity
                elif not instance.is_fully_connected:
                    parent_name = product_parents[product_name]
                    if parent_name is not None:
                        # only check if parent is feasible
                        if not product_source_infeasibility.get((parent_name, 0),
                                                                False) and not product_source_infeasibility.get(
                            (parent_name, 1), False):
                            found_invalid_connectivity = False
                            (product_site_name1, _, _) = \
                                product_source_assigned_site_supplier_region[(product_name, 0)][0]
                            (product_site_name2, _, _) = \
                                product_source_assigned_site_supplier_region[(product_name, 1)][0]
                            i_site1 = site_name_to_index[product_site_name1]
                            i_site2 = site_name_to_index[product_site_name2]
                            (parent_site_name1, _, _) = product_source_assigned_site_supplier_region[(parent_name, 0)][
                                0]
                            (parent_site_name2, _, _) = product_source_assigned_site_supplier_region[(parent_name, 1)][
                                0]
                            j_site1 = site_name_to_index[parent_site_name1]
                            j_site2 = site_name_to_index[parent_site_name2]
                            if not has_valid_connectivity(product_name, i_site1, 0, j_site1,
                                                          0) or not has_valid_connectivity(product_name, i_site1, 0,
                                                                                           j_site2, 1):
                                product_source_infeasibility[(product_name, 0)] = "connectivity"
                                found_invalid_connectivity = True
                            if not has_valid_connectivity(product_name, i_site2, 1, j_site1,
                                                          0) or not has_valid_connectivity(product_name, i_site2, 1,
                                                                                           j_site2, 1):
                                product_source_infeasibility[(product_name, 1)] = "connectivity"
                                found_invalid_connectivity = True
                            if found_invalid_connectivity:
                                continue

                # only collect valid workshares (if we are still in the loop, no infeasibility has been detected yet)
                rel_value = product_rel_values[product_name]
                site_ws[site_name1] += A(0, instance) * rel_value
                site_ws[site_name2] += A(1, instance) * rel_value
                supplier_ws[supplier_name1] += A(0, instance) * rel_value
                supplier_ws[supplier_name2] += A(1, instance) * rel_value
        site_infeasibility = {}
        for site_name in sites:
            if site_ws[site_name] < site_ws_limits[site_name][0]:
                site_infeasibility[site_name] = "underassigned"
                # print("site", site_name, "is underassigned", site_ws[site_name], "<>", site_ws_limits[site_name])
            if site_ws[site_name] > site_ws_limits[site_name][1]:
                site_infeasibility[site_name] = "overassigned"
                # print("site", site_name, "is overassigned", site_ws[site_name], "<>", site_ws_limits[site_name])
        supplier_infeasibility = {}
        for supplier_name in suppliers:
            if supplier_ws[supplier_name] < supplier_ws_limits[supplier_name][0]:
                supplier_infeasibility[supplier_name] = "underassigned"
                # print("supplier", supplier_name, "is underassigned", supplier_ws[supplier_name], "<>", supplier_ws_limits[supplier_name])
            if supplier_ws[supplier_name] > supplier_ws_limits[supplier_name][1]:
                supplier_infeasibility[supplier_name] = "overassigned"
                # print("supplier", supplier_name, "is overassigned", supplier_ws[supplier_name], "<>", supplier_ws_limits[supplier_name])
        # shuffle (induce more randomness for the order in which problems are resolved
        product_source_infeasibility = shuffle_dict(product_source_infeasibility)
        site_infeasibility = shuffle_dict(site_infeasibility)
        supplier_infeasibility = shuffle_dict(supplier_infeasibility)
        return product_source_infeasibility, site_infeasibility, site_ws, supplier_infeasibility, supplier_ws

    def process_invalid_sites_and_suppliers(product_source_assigned_site_supplier_region, product_source_infeasibility,
                                            site_infeasibility, site_ws, supplier_infeasibility, supplier_ws):
        # print("process invalid sites/suppliers")
        choose_invalid_products = set()

        # iterate invalid sites
        for infeasible_site_name, infeasibility in site_infeasibility.items():

            # randomly select products to remove from site until we have enough
            if infeasibility == "overassigned":

                # check which products are assigned to site
                selectable_products = []
                for (
                        product_name,
                        double_source_index), assignments in product_source_assigned_site_supplier_region.items():
                    if len(assignments) == 1:
                        (site_name, supplier_name, region_name) = assignments[0]
                        if site_name == infeasible_site_name and (
                                product_name, double_source_index) not in product_source_infeasibility:
                            selectable_products.append((product_name, double_source_index))
                selectable_products = shuffle_list(selectable_products)

                # process selectable products
                ws_goal = site_ws[infeasible_site_name]
                for product_name, double_source_index in selectable_products:
                    choose_invalid_products.add((product_name, double_source_index))
                    rel_value = product_rel_values[product_name]
                    a = A(double_source_index, instance)
                    ws_goal -= a * rel_value
                    if ws_goal <= site_ws_limits[infeasible_site_name][1]:
                        break
            # randomly select products to add to site until we have enough
            if infeasibility == "underassigned":

                # check which products are not assigned to site, but can be
                selectable_products = []
                for (
                        product_name,
                        double_source_index), assignments in product_source_assigned_site_supplier_region.items():
                    if len(assignments) == 1:
                        (site_name, supplier_name, region_name) = assignments[0]
                        if site_name != infeasible_site_name and (
                                product_name, double_source_index) not in product_source_infeasibility:
                            if infeasible_site_name in feasible_sites_per_product[product_name]:
                                selectable_products.append((product_name, double_source_index))
                selectable_products = shuffle_list(selectable_products)

                # process selectable products
                ws_goal = site_ws[infeasible_site_name]
                for product_name, double_source_index in selectable_products:
                    choose_invalid_products.add((product_name, double_source_index))
                    rel_value = product_rel_values[product_name]
                    a = A(double_source_index, instance)
                    ws_goal += a * rel_value
                    if ws_goal >= site_ws_limits[infeasible_site_name][0]:
                        break

        # iterate invalid suppliers
        for infeasible_supplier_name, infeasibility in supplier_infeasibility.items():

            # randomly select products to remove from site until we have enough
            if infeasibility == "overassigned":

                # check which products are assigned to supplier
                selectable_products = []
                for (
                        product_name,
                        double_source_index), assignments in product_source_assigned_site_supplier_region.items():
                    if len(assignments) == 1:
                        (site_name, supplier_name, region_name) = assignments[0]
                        if supplier_name == infeasible_supplier_name and (
                                product_name, double_source_index) not in product_source_infeasibility:
                            selectable_products.append((product_name, double_source_index))
                selectable_products = shuffle_list(selectable_products)

                # process selectable products
                ws_goal = supplier_ws[infeasible_supplier_name]
                for product_name, double_source_index in selectable_products:
                    choose_invalid_products.add((product_name, double_source_index))
                    rel_value = product_rel_values[product_name]
                    a = A(double_source_index, instance)
                    ws_goal -= a * rel_value
                    if ws_goal <= supplier_ws_limits[infeasible_supplier_name][1]:
                        break

            # randomly select products to add to supplier until we have enough
            if infeasibility == "underassigned":

                # check which products are not assigned to supplier, but can be
                selectable_products = []
                for (
                        product_name,
                        double_source_index), assignments in product_source_assigned_site_supplier_region.items():
                    if len(assignments) == 1:
                        (site_name, supplier_name, region_name) = assignments[0]
                        if supplier_name != infeasible_supplier_name and (
                                product_name, double_source_index) not in product_source_infeasibility:
                            if infeasible_supplier_name in feasible_suppliers_per_product[product_name]:
                                selectable_products.append((product_name, double_source_index))
                selectable_products = shuffle_list(selectable_products)

                # process selectable products
                ws_goal = supplier_ws[infeasible_supplier_name]
                for product_name, double_source_index in selectable_products:
                    choose_invalid_products.add((product_name, double_source_index))
                    rel_value = product_rel_values[product_name]
                    a = A(double_source_index, instance)
                    ws_goal += a * rel_value
                    if ws_goal >= supplier_ws_limits[infeasible_supplier_name][0]:
                        break

                # print("infeasible_supplier_name", infeasible_supplier_name)
                # print("\t", selectable_products, "-->", choose_invalid_products)
                # print("\t", supplier_ws[infeasible_supplier_name], "<>", supplier_ws_limits[infeasible_supplier_name], "goal", ws_goal)

        # process selected products
        for (product_name, double_source_index) in choose_invalid_products:
            product_source_assigned_site_supplier_region[
                (product_name, double_source_index)] = []  # remove assignments for those products
            # print("remove assignment for", (product_name, double_source_index))
            # NOTE: site_ws, supplier_ws have changed at this point, recalculate

        return product_source_assigned_site_supplier_region

    def process_invalid_products(product_source_assigned_site_supplier_region, product_source_infeasibility,
                                 site_infeasibility, site_ws, supplier_infeasibility, supplier_ws):
        # print("process invalid products")

        # sort by level: from top to bottom
        product_source_infeasibility = dict(sorted([item for item in product_source_infeasibility.items()],
                                                   key=lambda item: product_levels[item[0][0]]))

        # iterate invalid products
        for (product_name, double_source_index), infeasibility in product_source_infeasibility.items():

            if infeasibility == "overassigned":
                num_assignments = len(product_source_assigned_site_supplier_region[(product_name, double_source_index)])
                selected_index = rng.choice(np.arange(num_assignments))
                new_assignment = product_source_assigned_site_supplier_region[(product_name, double_source_index)][
                    selected_index]

            elif infeasibility in ["unassigned", "sites", "regions", "connectivity"]:

                df_solution_space_product = solution_space_per_product[product_name]

                rel_value = product_rel_values[product_name]
                a = A(double_source_index, instance)
                exclusive_regions = product_exclusive_regions[product_name]
                assert product_exclusive_double_sourcing[product_name]  # presume that double sourcing is in place

                # check other source
                if (product_name, 1 - double_source_index) not in product_source_infeasibility:
                    other_site_name, other_supplier_name, other_region_name = \
                        product_source_assigned_site_supplier_region[(product_name, 1 - double_source_index)][0]
                    if not exclusive_regions:
                        other_region_name = None
                else:
                    other_site_name, other_supplier_name, other_region_name = None, None, None

                # filter feasible sites and suppliers
                feasible_sites = set([site for site in df_solution_space_product["site"] if
                                      site_ws[site] + a * rel_value <= site_ws_limits[site][
                                          1] and site != other_site_name and site_regions[site] != other_region_name])
                feasible_suppliers = set([supplier for supplier in df_solution_space_product["supplier"] if
                                          supplier_ws[supplier] + a * rel_value <= supplier_ws_limits[supplier][1]])

                # connectivity
                product_index = product_name_to_index[product_name]
                if not instance.is_fully_connected and product_index in instance.non_connectivity_cache:
                    parent_name = product_parents[product_name]
                    if parent_name is not None:
                        # check only if parent is feasible
                        if not product_source_infeasibility.get((parent_name, 0),
                                                                False) and not product_source_infeasibility.get(
                            (parent_name, 1), False):
                            forbidden_sites = set()
                            if product_is_immovable[product_name]:
                                parent_site_name, _, _ = \
                                    product_source_assigned_site_supplier_region[(parent_name, double_source_index)][0]
                                parent_site_index = site_name_to_index[parent_site_name]
                                for i_site_index, j_site_index_list in instance.non_connectivity_cache[
                                    product_index].items():
                                    if site_index_to_name[
                                        i_site_index] in feasible_sites and parent_site_index in j_site_index_list:
                                        # no connection between product site and parent site
                                        forbidden_sites.add(site_index_to_name[i_site_index])
                            else:
                                parent_site1_name, _, _ = \
                                    product_source_assigned_site_supplier_region[(parent_name, 0)][0]
                                parent_site2_name, _, _ = \
                                    product_source_assigned_site_supplier_region[(parent_name, 1)][0]
                                parent_site1_index = site_name_to_index[parent_site1_name]
                                parent_site2_index = site_name_to_index[parent_site2_name]
                                for i_site_index, j_site_index_list in instance.non_connectivity_cache[
                                    product_index].items():
                                    if site_index_to_name[i_site_index] in feasible_sites and (
                                            parent_site1_index in j_site_index_list or parent_site2_index in j_site_index_list):
                                        # no connection between product site and parent site
                                        forbidden_sites.add(site_index_to_name[i_site_index])
                            feasible_sites -= forbidden_sites

                # perform filtering
                site_query = ('(' + '|'.join([f'site=="{name}"' for name in feasible_sites]) + ')') if len(
                    feasible_sites) > 0 else "index<0"  # invalid condition
                supplier_query = ('(' + '|'.join([f'supplier=="{name}"' for name in feasible_suppliers]) + ')') if len(
                    feasible_suppliers) > 0 else "index<0"  # invalid condition
                df_feasible_solution_space_product = df_solution_space_product.query(site_query + "&" + supplier_query)
                if len(df_feasible_solution_space_product) == 0:
                    # print(f"product {product_name} cannot be assigned")
                    new_assignment = None
                else:
                    # randomly choose a feasible site

                    # A) if spossible, prefer underassigned sites/suppliers
                    underassigned_sites = [site_name for site_name, infeasibility in site_infeasibility.items() if
                                           infeasibility == "underassigned"]
                    underassigned_suppliers = [supplier_name for supplier_name, infeasibility in
                                               supplier_infeasibility.items() if infeasibility == "underassigned"]
                    site_query = ('(' + '|'.join([f'site=="{name}"' for name in underassigned_sites]) + ')') if len(
                        underassigned_sites) > 0 else "index>=0"  # valid condition
                    supplier_query = (
                            '(' + '|'.join([f'supplier=="{name}"' for name in underassigned_suppliers]) + ')') if len(
                        underassigned_suppliers) > 0 else "index>=0"  # valid condition
                    df_prefereded_feasible_solution_space_product = df_feasible_solution_space_product.query(
                        site_query + "|" + supplier_query)
                    if len(df_prefereded_feasible_solution_space_product) > 0:
                        prefered_feasible_option_indices = df_prefereded_feasible_solution_space_product[
                            "index"].tolist()
                        index = rng.choice(prefered_feasible_option_indices)
                        chosen_option = df_prefereded_feasible_solution_space_product.query(f"index=={index}").iloc[0]
                    else:
                        # B) alteratively, avoid overassigned sites
                        overassigned_sites = [site_name for site_name, infeasibility in site_infeasibility.items() if
                                              infeasibility == "overassigned"]
                        overassigned_suppliers = [supplier_name for supplier_name, infeasibility in
                                                  supplier_infeasibility.items() if infeasibility == "overassigned"]
                        site_query = ('(' + '|'.join([f'site!="{name}"' for name in overassigned_sites]) + ')') if len(
                            overassigned_sites) > 0 else "index>=0"  # valid condition
                        supplier_query = ('(' + '|'.join(
                            [f'supplier!="{name}"' for name in overassigned_suppliers]) + ')') if len(
                            overassigned_suppliers) > 0 else "index>=0"  # valid condition
                        df_careful_feasible_solution_space_product = df_feasible_solution_space_product.query(
                            site_query + "&" + supplier_query)
                        if len(df_careful_feasible_solution_space_product) > 0:
                            careful_feasible_option_indices = df_careful_feasible_solution_space_product[
                                "index"].tolist()
                            index = rng.choice(careful_feasible_option_indices)
                            chosen_option = df_careful_feasible_solution_space_product.query(f"index=={index}").iloc[0]
                        else:
                            # C) as a fallback, just choose any feasible site
                            feasible_option_indices = df_feasible_solution_space_product["index"].tolist()
                            index = rng.choice(feasible_option_indices)
                            chosen_option = df_feasible_solution_space_product.query(f"index=={index}").iloc[0]

                    # process chosen option
                    new_assignment = (chosen_option["site"], chosen_option["supplier"], chosen_option["region"])

            # apply
            if new_assignment is not None:
                product_source_assigned_site_supplier_region[(product_name, double_source_index)] = [new_assignment]
                product_source_infeasibility[(product_name, double_source_index)] = False
                # print("new assignment for", (product_name, double_source_index), ':', new_assignment)

        return product_source_assigned_site_supplier_region

    def convert_to_x(product_source_assigned_site_supplier_region):
        # convert to dict
        x_corrected = {variable_name: 0 for variable_name in instance.variables if '__a' not in variable_name}
        for (product_name, double_source_index), assignments in product_source_assigned_site_supplier_region.items():
            assert len(assignments) == 1
            (site_name, supplier_name, _) = assignments[0]
            product_index = product_name_to_index[product_name]
            site_index = site_name_to_index[site_name]
            supplier_index = supplier_name_to_index[supplier_name]
            key = (product_index, double_source_index, site_index, supplier_index)
            name = instance.variable_indices_to_name(*key)
            if name in x_corrected:
                x_corrected[name] = 1
            else:
                assert name in instance.constants and instance.constants[name] == 1, name
        if any(['__a' in variable_name for variable_name in instance.model.variables]):
            # instance contains ancilla variables that need to be found
            if not ignore_ancillas:
                ancilla_solution = find_correct_ancillas_for_solution(instance, x_corrected)
                x_corrected.update(ancilla_solution)
            else:
                if verbose:
                    print("ancillas ignored")
        return x_corrected

    # single fix iteration
    def fix_iteration(product_source_assigned_site_supplier_region):
        # check assignments
        product_source_infeasibility, site_infeasibility, site_ws, supplier_infeasibility, supplier_ws = check_assignments(
            product_source_assigned_site_supplier_region)
        if len(product_source_infeasibility) == 0 and len(site_infeasibility) == 0 and len(supplier_infeasibility) == 0:
            return True, product_source_assigned_site_supplier_region  # early return if feasibility is achieved

        # fix sites/suppliers
        product_source_assigned_site_supplier_region = process_invalid_sites_and_suppliers(
            product_source_assigned_site_supplier_region, product_source_infeasibility, site_infeasibility, site_ws,
            supplier_infeasibility, supplier_ws)

        #  check assignments (again!)
        product_source_infeasibility, site_infeasibility, site_ws, supplier_infeasibility, supplier_ws = check_assignments(
            product_source_assigned_site_supplier_region)
        # (no need to chek if we can leave early, feasibility must be wrong at this point)

        # fix products
        product_source_assigned_site_supplier_region = process_invalid_products(
            product_source_assigned_site_supplier_region, product_source_infeasibility, site_infeasibility, site_ws,
            supplier_infeasibility, supplier_ws)

        # check assignments (again^2!)
        product_source_infeasibility, site_infeasibility, site_ws, supplier_infeasibility, supplier_ws = check_assignments(
            product_source_assigned_site_supplier_region)
        is_corrected = len(product_source_infeasibility) == 0 and len(site_infeasibility) == 0 and len(
            supplier_infeasibility) == 0
        return is_corrected, product_source_assigned_site_supplier_region

    is_corrected = False
    with tqdm(total=max_iterations, desc="try to fix", disable=not verbose) as pbar:
        for _ in range(max_iterations):
            is_corrected, product_source_assigned_site_supplier_region = fix_iteration(
                product_source_assigned_site_supplier_region)
            if is_corrected:
                pbar.n = max_iterations
                pbar.update(0)
                break
            pbar.update(1)

    # return result, if possible
    if is_corrected:
        x_corrected = convert_to_x(product_source_assigned_site_supplier_region)
        if instance.model.is_solution_valid(x_corrected):  # final check: should always be True
            return x_corrected

    # failure
    return None


def informed_local_search(model, instance, x, num_iterations, impatience, seed=None, verbose=True):
    # guess a new feasible solution
    # NOTE: ONLY WORKS FOR FULL INSTANCE, NOT SUB INSTANCES
    # impacience: set to 0 to search everything, set to 1 to seach just 1 option, everything in between is a compromise between the extremes

    # random number generator
    rng = np.random.RandomState(seed)

    # make a copy of original solution
    x = x.copy()

    # gather
    products = model.data_processor.df_products["name"].tolist()
    product_parents = {row["name"]: None if pd.isna(row["parent"]) else row["parent"] for _, row in
                       model.data_processor.df_products.iterrows()}
    product_index_to_name = {row["index"]: row["name"] for _, row in model.data_processor.df_products.iterrows()}
    products_with_potential = [row["index"] for _, row in model.data_processor.df_products.iterrows() if
                               len(instance.df_solution_space.query(f"product_index=={row['index']}")) >= 2]
    product_is_immovable = {row["name"]: row["is_immovable"] for _, row in
                            model.data_processor.df_products.iterrows()}
    site_name_to_index = {row["name"]: row["index"] for _, row in model.data_processor.df_production_sites.iterrows()}

    product_exclusive_regions = {
        product_name: len(set(instance.df_solution_space.query(f"product=='{product_name}'")["region"].tolist())) >= 2
        for
        product_name in products}

    def find_best_options_for_product(product_index, x):
        product_name = product_index_to_name[product_index]
        parent_name = product_parents[product_name]

        # parent assignment
        if not instance.is_fully_connected and parent_name is not None:
            parent_site_assignments = {}
            for name, value in list(x.items()) + list(instance.constants.items()):
                if '__a' in name:
                    continue
                if value == 1:
                    product_index_x, double_source_index_x, site_index_x, _ = instance.variable_name_to_indices(
                        name)
                    if product_index_to_name[product_index_x] == parent_name:
                        parent_site_assignments[double_source_index_x] = site_index_x
                        if len(parent_site_assignments) == 2:
                            break
            assert len(parent_site_assignments) == 2, (product_name, parent_name, parent_site_assignments)

        # pepare test solution
        x_test_solution = x.copy()
        for name, value in list(x.items()) + list(instance.constants.items()):
            if '__a' in name:
                continue
            product_index_test, _, _, _ = instance.variable_name_to_indices(name)
            if product_index == product_index_test:
                x_test_solution[name] = 0

        # setup best choice
        best_obj = instance.model.value(x)
        best_x = x.copy()

        # iterate options (option for primary source)
        for _, row1 in instance.df_solution_space.query(f"product_index=={product_index}").iterrows():
            site_name1 = row1["site"]
            region_name1 = row1["region"]

            # check connectivity
            site_index1 = site_name_to_index[site_name1]
            if not instance.is_fully_connected and product_index in instance.non_connectivity_cache and site_index1 in \
                    instance.non_connectivity_cache[product_index] and parent_name is not None:
                site_is_forbidden = False
                for j_site_index in instance.non_connectivity_cache[product_index][site_index1]:
                    if j_site_index == parent_site_assignments[0] or (
                            not product_is_immovable[product_name] and j_site_index == parent_site_assignments[1]):
                        site_is_forbidden = True
                        break
            else:
                site_is_forbidden = False
            if site_is_forbidden:
                # print(product_name, "site 0", site_index, "to", parent_site_assignments, "is forbidden")
                continue

            # iterate options (option for secondary source)
            for _, row2 in instance.df_solution_space.query(f"product_index=={product_index}").iterrows():
                site_name2 = row2["site"]
                region_name2 = row2["region"]
                if site_name1 == site_name2:
                    continue
                if product_exclusive_regions[product_name] and region_name1 == region_name2:
                    continue

                # check connectivity
                site_index2 = site_name_to_index[site_name2]
                if not instance.is_fully_connected and product_index in instance.non_connectivity_cache and site_index2 in \
                        instance.non_connectivity_cache[product_index] and parent_name is not None:
                    site_is_forbidden = False
                    for j_site_index in instance.non_connectivity_cache[product_index][site_index2]:
                        if j_site_index == parent_site_assignments[1] or (
                                not product_is_immovable[product_name] and j_site_index == parent_site_assignments[0]):
                            site_is_forbidden = True
                            break
                else:
                    site_is_forbidden = False
                if site_is_forbidden:
                    # print(product_name, "site 1", site_index,"to", parent_site_assignments, "is forbidden")
                    continue

                # feasible option: (row1, row2)
                option1 = row1  # primary source
                option2 = row2  # secondary source

                # explore option
                var_name1 = instance.variable_indices_to_name(product_index, 0, option1['site_index'],
                                                              option1['supplier_index'])
                var_name2 = instance.variable_indices_to_name(product_index, 1, option2['site_index'],
                                                              option2['supplier_index'])
                x_test_solution[var_name1] = 1
                x_test_solution[var_name2] = 1
                try:
                    ancilla_solution = find_correct_ancillas_for_solution(instance, x_test_solution)
                except:
                    continue
                x_test_solution.update(ancilla_solution)
                obj = instance.model.value(x_test_solution)
                if obj < best_obj:
                    best_obj = obj
                    best_x = x_test_solution.copy()
                    if rng.uniform(0, 1) < impatience:
                        return best_x  # early exit!
                x_test_solution[var_name1] = 0
                x_test_solution[var_name2] = 0

        return best_x

    for _ in tqdm(range(num_iterations), total=num_iterations, desc="refine", disable=not verbose):
        product_index = rng.choice(products_with_potential)
        best_x = find_best_options_for_product(product_index, x)
        x.update(best_x)
        assert instance.model.is_solution_valid(x)

    return x


def evaluate_solution(model, instance, x_solution, allow_infasible_solutions=False):
    # evaluate objecives of a solution based on the original data
    if not allow_infasible_solutions:
        assert instance.model.is_solution_valid(x_solution)

    # load
    df_cost_matrix, df_solution_space, df_paths = model.data_processor.evaluate(instance.weights[:3])

    # convert solution vector to assignments
    assignments = {}
    for name, value in list(x_solution.items()) + list(instance.constants.items()):
        if '__a' in name:
            continue  # ignore ancilla variables
        if value == 1:
            product_index, double_source_index, site_index, supplier_index = instance.variable_name_to_indices(name)
            if product_index not in assignments:
                assignments[product_index] = {0: None, 1: None}
            assignments[product_index][double_source_index] = (site_index, supplier_index)
    assignments = dict(sorted([item for item in assignments.items()], key=lambda item: item[0]))

    # total objective
    obj = 0

    # co2, eur, time
    co2, eur, time = 0, 0, 0
    supplier_ws = {supplier_index: 0 for supplier_index in set(instance.df_solution_space["supplier_index"].tolist())}
    for product_index, product_assignment_dict in assignments.items():
        for double_source_index, (site_index, supplier_index) in product_assignment_dict.items():
            a = A(double_source_index, instance)
            rel_val = get_rel_value(model.data_processor.df_products, product_index)

            # workshare contribution
            supplier_ws[supplier_index] += a * rel_val

            # transportation costs for product
            if product_index in model.phi_inv:

                # parent
                parent_index = model.phi_inv[product_index]
                parent_product_assignment_dict = assignments[parent_index]
                for parent_double_source_index, (
                        parent_site_index, parent_supplier_index) in parent_product_assignment_dict.items():
                    if site_index == parent_site_index:
                        # same-site transportation is free
                        pass
                    else:
                        df_costs = df_cost_matrix.query(
                            f"product_index=={product_index}&start_site_index=={site_index}&end_site_index=={parent_site_index}")
                        assert len(df_costs) <= 1
                        if len(df_costs) == 1:
                            df_cost = df_costs.iloc[0]
                            co2 += a * df_cost["co2"]
                            eur += a * df_cost["eur"]
                            time += a * df_cost["time"]
                            obj += a * df_cost["scalarized_cost"]
                            assert abs(df_cost["scalarized_cost"] - (
                                    instance.weights[0] * df_cost["co2"] + instance.weights[1] * df_cost["eur"] +
                                    instance.weights[2] * df_cost["time"])) < 1e-9
                        else:
                            pass  # no connectivity, ignore costs (in a feasible solution this can only come from an immovable product)


            else:
                # no parent
                pass

    # evaluate ws
    ws = 0
    for supplier_index, supplier_ws in supplier_ws.items():
        supplier_target_ws = model.data_processor.df_suppliers.query(f"index=={supplier_index}").iloc[0][
            "targetWorkshare_supplier"]
        ws += (supplier_ws - supplier_target_ws) ** 2
    ws *= model.ws_obj_factor  # !
    obj += instance.weights[3] * ws

    # rescaling
    if instance.objective_normalization_limits_dict is not None:
        min_value, max_value = instance.objective_normalization_limits_dict[0]
        co2 = (co2 - min_value) / (max_value - min_value)
        min_value, max_value = instance.objective_normalization_limits_dict[1]
        eur = (eur - min_value) / (max_value - min_value)
        min_value, max_value = instance.objective_normalization_limits_dict[2]
        time = (time - min_value) / (max_value - min_value)
        min_value, max_value = instance.objective_normalization_limits_dict[3]
        ws = (ws - min_value) / (max_value - min_value)
        obj = model.rescale_total_objective(instance.weights, instance.objective_normalization_limits_dict, obj)

    # sanity check
    if not allow_infasible_solutions:
        error = abs(instance.model.value(x_solution) - obj)
        assert error < 1e-6, error

    # return objectives
    return (co2, eur, time, ws), obj


def save_result(file_path, model, instance, x, suppl_data=None):
    try:
        (co2, eur, time, ws), obj = evaluate_solution(model, instance, x)
    except:
        co2, eur, time, ws, obj = None, None, None, None, None
    metadata = model.data_processor.metadata.copy()
    metadata.update(dict(CON_INFO=instance.con_info,
                         WS_OBJ_FACTOR=model.ws_obj_factor, MODEL_VERSION=model.model_version,
                         TIMESTAMP=datetime.now().timestamp()))
    result_dict = dict(eval=dict(co2=co2, eur=eur, time=time, ws=ws, obj=obj),
                       settings=dict(weights=instance.weights, alpha_pq=instance.alpha_pq,
                                     lambda_values=instance.lambda_values,
                                     enable_box_constraint_lambda_rescaling=instance.enable_box_constraint_lambda_rescaling,
                                     objective_normalization_limits_dict=instance.objective_normalization_limits_dict),
                       x=x,
                       metadata=metadata,
                       suppl_data=suppl_data
                       )
    with open(file_path, 'w') as fh:
        json.dump(result_dict, fh, default=str)
    return result_dict

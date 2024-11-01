import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

CACHE_PATH = "cache"
DATA_PATH = "data"
INTERMEDIATE_SITES_MUST_BE_WAREHOUSES = True
CO2_SCALE = 1
EUR_SCALE = 1
TIME_SCALE = 1
PROCESS_VERSION = "FINAL"


def gather_path_data(df_routes, route_indices):
    route_names = []
    start_site_name, end_site_name = None, None
    for idx, route_idx in enumerate(route_indices):
        df_route = df_routes.query(f"index=={route_idx}")
        assert len(df_route) == 1
        route = df_route.iloc[0]
        route_name = route["name"]
        route_names.append(route_name)
        if idx == 0:
            start_site_name = route['start']
        if idx == len(route_indices) - 1:
            end_site_name = route['end']
    path_data = {'route_names': route_names, 'start_site': start_site_name, 'end_site': end_site_name}
    return path_data


def get_path_kpis(df_routes, route_indices, pathfinding_data_for_product):
    co2, eur, time = 0, 0, 0
    for idx, route_idx in enumerate(route_indices):
        df_route = df_routes.query(f"index=={route_idx}")
        assert len(df_route) == 1
        route = df_route.iloc[0]
        start_site_name = route['start']
        end_site_name = route['end']
        key = (start_site_name, end_site_name)
        co2_route, eur_route, time_route = pathfinding_data_for_product[key][route_idx]
        co2 += co2_route
        eur += eur_route
        time += time_route
    return co2, eur, time


class DataProcessor():

    def __init__(self, data_path=None, cache_path=None, verbose=True):
        self.intermediate_sites_must_be_warehouses = INTERMEDIATE_SITES_MUST_BE_WAREHOUSES
        assert self.intermediate_sites_must_be_warehouses  # we agreed on this option
        self.kpi_names = ["co2", "eur", "time"]
        if data_path is None:
            data_path = DATA_PATH
        self.data_path = data_path
        if cache_path is None:
            cache_path = CACHE_PATH
        self.cache_path = cache_path
        self.verbose = verbose
        self._initialize_or_load_cached()

    @property
    def number_of_kpis(self):
        return len(self.kpi_names)

    def _initialize_or_load_cached(self):
        # load data
        self.df_products = pd.read_csv(os.path.join(self.data_path, 'data-products.csv'))
        self.df_production_sites = pd.read_csv(os.path.join(self.data_path, 'data-production-sites.csv'))
        self.df_warehouses = pd.read_csv(os.path.join(self.data_path, 'data-warehouses.csv'))
        self.df_suppliers = pd.read_csv(os.path.join(self.data_path, 'data-suppliers.csv'))
        self.df_regions = pd.read_csv(os.path.join(self.data_path, 'data-regions.csv'))
        self.df_routes = pd.read_csv(os.path.join(self.data_path, 'data-routes.csv'))
        self.df_cargotypes = pd.read_csv(os.path.join(self.data_path, 'data-cargotypes.csv'))

        # objective scales
        self.co2_obj_factor = CO2_SCALE
        self.eur_obj_factor = EUR_SCALE
        self.time_obj_factor = TIME_SCALE / (
                (np.max(self.df_products["level"].tolist()) - np.min(self.df_products["level"].tolist())) / 2)

        # meta data
        try:
            with open(os.path.join(self.data_path, 'meta.json'), 'r') as fh:
                self.metadata = json.load(fh)
        except:
            self.metadata = dict()
        self.metadata.update(
            dict(CO2_OBJ_FACTOR=self.co2_obj_factor, EUR_OBJ_FACTOR=self.eur_obj_factor,
                 TIME_OBJ_FACTOR=self.time_obj_factor,
                 INTERMEDIATE_SITES_MUST_BE_WAREHOUSES=INTERMEDIATE_SITES_MUST_BE_WAREHOUSES,
                 PROCESS_VERSION=PROCESS_VERSION))

        # cache dir
        os.makedirs(self.cache_path, exist_ok=True)

        # kpi calculation method
        def calculate_kpi(value_per_distance, distance):
            return value_per_distance * distance

        def calculate_kpi_per_volume_fraction(value_per_distance, distance, product_volume, total_cargo_volume):
            # product_volume is optional here and could also be factored in after pathfinding
            if np.isnan(total_cargo_volume):
                factor = 1
            else:
                factor = product_volume / total_cargo_volume
                assert factor > 0 and factor <= 1
            cost = calculate_kpi(value_per_distance, distance) * factor
            return float(cost)

        def calculate_kpi_with_priority_factor(value_per_distance, distance, product_level):
            # priority is optional here and could also be factored in after pathfinding
            priority = product_level  # >= 0 (but level 0 is never transported)
            cost = calculate_kpi(value_per_distance, distance) * priority
            return float(cost)

        # for each product, gather allowed production sites (for allowed start sites)
        file_path = os.path.join(self.cache_path, 'allowed_production_site_per_product.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as fh:
                allowed_production_site_per_product = json.load(fh)
        else:
            allowed_production_site_per_product = {}
            for _, product_row in self.df_products.iterrows():
                product_name = product_row["name"]
                allowed_production_site_per_product[product_name] = []
                for _, site_row in self.df_production_sites.iterrows():
                    site_name = site_row["name"]
                    supplier1_products = site_row["supplier1_products"].split(";")
                    supplier2_products = site_row["supplier2_products"]
                    if not pd.isna(supplier2_products):
                        supplier2_products = supplier2_products.split(";")
                    else:
                        supplier2_products = []
                    if product_name in supplier1_products + supplier2_products:
                        allowed_production_site_per_product[product_name].append(site_name)
                        allowed_production_site_per_product[product_name] = sorted(
                            allowed_production_site_per_product[product_name])
            with open(file_path, 'w') as fh:
                json.dump(allowed_production_site_per_product, fh)

        # for each product, gather site suppliers
        file_path = os.path.join(self.cache_path, 'allowed_production_site_suppliers_per_product.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as fh:
                allowed_production_site_suppliers_per_product = json.load(fh)
        else:
            allowed_production_site_suppliers_per_product = {}  # (product name) -> (site name) -> [list of suppliers]
            for _, product_row in self.df_products.iterrows():
                product_name = product_row["name"]
                allowed_production_site_suppliers_per_product[product_name] = {}
                for _, site_row in self.df_production_sites.iterrows():
                    site_name = site_row["name"]
                    supplier1_products = site_row["supplier1_products"].split(";")
                    supplier2_products = site_row["supplier2_products"]
                    if not pd.isna(supplier2_products):
                        supplier2_products = supplier2_products.split(";")
                    else:
                        supplier2_products = []
                    if product_name in supplier1_products + supplier2_products:
                        allowed_production_site_suppliers_per_product[product_name][site_name] = []
                        if product_name in supplier1_products:
                            supplier1 = site_row["supplier1"]
                            allowed_production_site_suppliers_per_product[product_name][site_name].append(supplier1)
                        if product_name in supplier2_products:
                            supplier2 = site_row["supplier2"]
                            allowed_production_site_suppliers_per_product[product_name][site_name].append(supplier2)
                        allowed_production_site_suppliers_per_product[product_name][site_name] = sorted(
                            list(set(allowed_production_site_suppliers_per_product[product_name][site_name])))
            with open(file_path, 'w') as fh:
                json.dump(allowed_production_site_suppliers_per_product, fh)

        # for each product, gather allowed cargotypes (for allowed routes)
        file_path = os.path.join(self.cache_path, 'allowed_cargotypes_per_product.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as fh:
                allowed_cargotypes_per_product = json.load(fh)
        else:
            allowed_cargotypes_per_product = {}
            for _, product_row in self.df_products.iterrows():
                product_name = product_row["name"]
                allowed_cargotypes_per_product[product_name] = []
                for _, cargotype_row in self.df_cargotypes.iterrows():
                    cargotype_name = cargotype_row["name"]
                    cargotype_products = cargotype_row["products"].split(";")
                    if product_name in cargotype_products:
                        allowed_cargotypes_per_product[product_name].append(cargotype_name)
                        allowed_cargotypes_per_product[product_name] = sorted(
                            allowed_cargotypes_per_product[product_name])
                with open(file_path, 'w') as fh:
                    json.dump(allowed_cargotypes_per_product, fh)

        # for each product, gather pathfinding data
        file_path = os.path.join(self.cache_path, 'pathfinding_data.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as fh:
                json_dict = json.load(fh)
                pathfinding_data = {
                    key: {tuple([k.strip()[1:-1] for k in key2_str[1:-1].split(",")]): value2 for key2_str, value2 in
                          value.items()} for key, value in json_dict.items()}
        else:
            pathfinding_data = {}
            kpis_list = []
            for product_idx, product_row in tqdm(self.df_products.iterrows(), total=len(self.df_products),
                                                 desc="preprocess pathfinding data", disable=not self.verbose):
                product_name = product_row["name"]
                product_length = product_row['length']
                product_width = product_row['width']
                product_height = product_row['height']
                product_volume = product_length * product_width * product_height

                product_level = product_row['level']

                pathfinding_data_for_product = {}
                for route_idx, route_row in self.df_routes.iterrows():

                    # cargo types
                    cargo_type1 = route_row["cargo_type1"]
                    cargo_type2 = route_row["cargo_type2"]

                    if cargo_type1 in allowed_cargotypes_per_product[product_name] or cargo_type2 in \
                            allowed_cargotypes_per_product[
                                product_name]:

                        # cargo properties
                        if not pd.isna(cargo_type1):
                            cargo1 = self.df_cargotypes.query(f"name=='{cargo_type1}'").iloc[0]
                            cargo_type1_length = cargo1['length']
                            cargo_type1_width = cargo1['width']
                            cargo_type1_height = cargo1['height']
                            cargo_type1_count = cargo1['count']
                            cargo_type1_volume = cargo_type1_length * cargo_type1_width * cargo_type1_height * cargo_type1_count
                        else:
                            cargo_type1_volume = np.nan
                        if not pd.isna(cargo_type2):
                            cargo2 = self.df_cargotypes.query(f"name=='{cargo_type2}'").iloc[0]
                            cargo_type2_length = cargo2['length']
                            cargo_type2_width = cargo2['width']
                            cargo_type2_height = cargo2['height']
                            cargo_type2_count = cargo2['count']
                            cargo_type2_volume = cargo_type2_length * cargo_type2_width * cargo_type2_height * cargo_type2_count
                        else:
                            cargo_type2_volume = np.nan
                        if not np.isnan(cargo_type1_volume) and not np.isnan(cargo_type2_volume):
                            total_cargo_volume = cargo_type1_volume + cargo_type2_volume
                        elif not np.isnan(cargo_type1_volume) and np.isnan(cargo_type2_volume):
                            total_cargo_volume = cargo_type1_volume
                        elif np.isnan(cargo_type1_volume) and not np.isnan(cargo_type2_volume):
                            total_cargo_volume = cargo_type2_volume
                        else:
                            total_cargo_volume = np.nan

                        # locations
                        start_site_name = route_row["start"]
                        end_site_name = route_row["end"]

                        # KPIs
                        co2_per_distance = route_row["co2_per_distance"]
                        eur_per_distance = route_row["eur_per_distance"]
                        time_per_distance = route_row["time_per_distance"]
                        distance = route_row["distance"]
                        co2 = calculate_kpi_per_volume_fraction(co2_per_distance, distance, product_volume,
                                                                total_cargo_volume)
                        eur = calculate_kpi_per_volume_fraction(eur_per_distance, distance, product_volume,
                                                                total_cargo_volume)
                        time = calculate_kpi_with_priority_factor(time_per_distance, distance, product_level)
                        kpis = [co2, eur, time]  # KPI order: emission, money, time
                        kpis_list.append(kpis)

                        # add route
                        key = (start_site_name, end_site_name)
                        if key not in pathfinding_data_for_product:
                            pathfinding_data_for_product[key] = {}
                        pathfinding_data_for_product[key][
                            route_idx] = kpis  # (start, end) -> (route_idx) -> (co2, eur, time) = min!

                pathfinding_data[product_name] = pathfinding_data_for_product

            # rescale: transform all kpis to [minimum>0, 1] and multiply global factor
            kpis_list = np.array(kpis_list)
            objective_factors = [self.co2_obj_factor, self.eur_obj_factor, self.time_obj_factor]
            max_values = [float(np.max(kpis_list[:, n])) for n in range(self.number_of_kpis)]
            for product_name in pathfinding_data.keys():
                for key in pathfinding_data[product_name].keys():
                    for route_idx in pathfinding_data[product_name][key].keys():
                        kpis = pathfinding_data[product_name][key][route_idx]
                        rescaled_kpis = [kpis[n] / max_values[n] * objective_factors[n] for n in
                                         range(self.number_of_kpis)]  # linearly rescaled, no shift
                        pathfinding_data[product_name][key][route_idx] = rescaled_kpis
            print(f"rescaling pathfinding_data with max_values: {max_values}")

            with open(file_path, 'w') as fh:
                json_dict = {key: {str(key2): value2 for key2, value2 in value.items()} for key, value in
                             pathfinding_data.items()}
                json.dump(json_dict, fh, default=str)

        # assign pre-calculated
        self.allowed_production_site_per_product = allowed_production_site_per_product
        self.allowed_production_site_suppliers_per_product = allowed_production_site_suppliers_per_product
        self.allowed_cargotypes_per_product = allowed_cargotypes_per_product
        self.pathfinding_data = pathfinding_data

        # evaluate product parents
        self.product_parents = {}
        for _, product_row in self.df_products.iterrows():
            product_name = product_row['name']
            parent_name = product_row['parent']
            if not pd.isna(parent_name):
                self.product_parents[product_name] = parent_name
            else:
                self.product_parents[product_name] = None

    def _pathfinder(self, weights):
        # list of production site names and warehouse names (for allowed routes)
        production_sites = self.df_production_sites["name"].tolist()
        warehouses = self.df_warehouses["name"].tolist()

        def generate_scalarized_cost_graph_for_path(product_name, path_start_site_name, path_end_site_name, weights,
                                                    intermediate_sites_must_be_warehouses):
            G = nx.DiGraph()
            G.add_nodes_from(production_sites)
            G.add_nodes_from(warehouses)
            for (start_site_name, end_site_name), route_dict in self.pathfinding_data[product_name].items():
                # check if intermediate site is a warehouse
                if (not intermediate_sites_must_be_warehouses) or (end_site_name in warehouses + [path_end_site_name]):
                    # take best of all valid connections
                    costs = [sum([weight * kpi for weight, kpi in zip(weights, kpis)]) for kpis in
                             route_dict.values()]
                    route_indices = list(route_dict.keys())
                    best_cost_idx = np.argmin(costs)
                    G.add_edge(start_site_name, end_site_name, cost=costs[best_cost_idx],
                               route_idx=route_indices[best_cost_idx])
            return G

        def get_best_path(product_name, path_start_site_name, path_end_site_name, weights,
                          intermediate_sites_must_be_warehouses):

            # check if same location
            if path_start_site_name == path_end_site_name:
                return 0, []

            # build graph for product with specific start and end site (to determine intermediate nodes)
            G = generate_scalarized_cost_graph_for_path(product_name, path_start_site_name, path_end_site_name, weights,
                                                        intermediate_sites_must_be_warehouses)

            # get best path with graph
            try:
                path = nx.shortest_path(G, source=path_start_site_name, target=path_end_site_name, weight='cost')
            except nx.NetworkXNoPath:
                return None, None
            total_cost = 0
            for site_from, site_to in zip(path[::], path[1::]):
                total_cost += G.edges[site_from, site_to]['cost']
            route_indices = []
            for site_from, site_to in zip(path[::], path[1::]):
                route_indices.append(G.edges[site_from, site_to]['route_idx'])
            return total_cost, route_indices

        def get_effective_paths_for_scalarized_kpis(weights, intermediate_sites_must_be_warehouses):
            effective_paths_data = {'product': [], 'start': [], 'end': []}
            for kpi_name in self.kpi_names:
                effective_paths_data[kpi_name] = []
            effective_paths_data['scalarized_cost'] = []
            for kpi_name in self.kpi_names:
                effective_paths_data[f'weight_{kpi_name}'] = []
            effective_paths_data['route_indices'] = []

            for product_idx, product_row in tqdm(self.df_products.iterrows(), total=len(self.df_products),
                                                 desc='scalarized best routes', disable=not self.verbose):
                product_name = product_row["name"]
                parent_product_name = self.product_parents[product_name]

                for start_site_idx, start_site_row in self.df_production_sites.iterrows():
                    start_site_name = start_site_row["name"]

                    if start_site_name in self.allowed_production_site_per_product[product_name]:

                        for end_site_idx, end_site_row in self.df_production_sites.iterrows():
                            end_site_name = end_site_row["name"]

                            if parent_product_name is not None and end_site_name in \
                                    self.allowed_production_site_per_product[parent_product_name]:
                                # only consider routes with a target destination that the parent can be assigned to

                                # best route from start_site_name to end_site_name for product_name
                                total_cost, route_indices = get_best_path(product_name, start_site_name, end_site_name,
                                                                          weights,
                                                                          intermediate_sites_must_be_warehouses)
                                if total_cost is not None:
                                    effective_paths_data['product'].append(product_name)
                                    effective_paths_data['start'].append(start_site_name)
                                    effective_paths_data['end'].append(end_site_name)
                                    kpis = get_path_kpis(self.df_routes, route_indices,
                                                         self.pathfinding_data[product_name])
                                    for kpi_name, kpi in zip(self.kpi_names, kpis):
                                        effective_paths_data[kpi_name].append(float(kpi))
                                    for weight, kpi_name in zip(weights, self.kpi_names):
                                        effective_paths_data[f'weight_{kpi_name}'].append(weight)  # constant
                                    effective_paths_data['scalarized_cost'].append(total_cost)
                                    effective_paths_data['route_indices'].append(
                                        ";".join([str(route_idx) for route_idx in route_indices]))
            return effective_paths_data  # includes self-connectivity

        # evaluate
        effective_paths_data = get_effective_paths_for_scalarized_kpis(weights,
                                                                       self.intermediate_sites_must_be_warehouses)
        df_paths = pd.DataFrame(effective_paths_data)
        df_paths = df_paths.sort_values(["product", "start", "end"])
        df_paths = df_paths.reset_index(drop=True)
        df_paths.index.name = 'index'

        return df_paths

    def _get_feasible_production_site_per_product(self, df_paths):

        # for each product, gather feasible sites (sites with an actual connection)
        feasible_production_site_per_product = {key: list(value.keys()) for key, value in
                                                self.allowed_production_site_suppliers_per_product.items()}
        requires_feasibility_update = {product_name: True for product_name in self.df_products["name"]}

        def get_parts(product_name):
            parts = self.df_products.query(f"name=='{product_name}'").iloc[0]["parts"]
            if pd.isna(parts):
                parts = []
            else:
                parts = parts.split(";")
            return parts

        def determine_feasible_sites_parts(product_name, part_name):
            product_sites = feasible_production_site_per_product[product_name]
            part_sites = feasible_production_site_per_product[part_name]
            joint_manufacure_sites = set(product_sites).intersection(part_sites)
            reachable_sites = set([product_site for product_site in product_sites if
                                   len(df_paths.query(f'product=="{part_name}"&end=="{product_site}"')) > 0])
            feasible_sites = set.union(joint_manufacure_sites, reachable_sites)
            return feasible_sites

        def determine_feasible_sites_parent(product_name, part_name):
            product_sites = feasible_production_site_per_product[product_name]
            part_sites = feasible_production_site_per_product[part_name]
            joint_manufacure_sites = set(product_sites).intersection(part_sites)
            if len(product_sites) > 0:
                end_condition = "(" + "|".join([f'end=="{product_site}"' for product_site in product_sites]) + ")"
                reachable_sites = set(df_paths.query(f'product=="{part_name}"&{end_condition}')["start"].tolist())
                feasible_sites = set.union(joint_manufacure_sites, reachable_sites)
            else:
                feasible_sites = []
            return feasible_sites

        def determine_joint_feasible_sites(product_name):
            parts = get_parts(product_name)
            if len(parts) == 0:
                joint_feasible_sites = feasible_production_site_per_product[product_name]
            else:
                feasible_sites_list = []
                for part_name in parts:
                    feasible_sites_list.append(determine_feasible_sites_parts(product_name, part_name))
                    # DEBUG
                    # print(product_name, "feasibility")
                    # print(' -> ', part_name, feasible_sites_list[-1])
                joint_feasible_sites = set.intersection(*feasible_sites_list)
                # DEBUG
                # print(" ==> ", joint_feasible_sites)
            return list(joint_feasible_sites)

        def set_requires_feasibility_update(product_name, value=True):
            requires_feasibility_update[product_name] = value
            for part_name in get_parts(product_name):
                set_requires_feasibility_update(part_name, value)

        def update_feasibility(product_name):
            joint_feasible_sites = determine_joint_feasible_sites(product_name)
            feasible_production_site_per_product[product_name] = sorted(joint_feasible_sites)
            parts = get_parts(product_name)
            for part_name in parts:
                new_feasible_sites = sorted(list(determine_feasible_sites_parent(product_name, part_name)))
                if feasible_production_site_per_product[part_name] != new_feasible_sites:
                    # DEBUG
                    # print(part_name, "feasibility reduced by", product_name, ', removed sites:', [site for site in feasible_production_site_per_product[part_name] if site not in new_feasible_sites])
                    feasible_production_site_per_product[part_name] = new_feasible_sites
                    set_requires_feasibility_update(part_name)
            requires_feasibility_update[product_name] = False

        max_level = max(self.df_products["level"])
        max_iters = 100
        it = 0
        while (any(list(requires_feasibility_update.values())) and it < max_iters):
            updated = False
            for level in range(0, max_level + 1):
                products_on_level = self.df_products.query(f"level=={level}")["name"].tolist()
                for product_name in products_on_level:
                    if requires_feasibility_update[product_name]:
                        update_feasibility(product_name)
                        updated = True
                if updated:
                    break
            it += 1

        feasible_production_site_per_product = dict(
            sorted([(key, value) for key, value in feasible_production_site_per_product.items()],
                   key=lambda item: item[0]))  # sort dict
        assert all([key in feasible_production_site_per_product for key in
                    self.allowed_production_site_suppliers_per_product.keys()])  # sanity check
        assert all([len(value) > 0 for value in
                    feasible_production_site_per_product.values()])  # check: at least one feasible site per product

        return feasible_production_site_per_product

    def _get_cost_matrix(self, df_paths, feasible_production_site_per_product):

        # index maps
        product_name_to_index_map = {row['name']: row['index'] for _, row in self.df_products.iterrows()}
        site_name_to_index_map = {row['name']: row['index'] for _, row in self.df_production_sites.iterrows()}
        products = list(product_name_to_index_map.keys())

        # cost matrix
        cost_matrix_data = {'product': [], 'product_index': [],
                            'start_site': [], 'start_site_index': [],
                            'end_site': [], 'end_site_index': [],
                            'co2': [], 'eur': [], 'time': [], 'scalarized_cost': [],
                            'route_indices': []}

        for product_name in tqdm(products, desc='cost matrix', disable=not self.verbose):
            parent = self.product_parents[product_name]
            if parent is None:
                continue
            for start_site in feasible_production_site_per_product[product_name]:
                for end_site in feasible_production_site_per_product[parent]:
                    # if start_site == end_site:
                    #    continue  # no self-connectivity
                    df_path = df_paths.query(f'product=="{product_name}"&start=="{start_site}"&end=="{end_site}"')
                    if len(df_path) == 0:
                        continue  # no available path between start site and end site for the product
                    assert len(df_path) == 1
                    path_data = df_path.iloc[0]
                    co2 = path_data.get('co2', None)
                    eur = path_data.get('eur', None)
                    time = path_data.get('time', None)
                    scalarized_cost = path_data.get('scalarized_cost', None)
                    cost_matrix_data['product'].append(product_name)
                    cost_matrix_data['start_site'].append(start_site)
                    cost_matrix_data['end_site'].append(end_site)
                    product_idx = product_name_to_index_map[product_name]
                    start_site_idx = site_name_to_index_map[start_site]
                    end_site_idx = site_name_to_index_map[end_site]
                    cost_matrix_data['product_index'].append(product_idx)
                    cost_matrix_data['start_site_index'].append(start_site_idx)
                    cost_matrix_data['end_site_index'].append(end_site_idx)
                    cost_matrix_data['co2'].append(
                        co2)  # rescaled kpi 1: rescaled with a fractor (no shift) to [min>0, 1]
                    cost_matrix_data['eur'].append(
                        eur)  # rescaled kpi 2: rescaled with a fractor (no shift) to [min>0, 1]
                    cost_matrix_data['time'].append(
                        time)  # rescaled kpi 3: rescaled with a fractor (no shift) to [min>0, 1]
                    cost_matrix_data['scalarized_cost'].append(scalarized_cost)  # rescaled kpis * weight
                    route_indices = path_data['route_indices']
                    cost_matrix_data['route_indices'].append(route_indices)
        df_cost_matrix = pd.DataFrame(cost_matrix_data)
        df_cost_matrix = df_cost_matrix.sort_values(["product", "start_site", "end_site"])
        df_cost_matrix = df_cost_matrix.reset_index(drop=True)
        df_cost_matrix.index.name = 'index'

        return df_cost_matrix  # includes self-connectivity

    def _get_solution_space(self, feasible_production_site_per_product):

        # maps
        product_name_to_index_map = {row['name']: row['index'] for _, row in self.df_products.iterrows()}
        site_name_to_index_map = {row['name']: row['index'] for _, row in self.df_production_sites.iterrows()}
        supplier_name_to_index_map = {row['name']: row['index'] for _, row in self.df_suppliers.iterrows()}
        country_name_to_index_map = {row['name']: row['index'] for _, row in self.df_regions.iterrows()}
        product_names = list(product_name_to_index_map.keys())

        # solution space
        solution_space_data = {'product': [], 'product_index': [],
                               'site': [], 'site_index': [],
                               'region': [], 'region_index': [],
                               'supplier': [], 'supplier_index': []}
        for product_name in tqdm(product_names, desc='solution space', disable=not self.verbose):
            for production_site in feasible_production_site_per_product[product_name]:
                site_country = self.df_production_sites.query(f'name=="{production_site}"')["country"].iloc[0]
                for supplier in self.allowed_production_site_suppliers_per_product[product_name][production_site]:
                    solution_space_data['product'].append(product_name)
                    solution_space_data['site'].append(production_site)
                    solution_space_data['region'].append(site_country)
                    solution_space_data['supplier'].append(supplier)
                    product_idx = product_name_to_index_map[product_name]
                    site_idx = site_name_to_index_map[production_site]
                    region_idx = country_name_to_index_map[site_country]
                    supplier_idx = supplier_name_to_index_map[supplier]
                    solution_space_data['product_index'].append(product_idx)
                    solution_space_data['site_index'].append(site_idx)
                    solution_space_data['region_index'].append(region_idx)
                    solution_space_data['supplier_index'].append(supplier_idx)
        df_solution_space = pd.DataFrame(solution_space_data)
        df_solution_space = df_solution_space.sort_values(["product", "site", "supplier"])
        df_solution_space = df_solution_space.reset_index(drop=True)
        df_solution_space.index.name = 'index'

        return df_solution_space

    def evaluate(self, weights):
        assert len(weights) == self.number_of_kpis
        cost_file_path = os.path.join(self.cache_path,
                                      f'costmatrix-{";".join([f"{weight:.9f}" for weight in weights])}.csv')
        solutionspace_file_path = os.path.join(self.cache_path,
                                               f'solutionspace-{";".join([f"{weight:.9f}" for weight in weights])}.csv')
        paths_file_path = os.path.join(self.cache_path,
                                       f'paths-{";".join([f"{weight:.9f}" for weight in weights])}.csv')

        if not os.path.exists(paths_file_path):
            df_paths = self._pathfinder(weights)
            df_paths.to_csv(paths_file_path)
        df_paths = pd.read_csv(paths_file_path)
        feasible_production_site_per_product = None

        if not os.path.exists(solutionspace_file_path):
            if feasible_production_site_per_product is None:
                feasible_production_site_per_product = self._get_feasible_production_site_per_product(df_paths)
            df_solution_space = self._get_solution_space(feasible_production_site_per_product)
            df_solution_space.to_csv(solutionspace_file_path)
        df_solution_space = pd.read_csv(solutionspace_file_path)

        if not os.path.exists(cost_file_path):
            if feasible_production_site_per_product is None:
                feasible_production_site_per_product = self._get_feasible_production_site_per_product(df_paths)
            df_cost_matrix = self._get_cost_matrix(df_paths, feasible_production_site_per_product)
            df_cost_matrix.to_csv(cost_file_path)
        df_cost_matrix = pd.read_csv(cost_file_path)

        return df_cost_matrix, df_solution_space, df_paths

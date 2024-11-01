import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_PATH = '../airbus-data/24-10-21-is-academic-exchange-csv-files'
DATA_PATH = 'data'
INCLUDE_AUX = True
ROOT_PRODUCT = 'Single Aisle Aircraft'
IMMOVABLE_PRODUCTS = ['Single Aisle Aircraft', 'S123456 Full Fuselage']
PRODUCT_REL_VALUE_FRACTION = 0.1
DATA_VERSION = "FINAL"


# COMMON ##########################################################################################################################

def load_csv(file, input_path='.', sort=True, show=True):
    file_path = os.path.join(input_path, file)
    print(f'load file: {file_path}')
    df = pd.read_csv(file_path)
    if show:
        print(f'           [columns: {", ".join(df.columns.tolist())} | size: {df.shape}]')
    if sort:
        if 'id' in df and 'name' in df:
            df = df.sort_values(['id', 'name'])
        elif 'id' in df:
            df = df.sort_values(['id'])
        elif 'name' in df:
            df = df.sort_values(['name'])
        df = df.reset_index(drop=True)
    return df


def save_csv(df, name, output_path='.', show=True, makedir=True, sort_column="name"):
    if makedir:
        os.makedirs(output_path, exist_ok=True)
    file_name = f'{name}.csv'
    file_path = os.path.join(output_path, file_name)
    if sort_column is not None:
        df = df.sort_values(sort_column)
        df = df.reset_index(drop=True)
    df.index.name = 'index'
    df.to_csv(file_path)
    print(f'save file: {file_path}')
    if show:
        print(f'           [columns: {", ".join(df.columns.tolist())} | size: {df.shape}]')
    return name


# PRODUCTS ########################################################################################################################
def convert_products(include_aux=INCLUDE_AUX):
    # load
    df_products = load_csv('products.csv', RAW_PATH, sort=True)
    df_products_to_parts = load_csv('products-to-parts.csv', RAW_PATH, sort=True)

    # cubify
    df_products.loc[pd.isna(df_products.width), "width"] = df_products.loc[pd.isna(df_products.width), "diameter"]
    df_products.loc[pd.isna(df_products.height), "height"] = df_products.loc[pd.isna(df_products.height), "diameter"]

    # aux
    if include_aux:
        df_products['#'] = ""

    # PBS
    PBS = {}
    for idx in range(len(df_products_to_parts)):
        product = df_products_to_parts.iloc[idx]
        uuid = product['id']
        name = product['name']
        part_uuid = product['part']
        if name not in PBS:
            PBS[name] = {'uuid': uuid, 'parts': []}
        if not pd.isna(part_uuid):
            product = df_products.query(f'id=="{part_uuid}"').iloc[0]
            part_name = product['name']
            PBS[name]['parts'].append(part_name)
    for key, value in PBS.items():
        parts = ";".join(sorted(value['parts']))
        df_products.loc[df_products.name == key, 'parts'] = parts
        if include_aux:
            aux_info = json.dumps({'uuid': value["uuid"]})
            df_products.loc[df_products.name == key, '#'] = aux_info

    # set parents
    product_parents = {}
    for _, product_row in df_products.iterrows():
        product_name = product_row["name"]
        for _, product2_row in df_products.iterrows():
            product2_name = product2_row["name"]
            parts = product2_row["parts"]
            if not pd.isna(parts):
                parts = parts.split(";")
            else:
                parts = []
            if product_name in parts:
                assert product_name not in product_parents  # only 1 parent
                product_parents[product_name] = product2_name
        if product_name not in product_parents:
            product_parents[product_name] = ""
    df_products['parent'] = list(product_parents.values())

    # set level
    product_levels = {name: None for name in df_products["name"]}

    def traverse_tree(product_name, level):
        product_levels[product_name] = level
        parts = PBS[product_name]["parts"]
        for part_name in parts:
            traverse_tree(part_name, level=level + 1)

    traverse_tree(ROOT_PRODUCT, level=0)
    df_products['level'] = list(product_levels.values())

    # value
    values = df_products['valueAdded'].tolist()
    total_value = sum(values)
    df_products['abs_value'] = values
    df_products['rel_value'] = [100.0 * value / total_value for value in values]
    df_products['rel_value_p'] = [int(rel_value // PRODUCT_REL_VALUE_FRACTION) for rel_value in
                                  df_products['rel_value'].tolist()]
    df_products['rel_value_q'] = [int(1 / PRODUCT_REL_VALUE_FRACTION)] * len(values)
    max_error = max([abs(val - p / q) for val, p, q in
                     zip(df_products['rel_value'].tolist(), df_products['rel_value_p'].tolist(),
                         df_products['rel_value_q'].tolist())])
    print(
        f"maximum rel_value error for the rational number transformation (p / {1 / PRODUCT_REL_VALUE_FRACTION}): {max_error}")
    assert max_error < 0.2  # 0.051

    # immovability
    df_products['is_immovable'] = [name in IMMOVABLE_PRODUCTS for name in df_products["name"]]

    # build dataframe
    df_products_data = df_products[
        ["name", "length", "width", "height", "abs_value", "rel_value", "rel_value_p", "rel_value_q", "parts", "parent",
         "level", "is_immovable"] + (
            ['#'] if include_aux else [])]

    # save
    save_csv(df_products_data, 'data-products', DATA_PATH)


# LOCATIONS AND SUPPLIERS #########################################################################################################
def convert_locations_and_suppliers(include_aux=INCLUDE_AUX):
    def get_property_by_str_key(df, key, property_name, key_name="name", return_list=False):
        properties = list(set(df.query(f'{key_name}=="{key}"')[property_name].tolist()))
        if return_list:
            return properties
        assert len(properties) == 1, (key, property_name, key_name, properties)
        return properties[0]

    def get_country(df, name):
        country_uuid = get_property_by_str_key(df, name, "country")
        return get_property_by_str_key(df_countries, country_uuid, "name", key_name="id")

    def get_continent(df, name):
        country_uuid = get_property_by_str_key(df, name, "country")
        continent_uuid = get_property_by_str_key(df_countries, country_uuid, "continent", key_name="id")
        return get_property_by_str_key(df_contintents, continent_uuid, "name", key_name="id")

    def get_countries_or_continents(df, name):
        country_or_continent_uuid_list = get_property_by_str_key(df, name, "country", return_list=True)
        output_list = []
        for uuid in country_or_continent_uuid_list:
            try:
                name = get_property_by_str_key(df_countries, uuid, "name", key_name="id")
            except:
                name = get_property_by_str_key(df_contintents, uuid, "name", key_name="id")
            output_list.append(name)
        return output_list

    # load
    df_products = load_csv('products.csv', RAW_PATH, sort=True)
    df_production_locations = load_csv('production-locations.csv', RAW_PATH, sort=True)
    df_warehouse_locations = load_csv('warehouse-locations.csv', RAW_PATH, sort=True)
    df_countries = load_csv('countries.csv', RAW_PATH, sort=True)
    df_contintents = load_csv('continents.csv', RAW_PATH, sort=True)
    df_manufacturing_resources = load_csv('manufacturing-resources.csv', RAW_PATH, sort=True)
    df_suppliers = load_csv('suppliers.csv', RAW_PATH, sort=True)
    # df_recipes = load_csv('recipes.csv', RAW_PATH, sort=True)

    # names
    prod_names = sorted(list(set(df_production_locations["name"])))
    ware_names = sorted(list(set(df_warehouse_locations["name"])))

    # production site workshares
    prod_minimumWorkshare = [get_property_by_str_key(df_production_locations, name, 'minimumWorkshare') for name in
                             prod_names]
    prod_maximumWorkshare = [get_property_by_str_key(df_production_locations, name, 'maximumWorkshare') for name in
                             prod_names]
    true_prod_minimumWorkshare, true_prod_maximumWorkshare = [], []
    for prod_name, min_ws, max_ws in zip(prod_names, prod_minimumWorkshare, prod_maximumWorkshare):
        if min_ws <= max_ws:
            true_prod_minimumWorkshare.append(min_ws)
            true_prod_maximumWorkshare.append(max_ws)
        else:
            true_prod_minimumWorkshare.append(max_ws)
            true_prod_maximumWorkshare.append(min_ws)
            print(f"corrected location '{prod_name}' workshare: switch minimum ({min_ws}) and maximum ({max_ws})")

    # regions
    prod_countries = [get_country(df_production_locations, name) for name in prod_names]
    ware_countries = [get_country(df_warehouse_locations, name) for name in ware_names]
    prod_continents = [get_continent(df_production_locations, name) for name in prod_names]
    ware_continents = [get_continent(df_warehouse_locations, name) for name in ware_names]

    # durations
    print(
        "production durations are presumed to be constant = 0 instead of extracting them from recipes.csv because of missing uuids.")
    duration_list = [0] * len(prod_names)

    # supplier data
    suppliers, products, locations = [], [], []
    for _, row in tqdm(df_manufacturing_resources.iterrows(), total=len(df_manufacturing_resources),
                       desc='manufacturing resources'):
        supplier_uuid = row['supplier']
        supplier_name = get_property_by_str_key(df_suppliers, supplier_uuid, "name", key_name="id")
        product_uuid = row['product']
        product_name = get_property_by_str_key(df_products, product_uuid, "name", key_name="id")
        location_uuid = row['location']
        location_name = get_property_by_str_key(df_production_locations, location_uuid, "name", key_name="id")
        suppliers.append(supplier_name)
        products.append(product_name)
        locations.append(location_name)
    df_supplier_product_locations = pd.DataFrame({'supplier': suppliers, 'product': products, 'location': locations})
    prod_suppliers_1, prod_suppliers_2 = [], []
    for name in prod_names:
        suppliers = get_property_by_str_key(df_supplier_product_locations, name, "supplier", key_name="location",
                                            return_list=True)
        assert len(suppliers) <= 2
        if len(suppliers) == 1:
            supplier_1 = suppliers[0]
            supplier_2 = ""
        else:
            supplier_1 = suppliers[0]
            supplier_2 = suppliers[1]
        prod_suppliers_1.append(supplier_1)
        prod_suppliers_2.append(supplier_2)

    # supplier workshares
    supplier_names = list(set(df_supplier_product_locations["supplier"]))
    suppl_minimumWorkshare = [get_property_by_str_key(df_suppliers, supplier_name, "minimumWorkshare") for supplier_name
                              in
                              supplier_names]
    suppl_targetWorkshare = [get_property_by_str_key(df_suppliers, supplier_name, "targetWorkshare") for supplier_name
                             in
                             supplier_names]
    suppl_maximumWorkshare = [get_property_by_str_key(df_suppliers, supplier_name, "maximumWorkshare") for supplier_name
                              in
                              supplier_names]
    true_suppl_minimumWorkshare, true_suppl_targetWorkshare, true_suppl_maximumWorkshare = [], [], []
    for supplier_name, min_ws, target_ws, max_ws in zip(supplier_names, suppl_minimumWorkshare, suppl_targetWorkshare,
                                                        suppl_maximumWorkshare):
        if min_ws <= max_ws:
            true_suppl_minimumWorkshare.append(min_ws)
            true_suppl_maximumWorkshare.append(max_ws)
        else:
            true_suppl_minimumWorkshare.append(max_ws)
            true_suppl_maximumWorkshare.append(min_ws)
            print(f"corrected supplier '{supplier_name}' workshare: switch minimum ({min_ws}) and maximum ({max_ws})")
        if target_ws > max_ws:
            true_suppl_targetWorkshare.append(max_ws)
            print(f"corrected supplier '{supplier_name}' workshare: target ({target_ws}) set to maximum ({max_ws})")
        elif target_ws < min_ws:
            true_suppl_targetWorkshare.append(min_ws)
            print(f"corrected supplier '{supplier_name}' workshare: target ({target_ws}) set to minimum ({min_ws})")
        else:
            true_suppl_targetWorkshare.append(target_ws)

    # manual correction of workshares
    print("manual correction of workshares to ensure feasibility")
    df_products_data = load_csv('data-products.csv', DATA_PATH)

    def get_rel_value(product_name):
        rel_value_p = df_products_data.query(f"name=='{product_name}'").iloc[0]["rel_value_p"]
        rel_value_q = df_products_data.query(f"name=='{product_name}'").iloc[0]["rel_value_q"]
        rel_value = rel_value_p / rel_value_q
        return rel_value

    # manual correction of workshare requierments to enable feasibility
    def recalculate_workshare_from_sum(product_name_list):
        return int(np.ceil(sum([get_rel_value(product_name) for product_name in product_name_list])))

    if '24-10-11-is-academic-exchange-csv-files' in RAW_PATH or '24-10-21-is-academic-exchange-csv-files' in RAW_PATH:
        for supplier_index, supplier_name in enumerate(supplier_names):
            if supplier_name == "Airbus Operations Germany":
                true_suppl_maximumWorkshare[supplier_index] = recalculate_workshare_from_sum(
                    ['Horizontal Tailplane', 'S123456 Full Fuselage', 'Single Aisle Aircraft'])
                print(
                    f"manually correct supplier {supplier_name} maximum workshare: {true_suppl_maximumWorkshare[supplier_index]}")
            if supplier_name == "Airbus Operations UK":
                true_suppl_maximumWorkshare[supplier_index] = recalculate_workshare_from_sum(
                    ['Front Landing Gear', 'Rear Landing Gear Left', 'Rear Landing Gear Right', 'Wing Left',
                     'Wing Right'])
                print(
                    f"manually correct supplier {supplier_name} maximum workshare: {true_suppl_maximumWorkshare[supplier_index]}")
        for site_index, site_name in enumerate(prod_names):
            if site_name == "Hamburg FAL":
                true_prod_maximumWorkshare[site_index] = recalculate_workshare_from_sum(
                    ['S123456 Full Fuselage', 'Single Aisle Aircraft'])
                print(
                    f"manually correct production site {site_name} maximum workshare: {true_prod_maximumWorkshare[site_index]}")
            if site_name == "Broughton":
                true_prod_maximumWorkshare[site_index] = recalculate_workshare_from_sum(['Wing Left', 'Wing Right'])
                print(
                    f"manually correct production site {site_name} maximum workshare: {true_prod_maximumWorkshare[site_index]}")
            if site_name == "Belfast":
                true_prod_maximumWorkshare[site_index] = recalculate_workshare_from_sum(['Wing Left', 'Wing Right'])
                print(
                    f"manually correct production site {site_name} maximum workshare: {true_prod_maximumWorkshare[site_index]}")
            if site_name == "Stade":
                true_prod_minimumWorkshare[site_index] = 0
                print(
                    f"manually correct production site {site_name} minimum workshare: {true_prod_minimumWorkshare[site_index]}")
            if site_name == "Bremen":
                true_prod_minimumWorkshare[site_index] = 0
                print(
                    f"manually correct production site {site_name} minimum workshare: {true_prod_minimumWorkshare[site_index]}")
    if '24-10-21-is-academic-exchange-csv-files' in RAW_PATH:
        for site_index, site_name in enumerate(prod_names):
            if site_name == "Toulouse FAL":
                true_prod_maximumWorkshare[site_index] = recalculate_workshare_from_sum(
                    ['S123456 Full Fuselage', 'Single Aisle Aircraft'])
                print(
                    f"manually correct production site {site_name} maximum workshare: {true_prod_maximumWorkshare[site_index]}")

    # regions
    country_names = df_countries["name"].tolist()
    country_uuids = df_countries["id"].tolist()
    country_contintent_uuids = df_countries["continent"].tolist()
    country_continent_names = [get_property_by_str_key(df_contintents, continent_uuid, "name", key_name="id") for
                               continent_uuid in country_contintent_uuids]

    # feasible products per site
    feasible_products_supplier1_list = []
    feasible_products_supplier2_list = []
    for supplier1, supplier2, site_name in zip(prod_suppliers_1, prod_suppliers_2, prod_names):
        feasible_products = set(df_supplier_product_locations.query(f"location=='{site_name}'")["product"].tolist())
        feasible_products_supplier1 = set(
            df_supplier_product_locations.query(f"location=='{site_name}'&supplier=='{supplier1}'")[
                "product"].tolist())
        feasible_products_supplier2 = set(
            df_supplier_product_locations.query(f"location=='{site_name}'&supplier=='{supplier2}'")["product"].tolist())
        assert feasible_products == set.union(feasible_products_supplier1, feasible_products_supplier2)
        feasible_products_supplier1_list.append(";".join(sorted(list(feasible_products_supplier1))))
        feasible_products_supplier2_list.append(";".join(sorted(list(feasible_products_supplier2))))

    # aux infos
    if include_aux:
        prod_uuids = [get_property_by_str_key(df_production_locations, name, 'id') for name in prod_names]
        ware_uuids = [get_property_by_str_key(df_warehouse_locations, name, 'id') for name in ware_names]
        suppl_uuids = [get_property_by_str_key(df_suppliers, name, 'id') for name in supplier_names]
        prod_latitudes = [get_property_by_str_key(df_production_locations, name, 'latitude') for name in prod_names]
        ware_latitudes = [get_property_by_str_key(df_warehouse_locations, name, 'latitude') for name in ware_names]
        prod_longitudes = [get_property_by_str_key(df_production_locations, name, 'longitude') for name in prod_names]
        ware_longitudes = [get_property_by_str_key(df_warehouse_locations, name, 'longitude') for name in ware_names]
        prod_country_uuids = [get_property_by_str_key(df_production_locations, name, 'country') for name in prod_names]
        ware_country_uuids = [get_property_by_str_key(df_warehouse_locations, name, 'country') for name in ware_names]
        prod_countries_aux = [get_property_by_str_key(df_countries, country_uuid, "name", key_name="id") for
                              country_uuid in
                              prod_country_uuids]
        assert prod_countries == prod_countries_aux
        ware_countries_aux = [get_property_by_str_key(df_countries, country_uuid, "name", key_name="id") for
                              country_uuid in
                              ware_country_uuids]
        assert ware_countries == ware_countries_aux
        prod_continent_uuids = [get_property_by_str_key(df_countries, country_uuid, "continent", key_name="id") for
                                country_uuid
                                in prod_country_uuids]
        ware_continent_uuids = [get_property_by_str_key(df_countries, country_uuid, "continent", key_name="id") for
                                country_uuid
                                in ware_country_uuids]
        prod_continents_aux = [get_property_by_str_key(df_contintents, continent_uuid, "name", key_name="id") for
                               continent_uuid in
                               prod_continent_uuids]
        assert prod_continents == prod_continents_aux
        ware_continents_aux = [get_property_by_str_key(df_contintents, continent_uuid, "name", key_name="id") for
                               continent_uuid in
                               ware_continent_uuids]
        assert ware_continents == ware_continents_aux
        suppl_locations = [get_property_by_str_key(df_suppliers, name, 'location', return_list=True) for name in
                           supplier_names]
        suppl_countries = [get_countries_or_continents(df_suppliers, name) for name in supplier_names]
        prod_aux_infos = [json.dumps(
            {'uuid': uuid, 'latitude': latitude, 'longitude': longitude, 'country_name': country,
             'continent_name': continent})
            for uuid, latitude, longitude, country, continent in
            zip(prod_uuids, prod_latitudes, prod_longitudes, prod_countries, prod_continents)]
        ware_aux_infos = [json.dumps(
            {'uuid': uuid, 'latitude': latitude, 'longitude': longitude, 'country_name': country,
             'continent_name': continent})
            for uuid, latitude, longitude, country, continent in
            zip(ware_uuids, ware_latitudes, ware_longitudes, ware_countries, ware_continents)]
        suppl_aux_infos = [json.dumps({'uuid': uuid, 'location': locations, 'country_name': countries}) for
                           uuid, locations, countries in zip(suppl_uuids, suppl_locations, suppl_countries)]
        reg_aux_infos = [json.dumps(
            {'country_uuid': country_uuid, 'continent_uuid': contintent_uuid})
            for country_uuid, contintent_uuid in
            zip(country_uuids, country_contintent_uuids)]

    # build dataframes
    production_site_data = {'name': prod_names,
                            'country': prod_countries,
                            'continent': prod_continents,
                            'minimumWorkshare_site': true_prod_minimumWorkshare,
                            'maximumWorkshare_site': true_prod_maximumWorkshare,
                            'production_duration': duration_list,
                            'supplier1': prod_suppliers_1,
                            'supplier2': prod_suppliers_2,
                            'supplier1_products': feasible_products_supplier1_list,
                            'supplier2_products': feasible_products_supplier2_list}
    if include_aux:
        production_site_data.update({'#': prod_aux_infos})
    df_production_sites = pd.DataFrame(production_site_data)

    warehouses_data = {'name': ware_names,
                       'country': ware_countries,
                       'continent': ware_continents}
    if include_aux:
        warehouses_data.update({'#': ware_aux_infos})
    df_warehouses = pd.DataFrame(warehouses_data)

    suppliers_data = {'name': supplier_names,
                      'minimumWorkshare_supplier': true_suppl_minimumWorkshare,
                      'targetWorkshare_supplier': true_suppl_targetWorkshare,
                      'maximumWorkshare_supplier': true_suppl_maximumWorkshare}
    if include_aux:
        suppliers_data.update({'#': suppl_aux_infos})
    df_suppliers = pd.DataFrame(suppliers_data)

    regions_data = {'name': country_names,
                    'continent': country_continent_names}
    if include_aux:
        regions_data.update({'#': reg_aux_infos})
    df_regions = pd.DataFrame(regions_data)

    # save
    save_csv(df_production_sites, 'data-production-sites', DATA_PATH)
    save_csv(df_warehouses, 'data-warehouses', DATA_PATH)
    save_csv(df_suppliers, 'data-suppliers', DATA_PATH)
    save_csv(df_supplier_product_locations, 'data-supplier-product-locations', DATA_PATH,
             sort_column=["supplier", "product", "location"])
    save_csv(df_regions, 'data-regions', DATA_PATH)


# ROUTES AND TRANSPORTATION METHODS ###############################################################################################
def convert_routes_and_transportation_methods(include_aux=INCLUDE_AUX):
    def id_query(df, uuid):
        return df[(df["id"].values == uuid)]

    def get_location_name(uuid):
        if pd.isna(uuid):
            location_name = np.nan
        else:
            locations_prod = id_query(df_production_locations, uuid)
            locations_ware = id_query(df_warehouse_locations, uuid)
            locations = pd.concat([locations_prod, locations_ware])
            location_names = list(set(locations['name'].tolist()))
            assert len(location_names) == 1, uuid
            location_name = location_names[0]
        return location_name

    def get_product_name(uuid):
        if pd.isna(uuid):
            product_name = np.nan
        else:
            product_names = id_query(df_products, uuid)["name"].tolist()
            assert len(product_names) == 1, uuid
            product_name = product_names[0]
        return product_name

    def get_cargo_data(uuid):
        if pd.isna(uuid):
            cargo_uuid, count, length, width, height, capacity_name, cargo_name = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            cargo_capacities = id_query(df_cargo_capacities, uuid)
            assert len(cargo_capacities) == 1, uuid
            cargo_capacity = cargo_capacities.iloc[0]
            count = int(cargo_capacity['cargoCount'])
            cargo_uuid = cargo_capacity['cargo']
            cargos = df_cargos.query(f'id=="{cargo_uuid}"')
            # assert len(cargos) == 1, cargo_uuid
            # different cargo rows now refer to different products that are allowed to be transported
            # if it exists, just use the first row. otherwise, the cargo does not seem to be defined.
            ####### CHANGES HERE
            if len(cargos) >= 1:
                cargo = cargos.iloc[0]
                length = int(cargo['length'])
                width = int(cargo['width'])
                height = int(cargo['height'])
                capacity_name = cargo_capacity['name']
                cargo_name = cargo['name']
            else:
                print(f"cargo {cargo_uuid} of cargo_capacity {uuid} undefined")
                cargo_uuid, count, length, width, height, capacity_name, cargo_name = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return {'count': count, 'length': length, 'width': width,
                'height': height}, capacity_name, cargo_name, cargo_uuid

    def check_if_product_fits(product_dimensions, cargo_dimensions):
        if not any([np.isnan(cargo_dimensions[idx]) for idx in range(3)]):
            return True  # treat NaN cargo dimension as arbitrarily large size
        if not any([np.isnan(product_dimensions[idx]) for idx in range(3)]):
            return True  # treat NaN product dimension as arbitrarily small size
        product_dimensions = sorted(product_dimensions)
        cargo_dimensions = sorted(cargo_dimensions)
        return all([product_dimensions[idx] <= cargo_dimensions[idx] for idx in range(3)])  # check dimensions
        # return np.prod(product_dimensions) <= np.prod(cargo_dimensions): # check volume

    def check_product_in_cargo_feasibility(cargo_data, product_name, product_data_dict):

        if pd.isna(product_name):
            return False  # invalid product
        cargo_length = float(cargo_data['length'])
        cargo_width = float(cargo_data['width'])
        cargo_height = float(cargo_data['height'])
        cargo_dimensions = sorted([cargo_length, cargo_width, cargo_height])
        product_data = product_data_dict[product_name]
        product_length = float(product_data["length"])
        product_width = float(product_data["width"])
        product_height = float(product_data["height"])
        product_dimensions = sorted([product_length, product_width, product_height])
        if not check_if_product_fits(product_dimensions, cargo_dimensions):
            return False  # does not fit!
        return True

    def fine_tune_cargo_type_products(value, products, cargo_length, cargo_width, cargo_height):
        # no fine tuning!
        return products, []

    # load
    df_routes = load_csv('routes.csv', RAW_PATH, sort=True)
    df_transportation_resources = load_csv('transportation-resources.csv', RAW_PATH, sort=True)
    df_cargo_capacities = load_csv('cargo-capacities.csv', RAW_PATH, sort=True)
    df_cargos = load_csv('cargos.csv', RAW_PATH, sort=True)
    df_production_locations = load_csv('production-locations.csv', RAW_PATH, sort=True)
    df_warehouse_locations = load_csv('warehouse-locations.csv', RAW_PATH, sort=True)
    df_products = load_csv('products.csv', RAW_PATH, sort=True)
    df_data_products = load_csv('data-products.csv', DATA_PATH, sort=True)

    # extract product data
    product_data_dict = {product_name: df_data_products.query(f'name=="{product_name}"').iloc[0] for product_name in
                         df_data_products["name"]}
    # all_product_names = sorted(df_data_products["name"].tolist())

    # locations
    all_production_location_names = sorted(df_production_locations["name"].tolist())

    # eliminate transport regions
    df_transportation_resources = df_transportation_resources.drop_duplicates(
        subset=[col for col in df_transportation_resources.columns if col != 'transportRegion'])
    df_transportation_resources = df_transportation_resources.drop(['transportRegion'], axis=1)

    # extract cargo types
    cargo_type_collections = {}
    excluded_transportation_methods = {}  # (cargo_name, product_name) -> uuid list
    sorted_uuid_set = sorted(list(set(df_transportation_resources["id"])))
    for uuid in tqdm(sorted_uuid_set, desc='cargos'):
        cargo_type_collection = {}
        df_transportation_resource = df_transportation_resources[df_transportation_resources.id == uuid]
        products = df_transportation_resource["product"]
        cargo_capacities = df_transportation_resource["cargoCapacity"]
        for product_uuid, cargo_capactiy_uuid in zip(products, cargo_capacities):
            product_name = get_product_name(product_uuid)
            cargo_data, capacity_name, cargo_name, cargo_uuid = get_cargo_data(cargo_capactiy_uuid)

            # check if product is valid and fits into the cargo
            if not check_product_in_cargo_feasibility(cargo_data, product_name, product_data_dict):
                if (cargo_name, product_name) not in excluded_transportation_methods:
                    excluded_transportation_methods[(cargo_name, product_name)] = []
                excluded_transportation_methods[(cargo_name, product_name)].append(uuid)
                continue  # no need to consider invalid/unfitting products

            key = tuple(cargo_data.values())
            if key not in cargo_type_collection:
                cargo_type_collection[key] = {'products': [],
                                              'transportation_resource_uuids': [], 'cargo_capactiy_uuids': [],
                                              'capacity_names': [], 'cargo_names': [], 'cargo_uuids': []}
            if product_name not in cargo_type_collection[key]['products']:
                cargo_type_collection[key]['products'].append(product_name)
            if uuid not in cargo_type_collection[key]['transportation_resource_uuids']:
                cargo_type_collection[key]['transportation_resource_uuids'].append(uuid)
            if cargo_capactiy_uuid not in cargo_type_collection[key]['cargo_capactiy_uuids']:
                cargo_type_collection[key]['cargo_capactiy_uuids'].append(cargo_capactiy_uuid)
            if capacity_name not in cargo_type_collection[key]['capacity_names']:
                cargo_type_collection[key]['capacity_names'].append(capacity_name)
            if cargo_name not in cargo_type_collection[key]['cargo_names']:
                cargo_type_collection[key]['cargo_names'].append(cargo_name)
            if cargo_uuid not in cargo_type_collection[key]['cargo_uuids']:
                cargo_type_collection[key]['cargo_uuids'].append(cargo_uuid)
        for key, value in cargo_type_collection.items():
            transportation_resource_uuids = value['transportation_resource_uuids']
            cargo_capactiy_uuids = value['cargo_capactiy_uuids']
            capacity_names = value['capacity_names']
            cargo_names = value['cargo_names']
            cargo_uuids = value['cargo_uuids']
            products = value['products']
            value_key = [";".join(sorted(products))] if products != [np.nan] else [np.nan]
            joint_key = tuple(list(key) + value_key)  # cargo properties + products
            if joint_key not in cargo_type_collections:
                cargo_type_collections[joint_key] = {'idx': len(cargo_type_collections),
                                                     'transportation_resource_uuids': [], 'cargo_capactiy_uuids': [],
                                                     'capacity_names': [], 'cargo_names': [], 'cargo_uuids': []}
            if transportation_resource_uuids not in cargo_type_collections[joint_key]['transportation_resource_uuids']:
                cargo_type_collections[joint_key]['transportation_resource_uuids'] += transportation_resource_uuids
            if cargo_capactiy_uuids not in cargo_type_collections[joint_key]['cargo_capactiy_uuids']:
                cargo_type_collections[joint_key]['cargo_capactiy_uuids'] += cargo_capactiy_uuids
            if capacity_names not in cargo_type_collections[joint_key]['capacity_names']:
                cargo_type_collections[joint_key]['capacity_names'] += capacity_names
            if cargo_names not in cargo_type_collections[joint_key]['cargo_names']:
                cargo_type_collections[joint_key]['cargo_names'] += cargo_names
            if cargo_uuids not in cargo_type_collections[joint_key]['cargo_uuids']:
                cargo_type_collections[joint_key]['cargo_uuids'] += cargo_uuids

    print(
        f"excluded transportation methods ({len(excluded_transportation_methods)}): {excluded_transportation_methods}")

    # def cargo_names
    def cargo_type_name(cargo_type_idx):
        return f'cargo_type_{cargo_type_idx:02d}'

    # traverse all paths
    paths = list(zip(df_routes["sourceLocation"].tolist(), df_routes["destinationLocation"].tolist()))
    paths = sorted(list(set(paths)))
    route_data = {'name': [], 'start': [], 'end': [],
                  'co2_per_distance': [], 'eur_per_distance': [], 'time_per_distance': [], 'distance': [],
                  'maximum_instances': []}
    max_cargos = 2
    for n in range(max_cargos):
        route_data.update({f'cargo_type{n + 1}': []})
    if include_aux:
        route_data['#'] = []
    for idx, path in tqdm(enumerate(paths), total=len(paths), desc='routes'):
        # locations
        start_uuid, end_uuid = path
        start = get_location_name(start_uuid)
        end = get_location_name(end_uuid)

        # transports
        df_path = df_routes[
            (df_routes["sourceLocation"].values == path[0]) & (df_routes["destinationLocation"].values == path[1])]
        assert len(df_path) >= 1
        route_uuids = df_path["id"].tolist()
        route_names = df_path["name"].tolist()

        # methods and distances
        transport_methods = df_path['transportationResource']
        assert len(transport_methods) == len(set(transport_methods))
        distances = [float(distance) for distance in df_path["distance"].tolist()]

        # compile
        for route_uuid, route_name, transport_resource_uuid, distance in zip(route_uuids, route_names,
                                                                             transport_methods, distances):
            resources = id_query(df_transportation_resources, transport_resource_uuid)

            # kpis
            co2s = resources["co2Emissions"].tolist()
            assert len(set(co2s)) == 1
            co2 = float(co2s[0])
            eurs = resources["recurringCosts"].tolist()
            assert len(set(eurs)) == 1
            eur = float(eurs[0])
            speeds = resources["speed"].tolist()
            assert len(set(speeds)) == 1
            speed = float(speeds[0])

            # capacity
            poolMachines_list = resources["poolMachines"].tolist()
            assert len(set(poolMachines_list)) == 1
            maximum_instances = float(poolMachines_list[0])

            # products
            products = resources["product"].tolist()
            product_names = [get_product_name(uuid) for uuid in products]
            cargo_capacity_uuids = resources['cargoCapacity'].tolist()
            assert len(set(cargo_capacity_uuids)) <= max_cargos
            cargo_dict = {}
            for product_name, cargo_capacity_uuid in zip(product_names, cargo_capacity_uuids):
                cargo_data, capacity_name, cargo_name, cargo_uuid = get_cargo_data(cargo_capacity_uuid)

                # check if product is valid and fits into the cargo
                if not check_product_in_cargo_feasibility(cargo_data, product_name, product_data_dict):
                    assert (cargo_name, product_name) in excluded_transportation_methods, (cargo_name, product_name)
                    continue  # no need to consider invalid/unfitting products
                else:
                    assert (cargo_name, product_name) not in excluded_transportation_methods, (cargo_name, product_name)

                if cargo_capacity_uuid not in cargo_dict:
                    cargo_dict[cargo_capacity_uuid] = {'products': [], 'cargo_data': [], 'cargo_name': cargo_name}
                    cargo_dict[cargo_capacity_uuid]['cargo_data'] = cargo_data
                else:
                    assert cargo_data['count'] == cargo_dict[cargo_capacity_uuid]['cargo_data'][
                        'count'], cargo_capacity_uuid
                    assert cargo_data['length'] == cargo_dict[cargo_capacity_uuid]['cargo_data'][
                        'length'], cargo_capacity_uuid
                    assert cargo_data['width'] == cargo_dict[cargo_capacity_uuid]['cargo_data'][
                        'width'], cargo_capacity_uuid
                    assert cargo_data['height'] == cargo_dict[cargo_capacity_uuid]['cargo_data'][
                        'height'], cargo_capacity_uuid
                    assert cargo_name == cargo_dict[cargo_capacity_uuid]['cargo_name'], cargo_capacity_uuid
                if product_name not in cargo_dict[cargo_capacity_uuid]['products']:
                    cargo_dict[cargo_capacity_uuid]['products'].append(product_name)

            route_data['start'].append(start)
            route_data['end'].append(end)
            route_data['co2_per_distance'].append(co2)
            route_data['eur_per_distance'].append(eur)
            route_data['time_per_distance'].append(speed)
            route_data['distance'].append(distance)
            route_data['maximum_instances'].append(maximum_instances)
            if include_aux:
                aux_dict = {'route_name': route_name, 'route_uuid': route_uuid,
                            'start_uuid': start_uuid, 'end_uuid': end_uuid,
                            'transport_resource_uuid': transport_resource_uuid}
                route_data['#'].append(aux_dict)
            cargo_types_for_route = []
            keys = list(cargo_dict.keys())  # cargo_uuid
            values = list(cargo_dict.values())  # { products, cargo_data, cargo_name }
            for n in range(max_cargos):
                if len(values) > n:
                    value = values[n]
                    try:
                        product_names_str = ";".join(sorted(value['products']))
                    except:
                        product_names_str = np.nan
                        assert all([pd.isna(v) for v in value['products']])
                    key = list(value['cargo_data'].values())
                    value_key = [product_names_str]
                    joint_key = tuple(key + value_key)  # as above: cargo properties + products
                    cargo_type_idx = cargo_type_collections[joint_key]['idx']
                    cargo_type = cargo_type_name(cargo_type_idx)
                    cargo_capacity_uuid = keys[n]
                    cargo_name = value['cargo_name']
                else:
                    cargo_type = ""
                    cargo_capacity_uuid = ""
                    cargo_name = ""
                route_data[f'cargo_type{n + 1}'].append(cargo_type)
                if include_aux:
                    route_data['#'][-1][f'cargo_capacity_{n}_uuid'] = cargo_capacity_uuid
                    route_data['#'][-1][f'cargo_{n}_name'] = cargo_name
                cargo_types_for_route.append(cargo_type)

            # name
            start_suffix = '' if start in all_production_location_names else '⌂'
            end_suffix = '' if end in all_production_location_names else '⌂'
            name = f'{start}{start_suffix}➥[{";".join([cargo_type for cargo_type in cargo_types_for_route if len(cargo_type) > 0])}]➦{end}{end_suffix}'
            route_data['name'].append(name)

    # prepare aux infos
    if include_aux:
        route_data['#'] = [json.dumps(aux_dict) for aux_dict in route_data['#']]

    # prepare data frames
    df_route_data = pd.DataFrame(route_data)
    df_route_data = df_route_data.sort_values(['start', 'end', 'distance', 'cargo_type1'])
    df_route_data = df_route_data.reset_index(drop=True)
    excluded_cargo_types = []
    extended_cargo_types = {}
    cargo_type_data = {'name': [], 'count': [], 'length': [], 'width': [], 'height': [], 'products': [],
                       'description': []}
    if include_aux:
        cargo_type_data['#'] = []
    for key, value in cargo_type_collections.items():
        count, length, width, height, products = key
        cargo_type_idx = value['idx']
        capacity_names = value['capacity_names']
        description = ";".join([str(capacity_name) for capacity_name in capacity_names])
        name = cargo_type_name(cargo_type_idx)
        fine_tuned_products, new_product_names = fine_tune_cargo_type_products(value, products, length, width, height)
        if len(new_product_names) > 0:
            extended_cargo_types[name] = new_product_names
        exclusion_criteria = [products]  # alternative: [count, length, width, height, products]
        is_valid = not any([pd.isna(value) for value in exclusion_criteria])
        if not is_valid:
            excluded_cargo_types.append(name)
        cargo_type_data['name'].append(name)
        cargo_type_data['count'].append(count)
        cargo_type_data['length'].append(length)
        cargo_type_data['width'].append(width)
        cargo_type_data['height'].append(height)
        cargo_type_data['products'].append(fine_tuned_products)
        cargo_type_data['description'].append(description)
        transportation_resource_uuids = value['transportation_resource_uuids']
        cargo_capactiy_uuids = value['cargo_capactiy_uuids']
        cargo_names = value['cargo_names']
        cargo_uuids = value['cargo_uuids']
        if include_aux:
            aux_dict = {'transportation_resource_uuids': transportation_resource_uuids,
                        'cargo_capactiy_uuids': cargo_capactiy_uuids,
                        'cargo_uuids': cargo_uuids,
                        'capacity_names': capacity_names,
                        'cargo_names': cargo_names}
            cargo_type_data['#'].append(json.dumps(aux_dict))
    df_cargo_type_data = pd.DataFrame(cargo_type_data)

    # remove routes based on excluded cargo types
    for excluded_cargo_type in excluded_cargo_types:
        for n in range(max_cargos):
            df_route_data.loc[df_route_data[f"cargo_type{n + 1}"] == excluded_cargo_type, f"cargo_type{n + 1}"] = ""
    drop_indices = df_route_data.query("&".join([f"cargo_type{n + 1}==''" for n in range(max_cargos)])).index.tolist()
    print(f'number of excluded routes:         {len(drop_indices)} (of {len(df_route_data)})')
    df_route_data.drop(drop_indices, inplace=True)
    print(f'excluded invalid cargo types:      {excluded_cargo_types}')
    print(f'extended products for cargo types: {extended_cargo_types}')

    # save
    save_csv(df_route_data, 'data-routes', DATA_PATH)
    save_csv(df_cargo_type_data, 'data-cargotypes', DATA_PATH)


def save_metadata():
    file_path = os.path.join(DATA_PATH, 'meta.json')
    metadata_dict = dict(RAW_PATH=RAW_PATH, PRODUCT_REL_VALUE_FRACTION=PRODUCT_REL_VALUE_FRACTION,
                         DATA_VERSION=DATA_VERSION)
    with open(file_path, 'w') as fh:
        json.dump(metadata_dict, fh)


# MAIN ############################################################################################################################

def process_data():
    print("\nPRODUCTS")
    print("========")
    convert_products()

    print("\nLOCATIONS AND SUPPLIERS")
    print("=======================")
    convert_locations_and_suppliers()

    print("\nROUTES AND TRANSPORTATION METHODS")
    print("=================================")
    convert_routes_and_transportation_methods()

    save_metadata()


if __name__ == "__main__":
    process_data()

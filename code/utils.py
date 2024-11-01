import json

import geodatasets
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString

from tools import A, get_rel_value


def visualize_solution(model, instance, x_solution):
    if instance.model.is_solution_valid(x_solution):
        print("SOLUTION IS FEASIBLE")
    else:
        print("SOLUTION IS INFEASIBLE")
    print(f"OBJECTIVE: {instance.model.value(x_solution)}")

    # get from solution space
    feasible_suppliers = list(set(instance.df_solution_space["supplier"].tolist()))
    feasible_sites = list(set(instance.df_solution_space["site"].tolist()))

    # compile
    solution_data = {'product_index': [], 'double_source_index': [], 'site_index': [], 'supplier_index': [],
                     'value': []}
    num_ancilla_variables = 0
    for name, value in x_solution.items():
        if '__a' in name:
            # skip ancilla
            num_ancilla_variables += 1
            continue
        product_index, double_source_index, site_index, supplier_index = instance.variable_name_to_indices(name)
        solution_data['product_index'].append(product_index)
        solution_data['double_source_index'].append(double_source_index)
        solution_data['site_index'].append(site_index)
        solution_data['supplier_index'].append(supplier_index)
        solution_data['value'].append(value)
    for name, value in instance.constants.items():
        product_index, double_source_index, site_index, supplier_index = instance.variable_name_to_indices(name)
        solution_data['product_index'].append(product_index)
        solution_data['double_source_index'].append(double_source_index)
        solution_data['site_index'].append(site_index)
        solution_data['supplier_index'].append(supplier_index)
        solution_data['value'].append(value)
    df_solution = pd.DataFrame(solution_data)

    print(f'number of ignored ancilla variables: {num_ancilla_variables}')

    df_products = model.data_processor.df_products
    df_production_sites = model.data_processor.df_production_sites
    df_suppliers = model.data_processor.df_suppliers

    D = {}  # product index -> number of sources
    for product_index in df_products["index"].tolist():
        number_of_options = len(instance.df_solution_space.query(f"product_index=={product_index}"))
        D[product_index] = 1 if number_of_options == 1 else 2
    N = {}  # product index -> number of feasible regions
    for product_index in df_products["index"].tolist():
        regions = set(instance.df_solution_space.query(f"product_index=={product_index}")["region"].tolist())
        N[product_index] = len(regions)

    # gather solution_product_sites
    solution_product_sites = dict()  # product -> [double_source_index] -> site_index
    print("gather solution_product_sites")
    for _, row in df_products.iterrows():
        product_index = row['index']
        product_name = row['name']
        d = D[product_index]
        n = N[product_index]
        solution_product_sites[product_name] = dict()
        df_prod = df_solution.query(f'value==1&product_index=={product_index}')
        if not len(df_prod) == 2:
            print('invalid assignment:', product_name, 'has', len(df_prod), ' assignments')
        if d == 1:
            if not len(set(df_prod['site_index'])) == 1:
                print('invalid single source assignment:', product_name)
            site_index = df_prod.iloc[0]["site_index"]
            site_name = df_production_sites.query(f'index=={site_index}').iloc[0]["name"]
            solution_product_sites[product_name] = [site_name]
            # print(f"product {product_name:30} assigned to site {site_name} (single source)")
        elif d == 2:
            if not len(set(df_prod['site_index'])) == 2:
                print('invalid double source assignment:', product_name)
            site_regions = []
            source_dict = dict()
            for _, row_prod in df_prod.iterrows():
                site_index = row_prod['site_index']
                site_name = df_production_sites.query(f'index=={site_index}').iloc[0]["name"]
                site_region = df_production_sites.query(f'index=={site_index}').iloc[0]["country"]
                double_source_index = row_prod['double_source_index']
                source_dict[int(double_source_index)] = site_name
                site_regions.append(site_region)
            if n > 2:
                if not len(set(site_regions)) == 2:
                    print('invalid assignment:', product_name, 'is produced in', len(site_regions), ' regions')
            if len(source_dict) != 2:
                print('invalid assignment:', product_name, 'is produced in', len(source_dict), ' sites')
            else:
                solution_product_sites[product_name] = [source_dict[0], source_dict[1]]
                # print(f"product {product_name:30} assigned to sites {source_dict[0]} ({site_regions[0]}) & {source_dict[1]} ({site_regions[1]})")

    # gather solution_product_suppliers
    solution_product_supplier = dict()  # product -> [double_source_index] -> [supplier_index]
    print("gather solution_product_suppliers")
    for _, row in df_products.iterrows():
        product_index = row['index']
        product_name = row['name']
        d = D[product_index]
        # n = N[product_index]
        solution_product_supplier[product_name] = dict()
        df_prod = df_solution.query(f'value==1&product_index=={product_index}')
        if not len(df_prod) == 2:
            print('invalid assignment:', product_name, 'has', len(df_prod), ' assignments')
        if d == 1:
            if not len(set(df_prod['supplier_index'])) == 1:
                print('invalid single source assignment:', product_name)
            supplier_index = df_prod.iloc[0]["supplier_index"]
            supplier = df_suppliers.query(f'index=={supplier_index}').iloc[0]["name"]
            solution_product_supplier[product_name] = [supplier]
            # print(f"product {product_name:30} assigned to supplier {supplier} (single source)")
        elif d == 2:
            source_dict = dict()
            for _, row_prod in df_prod.iterrows():
                supplier_index = row_prod['supplier_index']
                supplier = df_suppliers.query(f'index=={supplier_index}').iloc[0]["name"]
                double_source_index = row_prod['double_source_index']
                source_dict[int(double_source_index)] = supplier
            if len(source_dict) == 2:
                solution_product_supplier[product_name] = [source_dict[0], source_dict[1]]
                # print(f"product {product_name:30} assigned to suppliers {source_dict[0]} & {source_dict[1]}")

    # lists
    site_names = df_production_sites["name"].tolist()
    assigned_sites = set.union(*[set(value) for value in solution_product_sites.values()])
    supplier_names = df_suppliers["name"].tolist()

    # MAP PLOT
    size = .45

    # locations
    LAT, LONG = [], []
    for _, row in df_production_sites.iterrows():
        aux = json.loads(row['#'])
        latitude = aux['latitude']
        longitude = aux['longitude']
        LAT.append(latitude)
        LONG.append(longitude)
    gdf_locations_site = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy(LONG, LAT))
    gdf_locations_site['color'] = gdf_locations_site.index.map(
        lambda idx: 'red' if site_names[idx] in assigned_sites else 'cyan')
    gdf_locations_site['markersize'] = gdf_locations_site.index.map(
        lambda idx: 250 if site_names[idx] in assigned_sites else 100)

    # get size
    x_lims = [min(LONG) - 2, max(LONG) + 2]
    y_lims = [min(LAT) - 2, max(LAT) + 2]
    figsize = ((x_lims[1] - x_lims[0]) * size, (y_lims[1] - y_lims[0]) * size)

    # connections
    LAT, LONG, IDX = [], [], []
    idx = 0
    for _, row in df_products.iterrows():
        parent_name = row['parent']
        if pd.isna(parent_name):
            continue
        product_name = row['name']
        for start_site_name in solution_product_sites[product_name]:
            aux1 = json.loads(df_production_sites.query(f"name=='{start_site_name}'").iloc[0]['#'])
            latitude1 = aux1['latitude']
            longitude1 = aux1['longitude']
            for end_site_name in solution_product_sites[parent_name]:
                aux2 = json.loads(df_production_sites.query(f"name=='{end_site_name}'").iloc[0]['#'])
                latitude2 = aux2['latitude']
                longitude2 = aux2['longitude']
                LAT.append(latitude1)
                LAT.append(latitude2)
                LONG.append(longitude1)
                LONG.append(longitude2)
                IDX.append(idx)
                IDX.append(idx)
                idx += 1
    df = pd.DataFrame({'IDX': IDX, 'LONG': LONG, 'LAT': LAT})
    gdf_routes = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(LONG, LAT))
    lines = gdf_routes.groupby(['IDX'])['geometry'].apply(lambda x: LineString(x.tolist()))
    gdf_lines = geopandas.GeoDataFrame(lines, geometry='geometry', crs="EPSG:4326")
    gdf_lines.reset_index(inplace=True)

    # world map
    world = geopandas.read_file(geodatasets.get_path("naturalearth land"))
    ax = world.plot(figsize=figsize, color=[.9, .9, .9])

    # lines
    # gdf_lines.plot(ax=ax, color='blue', alpha=.2, linewidth=1)
    for _, line in gdf_lines["geometry"].items():
        start_xy = [line.xy[0][0], line.xy[1][0]]
        end_xy = [line.xy[0][1], line.xy[1][1]]
        offset = 0
        ax.annotate(
            "",
            xy=(start_xy[0] + offset, start_xy[1] + offset),
            xytext=(end_xy[0] + offset, end_xy[1] + offset),
            arrowprops={"color": 'blue', 'alpha': .1, 'linewidth': .25},
        )

    # locations
    gdf_locations_site.plot(ax=ax, color=gdf_locations_site['color'], marker='^',
                            markersize=gdf_locations_site['markersize'])

    # labels
    for x_coord, y_coord, site_name in zip(gdf_locations_site.geometry.x, gdf_locations_site.geometry.y, site_names):
        labels = []
        for product_name, sites in solution_product_sites.items():
            for assigned_site_name in sites:
                if assigned_site_name == site_name:
                    labels.append(product_name)
        if len(labels) > 0:
            label = f'{site_name}:\n' + "\n".join(sorted(list(set(labels))))
            ax.annotate(label, xy=(x_coord, y_coord), xytext=(6, 0), textcoords="offset points", color='red',
                        weight='bold')
        else:
            label = f'({site_name})'
            ax.annotate(label, xy=(x_coord, y_coord), xytext=(6, 0), textcoords="offset points", color='cyan')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    plt.axis('off')
    plt.show()

    # SITE WORK SHARE PLOT
    site_workshare = {site_name: 0 for site_name in site_names}
    for product, sites in solution_product_sites.items():
        rel_value = get_rel_value(model.data_processor.df_products, product)
        if len(sites) == 1:
            site_workshare[sites[0]] += rel_value
        elif len(sites) == 2:
            site_workshare[sites[0]] += instance.alpha * rel_value
            site_workshare[sites[1]] += (1 - instance.alpha) * rel_value
        else:
            continue  # invalid
    site_workshare = list(site_workshare.values())
    site_min_workshare = df_production_sites['minimumWorkshare_site'].tolist()
    site_max_workshare = df_production_sites['maximumWorkshare_site'].tolist()
    site_valid = []
    for name, s, min_s, max_s in zip(site_names, site_workshare, site_min_workshare, site_max_workshare):
        if name not in feasible_sites:
            site_valid.append("gray")
        else:
            if (s >= min_s and s <= max_s):
                site_valid.append("green")
            else:
                site_valid.append("red")
    plt.figure()
    plt.scatter(range(len(site_names)), site_workshare, marker='.', c=site_valid)
    plt.scatter(range(len(site_names)), site_min_workshare, marker='2')
    plt.scatter(range(len(site_names)), site_max_workshare, marker='1')
    plt.xticks(range(len(site_names)), site_names, rotation=90)
    for tick_idx in range(len(site_names)):
        plt.gca().get_xticklabels()[tick_idx].set_color(site_valid[tick_idx])
    plt.yticks(range(max(site_max_workshare) + 1))
    plt.ylabel('site work share')
    plt.grid(alpha=.25)
    plt.show()

    # SUPPLIER WORK SHARE PLOT
    supplier_workshare = {supplier_name: 0 for supplier_name in supplier_names}
    for product, suppliers in solution_product_supplier.items():
        rel_value = get_rel_value(model.data_processor.df_products, product)
        if len(suppliers) == 1:
            supplier_workshare[suppliers[0]] += rel_value
        elif len(suppliers) == 2:
            supplier_workshare[suppliers[0]] += instance.alpha * rel_value
            supplier_workshare[suppliers[1]] += (1 - instance.alpha) * rel_value
        else:
            continue  # invalid
    supplier_workshare = list(supplier_workshare.values())
    supplier_min_workshare = df_suppliers['minimumWorkshare_supplier'].tolist()
    supplier_max_workshare = df_suppliers['maximumWorkshare_supplier'].tolist()
    supplier_target_workshare = df_suppliers['targetWorkshare_supplier'].tolist()
    supplier_valid = []
    for name, s, min_s, max_s in zip(supplier_names, site_workshare, site_min_workshare, site_max_workshare):
        if name not in feasible_suppliers:
            supplier_valid.append("gray")
        else:
            if (s >= min_s and s <= max_s):
                supplier_valid.append("green")
            else:
                supplier_valid.append("red")
    plt.figure()
    plt.scatter(range(len(supplier_names)), supplier_workshare, marker='.', c=supplier_valid)
    plt.scatter(range(len(supplier_names)), supplier_min_workshare, marker='2')
    plt.scatter(range(len(supplier_names)), supplier_max_workshare, marker='1')
    plt.scatter(range(len(supplier_names)), supplier_target_workshare, marker='_', c='black')
    plt.xticks(range(len(supplier_names)), supplier_names, rotation=90)
    for tick_idx in range(len(supplier_names)):
        plt.gca().get_xticklabels()[tick_idx].set_color(supplier_valid[tick_idx])
    plt.yticks(range(max(supplier_max_workshare) + 1)[::2])
    plt.ylabel('supplier work share')
    plt.grid(alpha=.25)
    plt.show()

    # REGIONS PLOT
    regions = model.data_processor.df_regions["name"].tolist()
    plt.figure(figsize=(8, 8))
    products = []
    for product, sites in solution_product_sites.items():
        y = len(products)
        products.append(product)
        regions_to_choose = len(set(instance.df_solution_space.query(f"product=='{product}'")["region"].tolist()))
        if len(sites) == 1:
            region1 = model.data_processor.df_production_sites.query(f"name=='{sites[0]}'").iloc[0]["country"]
            region1_x = regions.index(region1)
            valid = regions_to_choose == 1
            c = 'green' if valid else 'red'
            plt.plot([region1_x], [y], c=c, marker='x')
        elif len(sites) == 2:
            region1 = model.data_processor.df_production_sites.query(f"name=='{sites[0]}'").iloc[0]["country"]
            region2 = model.data_processor.df_production_sites.query(f"name=='{sites[1]}'").iloc[0]["country"]
            region1_x = regions.index(region1)
            region2_x = regions.index(region2)
            valid = regions_to_choose == 1 or region1 != region2
            c = 'green' if valid else 'red'
            plt.plot([region1_x, region2_x], [y, y], c=c)
        else:
            continue  # invalid
    plt.xticks(list(range(len(regions))), regions, rotation=90)
    plt.yticks(list(range(len(products))), products)
    plt.grid(alpha=.25)
    plt.show()

    return df_solution


def solution_to_report(model, instance, x_solution):
    # convert solution into a text report
    if instance.model.is_solution_valid(x_solution):
        print("SOLUTION IS FEASIBLE")
    else:
        print("SOLUTION IS INFEASIBLE")
    print(f"OBJECTIVE: {instance.model.value(x_solution)}")

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

    # report
    report = {product_name: [] for product_name in model.data_processor.df_products["name"].tolist()}
    for product_index, product_assignment_dict in assignments.items():
        product_name = model.data_processor.df_products.query(f"index=={product_index}").iloc[0]["name"]

        for double_source_index, value in product_assignment_dict.items():
            if value is None:
                report[product_name].append("invalid assignment")
                continue

            (site_index, supplier_index) = value
            a = A(double_source_index, instance)
            site_name = model.data_processor.df_production_sites.query(f"index=={site_index}").iloc[0]["name"]
            supplier_name = model.data_processor.df_suppliers.query(f"index=={supplier_index}").iloc[0]["name"]
            source_description = "primary" if double_source_index == 0 else "secondary"
            report[product_name].append(
                f'{source_description} production ({a * 100:.0f}%) at "{site_name}" with "{supplier_name}"')

            # transportation costs for product
            if product_index in model.phi_inv:

                # parent
                parent_index = model.phi_inv[product_index]
                if parent_index not in assignments:
                    report[product_name].append("invalid parent assignment")
                    continue
                parent_product_assignment_dict = assignments[parent_index]
                parent_name = model.data_processor.df_products.query(f"index=={parent_index}").iloc[0]["name"]
                for parent_double_source_index, parent_value in parent_product_assignment_dict.items():
                    if parent_value is None:
                        report[product_name].append("invalid parent assignment")
                        continue

                    (parent_site_index, parent_supplier_index) = parent_value
                    parent_source_description = "primary" if parent_double_source_index == 0 else "secondary"
                    parent_site_name = \
                        model.data_processor.df_production_sites.query(f"index=={parent_site_index}").iloc[0]["name"]
                    if site_index == parent_site_index:
                        # no transportation
                        report[product_name].append(
                            f'no transportation of {source_description} source to {parent_source_description} parent source ("{parent_name}") necessary')
                    else:
                        df_costs = df_cost_matrix.query(
                            f"product_index=={product_index}&start_site_index=={site_index}&end_site_index=={parent_site_index}")
                        if len(df_costs) == 0:
                            report[product_name].append(
                                f'no transportation of {source_description} source to {parent_source_description} parent source ("{parent_name}") possible')
                        else:
                            assert len(df_costs) == 1
                            df_cost = df_costs.iloc[0]
                            route_indices = df_cost['route_indices'].split(";")
                            paths = []
                            for idx, route_index in enumerate(route_indices):
                                route = model.data_processor.df_routes.query(f"index=={route_index}").iloc[0]
                                start_site_name = '"' + route["start"] + '"' + (
                                    f' (production site)' if idx == 0 else "")
                                end_site_name = '"' + route["end"] + '"' + (
                                    f' (production site of "{parent_name}")' if idx == len(route_indices) - 1 else "")
                                cargo_type1 = route["cargo_type1"]
                                cargo_type2 = route["cargo_type2"]
                                descriptions = []
                                if not pd.isna(cargo_type1):
                                    descriptions.append(
                                        model.data_processor.df_cargotypes.query(f"name=='{cargo_type1}'").iloc[0][
                                            "description"].replace(";", "/"))
                                if not pd.isna(cargo_type2):
                                    descriptions.append(
                                        model.data_processor.df_cargotypes.query(f"name=='{cargo_type2}'").iloc[0][
                                            "description"].replace(";", "/"))
                                transportation_description = " and ".join(
                                    [('"' + description + '"') for description in descriptions])
                                paths.append(
                                    f'from {start_site_name} to {end_site_name} via {transportation_description}')
                            report[product_name].append(
                                f'transportation of {source_description} source from "{site_name}" to {parent_source_description} parent source ("{parent_name}") at "{parent_site_name}" using the route ' + " and ".join(
                                    paths))
            else:
                # no parent
                pass

    # to string
    report_str = ""
    for key, tags in report.items():
        report_str += f"{key}\n"
        report_str += f"{'=' * len(key)}\n"
        for tag_idx, tag in enumerate(tags):
            report_str += f"[{tag_idx + 1:d}] {tag}\n"
        report_str += "\n"

    # return report string
    return report_str

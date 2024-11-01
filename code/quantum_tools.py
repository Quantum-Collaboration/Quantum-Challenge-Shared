from collections import defaultdict

import networkx as nx


def process_sample_dict(samples, n_items):
    results = defaultdict(int)
    for sample in samples:
        if n_items == 1:
            sample = [sample]
        results["".join(str(i) for i in sample)[:n_items]] += 1
    results = dict(sorted([item for item in results.items()], key=lambda item: -item[1]))
    return results


def process_sample_list(sample_list):
    results = dict()
    for sample_spin in zip(*sample_list):
        sample_bin = [1 if s < 0 else 0 for s in sample_spin]
        key = "".join(str(b) for b in sample_bin)
        if key not in results:
            results[key] = 0
        results[key] += 1
    results = dict(sorted([item for item in results.items()], key=lambda item: -item[1]))
    return results


def build_mixer_graph(instance, allowed_indices=None):
    # find allowed names
    if allowed_indices is None:
        allowed_names = None
    else:
        allowed_names = [instance.model.reverse_mapping[idx] for idx in allowed_indices]

    # build graph
    mixer_graph = nx.Graph()
    for variable_name in instance.variables:
        if allowed_names is not None and variable_name not in allowed_names:
            continue
        mixer_graph.add_node(variable_name)
    for idx, variable_name1 in enumerate(instance.variables):
        if '__a' in variable_name1:
            continue
        if allowed_names is not None and variable_name1 not in allowed_names:
            continue
        product_index1, double_source_index1, site_index1, supplier_index1 = instance.variable_name_to_indices(
            variable_name1)
        for variable_name2 in list(instance.variables)[idx + 1:]:
            if variable_name1 == variable_name2:
                continue
            if '__a' in variable_name2:
                continue
            if allowed_names is not None and variable_name2 not in allowed_names:
                continue
            product_index2, double_source_index2, site_index2, supplier_index2 = instance.variable_name_to_indices(
                variable_name2)
            if product_index1 == product_index2 and double_source_index1 == double_source_index2:
                mixer_graph.add_edge(instance.model.mapping[variable_name1], instance.model.mapping[variable_name2])
                # print(variable_name1, variable_name2)

    return mixer_graph

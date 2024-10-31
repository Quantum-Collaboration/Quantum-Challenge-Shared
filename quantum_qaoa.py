import math
from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import qaoa

from quantum_tools import build_mixer_graph
from quantum_tools import process_sample_list


def from_Q_to_Ising(Q):
    n_qubits = len(Q)
    h = defaultdict(int)
    J = defaultdict(int)
    c = 0

    # Loop over each qubit (variable) in the QUBO matrix
    for i in range(n_qubits):
        h[(i,)] -= Q[i, i] / 2
        c += Q[i, i] / 2
        for j in range(n_qubits):
            if i == j:
                continue
            h[(i,)] -= Q[i, j] / 4
            h[(j,)] -= Q[i, j] / 4
            J[(i, j)] += Q[i, j] / 4
            c += Q[i, j] / 4

    return h, J, c


def qaoa_circuit(gammas, betas, h, J, indices_qaoa, edge_list=None):
    wmax = max(
        np.max(np.abs(list(h.values()))), np.max(np.abs(list(J.values())))
    )
    if edge_list is not None:
        mixer_graph = nx.from_edgelist(edge_list)
        wire_map = {idx_qaoa: idx for idx, idx_qaoa in enumerate(indices_qaoa)}
    else:
        mixer_graph = None
    num_qubits = len(h)
    wires = list(range(num_qubits))
    p = len(gammas)
    if mixer_graph is None:
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
    else:
        for node_set in nx.connected_components(mixer_graph):
            k = 1
            state = np.zeros(2 ** len(node_set))
            const = 1 / np.sqrt(math.comb(len(node_set), k))
            for comb in product(range(2), repeat=len(node_set)):
                if sum(comb) == k:
                    idx = int(''.join([str(c) for c in comb]), 2)
                    state[idx] = const
            qml.StatePrep(state, wires=[wire_map[node_idx] for node_idx in node_set])
    for layer in range(p):
        # cost
        for ki, v in h.items():
            qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
        for kij, vij in J.items():
            qml.CNOT(wires=[kij[0], kij[1]])
            qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
            qml.CNOT(wires=[kij[0], kij[1]])
        # mixer
        if edge_list is None:
            for i in range(num_qubits):
                qml.RX(-2 * betas[layer], wires=i)
        else:
            mixer_op = qaoa.xy_mixer(mixer_graph)
            mixer_op = qml.map_wires(mixer_op, wire_map=wire_map)
            qml.apply(mixer_op)
    return [qml.sample(qml.PauliZ(i)) for i in wires]  # qml.sample(wires=wires)


def run_qaoa(instance, p, use_custom_mixer, dev_kwargs, N_qaoa=25, seed=None, select_indices=None):
    # prepare matrix
    N = len(instance.variables)
    Q_full = np.zeros((N, N))
    for (i, j), value in instance.qubo_matrix.items():
        Q_full[i, j] = value
    if select_indices is None:
        if N <= N_qaoa:
            indices_qaoa = np.arange(N).tolist()
            N_qaoa = N
        else:
            indices_qaoa = np.random.RandomState(seed).choice(np.arange(N), size=N_qaoa).tolist()  # fallback option
    else:
        indices_qaoa = [idx for idx in select_indices if idx in range(N)]
        assert N_qaoa >= len(indices_qaoa)
    Q = Q_full[np.ix_(indices_qaoa, indices_qaoa)]  # (<=N_qaoa, <=N_qaoa)

    # setup quantum algorithm
    if p > 1:
        gammas = np.linspace(0, 1, p)
        betas = gammas[::-1]
    else:
        gammas = [0.5]
        betas = [0.5]
    mixer_graph = build_mixer_graph(instance, allowed_indices=indices_qaoa)
    edge_list = [edge for edge in mixer_graph.edges] if use_custom_mixer else None
    h, J, c = from_Q_to_Ising(Q)

    # run circuit
    dev = qml.device(**dev_kwargs, wires=N_qaoa)
    qnode = qml.QNode(qaoa_circuit, dev, interface=None)
    sample_list = qnode(gammas, betas, h, J, indices_qaoa, edge_list)

    # process results
    samp_dict = process_sample_list(sample_list)
    x_qaoa = [int(bit) for bit in list(samp_dict.keys())[0]]
    x = {instance.model.reverse_mapping[idx_qaoa]: x_qaoa[idx] for idx, idx_qaoa in enumerate(indices_qaoa)}
    return x, samp_dict

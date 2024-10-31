import math
import warnings
from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import qaoa
from scipy.linalg import ishermitian
from scipy.optimize import dual_annealing
from tqdm import tqdm

from quantum_tools import build_mixer_graph
from quantum_tools import process_sample_list

warnings.filterwarnings('ignore', module='pennylane')  # no autograd required

MINIMUM_SHOTS_PER_CIRCUIT = 8  # fot testing only


def from_Q_to_Ising(Q_full, indices_qaoa, indices_logn, x):
    Q = Q_full[np.ix_(indices_qaoa, indices_qaoa)]  # (N1, N1)

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

    # Correlation
    for i1 in range(len(indices_qaoa)):
        for i2 in range(len(indices_logn)):
            h[(i1,)] -= Q_full[i1, i2] * x[i2] / 2
            h[(i1,)] -= Q_full[i2, i1] * x[i2] / 2
            c += Q_full[i1, i2] * x[i2] / 2
            c += Q_full[i2, i1] * x[i2] / 2

    return h, J, c


def from_Q_to_quadratic_Ising(Q_full, indices_logn, indices_qaoa, num_dummy_indices, x):
    Q = Q_full[np.ix_(indices_logn, indices_logn)]  # (N2, N2)

    n = len(Q)  # n_vars = 2*n
    J = defaultdict(int)
    c = 0

    # Loop over each qubit (variable) in the QUBO matrix
    for i in range(n):
        J[(i, i + n)] -= Q[i, i] / 2 / 2
        J[(i + n, i)] -= Q[i, i] / 2 / 2
        c += Q[i, i] / 2
        for j in range(n):
            if i == j:
                continue
            J[(i, i + n)] -= Q[i, j] / 4 / 2
            J[(i + n, i)] -= Q[i, j] / 4 / 2
            J[(j, j + n)] -= Q[i, j] / 4 / 2
            J[(j + n, j)] -= Q[i, j] / 4 / 2
            J[(i, j)] += Q[i, j] / 4
            c += Q[i, j] / 4

    # Correlation
    for i2 in range(len(indices_logn)):
        for i1 in range(len(indices_qaoa)):
            J[(i2, i2 + n)] -= Q_full[i1, i2] * x[i2] / 2 / 2
            J[(i2 + n, i2)] -= Q_full[i1, i2] * x[i2] / 2 / 2
            J[(i2, i2 + n)] -= Q_full[i2, i1] * x[i2] / 2 / 2
            J[(i2 + n, i2)] -= Q_full[i2, i1] * x[i2] / 2 / 2
            c += Q_full[i1, i2] * x[i2] / 2
            c += Q_full[i2, i1] * x[i2] / 2

    # Fill to power of 2
    for i in range(num_dummy_indices):
        J[(2 * n + i, 2 * n + i)] = 0
        J[(2 * n + i + num_dummy_indices, 2 * n + i + num_dummy_indices)] = 0

    return J, c


def generate_op(N2, J2, wires):
    A = np.zeros((2 * N2, 2 * N2))
    assert ishermitian(A)
    for (i, j), val in J2.items():
        A[i, j] = val
    op = qml.Hermitian(A, wires=wires)
    return op


def calc_elem(theta, q, pot=4):
    # see quantalg4

    def calc_nonl3(theta):
        peaks = np.linspace(0 + np.pi / 2, 2 * np.pi - np.pi / 2, 2)
        peak_dist = np.pi  # delta
        closest_peak_index = np.argmin(np.abs(theta - peaks))
        closest_peak = peaks[closest_peak_index]
        shift_01 = np.abs(closest_peak - theta) / (peak_dist / 2)
        sign_01 = np.sign(closest_peak - theta)
        shift = shift_01 ** pot  # [0,1] -> [0,1]
        return closest_peak + shift * sign_01 * (peak_dist / 2)

    def calc_S3(theta, q):
        theta = (theta / np.pi * 2 ** (q)) % 2 * np.pi
        return np.sin(calc_nonl3(theta) / 2 - np.pi / 4) ** 2

    S = calc_S3(theta, q)
    return np.exp(1j * S * np.pi)  # [-1,+1]


def eval_diag(thetas, N2):
    num_thetas = len(thetas)
    D = []
    for theta_idx, idx_array in enumerate(np.array_split(np.arange(0, N2), num_thetas)):
        theta = thetas[theta_idx]
        num = len(idx_array)
        D += [calc_elem(theta, q) for q in range(num)]
    return D  # [-1, +1]


def get_exp_from_samples_collection(samples_collection):
    exp = 0
    for (coefficient, samples_dict) in samples_collection:
        measurement_exp = 0
        for key, value in samples_dict.items():
            eigenvals = [-1 if k == '1' else 1 for k in key]
            eigenvals_prod = np.prod(eigenvals)
            measurement_exp += eigenvals_prod * value / sum(samples_dict.values())
        exp += measurement_exp * coefficient
    return exp


def merge_x(indices_qaoa, x_qaoa, indices_logn, x_logn):
    x = np.empty(len(x_qaoa) + len(x_logn))
    for idx1, index1 in enumerate(indices_qaoa):
        x[index1] = x_qaoa[idx1]
    for idx2, index2 in enumerate(indices_logn):
        x[index2] = x_logn[idx2]
    return x


def assemble_solution(Q_full, qaoa_bitstring, D, indices_qaoa, indices_logn):
    x_qaoa = [int(v) for v in qaoa_bitstring]
    x_logn = [1 if d < 0 else 0 for d in D]
    x = np.asarray(merge_x(indices_qaoa, x_qaoa, indices_logn, x_logn))
    energy = x @ Q_full @ x
    return x, energy


def cutoff_pauli_decomposition(measurement2_values, measurement2_paulis, cutoff=None):
    val_sum = 0
    abs_sum = sum([abs(value) for value in measurement2_values])
    effective_measurement2_values, effective_measurement2_paulis = [], []
    keep_count, cutoff_count = 0, 0
    for index in sorted(range(len(measurement2_values)), key=lambda index: -abs(measurement2_values[index])):
        val = measurement2_values[index]
        if cutoff is None or val_sum <= cutoff * abs_sum:
            val_sum += abs(val)
            effective_measurement2_values.append(val)
            effective_measurement2_paulis.append(measurement2_paulis[index])
            keep_count += 1
        else:
            cutoff_count += 1
    return effective_measurement2_values, effective_measurement2_paulis


def qaoa_circuit(gammas, betas, h1, J1, indices_qaoa, edge_list=None):
    wmax = max(
        np.max(np.abs(list(h1.values()))), np.max(np.abs(list(J1.values())))
    )
    if edge_list is not None:
        mixer_graph = nx.from_edgelist(edge_list)
        wire_map = {idx_qaoa: idx for idx, idx_qaoa in enumerate(indices_qaoa)}
    else:
        mixer_graph = None
    num_qubits1 = len(h1)
    wires1 = list(range(num_qubits1))
    p = len(gammas)
    if mixer_graph is None:
        for i in range(num_qubits1):
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
        for ki, v in h1.items():
            qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
        for kij, vij in J1.items():
            qml.CNOT(wires=[kij[0], kij[1]])
            qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
            qml.CNOT(wires=[kij[0], kij[1]])
        # mixer
        if edge_list is None:
            for i in range(num_qubits1):
                qml.RX(-2 * betas[layer], wires=i)
        else:
            mixer_op = qaoa.xy_mixer(mixer_graph)
            mixer_op = qml.map_wires(mixer_op, wire_map=wire_map)
            qml.apply(mixer_op)
    measurement_wires1 = wires1
    return num_qubits1, measurement_wires1


def logn_circuit(D, measurement2_pauli, wire_shift=0):
    N2 = len(D)
    logN2 = int(np.log2(N2))
    num_qubits2 = logN2 + 1
    wires2 = [i + wire_shift for i in range(num_qubits2)]
    for i in range(num_qubits2):
        qml.Hadamard(wires=i + wire_shift)
    C = [1. for q in range(2 ** logN2)]
    qml.DiagonalQubitUnitary(D + C, wires=wires2)
    operands = measurement2_pauli.operands
    measurement_wires2 = []
    for operand in operands:
        wire = operand.wires[0]
        name = operand.name
        if name == 'Identity':
            pass
        if name == 'PauliX':
            measurement_wires2.append(wire)
            qml.Hadamard(wires=wire)
        if name == 'PauliY':
            measurement_wires2.append(wire)
            qml.PhaseShift(phi=3 / 2 * np.pi, wires=wire)
            qml.Hadamard(wires=wire)
        if name == 'PauliZ':
            measurement_wires2.append(wire)
    return num_qubits2, measurement_wires2


def circuit_qaoa_logn_measure(gammas, betas, h1, J1, D, indices_qaoa, edge_list, measurement2_pauli):
    # Circuits
    num_qubits1, measurement_wires1 = qaoa_circuit(gammas, betas, h1, J1, indices_qaoa, edge_list)
    num_qubits2, measurement_wires2 = logn_circuit(D, measurement2_pauli, wire_shift=num_qubits1)

    # Measurement
    all_measurement_wires = measurement_wires1 + measurement_wires2
    return [qml.sample(qml.PauliZ(i)) for i in all_measurement_wires]  # qml.sample(wires=all_measurement_wires)


def circuit_qaoa_measure(gammas, betas, h1, J1, indices_qaoa, edge_list):
    # Circuits
    _, measurement_wires1 = qaoa_circuit(gammas, betas, h1, J1, indices_qaoa, edge_list)

    # Measurement
    return [qml.sample(qml.PauliZ(i)) for i in measurement_wires1]  # qml.sample(wires=all_measurement_wires1)


def preprocess_problem(instance, p, t, use_custom_mixer, rng, N_qaoa=25):
    # prepare matrix
    N = len(instance.variables)
    assert N > N_qaoa, f"problem is too small, run qaoa instead: {N} <= {N_qaoa}"
    Q_full = np.zeros((N, N))
    for (i, j), value in instance.qubo_matrix.items():
        Q_full[i, j] = value

    # choose indices: split in the order they arrive
    indices = np.arange(N)
    indices_qaoa = indices[:N_qaoa]
    indices_logn = indices[N_qaoa:]

    # setup quantum algorithm
    if p > 1:
        gammas = np.linspace(0, 1, p)
        betas = gammas[::-1]
    else:
        gammas = [0.5]
        betas = [0.5]
    if use_custom_mixer:
        mixer_graph = build_mixer_graph(instance, allowed_indices=indices_qaoa)
        edge_list = [edge for edge in mixer_graph.edges]
    else:
        edge_list = None

    # intialize theta
    thetas = rng.uniform(0, 2 * np.pi, t)

    # return
    return Q_full, edge_list, (indices_qaoa, indices_logn), (gammas, betas, thetas)


def evaluate_qaoa_logn(Q_full, indices_qaoa, indices_logn, x, gammas, betas, edge_list, D, cutoff, dev_kwargs,
                       total_shots, verbose):
    Q_full = (Q_full + np.transpose(Q_full)) / 2

    # prepare meaurement setup
    N1 = len(indices_qaoa)
    N2 = len(indices_logn)
    num_qubits1 = N1
    wires1 = list(range(N1))
    wire_shift = num_qubits1
    logN2 = int(np.ceil(np.log2(N2)))
    num_dummy_indices = 2 ** logN2 - N2
    num_qubits2 = logN2 + 1
    wires2 = [i + wire_shift for i in range(num_qubits2)]

    # prepare ising operators
    h1, J1, _ = from_Q_to_Ising(Q_full, indices_qaoa, indices_logn, x)
    J2, _ = from_Q_to_quadratic_Ising(Q_full, indices_logn, indices_qaoa, num_dummy_indices, x)

    # generate operator and decompose it
    op = generate_op(N2 + num_dummy_indices, J2, wires=wires2)
    decomposition = op.decomposition()
    measurement2_values, measurement2_paulis = decomposition[0].terms()
    measurement2_values, measurement2_paulis = cutoff_pauli_decomposition(measurement2_values, measurement2_paulis,
                                                                          cutoff)

    # choose effective shots
    effective_shots = max(total_shots // len(measurement2_values), MINIMUM_SHOTS_PER_CIRCUIT)

    # run measurements
    total_shots = 0
    total_qaoa_sample_dict = {}
    total_logn_sample_list = []
    min_energy, min_x = np.inf, None
    for measurement2_value, measurement2_pauli in tqdm(zip(measurement2_values, measurement2_paulis),
                                                       total=len(measurement2_paulis),
                                                       desc=f"run circuits [{effective_shots} shots/circuit]",
                                                       disable=not verbose):

        # run circuit
        dev = qml.device(**dev_kwargs, wires=len(wires1 + wires2), shots=effective_shots)
        qnode = qml.QNode(circuit_qaoa_logn_measure, dev, interface=None)
        sample_list = qnode(gammas, betas, h1, J1, D, indices_qaoa, edge_list, measurement2_pauli)

        # prepare processing
        measurement_wires1 = list(range(N1))
        measurement_wires2 = []
        for operand in measurement2_pauli.operands:
            wire = operand.wires[0]
            name = operand.name
            if name == 'Identity':
                pass
            if name == 'PauliX':
                measurement_wires2.append(wire)
            if name == 'PauliY':
                measurement_wires2.append(wire)
            if name == 'PauliZ':
                measurement_wires2.append(wire)

        # process solution
        samp_dict = process_sample_list(sample_list)
        total_shots += sum(samp_dict.values())
        samp_qaoa, samp_logn = {}, {}
        for key, value in samp_dict.items():
            # qaoa
            samp_qaoa_key = ''.join([key[idx] for idx in measurement_wires1])
            if samp_qaoa_key not in samp_qaoa:
                samp_qaoa[samp_qaoa_key] = 0
            samp_qaoa[samp_qaoa_key] += value
            x, energy = assemble_solution(Q_full, samp_qaoa_key, D, indices_qaoa, indices_logn)
            if energy < min_energy:
                min_energy = energy
                min_x = x

            # logn
            samp_logn_key = ''.join([key[N1 + idx] for idx in range(len(measurement_wires2))])
            if samp_logn_key not in samp_logn:
                samp_logn[samp_logn_key] = 0
            samp_logn[samp_logn_key] += value  # should be equally balanced (hadamard + only phase changes)

        # collect total qaoa samples
        for key, value in samp_qaoa.items():
            if key not in total_qaoa_sample_dict:
                total_qaoa_sample_dict[key] = 0
            total_qaoa_sample_dict[key] += value

        # collect total logn samples
        total_logn_sample_list.append((measurement2_value, samp_logn))

    # evaluate expectation value
    logn_exp = get_exp_from_samples_collection(total_logn_sample_list)

    return total_qaoa_sample_dict, logn_exp, min_energy, min_x


def evaluate_qaoa_logn_sim(Q_full, indices_qaoa, indices_logn, x, gammas, betas, edge_list, D, dev_kwargs, total_shots,
                           verbose):
    min_energy, min_x = np.inf, None

    # prepare ising operators
    h1, J1, _ = from_Q_to_Ising(Q_full, indices_qaoa, indices_logn, x)
    Q2 = Q_full[np.ix_(indices_logn, indices_logn)]

    # run circuit
    dev = qml.device(**dev_kwargs, wires=len(h1), shots=total_shots)
    qnode = qml.QNode(circuit_qaoa_measure, dev, interface=None)
    sample_list = qnode(gammas, betas, h1, J1, indices_qaoa, edge_list)

    # process measurement results
    samp_dict = process_sample_list(sample_list)
    for key, _ in samp_dict.items():
        x, energy = assemble_solution(Q_full, key, D, indices_qaoa, indices_logn)
        if energy < min_energy:
            min_energy = energy
            min_x = x

    # evaluate expectation value (simulated)
    x2 = np.array([1 if d < 0 else 0 for d in D])
    logn_exp = x2 @ Q2 @ x2

    return samp_dict, logn_exp, min_energy, min_x


def get_qaoa_logn_objective(thetas, history, best_sol, Q_full, indices_qaoa, indices_logn, gammas, betas, edge_list,
                            cutoff,
                            use_logn_sim, dev_kwargs, total_shots, verbose):
    # recover x vector
    x = history[list(history.keys())[-1]]['x']

    # precalculate diagonal
    D = eval_diag(thetas, len(indices_logn))

    # evaluate
    if not use_logn_sim:
        qaoa_samples, logn_exp, min_energy, min_x = evaluate_qaoa_logn(Q_full, indices_qaoa, indices_logn, x, gammas,
                                                                       betas, edge_list, D, cutoff, dev_kwargs,
                                                                       total_shots, verbose)
    else:
        qaoa_samples, logn_exp, min_energy, min_x = evaluate_qaoa_logn_sim(Q_full, indices_qaoa, indices_logn, x,
                                                                           gammas, betas, edge_list, D, dev_kwargs,
                                                                           total_shots, verbose)

    # check improvement
    if min_energy < best_sol['min_energy']:
        best_sol['min_energy'] = min_energy
        best_sol['min_x_vec'] = min_x

    # qaoa
    best_qaoa_key, _ = sorted([item for item in qaoa_samples.items()], key=lambda item: -item[1])[0]
    x, energy = assemble_solution(Q_full, best_qaoa_key, D, indices_qaoa, indices_logn)

    # store history
    history[len(history)] = dict(x=x.tolist(), energy=energy, logn_exp=logn_exp, qaoa_samples=qaoa_samples)

    # return objective
    return logn_exp


def run_qaoalogn(instance, p, t, da_maxiter, use_custom_mixer, cutoff, use_logn_sim, dev_kwargs, total_shots,
                 callback_fun=None, N_qaoa=25, seed=None, verbose=True):
    # random generator
    rng = np.random.RandomState(seed)
    solver_seed = rng.randint(0, 2 ** 31)

    # preprocess problem: full instance is processed
    Q_full, edge_list, (indices_qaoa, indices_logn), (gammas, betas, thetas) = preprocess_problem(instance, p, t,
                                                                                                  use_custom_mixer, rng,
                                                                                                  N_qaoa)

    # initial guess
    D = eval_diag(thetas, len(indices_logn))
    x_qaoa_init = rng.choice([0, 1], size=len(indices_qaoa)).tolist()
    x_init, energy_init = assemble_solution(Q_full, x_qaoa_init, D, indices_qaoa, indices_logn)
    best_sol = dict(min_energy=energy_init, min_x_vec=x_init)
    history = dict(init=dict(x=x_init, energy=energy_init))

    # start optimization
    args = [history, best_sol, Q_full, indices_qaoa, indices_logn, gammas, betas, edge_list, cutoff, use_logn_sim,
            dev_kwargs, total_shots, verbose]
    res = dual_annealing(get_qaoa_logn_objective, args=args, bounds=[(0, 2 * np.pi)] * len(thetas),
                         no_local_search=True, x0=thetas, maxiter=da_maxiter, seed=solver_seed, callback=callback_fun)

    # convert
    x = dict()
    for index, name in instance.model.reverse_mapping.items():
        x[name] = best_sol['min_x_vec'][index]

    # return
    return x, (best_sol, history, res)

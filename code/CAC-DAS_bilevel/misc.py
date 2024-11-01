import numpy as np
from itertools import product
import os

def generate_symmetric_matrix(N):
    # Step 1: Generate a random matrix with +1 and -1
    random_matrix = np.random.choice([-1, 1], size=(N, N))
    
    # Step 2: Make the matrix symmetric by copying the upper triangular part to the lower triangular part
    symmetric_matrix = np.triu(random_matrix) + np.triu(random_matrix, 1).T
    symmetric_matrix = symmetric_matrix - np.diag(np.diag(symmetric_matrix))

    return symmetric_matrix

def ising_hamiltonian(x, J, h):
    #H = -0.5*np.sum(np.outer(x, x) * J) - np.sum(x * h)
    H = -0.5 * np.sum(x * (J @ x) , 0) - np.sum(x * h[:,None],0)
    return H

def brute_force_ising(J, h):
    N = J.shape[0]
    
    # Generate all possible spin configurations (+1, -1) for N spins
    all_configs = list(product([-1, 1], repeat=N))
    
    # Initialize ground state energy and configuration
    ground_state_energy = np.inf
    ground_state_config = None
    
    # Iterate over all possible configurations
    for config in all_configs:
        config = np.array(config)[:,None]
        energy = ising_hamiltonian(config, J, h)[0]
        
        # Update ground state if a lower energy is found
        if energy < ground_state_energy:
            ground_state_energy = energy
            ground_state_config = config
    
    return ground_state_energy, ground_state_config

def qubo_hamiltonian(x, Q, g):
    #H = np.sum(np.outer(x, x) * Q) + np.sum(x * g)
    H = np.sum(x * (Q @ x) , 0) + np.sum(x * g[:,None],0)
    return H

def brute_force_qubo(Q,g):
    N = Q.shape[0]
    
    # Generate all possible spin configurations (+1, -1) for N spins
    all_configs = list(product([0, 1], repeat=N))
    
    # Initialize ground state energy and configuration
    ground_state_energy = np.inf
    ground_state_config = None
    
    # Iterate over all possible configurations
    for config in all_configs:
        config = np.array(config)[:,None]
        energy = qubo_hamiltonian(config, Q, g)[0]
        
        # Update ground state if a lower energy is found
        if energy < ground_state_energy:
            ground_state_energy = energy
            ground_state_config = config
    
    return ground_state_energy, ground_state_config

def ising_to_qubo(J, h):

    N = len(h)
    o = np.ones(N)
    
    Q = - 2 * J
    g = -2*h + 2 * J @ o
    
    B = o.T @ Q / 4 @ o + o.T @ g / 2
  
    return Q, g, B

def qubo_to_ising(Q, g):

    N = len(g)
    o = np.ones(N)
    
    J = - Q / 2
    h = - (Q / 2 @ o + g / 2)
    B = o.T @ Q / 4 @ o + o.T @ g / 2
    
    return J, h, B

def load_GSET_J(N,i):
    
    M = np.loadtxt(f'./../Data/GSET_{N}_21/GSET_{N}_21_{i+1}')
    
    J = np.zeros((N,N))
    for l in M:
        J[int(l[0])-1,int(l[1])-1] = l[2]
        
    J = J + J.T
    
    Hv = np.loadtxt(f'./../Data/GSET_{N}_21/GSET_{N}_21_SOL')
    H = Hv[i]
    
    return J
    
def load_GSET_H0(N,i,w):
    path = f'./../Data/GSET_{N}_21/GSET_{N}_21_SOL'
    file = open ( path , 'r')
    Cv = np.array([[float(num) for num in line.split(' ')] for line in file ])
    C0 = float(Cv[i])
    H0 = -4*C0 - np.sum(w)
    H0 = H0/2
    return H0

def save_matrix(Q, folder_path, name, thr):
    # Find the non-zero indices and values in Q
    Q[np.abs(Q)<thr]=0
    i, j = np.nonzero(Q)
    values = Q[i, j]
    
    # Create the Mx3 matrix A where M is the number of non-zero elements
    A = np.vstack((i, j, values)).T
    
    # Create the file path
    file_path = os.path.join(folder_path, f'{name}.txt')
    
    # Save matrix A in the text file
    np.savetxt(file_path, A, fmt='%d %d %f')
    
    return A, file_path

def load_matrix(file_path, N):
    # Load the Mx3 matrix from the file, skipping the header
    A = np.loadtxt(file_path, dtype=float)
    
    # Create an NxN matrix initialized with zeros
    Q_loaded = np.zeros((N, N))
    
    # Populate the NxN matrix using the data from A
    for row in A:
        i, j, value = int(row[0]), int(row[1]), row[2]
        Q_loaded[i, j] = value
    
    return Q_loaded

def convert_matrix(A):
    
    # Create an NxN matrix initialized with zeros
    Q_loaded = np.zeros((N, N))
    
    # Populate the NxN matrix using the data from A
    for row in A:
        i, j, value = int(row[0]), int(row[1]), row[2]
        Q_loaded[i, j] = value
    
    return Q_loaded
    
def reload_model(name_instance):
    
    Q = []
    g = []
    J = []
    h = []
    for i in range(5):
        
        file_path = f'./model/full_qubo/{name_instance}/g_{i}.txt'
        gi = np.loadtxt(file_path)
        g.append(gi.tolist())
        N = len(gi)
        
        file_path = f'./model/full_qubo/{name_instance}/Q_{i}.txt'
        Qi = load_matrix(file_path, N)
        Q.append(Qi.tolist())
        
        file_path = f'./model/full_Ising/{name_instance}/h_{i}.txt'
        hi = np.loadtxt(file_path)
        h.append(hi.tolist())
        
        file_path = f'./model/full_Ising/{name_instance}/J_{i}.txt'
        Ji = load_matrix(file_path, N)
        J.append(Ji.tolist())
        
    return Q,g,J,h
        
if __name__ == '__main__':

    if True:
        
        # Example usage
        N = 5  # Size of the matrix
        J = generate_symmetric_matrix(N)
        print(J)
        
        ground_state_energy, ground_state_config = brute_force_ising(J)
    
        print("Ground state configuration:", ground_state_config)
        print("Ground state energy:", ground_state_energy)

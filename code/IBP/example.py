import numpy as np
import bp_random_subtree as IBP





N = 500
M = 5000

#Random QUBO Problem
J_IDX = np.zeros((2,M), dtype = np.int64)
J_cost = np.zeros(M)

for m in range(M):
	i = int(np.random.rand()*N)
	j = (i  + int(np.random.rand()*(N-1))) % N
	J_IDX[:,m] = [i,j]
	J_cost[m] = 2*np.random.rand() - 1

h = 2*np.random.rand(N) - 1

#QUBO cost
def cost(x):
	
	return np.sum(x*h.reshape(N,1), axis = 0) + np.sum(x[J_IDX,:][0,:]*x[J_IDX,:][1,:]*J_cost.reshape(M,1), axis = 0)

	

#Iterate

#num parrallel runs
R = 10

#temperature
beta = 100.0

x = (np.random.rand(N,R) > 0.5)*1

for i in range(100):
	
	x = IBP.IBP_iterate(x, J_IDX, J_cost, h, N, R, beta, num_step = 1)
	print(cost(x))

	

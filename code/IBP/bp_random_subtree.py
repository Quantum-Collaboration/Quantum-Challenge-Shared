
import matplotlib.pyplot as plt
import numpy as np
import bp_log as bp
import time
#import bp
from numba import jit, njit
import numba

from qubovert import QUBO
from qubovert import boolean_var

from qubovert.sim import anneal_qubo

@jit(nopython = True)
def gen_subprob(IDX, C, N, M, x, R_par, used, idx_used, verbose = 0):
	N_ = len(idx_used)
	L = IDX.shape[1]
	IDX_ = -np.ones((M+N, L), dtype = np.int64)
	C_ = []
	
	k_used = 0
	
	local_field = np.zeros((len(idx_used), R_par))
	
	num_sum = 0
	for k, (idx,c) in enumerate(zip(IDX, C)):
		
		edge_unused = 1
		
		for i in idx:
			if(i >= 0):
				edge_unused = (1 - used[i])*edge_unused
		
		
		# if(edge_unused != 1):
# 			print("numba", k, edge_unused)
		if(edge_unused == 1):
			
			continue
		
		# if(used[idx[0]] >= 1 and used[idx[1]] >= 1):
# 			print([used[i] for i in idx])
		
		idx_ = -np.ones(L, dtype = np.int64)
		idx_len = 0
		SAT = np.ones(R_par)
		
		factor_idx = 0
		#print(idx)
		for _,i in enumerate(idx):
			
			if(i >= 0):
				if(used[i] == 0):
					SAT = SAT*x[i]
					#print(x[i][0])
				else:
					idx_[factor_idx] = idx_used.index(i)
					factor_idx +=1 
					idx_len += 1
		
		
		if( (np.max(SAT) > 0 or True) and idx_len > 1):
			
			IDX_[k_used, :] = idx_
			k_used += 1
			
			C_.append(c*SAT)
			#print(k_used, len(C_))
		
		
		
		if( (np.max(SAT) > 0 or True) and idx_len == 1):
			#print("added local field",k, idx_[0], c*SAT[0])
			num_sum += c
			local_field[idx_[0],:] += c*SAT
		
	#print(num_sum)
	
	for i in range(len(idx_used)):
		IDX_[k_used, :] = np.array([i] + [-1]*(L-1))
		k_used += 1
		
		C_.append(local_field[i,:])
		#print(k_used, len(C_))
	#print("local field numba", local_field[:,0])
	
	IDX_ = IDX_[:k_used, :]
	
	# print(idx_used)
# 	print(IDX_)
# 	print(C_)
	
	return IDX_, C_

def random_subproblem(IDX, C, N, M, factor_IDX, x, R_par = 1, seed = None, verbose = 0):
	
	
	
	idx_used = []
	
	degree = np.zeros(N, dtype = np.int32)
	used = np.zeros(N, dtype = np.int32)
	
	seed_node = 0
	while(np.min(degree + (used == 1)*10) <= 1 and len(idx_used) < N):
		
		p = (degree == 1)*1.0*(used == 0)
		if(np.sum(p) == 0):
			# print('sum p zero')
# 			print(p)
# 			print(degree)
# 			print(np.min(degree))
			p = np.ones(N)*(used == 0)
			if(seed_node >= 1):
				break
			seed_node += 1
		if(len(idx_used) > 500):
			break
		
		p = p/np.sum(p)
		
		ind = np.random.choice(range(N), p = p)
		
		if(not seed is None and len(idx_used) == 0):
			ind = seed % N
		#print(ind)
		for a in factor_IDX[ind]:
			for v in IDX[a]:
				if(v != ind):
					degree[v] += 1
		
		# print(p)
# 		print(degree)
# 		print(np.min(degree))
		
		idx_used.append(ind)
		used[ind] = 1
	tme = time.time()
	
	L = max([len(idx) for idx in IDX])
	IDX_np = -np.ones((M, L), dtype = np.int64)
	
	for m in range(M):
		IDX_np[m,:len(IDX[m])] = IDX[m]
	
	IDX_, C_ = gen_subprob(IDX_np, np.array(C), N, M, x, R_par, used, idx_used, verbose = verbose)
	
	IDX_ = [[_ for _ in idx if _>=0] for idx in IDX_]
	IDX_numba = IDX_
	C_numba = C_
	
	#for numba debug
	if(0):
		IDX_ = []
		C_ = []
		local_field = np.zeros((len(idx_used), R_par))
		num_sum = 0
		
		for k, (idx,c) in enumerate(zip(IDX, C)):
		
			edge_unused = 1
			for i in idx:
				if(i >= 0):
					edge_unused = (1 - used[i])*edge_unused
			if(edge_unused != 1):
				print(k, edge_unused)
			if(edge_unused == 1):
				continue
			# if(used[idx[0]] >= 1 and used[idx[1]] >= 1):
	# 			print([used[i] for i in idx])
			
			idx_ = []
			idx_len = 0
			SAT = np.ones(R_par)
			print(idx)
			for _,i in enumerate(idx):
				if(i >= 0):
					if(used[i] == 0):
						SAT = SAT*x[i]
						print(x[i][0])
					else:
						idx_.append(idx_used.index(i))
						
			
			if(np.max(SAT) > 0 and len(idx_) > 1):
				IDX_.append(idx_)
				
				
				C_.append(c*SAT)
				#print(k_used, len(C_))
			
			#print(k)
			if(k % 100 == 0):
				print("c", c)
				#print("SAT" ,SAT)
			if( (np.max(SAT) > 0 or True) and len(idx_) == 1):
				
				print("added local field",k, idx_[0], c*SAT[0])
				num_sum += c*SAT[0]
				local_field[idx_[0],:] += c*SAT
		
		for i in range(len(idx_used)):
			IDX_.append(np.array([i]))
			
			
			C_.append(local_field[i,:])
			#print(k_used, len(C_))
			
		print("num sum", num_sum)
		print("local field", local_field[:,0])
		
		for idx1, idx2 in zip(IDX_, IDX_numba):
			print(idx1, idx2)
		
		for c1, c2 in zip(C_, C_numba):
			print(c1 - c2)
	
	
	if(verbose >= 2):
		print("calc C", time.time() - tme)
	
	N_ = len(idx_used)
	M_ = len(IDX_)
	
	return IDX_, C_, N_, M_, idx_used





def run_subtree_bp(IDX, C, N, M, beta = 1.0, beta_anneal = 10.0, nt = 800*500, bp_step = 6, E0 = 0, check_valid_soln = None, R_par = 1, MH = True, verbose = 0, initial_x = None, random_seed = False):
	
	factor_IDX = [[0 for j in range(0)] for i in range(N)]
	
	
	for a in range(M):
		#print(IDX[a])
		for v in IDX[a]:
			if(v >= 0):
				factor_IDX[v].append(a)
	
	
	if(initial_x is None):
		x = (np.random.rand(N) > 0.5)*1
		if(R_par > 1):
			x = (np.random.rand(N,R_par) > 0.5)*1
	else:
		x = initial_x
	
	
	def E(x):
		E_ = 0
		for idx, c in zip(IDX, C):
			prd = 1
			for v in idx:
				prd *= x[v]
			
			E_ += prd*c
		
		return E_
	
	E_traj = []
	valid_traj = []
	time_prev = time.time()
	
	for i in range(nt):
		tme = time.time()
		
		seed = i
		if(random_seed):
			seed = None
		IDX_, C_, N_, M_, idx_used = random_subproblem(IDX, C, N, M, factor_IDX, x, R_par = R_par, seed = seed, verbose = verbose)
		if(verbose >= 1):
			print("generate subprob" , time.time() - tme)
			
			print("suprob size", N_, "iter number", i, "beta", beta/beta_anneal**(1 - float(i)/nt))
		
		tme = time.time()
		current_state = None
		if(MH):
			current_state = np.zeros((N_,R_par))
			for i_,i2 in enumerate(idx_used):
				current_state[i_,:] = x[i2,:]
			
		pv, x_ = bp.run_bp_wrap(IDX_, C_, N_, M_, beta/beta_anneal**(1 - float(i)/nt), beta_anneal = 1.0, nt = bp_step, sample = True, R_par = R_par, current_state = current_state, verbose = verbose)
		if(verbose >= 2):
			print("BP time " , time.time() - tme)
		#print(pv)
		tme = time.time()
		for ii in range(N_):
			#print(idx_used[i])
			#x[idx_used[i]] = (pv[i,0] > 0.5 + 0*np.random.rand())*1
			x[idx_used[ii]] = x_[ii]
		
		
		if(verbose >= 2):
			print("postprocess time " , time.time() - tme)
		
		tme = time.time()
		
		if(i % 50 == 0):
			val = None
			if(not check_valid_soln is None):
				if(R_par > 1):
					
					val = [0 for r in range(R_par)]
					if(i > nt*0.0):
						val = [check_valid_soln(x[:,r]) for r in range(R_par)]
				else:
					val = check_valid_soln(x)
			
			valid_traj.append(val)
			E_ = E(x)
			E_traj.append(E_)
			
			if(val is None):
				val = []
			
			if(verbose >= 1):
				print("E (avg)", np.average(E_) + E0, "(best)" , np.min(E_) + E0, "val % ", np.average(np.array(val)),  " tme ", time.time() - time_prev)
			time_prev = time.time()
			
			#if(verbose >= 0):
			if(False):
				if(R_par <= 1000):
					print("E", E_ + E0)
				else:
					print([E + E0 for (E,v) in zip(E_, val) if v])
			
		if(verbose >= 2):
			print("output time", time.time() - tme)
	
	
	return x, {"E": E_traj, "val": valid_traj}


if __name__ == "__main__":
	#random tree
	N = 500
	
	
	IDX = []
	C = []
	
	np.random.seed(0)
	
	# for i in range(1,N):
# 		j = int(np.random.rand()*i)
# 		#j = i-1
# 		IDX.append([i,j])
# 		C.append(1 - np.random.rand()*2)
	local_field = np.zeros(N)
	for i in range(int(0.02*N*(N-1)//2)):
		i = int(np.random.rand()*N)
		j = int(np.random.rand()*N)
		if(i != j):
			IDX.append([i,j])
			Jij = -1 - np.random.rand()*2*0
			C.append(-2*Jij)
			local_field[i] += 1*Jij
			local_field[j] += 1*Jij
	
	for i in range(N):
		IDX.append([i])
		C.append(1*(1 - np.random.rand()*2)*2 + local_field[i])
	
	IDX = [[int(x) for x in idx] for idx in IDX]
	for idx in IDX:
		print(idx)
	
	
	M =  len(IDX)
	beta = 10000.0
	beta_anneal = 1 + 0*50.0
	
	x_BP, info = run_subtree_bp(IDX, C, N, M, beta, beta_anneal = beta_anneal, nt = 100, R_par = 500)
	
	
	
	IDX = [[int(x) for x in idx if x >= 0] for idx in IDX]
	
		
	
	
	x = {i: boolean_var('x(%d)' % i) for i in range(N)}
	model = 0
	
	for idx, c in zip(IDX, C):
		prod = 1
		for v in idx:
			prod = prod*x[v]
		prod = prod*c
		
		model = model + prod
	
	
	print("SA")
	
	res = anneal_qubo(model, num_anneals=500, anneal_duration=100*400, temperature_range = (1.0*beta_anneal/beta, 1.0/beta))
	model_solution = res.best.state
	
	
	SA_solns = sorted([s for s in res if  model.is_solution_valid(s.state)])
	print([s.value for s in SA_solns])
	SA_best = SA_solns[0].state
	
	
	def E(x):
		E_ = 0
		for idx, c in zip(IDX, C):
			prd = 1
			for v in idx:
				prd *= x[v]
			
			E_ += prd*c
		
		return E_
	
	SA_best_vec = [SA_best['x(%d)' % i] for i in range(N)]
	print(SA_best_vec)
	print(E(SA_best_vec))
	
	BP_soln = x_BP
	print("BP_soln", BP_soln)
	
	BP_E = sorted(E(BP_soln))
	print(BP_E)
	print(np.min([np.min(E) for E in info["E"]]))
	
	plt.plot(range(len(info["E"])), info["E"])
	plt.show()
	plt.close()
	
	
	plt.yscale("symlog")
	plt.plot(range(len(info["E"])), info["E"] - np.min(info["E"]))
	plt.show()
	plt.close()
	
	SA_E = [s.value for s in SA_solns]
	plt.scatter(BP_E, SA_E)
	mn = min(np.min(BP_E), np.min(SA_E))
	mx = max(np.max(BP_E), np.max(SA_E))
	plt.plot([mn, mx], [mn, mx])
	
	plt.show()
	plt.close()
	
		
#Single IBP Iteration
#INPUT
#x: current state, numpy array, shape= (N,R)   val= {0,1}
#J_IDX: indices of nonzero elements in QUBO matrix shape= (2,M) val = 0...N
#J_cost: cost of nonzero elements in QUBO matrix shape= (M) val = float
#h: QUBO local field shape= (N) val = float
#N: number of vars
#R: number of parralel runs
#beta: inverse temperature to sample from
#num_step (optional): number of iterations, default to 1
#OUTPUT:
#x: new state, numpy array, shape= (N,R)   val= {0,1}

def IBP_iterate(x, J_IDX, J_cost, h, N, R, beta, num_step = 1, random_seed = True):
	
	IDX = []
	C = []
	M = J_IDX.shape[1]
	
	for m in range(M):
		IDX.append(J_IDX[:,m])
		C.append(J_cost[m])
	
	for i in range(N):
		IDX.append([i])
		C.append(h[i])
	
	#x_BP, info = run_subtree_bp(IDX, C, N, M, beta, beta_anneal = 1, nt = num_step, R_par = R, verbose = 0, initial_x = x, random_seed = random_seed)
	x_BP, info = run_subtree_bp(IDX, C, N, M+N, beta, beta_anneal = 1, nt = num_step, R_par = R, verbose = 0, initial_x = x, random_seed = random_seed)
	
	return x_BP

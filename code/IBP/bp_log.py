import numpy as np
from numba import jit, njit
import numba
from numba.typed import List
import time


from qubovert import QUBO
from qubovert import boolean_var

from qubovert.sim import anneal_qubo

def run_bp_wrap(IDX, C, N, M, beta_, beta_anneal = 1.0, nt = 100, sample = False, R_par = 1, current_state = None, verbose = 0):
	
	width = max(len(idx) for idx in IDX)
	IDX = [[idx[i] if i < len(idx) else -1 for i in range(width)] for idx in IDX]
	IDX = np.array(IDX, dtype = np.int64)
	
	if(R_par == 1):
		C = [float(c) for c in C]
	else:
		C = [np.array(c, dtype = np.float64) for c in C]
	
	tme = time.time()
	
	if(R_par == 1):
		out, mu, mu_iuv, factor_IDX, K, mu_v, mu_a, k_dict = run_bp(IDX, C, N, M, beta_, beta_anneal, nt, sample)
	
	if(R_par > 1):
		out, mu, mu_inv, factor_IDX, K, mu_v, mu_a, k_dict = run_bp_par(IDX, C, N, M, beta_, R_par, beta_anneal, nt, sample, verbose = verbose)
	
	if(verbose >= 2):
		print("BP iterations" , time.time() - tme, " (beta = %f)" % beta_)
	
	tme = time.time()
	p_v = out
	
	def f(a, x):
		
		out =  (-beta_*(np.prod(x))*C[a])
		if(R_par > 1):
			out = (-beta_*(np.prod(x, axis = 0)))*C[a]
		
		#print(x, out)
		return out
	
	if(sample):
		x = -np.ones(N)
		
		if(R_par > 1):
			x = -np.ones((N, R_par))
		
		def samp():
			
			indicator = None
			
			
			if(R_par == 1):
				indicator = (x < 0)*1
			else:
				indicator = (np.min(x, axis = 1) < 0)*1
			
			
			p_start = indicator
			
			index_start = np.random.choice(range(N), p = p_start/np.sum(p_start))
			
			index_set = []
			
			
			if(current_state is None):
				if(R_par == 1):
					x[index_start] = 1*(p_v[index_start,0] > np.random.rand())
				else:
					x[index_start, :] = 1*(p_v[index_start,0, :] > np.random.rand(R_par))
			else:
				#MH jump probability
				p_v_MH = p_v[index_start,:, :]/np.maximum(p_v[index_start,0, :], p_v[index_start,1, :])
				p_jump = p_v_MH[0,:]*(1 - current_state[index_start,:]) + p_v_MH[1,:]*(current_state[index_start,:])
				if(verbose >= 3):
					print(p_v_MH[:,:5])
					print(current_state[index_start,:5])
					print(p_jump[:5])
				
				jump = p_jump > np.random.rand(R_par)
				x[index_start, :] = current_state[index_start,:]*(1-jump) + (1-current_state[index_start,:])*jump
			
			
			index_set.append(index_start)
			
			#print(p_v[index_start,0], p_v[index_start,1])
			
			# mu_inv = []
# 			for k in range(K):
# 				mui = np.zeros(2)
# 				if(R_par > 1):
# 					mui = np.zeros((2, R_par))
# 				
# 				for a in factor_IDX[mu_v[k]]:
# 					if a != mu_a[k]:
# 						if(R_par == 1):
# 							mui += mu[k_dict[(a, mu_v[k])],:]
# 						else:
# 							mui += mu[k_dict[(a, mu_v[k])],:,:]
# 				
# 				mu_inv.append(mui)
			if(verbose >= 2):
				print("mu inv time" , time.time() - tme)
			
			#WARNING: may glitch for non-tree graphs
			#samples node "v_idx" and calls iterate on upsampled neighbors
			
			def iterate(v_idx, a):
				
				if(R_par == 1):
					if(x[v_idx] >= 0):
						return
				
				else:
					if(np.min(x[v_idx]) >= 0):
						return
				
				#a -> degree 2 factor node
				#conditional prob
				x_local_fixed = np.zeros(2)
				x_local_unfixed = np.zeros(2)
				
				if(R_par > 1):
					x_local_fixed = np.zeros((2, R_par))
					x_local_unfixed = np.zeros((2, R_par))
				
				
				for i,v in enumerate(IDX[a]):
					if(np.min(x[v]) >= 0):
						x_local_fixed[i] = x[v]
					else:
						x_local_unfixed[i] = 1
				
				
				#print(v_idx, IDX[a])
				#print(0*x_local_unfixed + x_local_fixed, 1*x_local_unfixed + x_local_fixed)
				f0 = f(a, 1*x_local_unfixed + x_local_fixed)
				f1 = f(a, 0*x_local_unfixed + x_local_fixed)
				
				
				
				#print(f0, f1, mu[k_dict[(a, v_idx)],0], mu[k_dict[(a, v_idx)],1])
				#print(mu_inv[k_dict[(a, v_idx)]][0], mu_inv[k_dict[(a, v_idx)]][1])
				
				# p0 = mu_inv[k_dict[(a, v_idx)]][0] + mu[k_dict[(a, v_idx)],0]
				# p1 = mu_inv[k_dict[(a, v_idx)]][1] + mu[k_dict[(a, v_idx)],1]
				
				p0 = mu_inv[k_dict[(a, v_idx)]][0] + f0
				p1 = mu_inv[k_dict[(a, v_idx)]][1] + f1
				
				if(R_par > 1):
					
					p0 = mu_inv[k_dict[(a, v_idx)]][0,:] + f0
					p1 = mu_inv[k_dict[(a, v_idx)]][1,:] + f1
				
				
				pmax = np.maximum(p0, p1)
				p0 = p0 - pmax
				p1 = p1 - pmax
				
				p0 = np.exp(p0)
				p1 = np.exp(p1)
				
				# p0 = p_v[v_idx,0]
	# 			p1 = p_v[v_idx,1]
				
				if(current_state is None):
					if(R_par == 1):
						x[v_idx] = (p0/(p0 + p1) > np.random.rand())*1
					else:
						x[v_idx, :] = (p0/(p0 + p1) > np.random.rand(R_par))*1
				else:
					
					#MH jump probability
					p_v_MH0 = p0/np.maximum(p0, p1)
					p_v_MH1 = p1/np.maximum(p0, p1)
					p_jump = p_v_MH0*(1 - current_state[v_idx,:]) + p_v_MH1*(current_state[v_idx,:])
					jump = p_jump > np.random.rand(R_par)
					x[v_idx, :] = current_state[v_idx,:]*(1-jump) + (1-current_state[v_idx,:])*jump
				
				
				for a_ in factor_IDX[v_idx]:
					if(a_ != a):
						idx = IDX[a_]
						if(len(idx) == 2 and np.min(idx) >= 0):
							b = 0
							#other_v = [v for v in idx if v != v_idx][0]
							other_v = idx[0]
							if(other_v == v_idx):
								other_v = idx[1]
							iterate(other_v, a_)
			
			
			for a_ in factor_IDX[index_start]:
				idx = IDX[a_]
				if(len(idx) == 2 and np.min(idx) >= 0):
					
					b = 0
					other_v = idx[0]
					if(other_v == index_start):
						other_v = idx[1]
					iterate(other_v, a_)
			
			#print(x)
		
		while(np.min(x) < 0):
			samp()
		
		if(verbose >= 2):
			print("sample time" , time.time() - tme)
		
		return out, x
	
	
	
	return out


#old not used
@jit(nopython = True)
def run_bp(IDX, C, N, M, beta_, beta_anneal = 1.0, nt = 100, sample = False):
	
	factor_IDX = [[0 for j in range(0)] for i in range(N)]
	
	K = 0
	
	mu_a = []
	mu_v = []
	
	k_dict = {}
	
	beta = 1.0
	
	idxlens = np.zeros(M, dtype = np.int32)
	
	for a in range(M):
		for v in IDX[a,:]:
			if(v >= 0):
				factor_IDX[v].append(a)
				K += 1
				
				mu_a.append(a)
				mu_v.append(v)
				
				k_dict[(a,v)] = K-1
				idxlens[a] += 1
	
	
	mu = np.random.rand(K,2)
	
	
	def f(a, x):
		
		out =  (-beta*(np.prod(1-x))*C[a])
		#print(x, out)
		return out
	
	mu_inv = [np.zeros(2) for i in range(K)]
	def step():
		
		#pass 1
		
		
		for k in range(K):
			mui = np.zeros(2)
			for a in factor_IDX[mu_v[k]]:
				if a != mu_a[k]:
					mui += mu[k_dict[(a, mu_v[k])],:]
			
			mu_inv[k] = mui
		
		
		#pass 2
		
		
		mu_new = np.zeros((K,2))
		
		for k in range(K):
			
			X = [np.zeros(idxlens[mu_a[k]])]
			
			for i in range(idxlens[mu_a[k]]):
				apnd = []
				for x in X:
					x = x.copy()
					x[i] = x[i] + 1
					apnd.append(x)
				X = X + apnd
			
			#print(X)
			
			sm = np.zeros(2) - 1*10.0**30
			
			def softmax(x,y):
				
				mx = 0*np.maximum(x,y)
				
				return np.log(np.exp(x - mx) + np.exp(y - mx)) + mx
			
			#to_exp_list = [(0,0.0) for i in range(0)]
			for x in X:
				mun = 0
				j = -1
				for i, v in enumerate(IDX[mu_a[k], :]):
					if(v >= 0):
						if v != mu_v[k]:
							mun += mu_inv[k_dict[(mu_a[k], v)]][int(x[i])]
						else:
							j = i
				
				to_exp = f(mu_a[k], x) + mun
				
				
				#print(x, to_exp, to_exp_reverse)
				
				#to_exp_list.append( (int(x[j]), to_exp) )
				#sm[int(x[j])] = softmax(sm[int(x[j])], to_exp)
				#sm[int(x[j])] = np.exp(sm[int(x[j])])
				mx = np.maximum(sm[int(x[j])] , to_exp)
				sm[int(x[j])] = np.log(np.exp(sm[int(x[j])] - mx) + np.exp(to_exp - mx)) + mx
				#sm[int(x[j])] = mx
				
				#sm[int(x[j])] = np.log(sm[int(x[j])])
			
			
			#mx = np.max([_[1] for _ in to_exp])
			
			# for i, e in to_exp():
# 				sm[i] += np.exp(e - mx)
			
			#print(sm)		
			mu_new[k,:] = sm
		
		
		
		for i in range(N//20+1):
			update_idx = int(np.random.rand()*K)
			
			#mu[update_idx] = mu_new[update_idx]
			
		mu_new = mu_new - np.sum(mu_new, axis = 1).reshape(-1,1)/2
		
		print(np.sum((mu - mu_new)**2))
		
		for i in range(K):
			mu[i] = mu[i]*0.0+ 1.0*mu_new[i]
		
		
		
		
	
	
	
	for i in range(nt):
		beta = beta_/beta_anneal**((nt-i)/nt)
		step()
		
		mu = mu - np.sum(mu, axis = 1).reshape(-1,1)/2
	
	#print(np.sum((mu - mu_new)**2))
	print(mu[0])
	
	p_v = np.zeros((N,2))
	
	
	for v in range(N):
		for a in factor_IDX[v]:
			p_v[v, :] += mu[k_dict[(a, v)]]
	
	#print(p_v)
	p_v = p_v - np.maximum(p_v[:,0], p_v[:,1]).reshape(-1,1)
	
	
	
	p_v = np.exp(p_v)
	#print(p_v)
	
	
	
	p_v = p_v/np.sum(p_v, axis = 1).reshape(-1,1)
	#print(mu)
	#print(p_v/np.sum(p_v, axis = 1).reshape(-1,1))
	
	#print(p_v)
	
		
		
	
	
	return p_v, mu, mu_inv, factor_IDX, K, mu_v, mu_a, k_dict







#currently used
@jit(nopython = True)
def run_bp_par(IDX, C, N, M, beta_, R, beta_anneal = 1.0, nt = 100, sample = False, verbose = 0):
	
	factor_IDX = [[0 for j in range(0)] for i in range(N)]
	
	K = 0
	
	mu_a = []
	mu_v = []
	
	k_dict = {}
	
	beta = 1.0
	
	idxlens = np.zeros(M, dtype = np.int32)
	
	for a in range(M):
		for v in IDX[a,:]:
			if(v >= 0):
				factor_IDX[v].append(a)
				K += 1
				
				mu_a.append(a)
				mu_v.append(v)
				
				k_dict[(a,v)] = K-1
				idxlens[a] += 1
	
	mu = np.random.rand(K,2,R)
	
	
	def f(a, x):
		
		out =  (-beta*(1 - np.minimum(np.sum(x, axis = 0), 1))*C[a])
		#out =  (-beta*(np.prod(1-x, axis = 0))*C[a])
		#print(x, out)
		return out
	
	mu_inv = [np.zeros((2,R)) for i in range(K)]
	def step():
		
		#pass 1
		
		
		for k in range(K):
			mui = np.zeros((2,R))
			for a in factor_IDX[mu_v[k]]:
				if a != mu_a[k]:
					mui += mu[k_dict[(a, mu_v[k])],:]
			
			mu_inv[k] = mui
		
		
		#pass 2
		
		
		mu_new = np.zeros((K,2,R))
		
		for k in range(K):
			
			X = [np.zeros(idxlens[mu_a[k]])]
			for i in range(idxlens[mu_a[k]]):
				apnd = []
				for x in X:
					x = x.copy()
					x[i] = x[i] + 1
					apnd.append(x)
				X = X + apnd
			
			#print(X)
			
			sm = np.zeros((2,R)) - 1*10.0**30
			
			def softmax(x,y):
				
				mx = 0*np.maximum(x,y)
				
				return np.log(np.exp(x - mx) + np.exp(y - mx)) + mx
			
			#to_exp_list = [(0,0.0) for i in range(0)]
			for x in X:
				mun = np.zeros(R)
				j = -1
				for i, v in enumerate(IDX[mu_a[k], :]):
					if(v >= 0):
						if v != mu_v[k]:
							mun += mu_inv[k_dict[(mu_a[k], v)]][int(x[i]),:]
						else:
							j = i
				
				to_exp = f(mu_a[k], x) + mun
				
				
				#print(x, to_exp, to_exp_reverse)
				
				#to_exp_list.append( (int(x[j]), to_exp) )
				#sm[int(x[j])] = softmax(sm[int(x[j])], to_exp)
				#sm[int(x[j])] = np.exp(sm[int(x[j])])
				mx = np.maximum(sm[int(x[j]),:] , to_exp)
				sm[int(x[j]),:] = np.log(np.exp(sm[int(x[j]),:] - mx) + np.exp(to_exp - mx)) + mx
				#sm[int(x[j])] = mx
				
				#sm[int(x[j])] = np.log(sm[int(x[j])])
			
			
			#mx = np.max([_[1] for _ in to_exp])
			
			# for i, e in to_exp():
# 				sm[i] += np.exp(e - mx)
			
			#print(sm)		
			mu_new[k,:,:] = sm
		
		
		
		
		mu_new = mu_new - np.sum(mu_new, axis = 1).reshape(K,1,R)/2
		
		if(verbose >= 3):
			print(np.sum((mu - mu_new)**2))
		
		for i in range(K):
			mu[i] = mu[i]*0.0+ 1.0*mu_new[i]
		
		
		
	for i in range(nt):
		beta = beta_/beta_anneal**((nt-i)/nt)
		step()
		
		mu = mu - np.sum(mu, axis = 1).reshape(K,1,R)/2
	
	if(verbose >= 3):
	#print(np.sum((mu - mu_new)**2))
		print(mu[0,:,0])
	
	
	
	p_v = np.zeros((N,2,R))
	
	
	for v in range(N):
		for a in factor_IDX[v]:
			p_v[v, :] += mu[k_dict[(a, v)]]
	
	#print(p_v)
	p_v = p_v - np.maximum(p_v[:,0,:], p_v[:,1,:]).reshape(N,1,R)
	if(verbose >= 3):
		print(p_v[0, :, 0])
	
	p_v = np.exp(p_v)
	#print(p_v)
	
	
	p_v = p_v/np.sum(p_v, axis = 1).reshape(N,1,R)
	#print(mu)
	#print(p_v/np.sum(p_v, axis = 1).reshape(-1,1))
	
	#print(p_v)
	
		
		
	
	
	return p_v, mu, mu_inv, factor_IDX, K, mu_v, mu_a, k_dict







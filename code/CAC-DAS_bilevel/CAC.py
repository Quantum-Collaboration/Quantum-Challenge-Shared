from functools import partial
import numpy as np
import misc

import jax
import jax.numpy as jnp

from jax.experimental import sparse

def run_cac_with_T(J, h, E0, eps, T, hyperparams, reps, Pin, seed, is_sparse=True, online_tuning=False, rho=0.5):

    R = list(hyperparams.values())[0].shape[0]
    N,R_ = np.shape(Pin)
    new_R = 100 * int(np.ceil(R / 100))
    
    J = np.array(J)
    h = np.array(h)

    if is_sparse:
        if J.ndim==2:
            J_in = sparse.BCOO.fromdense(J)  # assuming J is a dense array that needs conversion
            h_in = jnp.array(h)
        else:
            J_in = []
            h_in = []
            for j in range(np.shape(J)[0]):
                J_in.append(sparse.BCOO.fromdense(J[j,:,:]))
                h_in.append(jnp.array(h[j,:]))
    else:
        J_in = jnp.array(J)
        h_in = jnp.array(h)

    new_hyperparams = {name: jnp.array(np.pad(params, (0, new_R - R))) for name, params in hyperparams.items()}
    
    if online_tuning==False:
        new_T = jnp.array(np.pad(T, (0, new_R - R)))
        T_max = 100 * int(np.ceil(np.max(T) / 100))
    else:
        new_T = T
        T_max = T

    key = jax.random.key(seed)
    keys = jax.random.split(key, reps)

    P = jnp.array(Pin)
    
    E_opt, P_opt = jax.vmap(_run_cac_with_T, in_axes=(None, None, None, None, None, None, None, None, None, 0))(
        N, J_in, h_in, jnp.array(E0), jnp.array(eps), new_T, T_max, new_hyperparams, P, keys
    )

    E_opt = np.array(E_opt)[:, :R]
    P_opt = np.array(P_opt)[:, :, :R]
    
    return E_opt, P_opt


@partial(jax.jit, static_argnames='T_max')
def _run_cac_with_T(N, J, h, E0, eps, T, T_max, hyperparams, P, key):
    N = J[0].shape[0]

    print('compiling', hyperparams['beta'].shape[0], T_max)

    beta = hyperparams['beta']
    lamb1 = hyperparams['lamb1']
    lamb2 = hyperparams['lamb2']
    xi = hyperparams['xi']
    gamma = hyperparams['gamma']
    a = hyperparams['a']
    
    listcouplings = False
    if 'l1' in hyperparams:
        lv = [jnp.array(np.ones(len(hyperparams['l1']))),\
              jnp.array(hyperparams['l1']),jnp.array(hyperparams['l2']),\
                  jnp.array(hyperparams['l3']),jnp.array(hyperparams['l4'])]
        listcouplings = True

    R = beta.shape[0]

    def energy(s):
        
        if listcouplings:
            Js = jnp.zeros_like(s)
            for i in range(5):
                Js += lv[i][None,:] * (J[i] @ s + 2*h[i][:, None] )
        else:
            Js = J @ s + 2*h[:, None]# J_sparse @ s (resulting in N x R)
        return -0.5 * jnp.sum(s * Js, axis=0)
        
    def body_fn(carry, _):
        P, u, up, upp, e, E_opt, P_opt, i = carry

        lamb = lamb1 + jnp.float32(i) / jnp.float32(T) * (lamb2-lamb1)

        mP = jnp.mean(jnp.abs(P), axis=0)[None, :]

        if listcouplings:
            mu = jnp.zeros_like(P)
            for i in range(5):
                mu += lv[i][None,:] * (J[i] @ P + h[i][:, None] * mP)
                #mu += lv[i][None,:] * (J[i] @ P + 2*h[i][:, None] * mP)
        else:
            mu = J @ P + h[:, None] * mP
            #mu = J @ P + 2*h[:, None] * mP
    
        up, upp = u, up

        u = up - lamb * up + beta / eps * e * mu + gamma * (up - upp)
        e = e - (P ** 2 - a) * e * xi

        P = jnp.tanh(u)

        e = e / jnp.mean(e, axis=0)[None, :]
        e = jnp.abs(e)

        E = energy(jnp.sign(P))
        E_opt = jnp.where(jnp.logical_and(E < E_opt, i < T), E, E_opt)
        
        P_opt = jnp.where(jnp.logical_and(E <= E_opt, i < T), P, P_opt) ### This is not working 
 
        return (P, u, up, upp, e, E_opt, P_opt, i + 1), None

    P_opt = P
    u, up, upp = jnp.zeros((N, R)), jnp.zeros((N, R)), jnp.zeros((N, R))
    e = jnp.ones((N, R))
    E_opt = jnp.inf * jnp.ones((R,))

    E_opt, P_opt = jax.lax.scan(body_fn, (P, u, up, upp, e, E_opt, P_opt, 0), length=T_max)[0][5:7]

    return E_opt, P_opt

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange
import numpy as np


### SUPERVISED
@eqx.filter_jit
def loss_AD_sup(model, y, F, v, norm=True, **kwargs):
    """L_AD from paper v1, Hutchinson estimation 
    Reg = \sum_eps ||J_(F-Ftheta)(x).eps||_2^2 
        ~ ||J_(F-Ftheta)(x)||_F^2 

    If norm is True (usually easier to optimize):
        Reg = \sum_eps ||J_(F-Ftheta)(x). (eps / ||eps||) ||_2^2
            ~ 1/ndim ||J_(F-Ftheta)(x)||_F^2 

    with eps sampled in N(0,I) """
    y_shape = y.shape
    b, T, nc = y_shape[0], y_shape[1], y_shape[2]
    fun = lambda x: model.model_aug(x) - F(x)
    if norm: 
        v_norm = jnp.linalg.norm(v, ord=2, axis=1, keepdims=True) + 1e-8
        v = v / v_norm
    v_batched = jnp.broadcast_to(v[:, None, None, ...], (v.shape[0], b, T, *v.shape[1:]))
    # Define single-point JVP
    def jvp_single(x_i, v_i):
        """
        x are the primals i.e. points where we want to evaluate the JVP (on true trajectories),
        and v are the tangents i.e. the directions of the directional derivatives
        of the JVP (10 fixed random directions)
        """
        primals, tangents = jax.jvp(fun, (x_i,), (v_i,))
        return primals, tangents
    # Vectorize over (batch, time)
    jvp_batched = jax.vmap(jax.vmap(jvp_single, in_axes=(0, 0)), in_axes=(0, 0))
    # Vectorize over directions
    jvp_multi = jax.vmap(jvp_batched, in_axes=(None, 0)) 

    primals, tangents = jvp_multi(y, v_batched) # (num_directions, b, T, nc)
    res = jnp.mean(jnp.sum(tangents**2, axis=-1))
    return res

@eqx.filter_jit
def loss_Jacmatch_sup(model, y, F, **kwargs):
    """
    Jacmatch supervised, Reg = || (J_F-J_Ftheta)(x).F(x) ||_2^2
    Previously named: loss_AD_sup_local_norm
    """
    y_shape = y.shape
    b, T, nc = y_shape[0], y_shape[1], y_shape[2]
    fun = lambda x: model.model_aug(x) - F(x)
    # v = F(x)
    v = jax.vmap(F)(rearrange(y, 'b T nc -> (b T) nc'))
    v_batched = rearrange(v, '(b T) nc -> 1 b T nc', b=b)
    
    def jvp_single(x_i, v_i):
        primals, tangents = jax.jvp(fun, (x_i,), (v_i,))
        return primals, tangents

    # Vectorize over (batch, time)
    jvp_batched = jax.vmap(jax.vmap(jvp_single, in_axes=(0, 0)), in_axes=(0, 0))
    # Vectorize over directions
    jvp_multi = jax.vmap(jvp_batched, in_axes=(None, 0)) 

    primals, tangents = jvp_multi(y, v_batched) # (num_directions, b, T, nc)
    res = jnp.mean(jnp.sum(tangents**2, axis=-1))
    return res

@eqx.filter_jit
def loss_Accmatch_sup(model, y, F, **kwargs):
    """
    Accmatch supervised, Reg = || J_F(x)·F(x) - J_Ftheta(x)·Ftheta(x) ||_2^2
    """
    b, T, nc = y.shape
    Ftheta = model.model_aug  
    y_flat = rearrange(y, 'b T nc -> (b T) nc')
    # tangent vectors for the JVP          
    v_F      = jax.vmap(F)(y_flat) # F(x)
    v_Ftheta = jax.vmap(Ftheta)(y_flat) # Ftheta(x)                
    v_F      = rearrange(v_F,      '(b T) nc -> b T nc', b=b)   
    v_Ftheta = rearrange(v_Ftheta, '(b T) nc -> b T nc', b=b) 

    def make_jvp_batched(fun):
        def jvp_single(x_i, v_i):
            # x_i: (nc,), v_i: (nc,)  ->  tangent: (nc,)
            _, tangent = jax.jvp(fun, (x_i,), (v_i,))
            return tangent
        return jax.vmap(jax.vmap(jvp_single))   # vmap over T, then over b

    jvp_batched_F      = make_jvp_batched(funF)
    jvp_batched_Ftheta = make_jvp_batched(funFtheta)

    acc_F      = jvp_batched_F(y, v_F)           # J_F(x)·F(x)
    acc_Ftheta = jvp_batched_Ftheta(y, v_Ftheta) # J_Ftheta(x)·Ftheta(x)
    diff = acc_F - acc_Ftheta                    
    res = jnp.mean(jnp.sum(diff**2, axis=-1))
    return res

### UNSUPERVISED
@eqx.filter_jit
def loss_Fa(model, y, min_op, **kwargs):
    """ APHYNITY loss, Reg = ||F_theta(x)||_2^2 """
    #print('loss_Fa')
    y_in = rearrange(y, 'b T nc -> (b T) nc')
    aug_deriv = jax.vmap(model.model_aug)(y_in) 
    aug_deriv = rearrange(aug_deriv, '(b T) nc -> b T nc', b=y.shape[0])
    if min_op == 'l2_normalized':
        loss_op = ((jnp.linalg.norm(aug_deriv, ord=2, axis=2) / (jnp.linalg.norm(y, ord=2, axis=2) + 1e-5)) ** 2).mean()
    elif min_op == 'l2':
        loss_op = (jnp.linalg.norm(aug_deriv, ord=2, axis=2) ** 2).mean()
    else:
        loss_op = jnp.array(0.0)  # Default to zero if min_op is not recognized
    return loss_op

@eqx.filter_jit
def loss_FD_unsup(model, y, finite_diff="forward", **kwargs):
    """
    Reg = ||(J_F-J_Ftheta)(x) . (F(x)/||F(x)||) ||_2^2 unsupervised
    L_FD from paper v1"""
    dir_norm = jnp.linalg.norm(y[:,1:,:] - y[:,:-1,:], ord=2, axis=2)**2 + 1e-8
    # F_theta FD
    y_in = rearrange(y, 'b T ... -> (b T) ...')
    aug_deriv = jax.vmap(model.model_aug)(y_in) 
    aug_deriv = rearrange(aug_deriv, '(b T) ... -> b T ...', b=y.shape[0])
    FD_theta = (aug_deriv[:,1:,:] - aug_deriv[:,:-1,:]) 
    # F FD, unsupervised
    FD = (y[:,2:,:] - 2*y[:,1:-1,:] + y[:,:-2,:]) / model.dt 
    # alignment  
    if finite_diff == "forward":
        #print("forward FD")
        FD_theta = FD_theta[:,:-1,:] 
        dir_norm = dir_norm[:, :-1]
    elif finite_diff == "backward":
        #print("backward FD")
        FD_theta = FD_theta[:,1:,:] 
        dir_norm = dir_norm[:, 1:]
    FD_norm = (jnp.linalg.norm(FD_theta-FD, ord=2, axis=2)**2) / dir_norm
    return FD_norm.mean() 

@eqx.filter_jit
def loss_Jacmatch_unsup(model, y, **kwargs):
    """
    Reg = ||(J_F-J_Ftheta)(x).F(x)||_2^2 unsupervised
    Previously named: loss_FD_unsup_local_norm
    """
    y_shape = y.shape
    b, T, nc = y_shape[0], y_shape[1], y_shape[2]
    y_in = rearrange(y, 'b T ... -> (b T) ...')
    # Ftheta
    aug_deriv = jax.vmap(model.model_aug)(y_in) 
    aug_deriv = rearrange(aug_deriv, '(b T) ... -> b T ...', b=y.shape[0])
    # J_Ftheta(x).F(x), unsupervised using Finite Differences
    FD_theta = (aug_deriv[:,1:,:] - aug_deriv[:,:-1,:]) 
    FD_theta = FD_theta[:,:-1,:] / model.dt
    # J_F(x).F(x) unsupervised using Finite Differences
    FD = (y[:,2:,:] - 2*y[:,1:-1,:] + y[:,:-2,:]) / model.dt**2
    # Jacmatch
    jac_FD_norm = (jnp.linalg.norm(FD_theta-FD, ord=2, axis=2)**2) 
    return jac_FD_norm.mean()

@eqx.filter_jit
def loss_Lip(model, y, v, norm: bool, **kwargs): 
    """
    Reg = ||J_Ftheta||_F^2
    Previously named: loss_AD_single
    """
    y_shape = y.shape
    b, T, nc = y_shape[0], y_shape[1], y_shape[2]
    fun = lambda x: model.model_aug(x) 
    if norm:
        v_norm = jnp.linalg.norm(v, ord=2, axis=1, keepdims=True) + 1e-8
        v = v / v_norm
    v_batched = jnp.broadcast_to(v[:, Nonte, None, ...], (v.shape[0], b, T, *v.shape[1:]))

    def jvp_single(x_i, v_i):
        """ Single-point JVP
        x are the primals i.e. points where we want to evaluate the JVP (on true trajectories),
        and v are the tangents i.e. the directions of the directional derivatives
        of the JVP (10 fixed random directions)"""
        primals, tangents = jax.jvp(fun, (x_i,), (v_i,))
        return primals, tangents

    # Vectorize over (batch, time) and over directions
    jvp_batched = jax.vmap(jax.vmap(jvp_single, in_axes=(0, 0)), in_axes=(0, 0))
    jvp_multi = jax.vmap(jvp_batched, in_axes=(None, 0)) 

    primals, tangents = jvp_multi(y, v_batched) # (num_directions, b, T, nc)
    res = jnp.mean(jnp.sum(tangents**2, axis=-1))
    return res



LOSS_REGISTRY = {
    # supervised
    "AD": loss_AD_sup,
    "Jacmatch_sup": loss_Jacmatch_sup,
    "Accmatch_sup": loss_Accmatch_sup,

    # unsupervised
    "aphynity": loss_Fa,
    "lip": loss_Lip,
    "FD": loss_FD_unsup,
    "Jacmatch_unsup": loss_Jacmatch_unsup,
}
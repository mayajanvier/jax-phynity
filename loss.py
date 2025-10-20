import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange

def F_pendulum(x):
    return jnp.array([x[1], -((2*jnp.pi/12)**2)* jnp.sin(x[0]) - 0.2*x[1]])

def F_lorenz(var):
    beta = 8/3
    sigma = 10.
    rho = 28.
    x, y, z = var[0], var[1], var[2]
    dxdt = sigma * (y - x)
    dydt = rho * x - y - x * z
    dzdt = x * y - beta * z
    return jnp.array([dxdt, dydt, dzdt])

def F_twobody(s):
    x, y, x_prime, y_prime = s
    x_second = -x / (x**2 + y**2)**(3/2)
    y_second = -y / (x**2 + y**2)**(3/2)
    return jnp.array([x_prime, y_prime, x_second, y_second])

# Global variables 
F_dict = {
    "lorenz": F_lorenz,
    "twobody": F_twobody,
    "pendulum": F_pendulum
    }
key = jax.random.PRNGKey(0)
V_dict = {
    "lorenz": jax.random.normal(key, (10,3)),
    "twobody": jax.random.normal(key, (10,4)),
    "pendulum": jax.random.normal(key, (10,2))
    }

### Trajectory Losses
def MSEjax(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

@eqx.filter_jit
def loss_trajectory(model, y, epoch_rollout_index):
    print('loss_trajectory')
    x = y[:,0,:] # y0
    y_pred = jax.vmap(model)(x)
    y_pred = jax.lax.dynamic_slice(y_pred, (0, 0, 0), (y.shape[0], epoch_rollout_index, y.shape[2])) 
    return MSEjax(y_pred, y), y_pred

### Regularizations
@eqx.filter_jit
def loss_Fa(model, y, min_op):
    print('loss_Fa')
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
def loss_AD_sup(model, y, dataset_name: str):
    # TODO: check shapes, axis operations and define x and v here ? dataset ? 
    """
    x are the primals i.e. points where we want to evaluate the JVP (on true trajectories),
    and v are the tangents i.e. the directions of the directional derivatives
    of the JVP (10 fixed random directions)
    """
    print('loss AD supervised')
    b, T, nc = y.shape
    fun = lambda x: model.model_aug(x) - F_dict[dataset_name](x)
    v = V_dict[dataset_name]
    v_batched = jnp.broadcast_to(v[:, None, None, :], (v.shape[0], b, T, nc))
    # Define single-point JVP
    def jvp_single(x_i, v_i):
        primals, tangents = jax.jvp(fun, (x_i,), (v_i,))
        return primals, tangents
    # Vectorize over (batch, time)
    jvp_batched = jax.vmap(jax.vmap(jvp_single, in_axes=(0, 0)), in_axes=(0, 0))
    # Vectorize over directions
    jvp_multi = jax.vmap(jvp_batched, in_axes=(None, 0)) 

    primals, tangents = jvp_multi(y, v_batched) # (num_directions, b, T, nc)
    res = (jnp.linalg.norm(tangents, ord=2, axis=-1)**2).mean() # squared
    return res

@eqx.filter_jit
def loss_AD_unsup_global(model, x, v, L):
    # same x and v than golden standard here (?) 
    # TODO: check shapes, axis operations and which x and v ? 
    print("loss AD unsupervised")
    # JVP norms
    primals, tangents = jax.jvp(model.model_aug, x, v)
    tangents_norm = jnp.linalg.norm(tangents, ord=2, axis=2)
    res = (tangents_norm**2 - L**2).mean()
    return res 

@eqx.filter_jit
def loss_AD_unsup(model, y):
    # same x and v than golden standard here (?) 
    # TODO: check shapes, axis operations and which x and v ? 
    print("loss AD unsupervised")
    # Estimate Lipschitz constant based on trajectory samples 
    dy = y[:,1:,:] - y[:,:-1,:]
    L = jnp.max(jnp.linalg.norm(dy / model.dt, ord=2, axis=2), axis=1, keepdims=True) # batch max
    # forward (?)
    x = y[:,:-1,:]
    v = dy / jnp.linalg.norm(dy, ord=2, axis=2) 
    # JVP norms
    primals, tangents = jax.jvp(model.model_aug, x, v)
    tangents_norm = jnp.linalg.norm(tangents, ord=2, axis=2)
    res = (tangents_norm**2 - L**2).mean()
    return res 

@eqx.filter_jit
def loss_FD_unsup(model, y, finite_diff="forward"):
    #TODO: check shapes and axis operations
    print("loss FD unsupervised")
    dir_norm = jnp.linalg.norm(y[:,1:,:] - y[:,:-1,:], ord=2, axis=2)**2 + 1e-8
    # F_theta FD
    y_in = rearrange(y, 'b T nc -> (b T) nc')
    aug_deriv = jax.vmap(model.model_aug)(y_in) 
    aug_deriv = rearrange(aug_deriv, '(b T) nc -> b T nc', b=y.shape[0])
    FD_theta = (aug_deriv[:,1:,:] - aug_deriv[:,:-1,:]) 
    # F FD, unsupervised
    FD = (y[:,2:,:] - 2*y[:,1:-1,:] + y[:,:-2,:]) / model.dt 
    # alignment  
    if finite_diff == "forward":
        print("forward FD")
        FD_theta = FD_theta[:,:-1,:] 
        dir_norm = dir_norm[:, :-1]
    elif finite_diff == "backward":
        print("backward FD")
        FD_theta = FD_theta[:,1:,:] 
        dir_norm = dir_norm[:, 1:]
    FD_norm = (jnp.linalg.norm(FD_theta-FD, ord=2, axis=2)**2) / dir_norm
    return FD_norm.mean() 


### Compact 
def init_jit_aux_loss(aux_loss_names, min_op, dataset_name, finite_diff):
    """
    Initialize and jit auxiliary loss functions based on the provided names.
    """
    aux_losses_dict = {}
    for name in aux_loss_names:
        if name == 'loss_Fa':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa, min_op=min_op))
        elif name == 'loss_AD_sup':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_AD_sup, dataset_name=dataset_name))
        elif name == 'loss_AD_unsup':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_AD_unsup))
        elif name == 'loss_FD_unsup':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_FD_unsup, finite_diff=finite_diff))
        else:
            raise ValueError(f"Unknown auxiliary loss function: {name}")
    return aux_losses_dict

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, reg_loss_name, aux_losses_dict, lambda_, epoch_rollout_index, opt_mode):
    print('loss_fn')
    lossT, y_pred = loss_trajectory(model, y, epoch_rollout_index)
    losses_values_dict = {}
    for key, loss in aux_losses_dict.items():
        # if key == "loss_AD_unsup_global":
        #     losses_values_dict[key] = loss(model, x, v, L)
        # else:
        losses_values_dict[key] = loss(model, y)
    if reg_loss_name=='none':
        loss_op = 0.0
    else:
        loss_op = losses_values_dict[reg_loss_name]
    losses_values_dict["loss_traj"] = lossT
    if opt_mode == "constraint":
        loss_val = lossT * lambda_ + loss_op
    elif opt_mode == "traj":
        loss_val = lossT + loss_op * lambda_ 
    return loss_val, (y_pred, losses_values_dict)


### deprecated 
@eqx.filter_jit
def loss_Fa_primeX(model, y):
    print('loss_Fa_primeX')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:] # T nc
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedX(model, y, dataset_name: str, opt_mode: str):
    print('loss_Fa_prime_supervisedX')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        if dataset_name == 'lorenz':
            true_deriv = jax.vmap(F_lorenz)(y_in)
        elif dataset_name == 'pendulum':
            true_deriv = jax.vmap(F_pendulum)(y_in)
        elif dataset_name == 'twobody':
            true_deriv = jax.vmap(F_twobody)(y_in)
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        F_prime_true = jnp.abs(true_deriv[1:,:]-true_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        if opt_mode == 'constraint':
            F_prime = jnp.abs(F_prime - F_prime_true) 
        elif opt_mode == 'traj':
            F_prime = F_prime - F_prime_true
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedX_norm(model, y, dataset_name: str, opt_mode: str):
    print('loss_Fa_prime_supervisedX_norm')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        if dataset_name == 'lorenz':
            true_deriv = jax.vmap(F_lorenz)(y_in)
        elif dataset_name == 'pendulum':
            true_deriv = jax.vmap(F_pendulum)(y_in)
        elif dataset_name == 'twobody':
            true_deriv = jax.vmap(F_twobody)(y_in)
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        F_prime_true = jnp.abs(true_deriv[1:,:]-true_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        
        # Normalize each dimension (axis 0 = time)
        eps = 1e-8
        std_per_dim = jnp.std(F_prime_true, axis=0) + eps  # avoid division by 0
        if opt_mode == 'constraint':
            F_prime = jnp.abs(F_prime - F_prime_true) / std_per_dim
        elif opt_mode == 'traj':
            F_prime = (F_prime - F_prime_true) / std_per_dim
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedX_l2(model, y, dataset_name: str, opt_mode: str): # remove absolute value and do norm 2 as in loss_Fa
    print('loss_Fa_prime_supervisedX_l2')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        if dataset_name == 'lorenz':
            true_deriv = jax.vmap(F_lorenz)(y_in)
        elif dataset_name == 'pendulum':
            true_deriv = jax.vmap(F_pendulum)(y_in)
        elif dataset_name == 'twobody':
            true_deriv = jax.vmap(F_twobody)(y_in)
        F_prime = jnp.linalg.norm(aug_deriv[1:,:]-aug_deriv[:-1,:],2) / jnp.linalg.norm(y_in[1:,:] - y_in[:-1,:], 2)
        F_prime_true = jnp.linalg.norm(true_deriv[1:,:]-true_deriv[:-1,:],2) / jnp.linalg.norm(y_in[1:,:] - y_in[:-1,:])
        if opt_mode == 'constraint':
            F_prime = (jnp.linalg.norm(F_prime - F_prime_true, ord=2, axis=1) ** 2)
        elif opt_mode == 'traj':
            F_prime = F_prime - F_prime_true
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedX_direct(model, y, dataset_name: str): # remove absolute value and do norm 2 as in loss_Fa
    print('loss_Fa_prime_supervisedX_direct')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        if dataset_name == 'lorenz':
            true_deriv = jax.vmap(F_lorenz)(y_in)
        elif dataset_name == 'pendulum':
            true_deriv = jax.vmap(F_pendulum)(y_in)
        elif dataset_name == 'twobody':
            true_deriv = jax.vmap(F_twobody)(y_in)
        F_prime = (aug_deriv[1:,:]-aug_deriv[:-1,:]) / (y_in[1:,:] - y_in[:-1,:])
        F_prime_true = (true_deriv[1:,:]-true_deriv[:-1,:]) / (y_in[1:,:] - y_in[:-1,:])
        F_prime = (jnp.linalg.norm(F_prime - F_prime_true, ord=2, axis=1) ** 2)
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedYX(model, y, epoch_rollout_index, F_prime_true: jnp.ndarray):
    print('loss_Fa_prime_supervisedYX')
    loss_prime = 0.0
    y_theta_in = jax.vmap(model)(y[:,:,0])
    y_theta_in = jax.lax.dynamic_slice(y_theta_in, (0, 0, 0), (y.shape[0], y.shape[1], epoch_rollout_index))
    for k_batch in range(y.shape[0]):
        y_theta = y_theta_in[k_batch,:,:]
        aug_deriv = jax.vmap(model.model_aug)(y_theta) 
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / (jnp.abs(y_theta[1:,:] - y_theta[:-1,:])+1e-5)
        f_prime_true = F_prime_true[k_batch,:F_prime.shape[0],:]
        F_prime = jnp.abs(F_prime - f_prime_true)
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

def init_jit_aux_loss_old(aux_loss_names, min_op, dataset_name, opt_mode, finite_diff):
    """
    Initialize and jit auxiliary loss functions based on the provided names.
    """
    aux_losses_dict = {}
    for name in aux_loss_names:
        if name == 'loss_Fa':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa, min_op=min_op))
        elif name == 'loss_Fa_primeX':
            aux_losses_dict[name] = eqx.filter_jit(loss_Fa_primeX)
        elif name == 'loss_Fa_prime_supervisedX':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa_prime_supervisedX, dataset_name=dataset_name, opt_mode=opt_mode))
        elif name == 'loss_Fa_prime_supervisedX_norm':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa_prime_supervisedX_norm, dataset_name=dataset_name, opt_mode=opt_mode))
        elif name == 'loss_Fa_prime_supervisedX_l2':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa_prime_supervisedX_l2, dataset_name=dataset_name, opt_mode=opt_mode))
        elif name == 'loss_Fa_prime_supervisedX_direct':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa_prime_supervisedX_direct, dataset_name=dataset_name))
        elif name == 'loss_Fa_prime_supervisedYX':
            aux_losses_dict[name] = eqx.filter_jit(loss_Fa_prime_supervisedYX)
        else:
            raise ValueError(f"Unknown auxiliary loss function: {name}")
    return aux_losses_dict

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn_old(model, y, x, v, reg_loss_name, aux_losses_dict, lambda_, epoch_rollout_index, Fa_prime_true, opt_mode):
    print('loss_fn')
    lossT, y_pred = loss_trajectory(model, y, epoch_rollout_index)
    losses_values_dict = {}
    for key, loss in aux_losses_dict.items():
        if key == 'loss_Fa_prime_supervisedYX':
            losses_values_dict[key] = loss(model, y, epoch_rollout_index, F_prime_true=Fa_prime_true)
        else:
            losses_values_dict[key] = loss(model, y)
    if reg_loss_name=='none':
        loss_op = 0.0
    else:
        loss_op = losses_values_dict[reg_loss_name]
    losses_values_dict["loss_traj"] = lossT
    if opt_mode == "constraint":
        loss_val = lossT * lambda_ + loss_op
    elif opt_mode == "traj":
        loss_val = loss_op * lambda_ + lossT
    return loss_val, (y_pred, losses_values_dict)
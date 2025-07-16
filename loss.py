import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange

def F(x):
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

### Trajectory Losses
def MSEjax(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

def MSEjax_scaled(y_pred, y_true, scales = jnp.concatenate([jnp.array([0]),jnp.arange(1,0,-0.01)])):
    se = ((y_pred - y_true)**2)
    # pondéré par time 
    se = se * scales
    return se.mean()

def MSEjax_spatial(y_pred, y_true): # perform as MSEjax
    # mean over space
    se = ((y_pred - y_true)**2).mean(axis=2)
    return se.sum(axis=1).mean()

def MSEjax_halftime(y_pred, y_true):
    # mean over last half of interval
    se = ((y_pred - y_true)**2)[:,:,51:]
    return se.mean()

@eqx.filter_jit
def loss_trajectory(model, y, epoch_rollout_index):
    print('loss_trajectory')
    x = y[:,:,0] # y0
    y_pred = jax.vmap(model)(x)
    y_pred = jax.lax.dynamic_slice(y_pred, (0, 0, 0), (y.shape[0], y.shape[1], epoch_rollout_index))
    return MSEjax(y_pred, y), y_pred


### Regularizations
@eqx.filter_jit
def loss_Fa(model, y, min_op):
    print('loss_Fa')
    y_in = rearrange(y, 'b nc T -> (b T) nc')
    aug_deriv = jax.vmap(model.model_aug)(y_in) 
    aug_deriv = rearrange(aug_deriv, '(b T) nc -> b nc T', b=y.shape[0])
    if min_op == 'l2_normalized':
        loss_op = ((jnp.linalg.norm(aug_deriv, ord=2, axis=1) / (jnp.linalg.norm(y, ord=2, axis=1) + 1e-5)) ** 2).mean()
    elif min_op == 'l2':
        loss_op = (jnp.linalg.norm(aug_deriv, ord=2, axis=1) ** 2).mean()
    else:
        loss_op = jnp.array(0.0)  # Default to zero if min_op is not recognized
    return loss_op

@eqx.filter_jit
def loss_Fa_primeX(model, y):
    print('loss_Fa_primeX')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        y_in = rearrange(y_in, 'nc T -> (T) nc')
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedX(model, y, dataset_name: str):
    print('loss_Fa_prime_supervisedX')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        y_in = rearrange(y_in, 'nc T -> (T) nc')
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        if dataset_name == 'lorenz':
            true_deriv = jax.vmap(F_lorenz)(y_in)
        elif dataset_name == 'pendulum':
            true_deriv = jax.vmap(F)(y_in)
        elif dataset_name == 'twobody':
            true_deriv = jax.vmap(F_twobody)(y_in)
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        F_prime_true = jnp.abs(true_deriv[1:,:]-true_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        F_prime = jnp.abs(F_prime - F_prime_true)
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedX_direct(model, y, dataset_name: str): # remove absolute value and do norm 2 as in loss_Fa
    print('loss_Fa_prime_supervisedX_direct')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        y_in = rearrange(y_in, 'nc T -> (T) nc')
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        if dataset_name == 'lorenz':
            true_deriv = jax.vmap(F_lorenz)(y_in)
        elif dataset_name == 'pendulum':
            true_deriv = jax.vmap(F)(y_in)
        elif dataset_name == 'twobody':
            true_deriv = jax.vmap(F_twobody)(y_in)
        F_prime = aug_deriv[1:,:]-aug_deriv[:-1,:] / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        F_prime_true = true_deriv[1:,:]-true_deriv[:-1,:] / jnp.abs(y_in[1:,:] - y_in[:-1,:])
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
    y_theta_in = rearrange(y_theta_in, 'b nc T -> b (T) nc')
    for k_batch in range(y.shape[0]):
        y_theta = y_theta_in[k_batch,:,:]
        aug_deriv = jax.vmap(model.model_aug)(y_theta) 
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / (jnp.abs(y_theta[1:,:] - y_theta[:-1,:])+1e-5)
        f_prime_true = F_prime_true[k_batch,:F_prime.shape[0],:]
        F_prime = jnp.abs(F_prime - f_prime_true)
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime





### Final loss function
# @eqx.filter_value_and_grad(has_aux=True)
# @eqx.filter_jit
# def loss_fn(model, y, min_op, lambda_, epoch_rollout_index, model_phy_option: str, model_aug_option: bool, Fa_prime_true: jnp.ndarray = None):
#     print('loss_fn')
#     ### Trajectory loss
#     lossT, y_pred = loss_trajectory(model, y, epoch_rollout_index)
#     ### Regularization
#     if model_phy_option == "none": # none_aug
#         loss_op = loss_Fa(model, y, min_op)
#         return lossT, (lossT, loss_op, y_pred)
#     elif model_phy_option == "true": # true
#         return lossT, (lossT, jnp.array(0.0), y_pred)
#     elif model_phy_option == 'incomplete_no_Fa':
#         loss_op = loss_Fa(model, y, min_op)
#         return lossT, (lossT, loss_op, y_pred)
#     elif model_phy_option == 'none_Fa':
#         loss_op = loss_Fa(model, y, min_op)
#         return lossT * lambda_ + loss_op, (lossT, loss_op, y_pred)
#     elif model_phy_option == 'none_Fa_primeX':
#         loss_op = loss_Fa(model, y, min_op)
#         loss_prime = loss_Fa_primeX(model, y)
#         return lossT * lambda_ + loss_prime, (lossT, loss_op, y_pred)
#     elif model_phy_option == 'incomplete_Fa_prime':
#         loss_op = loss_Fa(model, y, min_op)
#         loss_prime = loss_Fa_primeX(model, y)
#         return lossT * lambda_ + loss_prime, (lossT, loss_op, y_pred)
#     elif model_phy_option == "none_Fa_prime_supYX":
#         loss_op = loss_Fa(model, y, min_op)
#         loss_prime = loss_Fa_prime_supervisedYX(model, y, epoch_rollout_index, Fa_prime_true)
#         return lossT * lambda_ + loss_prime, (lossT, loss_op, y_pred)
#     else:
#         if model_aug_option: # complete_aug or incomplete_aug
#             loss_op = loss_Fa(model, y, min_op)
#             return lossT * lambda_ + loss_op, (lossT,loss_op, y_pred)
#         else: # complete_physics or incomplete_physics
#             return lossT, (lossT, jnp.array(0.0), y_pred)
        

### Compact 
def init_jit_aux_loss(aux_loss_names, min_op, dataset_name):
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
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa_prime_supervisedX, dataset_name=dataset_name))
        elif name == 'loss_Fa_prime_supervisedX_direct':
            aux_losses_dict[name] = eqx.filter_jit(eqx.Partial(loss_Fa_prime_supervisedX_direct, dataset_name=dataset_name))
        elif name == 'loss_Fa_prime_supervisedYX':
            aux_losses_dict[name] = eqx.filter_jit(loss_Fa_prime_supervisedYX)
        else:
            raise ValueError(f"Unknown auxiliary loss function: {name}")
    return aux_losses_dict

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, reg_loss_name, aux_losses_dict, lambda_, epoch_rollout_index, Fa_prime_true):
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
    loss_val = lossT * lambda_ + loss_op
    return loss_val, (y_pred, losses_values_dict)
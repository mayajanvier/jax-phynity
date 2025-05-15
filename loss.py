import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange

def F(x):
    return jnp.array([x[1], -((2*jnp.pi/12)**2)* jnp.sin(x[0]) - 0.2*x[1]])

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
    print('loss_Fa_prime')
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
def loss_Fa_prime_supervisedX(model, y):
    print('loss_Fa_prime_supervised')
    loss_prime = 0.0
    for k_batch in range(y.shape[0]):
        y_in = y[k_batch,:,:]
        y_in = rearrange(y_in, 'nc T -> (T) nc')
        aug_deriv = jax.vmap(model.model_aug)(y_in) 
        true_deriv = jax.vmap(F)(y_in)
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        F_prime_true = jnp.abs(true_deriv[1:,:]-true_deriv[:-1,:]) / jnp.abs(y_in[1:,:] - y_in[:-1,:])
        F_prime = jnp.abs(F_prime - F_prime_true)
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime

@eqx.filter_jit
def loss_Fa_prime_supervisedYX(model, y, epoch_rollout_index, F_prime_true: jnp.ndarray):
    print('loss_Fa_prime_supervised2')
    loss_prime = 0.0
    y_theta_in = jax.vmap(model)(y[:,:,0])
    y_theta_in = jax.lax.dynamic_slice(y_theta_in, (0, 0, 0), (y.shape[0], y.shape[1], epoch_rollout_index))
    y_theta_in = rearrange(y_theta_in, 'b nc T -> b (T) nc')
    for k_batch in range(y.shape[0]):
        y_theta = y_theta_in[k_batch,:,:]
        aug_deriv = jax.vmap(model.model_aug)(y_theta) 
        F_prime = jnp.abs(aug_deriv[1:,:]-aug_deriv[:-1,:]) / jnp.abs(y_theta[1:,:] - y_theta[:-1,:])
        f_prime_true = F_prime_true[k_batch,:F_prime.shape[0],:]
        F_prime = jnp.abs(F_prime - f_prime_true)
        loss_prime += jnp.mean(F_prime)
    loss_prime /= y.shape[0]
    return loss_prime


### Final loss function
@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, min_op, lambda_, epoch_rollout_index, model_phy_option: str, model_aug_option: bool, Fa_prime_true: jnp.ndarray = None):
    print('loss_fn')
    lossT, y_pred = loss_trajectory(model, y, epoch_rollout_index)
    if model_phy_option == "none": # none_aug
        loss_op = loss_Fa(model, y, min_op)
        return lossT, (lossT, loss_op, y_pred)
    elif model_phy_option == "true": # true
        return lossT, (lossT, jnp.array(0.0), y_pred)
    elif model_phy_option == 'incomplete_no_Fa':
        loss_op = loss_Fa(model, y, min_op)
        return lossT, (lossT, loss_op, y_pred)
    elif model_phy_option == 'none_Fa':
        loss_op = loss_Fa(model, y, min_op)
        return lossT * lambda_ + loss_op, (lossT, loss_op, y_pred)
    # elif model_phy_option == 'none_Fa_prime':
    #     loss_op = loss_Fa(model, y, min_op)
    #     loss_prime = loss_Fa_prime(model, y)
    #     return lossT * lambda_ + loss_prime, (lossT, loss_op, y_pred)
    elif model_phy_option == 'incomplete_Fa_prime':
        loss_op = loss_Fa(model, y, min_op)
        loss_prime = loss_Fa_primeX(model, y)
        return lossT * lambda_ + loss_prime, (lossT, loss_op, y_pred)
    elif model_phy_option == "none_Fa_prime":
        loss_op = loss_Fa(model, y, min_op)
        loss_prime = loss_Fa_prime_supervisedYX(model, y, epoch_rollout_index, Fa_prime_true)
        return lossT * lambda_ + loss_prime, (lossT, loss_op, y_pred)
    else:
        if model_aug_option: # complete_aug or incomplete_aug
            loss_op = loss_Fa(model, y, min_op)
            return lossT * lambda_ + loss_op, (lossT,loss_op, y_pred)
        else: # complete_physics or incomplete_physics
            return lossT, (lossT, jnp.array(0.0), y_pred)
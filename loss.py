import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange
from utils import fft_diff_jax, fft_diff_jax_fast

### CONSTANTS
I = jnp.array([1.6, 1.0, 2 / 3]) # rigid body
m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81 # double pendulum
L_ks = 64 # KS domain length

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

def F_rigidbody(s): 
    y1, y2, y3 = s
    mat = jnp.array([
        [0, -y3, y2],
        [y3, 0, -y1],
        [-y2, y1, 0]])
    vect = jnp.array([y1/I[0], y2/I[1], y3/I[2]])
    dydt = mat @ vect
    return dydt

def dw1(s):
    theta1, theta2, w1, w2 = s
    return (
        -g * (2*m1 + m2) * jnp.sin(theta1) - m2 * g * jnp.sin(theta1 - 2*theta2) -
        2* jnp.sin(theta1-theta2) * m2 * (w2**2 * l2 + w1**2 * l1 * jnp.cos(theta1-theta2))
    ) /  (l1 * (2*m1 + m2 - m2 * jnp.cos(2*(theta1-theta2))))
    
def dw2(s):
    theta1, theta2, w1, w2 = s
    return (
        2 * jnp.sin(theta1 - theta2) * (
            w1**2 * l1 * (m1 + m2) +
            g * (m1 + m2) * jnp.cos(theta1) +
            w2**2 * l2 * m2 * jnp.cos(theta1 - theta2)
        )
    ) / (l2 * (2*m1 + m2 - m2 * jnp.cos(2*(theta1 - theta2))))

def F_doublependulum(s): 
    """Compute derivatives for double pendulum."""
    _, _, w1, w2 = s
    dtheta1_dt = w1
    dtheta2_dt = w2
    dw1_dt = dw1(s)
    dw2_dt = dw2(s)
    return jnp.array([dtheta1_dt, dtheta2_dt, dw1_dt, dw2_dt])


### KS 1D uses pseudospectral reconstruction in Brandsetter generated data 
def F_KS(u):
    # Compute the x derivatives using the pseudo-spectral method.
    ux = fft_diff_jax(u, period=L_ks)
    uxx = fft_diff_jax(u, period=L_ks, order=2)
    uxxxx = fft_diff_jax(u, period=L_ks, order=4)
    # Compute du/dt.
    dudt = - u*ux - uxx - uxxxx
    return dudt

def F_KS_fast(u):
    # Compute the x derivatives using the pseudo-spectral method.
    ux = fft_diff_jax_fast(u, period=L_ks)
    uxx = fft_diff_jax_fast(u, period=L_ks, order=2)
    uxxxx = fft_diff_jax_fast(u, period=L_ks, order=4)
    # Compute du/dt.
    dudt = - u*ux - uxx - uxxxx
    return dudt

# Global variables 
F_dict = {
    "lorenz": F_lorenz,
    "twobody": F_twobody,
    "pendulum": F_pendulum,
    "doublependulum": F_doublependulum,
    "rigidbody": F_rigidbody,
    "ks": F_KS_fast
    }

key = jax.random.PRNGKey(0)
V_dict = {
    "lorenz": jax.random.normal(key, (10,3)),
    "twobody": jax.random.normal(key, (10,4)),
    "pendulum": jax.random.normal(key, (10,2)),
    "doublependulum": jax.random.normal(key, (10,4)),
    "rigidbody": jax.random.normal(key, (10,3)),
    "ks": jax.random.normal(key, (10,256)),
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

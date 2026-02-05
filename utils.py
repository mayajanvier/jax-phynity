import os
import sys
import jax 
import jax.numpy as jnp
import equinox as eqx
from time import sleep 
from datetime import datetime
import json
import pandas as pd
import statistics
import wandb

# log utils 
class Logger(object):
    "Lumberjack class - duplicates sys.stdout to a log file and it's okay"
    def __init__(self, filename, mode="a"):
        self.stdout = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None

def log(train, epoch, iteration, metrics, nepoch):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=nepoch,
            i=iteration+1,
            max_i=len(train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=name, value=float(value))
            
        print(message)

def make_basedir(root, name_exp, timestamp=None, attempts=5):
    """Takes 5 shots at creating a folder from root,
    adding timestamp if desired.
    """
    for i in range(attempts):
        basedir = root
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            basedir = os.path.join(basedir, name_exp[:-8]+str(len(os.listdir(basedir)))+"_"+name_exp[-8:]) 
        try:
            os.makedirs(basedir)
            return basedir
        except:
            sleep(0.01)
    raise FileExistsError(root)

def compute_metric(net, train_data):
    metrics = {}
    metrics['param_error'] = statistics.mean(abs(v1-float(v2))/v1 for v1, v2 in zip(train_data.dataset.params.values(), net.get_pde_params().values()))
    metrics.update(net.get_pde_params())
    metrics.update({f'{k}_real': v for k, v in train_data.dataset.params.items() if k in metrics})
    return metrics

def log_wandb(net, dataloader, _lambda, loss_dict, split, epoch_rollout_index, log_param_error=True):
    if log_param_error:
        metric = compute_metric(net, dataloader)
        omega_error = abs(metric["omega0_square"] - metric["omega0_square_real"]) / metric["omega0_square_real"]
        alpha_error = abs(metric["alpha"] - metric["alpha_real"]) / metric['alpha_real']
        if split == 'train':
            wandb.log({
                    #"Train loss": loss_dict["loss_traj"],
                       "Lambda": _lambda,
                       #"Loss_Fa": loss_dict['loss_Fa'],
                       "Param error": metric["param_error"],
                       "Omega error":omega_error,
                       "Alpha error":alpha_error,
                       #"Train Fa_primeX": loss_dict["loss_Fa_primeX"],
                        "Rollout index": epoch_rollout_index} | loss_dict)
        elif split == 'val':
            wandb.log({"Test loss": loss_dict["loss_traj"], "Param error test": metric["param_error"]})
    else:
        if split == 'train':
            wandb.log({"Lambda": _lambda, "Rollout index": epoch_rollout_index} |loss_dict)
        elif split == 'val':
            wandb.log({"Test loss": loss_dict["loss_traj"]})

def save_loss_local(val_losses, train_losses, l_test, l_train, exp_path):
    val_losses.append(l_test['loss_traj'].item())
    train_losses.append(l_train['loss_traj'].item())
    L = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    L.to_csv(exp_path+'/loss.csv', index=False)

# jax utils
def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def l2normalize(v, eps=1e-12):
    # default is also Frobenius for matrix, L2 for vector
    return v / (jnp.linalg.norm(v) + eps) 

def orthogonal_init(key: jax.random.PRNGKey, shape: tuple, gain: float = 1.0) -> jax.Array:
    """Applies orthogonal initialization to a non-square matrix."""
    out_features, in_features = shape

    # Always generate a matrix of shape (max(out, in), in)
    size = max(out_features, in_features)
    mat = jax.random.normal(key, (size, in_features))
    q, r = jnp.linalg.qr(mat)
    q = q[:out_features]  # Truncate to desired out_features
    d = jnp.sign(jnp.diag(r))
    q = q * d  # Apply sign correction
    return gain * q

def init_linear_weight(model, init_fn, key, init_gain=0.2):
    """Applies a given weight initialization function to all eqx.nn.Linear layers in a model."""
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)  # Check if it's a Linear layer
    # Extract weights and biases from the model
    get_weights = lambda m: [x.weight
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]
    
    get_biases = lambda m: [x.bias 
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    
    weights = get_weights(model)  
    biases = get_biases(model)
    #print("weights", [weight.shape for weight in weights])

    # Initialization for weights
    new_weights = [init_fn(subkey, weight.shape, init_gain)  
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    #print("weights", [w.shape for w in new_weights])
    # Zero initialization for biases
    new_biases = [jnp.zeros_like(bias) for bias in biases]

    # Update the model
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)   
    return new_model

# fft utils

def fft_diff_jax(x, order=1, period=None):
    """
    JAX equivalent of scipy.fftpack.diff

    Parameters
    ----------
    x : array_like
        Periodic input sequence (1D).
    order : int
        Order of differentiation (negative = integration).
    period : float, optional
        Period of the signal. Default is 2*pi.

    Returns
    -------
    y : array
        Differentiated (or integrated) signal.
    """
    x = jnp.asarray(x)

    if order == 0:
        return x

    n = x.shape[0]

    # Handle complex input the same way SciPy does
    if jnp.iscomplexobj(x):
        return (
            fft_diff_jax(x.real, order, period)
            + 1j * fft_diff_jax(x.imag, order, period)
        )

    # Scaling constant
    if period is not None:
        c = 2 * jnp.pi / period
    else:
        c = 1.0

    # Fourier frequencies (integer modes)
    k = jnp.fft.fftfreq(n) * n

    # Fourier transform
    X = jnp.fft.fft(x)

    # Multiplier (i * c * k)^order
    ik = 1j * c * k

    if order > 0:
        multiplier = ik ** order
    else:
        # Integration: assume zero mean
        multiplier = jnp.zeros_like(ik)
        nonzero = k != 0
        multiplier = multiplier.at[nonzero].set(ik[nonzero] ** order)

    # Enforce y_0 = 0
    multiplier = multiplier.at[0].set(0.0)

    # Zero Nyquist mode for odd order & even n
    if (n % 2 == 0) and (order % 2 == 1):
        multiplier = multiplier.at[n // 2].set(0.0)

    Y = X * multiplier

    y = jnp.fft.ifft(Y)

    # SciPy returns real if input was real
    return y.real

@eqx.filter_jit
def fft_diff_jax_fast(x, order=1, period=2*jnp.pi):
    """
    JAX equivalent of scipy.fftpack.diff

    Parameters
    ----------
    x : array_like
        Periodic input sequence (1D).
    order : int
        Order of differentiation (negative = integration).
    period : float, optional
        Period of the signal. Default is 2*pi.

    Returns
    -------
    y : array
        Differentiated (or integrated) signal.
    """

    # if order == 0:
    #     return x

    n = x.shape[0]

    # Hope we don't need it 
    # Handle complex input the same way SciPy does
    # if jnp.iscomplexobj(x):
    #     print("complex")
    #     return (
    #         fft_diff_jax(x.real, order, period)
    #         + 1j * fft_diff_jax(x.imag, order, period)
    #     )

    # Scaling constant
    c = 2 * jnp.pi / period # default c=1.0

    # Fourier frequencies (integer modes)
    k = jnp.fft.fftfreq(n) * n

    # Fourier transform
    X = jnp.fft.fft(x)

    # Multiplier (i * c * k)^order
    ik = 1j * c * k

    #if order > 0:
    multiplier = ik ** order # we will only use order > 0 
    # else:
    #     # Integration: assume zero mean
    #     multiplier = jnp.zeros_like(ik)
    #     nonzero = k != 0
    #     multiplier = multiplier.at[nonzero].set(ik[nonzero] ** order)

    # Enforce y_0 = 0
    multiplier = multiplier.at[0].set(0.0)

    # Zero Nyquist mode for odd order & even n
    if (n % 2 == 0) and (order % 2 == 1):
        multiplier = multiplier.at[n // 2].set(0.0)

    Y = X * multiplier

    y = jnp.fft.ifft(Y)

    # SciPy returns real if input was real
    return y.real
    
if __name__ == '__main__':
    from networks import MLP
    # test weight initialization
    init_key, key = jax.random.split(jax.random.PRNGKey(0))
    model_aug = MLP(key=key, state_c=2, hidden=200)
    print(model_aug)
    model_aug = init_linear_weight(model_aug, orthogonal_init, key=init_key, init_gain=0.2)
    print(model_aug)
    w = model_aug.layers[2].weight
    print((w.T @ w)/(0.2**2)) # Identity matrix
    print(model_aug.layers[2].bias) # Zero bias

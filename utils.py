import os
import sys
import jax 
import jax.numpy as jnp
import equinox as eqx
from time import sleep 
from datetime import datetime
import json

# pure python utils from APHYNITY
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

def make_basedir(root, name_exp, timestamp=None, attempts=5):
    """Takes 5 shots at creating a folder from root,
    adding timestamp if desired.
    """
    for i in range(attempts):
        basedir = root
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            basedir = os.path.join(basedir, name_exp+ str(len(os.listdir(basedir)) + 1 -3)) # 3 data files 
        try:
            os.makedirs(basedir)
            return basedir
        except:
            sleep(0.01)
    raise FileExistsError(root)

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
    
    if out_features >= in_features:  # More output features than input features
        mat = jax.random.normal(key, (out_features, in_features))
        q, r = jnp.linalg.qr(mat)  # QR decomposition to get orthogonal columns
        d = jnp.sign(jnp.diag(r))  # Normalize the signs of the diagonal of r
        return gain * q * d  # Orthogonal matrix scaled by gain
    else:  # More input features than output features
        mat = jax.random.normal(key, (in_features, out_features))
        q, r = jnp.linalg.qr(mat.T)  # Transpose to get orthogonal rows
        d = jnp.sign(jnp.diag(r))  # Normalize the signs of the diagonal of r
        return gain * q.T * d  # Return transposed matrix scaled by gain

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

    # Initialization for weights
    new_weights = [init_fn(subkey, weight.shape, init_gain)  
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    # Zero initialization for biases
    new_biases = [jnp.zeros_like(bias) for bias in biases]

    # Update the model
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)   
    return new_model


if __name__ == '__main__':
    from networks import MLP
    # test weight initialization
    init_key, key = jax.random.split(jax.random.PRNGKey(0))
    model_aug = MLP(key=key, state_c=2, hidden=200)
    model_aug = init_linear_weight(model_aug, orthogonal_init, key=init_key, init_gain=0.2)
    print(model_aug)
    w = model_aug.layers[2].weight
    print((w.T @ w)/(0.2**2)) # Identity matrix
    print(model_aug.layers[2].bias) # Zero bias

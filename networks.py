import equinox as eqx
import jax
import jax.numpy as jnp
from collections import OrderedDict


### Physical model Fp
class DampedPendulumParamPDE(eqx.Module):
    # ParameterDict replaced, need to be jax object for array filtering (for gradient computation)
    is_complete: bool
    omega0_square_org: jax.Array 
    alpha_org: jax.Array
    omega0_square: jax.Array
    alpha: jax.Array


    def __init__(self, is_complete=False, real_params=None):
        super().__init__()
        self.is_complete = is_complete
        self.omega0_square_org = jnp.array(0.2)
        self.alpha_org = jnp.array(0.1)

        if real_params is not None:
            self.omega0_square = real_params["omega0_square"] 
            self.alpha = real_params["alpha"] 
        else:
            self.omega0_square = self.omega0_square_org
            if is_complete:
                self.alpha = self.alpha_org
            else:
                self.alpha = jnp.array(0.0)

    def __call__(self, state):
        q = state[:,0:1]
        p = state[:,1:2]

        dqdt = p
        dpdt = - self.omega0_square * jnp.sin(q) - self.alpha * p

        return jnp.concat([dqdt, dpdt], axis=1)
    
### Data driven model Fa    
class MLP(eqx.Module):
    layers: list # we need to define the type of the attributes of the class in jax

    def __init__(self, key, state_c, hidden, init_gain=0.2):
        super().__init__()
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(state_c, hidden, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(hidden, hidden, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(hidden, state_c, key=key3)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_derivatives(self, x):
        # batch management
        batch_size, nc, T = x.shape 
        x = jnp.permute_dims(x, (0, 2, 1))
        x = jnp.reshape(x, (batch_size * T, nc))
        x = jax.vmap(self.__call__)(x)
        x = jnp.reshape(x, (batch_size, T, nc))
        x = jnp.permute_dims(x, (0, 2, 1))
        return x
    


if __name__ == '__main__':
    nb_neurons = 200
    input = jax.random.normal(jax.random.PRNGKey(0), (2,))
    print(input.shape)
    model_aug = MLP(jax.random.PRNGKey(0), 2, nb_neurons)
    print(model_aug)
    output = model_aug(input)
    print(output.shape)

    model_phy = DampedPendulumParamPDE(is_complete=True, real_params=None) 
    print(model_phy)
    state = jax.random.normal(jax.random.PRNGKey(0), (1,2,3))
    out = model_phy(state)
    print(out.shape) # same shape as input
    print("params", model_phy.alpha, model_phy.omega0_square)
    print("params org", model_phy.alpha_org, model_phy.omega0_square_org)    
    

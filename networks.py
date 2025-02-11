import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange


### Physical model Fp
omega0_square_org = 0.2
alpha_org = 0.1 

class DampedPendulumParamPDE(eqx.Module):
    # ParameterDict replaced by jax.array, need to be jax object for array filtering (for gradient computation)
    is_complete: bool
    omega0_square: jax.Array
    alpha: jax.Array

    def __init__(self, is_complete=False, real_params=None):
        super().__init__()
        self.is_complete = is_complete

        if real_params is not None:
            # TODO: put in jnp.array or not since fixed in True ODE ? 
            self.omega0_square = real_params["omega0_square"]
            self.alpha = real_params["alpha"]
        else:
            self.omega0_square = jnp.array(omega0_square_org)
            if is_complete:
                self.alpha = jnp.array(alpha_org)
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
        x = rearrange(x, 'b nc T -> (b T) nc')
        x = jax.vmap(self.__call__)(x)
        x = rearrange(x, '(b T) nc -> b nc T', b=batch_size)
        return x
    


if __name__ == '__main__':
    nb_neurons = 200
    input = jax.random.normal(jax.random.PRNGKey(0), (25,2,40)) # batch, state, time
    print(input.shape)
    model_aug = MLP(jax.random.PRNGKey(0), 2, nb_neurons)
    print(model_aug)
    output = model_aug.get_derivatives(input)
    print(output.shape)

    model_phy = DampedPendulumParamPDE(is_complete=True, real_params=None) 
    print(model_phy)
    state = jax.random.normal(jax.random.PRNGKey(0), (1,2,3))
    out = model_phy(state)
    print(out.shape) # same shape as input
    print("params", model_phy.alpha, model_phy.omega0_square)  
    

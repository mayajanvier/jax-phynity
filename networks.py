import equinox as eqx
import jax
import jax.numpy as jnp
from collections import OrderedDict


### Physical model Fp
class DampedPendulumParamPDE(eqx.Module):
    is_complete: bool
    real_params: dict
    params_org: OrderedDict
    params: OrderedDict


    def __init__(self, is_complete=False, real_params=None):
        super().__init__()
        self.real_params = real_params
        self.is_complete = is_complete
        # TODO: ParameterDict replaced, see if problems later for derivatives
        self.params_org = OrderedDict({
            'omega0_square_org': jnp.array(0.2), 
            'alpha_org': jnp.array(0.1),
        })
        self.params = OrderedDict()
        if real_params is not None:
            self.params.update(real_params)

    def forward(self, state):
        if self.real_params is None: # Param ODE incomplete and complete have w0^2
            self.params['omega0_square'] = self.params_org['omega0_square_org']

        q = state[:,0:1]
        p = state[:,1:2]
        
        if self.is_complete: # Complete has damped pendulum 
            if self.real_params is None: # Only Param ODE complete have alpha
                self.params['alpha'] = self.params_org['alpha_org']
            (omega0_square, alpha) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * jnp.sin(q) - alpha * p
        else: # Incomplete is frictionless pendulum
            (omega0_square, ) = list(self.params.values())
            dqdt = p
            dpdt = - omega0_square * jnp.sin(q)

        return jnp.concat([dqdt, dpdt], axis=1)
    
### Data driven model Fa
class MLP(eqx.Module):
    layers: list # we need to define the type of the attributes of the class in jax
    state_c: int
    #initializer: jax.nn.initializers

    def __init__(self, key, state_c, hidden, init_gain=0.2):
        super().__init__()
        key1, key2, key3 = jax.random.split(key, 3)
        #self.initializer = jax.nn.initializers.orthogonal(scale=init_gain)
        self.state_c = state_c
        # orthogonal initialisation for the weights, biases to zero
        self.layers = [
            eqx.nn.Linear(state_c, hidden, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(hidden, hidden, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(hidden, state_c, key=key3)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_derivatives(self, x):
        batch_size, nc, T = x.shape 
        x = jnp.permute_dims(x, (0, 2, 1))
        x = jnp.reshape(x, (batch_size * T, nc))
        x = self.forward(x)
        x = jnp.reshape(x, (batch_size, T, self.state_c))
        x = jnp.permute_dims(x, (0, 2, 1))
        return x
    


if __name__ == '__main__':
    nb_neurons = 200
    input = jax.random.normal(jax.random.PRNGKey(0), (2,))
    print(input.shape)
    model = MLP(jax.random.PRNGKey(0), 2, nb_neurons)
    print(model)
    output = model.forward(input)
    print(output.shape)

    model = DampedPendulumParamPDE(is_complete=True, real_params=None) 
    print(model)
    state = jax.random.normal(jax.random.PRNGKey(0), (1,2,3))
    out = model.forward(state)
    print(out.shape) # same shape as input
    print(model.params, model.params_org)
    

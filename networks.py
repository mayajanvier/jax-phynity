import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

### Physical model Fp
omega0_square_org = 0.2
alpha_org = 0.1 
    
class PendulumParamPDE(eqx.Module):
    omega0_square: jax.Array # type makes it trainable
    alpha: jax.Array 
    is_damped: bool = eqx.static_field()  # Static field (not JAX-traceable)
    """ Unified pendulum for generation and inference """

    def __init__(self, is_damped=False, params={"alpha": 0.1, "omega0_square": 0.2}, is_true=False):
        super().__init__()
        self.is_damped = is_damped
        if is_true:
            self.omega0_square = params["omega0_square"] # float will not be trained
            self.alpha = params["alpha"] # float will not be trained
        else:
            self.omega0_square = jnp.array(params["omega0_square"], dtype=jnp.float32) # default unless precised
            if self.is_damped:
                self.alpha = jnp.array(params["alpha"], dtype=jnp.float32) # default unless precised
            else:
                self.alpha = 0.0 # float will not be trained

    def __call__(self, state): 
        # state should be (nc,)
        q, p = state
        dpdt = - self.omega0_square * jnp.sin(q) - self.alpha * p
        return jnp.array([p, dpdt]) 

class Lorenz(eqx.Module) :
    beta : jax.Array
    sigma : jax.Array
    rho : jax.Array

    def __init__(self, beta, sigma, rho) :
        super().__init__()
        self.beta = beta
        self.sigma = sigma
        self.rho = rho

    def __call__(self, t, s, args=None) :
        """
            x, y, z -> dxdt, dydt, dzdt
        """
        x, y, z = s
        dxdt = self.sigma*(y - x )
        dydt = self.rho * x - y - x*z
        dzdt =  x*y - self.beta*z
        return jnp.array([dxdt, dydt, dzdt])

class TwoBody(eqx.Module):
    def __call__(self, t, s):
        x, x_prime, y, y_prime = s
        x_second = -x/ (x**2 + y**2)**(3/2)
        y_second = -y/ (x**2 + y**2)**(3/2)
        return jnp.array([x_prime, x_second, y_prime, y_second])

    
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
        # shape (nc,)
        for layer in self.layers:
            x = layer(x)
        return x
    


if __name__ == '__main__':
    nb_neurons = 200
    input = jax.random.normal(jax.random.PRNGKey(0), (25,2,40)) # batch, state, time
    print(input.shape)
    model_aug = MLP(jax.random.PRNGKey(0), 2, nb_neurons)
    print(model_aug)
    output = jax.vmap(model_aug)(input)
    print(output.shape)

    model_phy = PendulumParamPDE(is_complete=True, real_params=None) 
    print(model_phy)
    state = jax.random.normal(jax.random.PRNGKey(0), (1,2,3))
    out = model_phy(state)
    print(out.shape) # same shape as input
    print("params", model_phy.alpha, model_phy.omega0_square)  
    

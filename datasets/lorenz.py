import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import os
import numpy as np
from datasets.base import ODEDataset

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux

MAX = np.iinfo(np.int32).max # maximum int value
# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# beta=8/3, sigma=10, rho=28
class Lorenz(ODEDataset):
    """Lorenz dataset generator using fixed-step RK solver."""

    def __init__(self, dt_num, num_steps_max, num_steps_rollout,
                 path, split, nb_traj, integration_method='RK4', **kwargs):
        super().__init__(
            name='lorenz',
            dt_num=dt_num,
            num_steps_max=num_steps_max,
            num_steps_rollout=num_steps_rollout,
            path=path,
            split=split,
            nb_traj=nb_traj,
            integration_method=integration_method,
        )

    def F(self, s, t):
        """
            x, y, z -> dxdt, dydt, dzdt
        """
        beta = 8/3
        sigma = 10.
        rho = 28.
        x, y, z = s
        dxdt = sigma*(y - x )
        dydt = rho * x - y - x*z
        dzdt =  x*y - beta*z
        return jnp.array([dxdt, dydt, dzdt])
    
    def _get_initial_condition(self, seed):
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX//2 - seed)
        else: # test
            key = random.PRNGKey(MAX - seed)

        keyX, keyY, keyZ = random.split(key, 3)
        x_rand = jax.random.normal(keyX) * 20.0
        y_rand = jax.random.normal(keyY) * 20.0
        z_rand = jax.random.normal(keyZ) * 20.0 + 20.0
        return jnp.array([x_rand, y_rand, z_rand]) 

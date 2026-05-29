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


class TwoBody(ODEDataset):
    """Two-body problem dataset generator using fixed-step RK solver."""

    def __init__(self, dt, num_steps_max, num_steps_rollout,
                 path, split, nb_traj, integration_method='RK4'):
        super().__init__(
            dataset_name='twobody',
            dt=dt,
            num_steps_max=num_steps_max,
            num_steps_rollout=num_steps_rollout,
            path=path,
            split=split,
            nb_traj=nb_traj,
            integration_method=integration_method,
        )

    def F(self, s, t):
        x, y, x_prime, y_prime = s
        r3 = (x**2 + y**2) ** (3 / 2)
        return jnp.array([x_prime, y_prime, -x / r3, -y / r3])

    def _get_initial_condition(self, seed):
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX // 2 - seed)
        else:  # test
            key = random.PRNGKey(MAX - seed)

        e = jax.random.uniform(key, shape=(), minval=0.5, maxval=0.7)
        return jnp.array([1 - e, 0.0, 0.0, jnp.sqrt((1 + e) / (1 - e))])

### With forcing
class TwoBodyForcing(ODEDataset):
    """Two-body problem + forcing dataset generator using fixed-step RK solver."""

    def __init__(self, dt, num_steps_max, num_steps_rollout,
                 path, split, nb_traj, integration_method='RK4'):
        super().__init__(
            dataset_name='twobody_forcing',
            dt=dt,
            num_steps_max=num_steps_max,
            num_steps_rollout=num_steps_rollout,
            path=path,
            split=split,
            nb_traj=nb_traj,
            integration_method=integration_method,
        )
    
    def F(self, s, t): 
        x, y, x_prime, y_prime, time = s
        forcing_magnitude = 0.01 * jnp.sin(2 * jnp.pi * t / 10)
        r = (x**2 + y**2) 
        x_second = -x / r**(3/2) - forcing_magnitude * y/r
        y_second = -y / r**(3/2) + forcing_magnitude * x/r
        return jnp.array([x_prime, y_prime, x_second, y_second, 1.0])

    def _get_initial_condition(self, seed):
        """Generate random initial conditions based on eccentricity."""
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX // 2 - seed)
        else:  # test
            key = random.PRNGKey(MAX - seed)

        e = jax.random.uniform(key, shape=(), minval=0.5, maxval=0.7)  # eccentricity
        return jnp.array([1 - e, 0.0, 0.0, jnp.sqrt((1 + e) / (1 - e)), 0.0])


if __name__ == '__main__':
    dataset = TwoBody(
        dt=0.01,
        num_steps_max=800,
        num_steps_rollout=2,
        path="data_test/twobody",
        split="train",
        nb_traj=40)
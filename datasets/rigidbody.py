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


class RigidBody(ODEDataset):
    """Rigidbody problem dataset generator using fixed-step RK solver."""

    def __init__(self, dt_num, num_steps_max, num_steps_rollout,
                 path, split, nb_traj, integration_method='RK4', **kwargs):
        super().__init__(
            name='rigidbody',
            dt_num=dt_num,
            num_steps_max=num_steps_max,
            num_steps_rollout=num_steps_rollout,
            path=path,
            split=split,
            nb_traj=nb_traj,
            integration_method=integration_method,
        )

    def F(self, s, t): 
        I = jnp.array([1.6, 1.0, 2 / 3]) # default White et al.
        y1, y2, y3 = s
        mat = jnp.array([
            [0, -y3, y2],
            [y3, 0, -y1],
            [-y2, y1, 0]])
        vect = jnp.array([y1/I[0], y2/I[1], y3/I[2]])
        dydt = mat @ vect
        return dydt
    
    def _get_initial_condition(self, seed):
        """Generate random initial conditions based on eccentricity."""
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX // 2 - seed)
        else:  # test
            key = random.PRNGKey(MAX - seed)

        phi = jax.random.uniform(key, shape=(), minval=0.5, maxval=1.5)  # initial angle
        return jnp.array([jnp.cos(phi), 0, jnp.sin(phi)])  # initial angular velocity


if __name__ == '__main__':
    dataset = RigidBody(
        dt=0.01,
        num_steps_max=1000,
        num_steps_rollout=2,
        path="data_test/rigidbody",
        split="train",
        nb_traj=40)
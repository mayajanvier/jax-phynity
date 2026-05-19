import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp
from einops import rearrange
import diffrax
from diffrax import diffeqsolve, ODETerm

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux
#from solvers.diffrax import RK_tableaux_diffrax

MAX = np.iinfo(np.int32).max # maximum int value

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

import os
import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random

# Assume RK_solver_fixed and RK_tableaux are defined elsewhere
class RigidBodyTrue:
    """Two-body problem dataset generator using fixed-step RK solver."""

    def __init__(self, dt, num_steps_max, num_steps_rollout, path, split, nb_traj, integration_method='RK4'):
        """
        Args:
            dt: float, time step
            num_steps: int, number of steps per trajectory
            path: str, directory path to save chunked trajectories
            split: str, 'train', 'val', or 'test'
            nb_traj: int, number of trajectories to generate
            integration_method: str, e.g. 'RK4', 'DOPRI5'
        """
        super().__init__()
        self.dt = dt
        self.num_steps_max = num_steps_max
        self.num_steps_rollout = num_steps_rollout
        self.nb_traj = nb_traj
        self.integration_method = integration_method
        self.split = split
        self.path = path
        self.I = jnp.array([1.6, 1.0, 2 / 3]) # default White et al.

        # Full trajectories are saved to a single .npy file
        self.data_path = f"datasets/rigidbody_full_{split}.npy"
        self.states = self._load_dataset()

    def __len__(self):
        return len(self.states)

    def F(self, s, t): 
            y1, y2, y3 = s
            mat = jnp.array([
                [0, -y3, y2],
                [y3, 0, -y1],
                [-y2, y1, 0]])
            vect = jnp.array([y1/self.I[0], y2/self.I[1], y3/self.I[2]])
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

    
    def _generate_and_save_dataset(self):
        """Generate full trajectories and save them in one .npy file."""
        all_states = []
        all_t = None
        num_steps = max(self.num_steps_rollout, self.num_steps_max) 
        for idx in range(self.nb_traj):
            y0 = self._get_initial_condition(idx)
            states, t, _, _ = RK_solver_fixed(
                fun=self.F,
                y0=y0,
                dt=self.dt,
                num_steps=num_steps,
                tableau=RK_tableaux[self.integration_method],
            )
            states = rearrange(states, 'nc T -> T nc')
            all_states.append(np.array(states))
            if all_t is None:
                all_t = np.array(t)

        all_states = np.stack(all_states)  # shape (nb_traj, T, nc)
        np.save(self.data_path, dict(states=all_states, t=all_t))
        print(f"Saved {self.nb_traj} trajectories to {self.data_path}")

    def _chunk_and_save_dataset(self):
        """Get full trajectories and save new dataset of chunked trajectories of num_steps_rollout steps."""
        # Load full trajectories
        if not os.path.exists(self.data_path):
            print(f"Generating {self.split} dataset...")
            self._generate_and_save_dataset()
        data = np.load(self.data_path, allow_pickle=True).item()
        print(data["states"].shape, data["states"][0].shape)  # (nb_traj, T, nc)
        
        if self.num_steps_rollout < self.num_steps_max: # needs chunking 
            # Create chunked dataset
            all_chunks = []
            T = data["states"][0].shape[0]
            print(T)
            for k in range(data['states'].shape[0]): 
                traj = data['states'][k]
                start, end = 0, self.num_steps_rollout+1
                while end < T+1:
                    print(start, end)
                    chunk = traj[start:end]
                    all_chunks.append(chunk)
                    start += self.num_steps_rollout
                    end = start + (self.num_steps_rollout+1)
            print(all_chunks[0].shape) # (num_steps_rollout+1, nc)
            all_chunks = np.stack(all_chunks)  # shape (num_chunks, num_steps_rollout, nc)
            print(all_chunks.shape)
            np.save(self.path, dict(states=all_chunks))
            print(f"Saved chunked dataset to {self.path}")
        else: # chunk=full
            np.save(self.path, dict(states=data['states']))
            print(f"Copied full dataset to {self.path}")

    def _load_dataset(self):
        """Load chunked dataset, generate if not existing."""
        if not os.path.exists(self.path+".npy"):
            print(f"Chunked dataset not found at {self.path}. Generating...")
            self._chunk_and_save_dataset()
        data = np.load(f"{self.path}.npy", allow_pickle=True).item()
        return data['states']

    def __getitem__(self, index):
        """Get one trajectory."""
        #states = self._load_dataset()
        return {"states": self.states[index]}
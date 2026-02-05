import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp
from einops import rearrange
import diffrax
from diffrax import diffeqsolve, ODETerm
import h5py

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

class KSTrue:
    """KS problem dataset generator using fixed-step RK solver."""

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

        # Full trajectories are saved to .h5 file
        self.data_path = f"data/KS_{split}.h5"
        self.states = self._load_dataset()

    def __len__(self):
        return len(self.states)

    def _chunk_and_save_dataset(self):
        """Get full trajectories and save new dataset of chunked trajectories of num_steps_rollout steps."""
        # Load full trajectories from .h5 file
        with h5py.File(self.data_path, "r") as f:
            if self.split == "test":
                states = f["test"]["pde_640-256"][:] # shape (nb_traj, T, nc)
            elif self.split == "train":
                states = f[self.split]["pde_140-256"][:]
            else:
                states = f["valid"]["pde_140-256"][:]
  
        data = {"states": states}
        print(data["states"].shape)  # (nb_traj, T, nc)
        
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
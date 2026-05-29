import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import h5py
from datasets.base import BaseDataset

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux
MAX = np.iinfo(np.int32).max # maximum int value

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

class KS(BaseDataset):
    """KS dataset from h5 files (generated with Brandsetter et al.)"""

    def __init__(self, *args, **kwargs):
        # full trajectories 
        self.data_path = f"data/KS_{kwargs['split']}.h5"
        super().__init__(
            *args,
            **kwargs
        )

    def _chunk_source_dataset(self):
        with h5py.File(self.data_path, "r") as f:
            if self.split == "test":
                states = f["test"]["pde_640-256"][:] # shape (nb_traj, T, nc)
            elif self.split == "train":
                states = f[self.split]["pde_140-256"][:]
            else:
                states = f["valid"]["pde_140-256"][:]
        return states


if __name__ == '__main__':
    dataset = KS(
        dt=0.01,
        num_steps_max=140,
        num_steps_rollout=140,
        path="data_test/ks",
        split="train",
        nb_traj=10)
import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp
from einops import rearrange
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

class Burgers(BaseDataset):
    """Burgers from h5 files, data from PDEBench"""

    def __init__(self, *args, **kwargs):
        # full trajectories 
        self.data_path = f"data/burgers_{kwargs['split']}.h5"
        super().__init__(
            *args,
            **kwargs
        )

    def _chunk_source_dataset(self):
        # Load full trajectories from .h5 file
        with h5py.File(self.data_path, "r") as f:
            states = f["pde"][:] # shape (nb_traj, T, nc)
        return states
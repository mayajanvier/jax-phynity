import numpy as np
import jax
import h5py
from datasets.base import BaseDataset

MAX = np.iinfo(np.int32).max # maximum int value

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

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
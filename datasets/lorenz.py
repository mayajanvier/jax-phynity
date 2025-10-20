import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp
from einops import rearrange
import os

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux
MAX = np.iinfo(np.int32).max # maximum int value

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# beta=8/3, sigma=10, rho=28
class LorenzTrue:
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
        self.beta = 8/3
        self.sigma = 10.
        self.rho = 28.
        self.dt = dt
        self.num_steps_max = num_steps_max
        self.num_steps_rollout = num_steps_rollout
        self.nb_traj = nb_traj
        self.integration_method = integration_method
        self.split = split
        self.path = path

        # Full trajectories are saved to a single .npy file
        self.data_path = f"/Users/mayajanvier/jax-phynity/datasets/lorenz_full_{split}.npy"

    def __len__(self):
        return self.nb_traj

    def F(self, s, t):
        """
            x, y, z -> dxdt, dydt, dzdt
        """
        x, y, z = s
        dxdt = self.sigma*(y - x )
        dydt = self.rho * x - y - x*z
        dzdt =  x*y - self.beta*z
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
    
    def _generate_and_save_dataset(self):
        """Generate full trajectories and save them in one .npy file."""
        all_states = []
        all_t = None
        for idx in range(self.nb_traj):
            y0 = self._get_initial_condition(idx)
            states, t, _, _ = RK_solver_fixed(
                fun=self.F,
                y0=y0,
                dt=self.dt,
                num_steps=self.num_steps_max,
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
        
        if self.num_steps_rollout != self.num_steps_max: # needs chunking 
            # Create chunked dataset
            all_chunks = []
            T = data["states"][0].shape[0]
            print(T)
            for k in range(data['states'].shape[0]): 
                traj = data['states'][k]
                start, end = 0, self.num_steps_rollout+1
                while end <= T+1:
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
        states = self._load_dataset()
        return {"states": states[index]}
    

class LorenzTrueShelve():

    def __init__(self, dt, num_steps, path, split, nb_traj, integration_method='RK4') :
        super().__init__()
        self.beta = 8/3
        self.sigma = 10.
        self.rho = 28.
        self.dt = dt # time step
        self.num_steps = num_steps 
        self.nb_traj = nb_traj  
        self.integration_method = integration_method     
        self.path = path # to save dataset
        self.split = split # train, val or test
        self.data = shelve.open(path) # to store trajectories

    def __len__(self):
        return self.nb_traj

    def F(self, s, t):
        """
            x, y, z -> dxdt, dydt, dzdt
        """
        x, y, z = s
        dxdt = self.sigma*(y - x )
        dydt = self.rho * x - y - x*z
        dzdt =  x*y - self.beta*z
        return jnp.array([dxdt, dydt, dzdt])
    
    def _get_initial_condition(self, seed):
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX//2 - seed)
        else: # test
            key = random.PRNGKey(MAX - seed)

        keyX, keyY, keyZ = random.split(key, 3)
        # before jax.random.norma(keyX) * 20.0
        x_rand = jax.random.normal(keyX) * 20.0
        y_rand = jax.random.normal(keyY) * 20.0
        z_rand = jax.random.normal(keyZ) * 20.0 + 20.0
        return jnp.array([x_rand, y_rand, z_rand]) 
        #return jax.random.normal(key, 3) * 0.1 + jnp.array([0.0, 0.0, 25.0])
    
    def __getitem__(self, index): 
        if self.data.get(str(index)) is None: # if trajectory is not saved
            #print("Generating trajectory ", index)
            y0 = self._get_initial_condition(index)
            states, t, global_err, err_list = RK_solver_fixed(
                fun=self.F,
                y0=y0,
                dt=self.dt,
                num_steps=self.num_steps,
                tableau = RK_tableaux[self.integration_method],
                )
            # save data as numpy array for Dataloader
            states = rearrange(states, 'nc T -> T nc') 
            self.data[str(index)] = states
            self.data['t'] = t
        else:
            #print("Loading trajectory ", index)
            t = self.data['t']
            states = self.data[str(index)] # get trajectory from shelve
        return {'states': np.array(states), 't': np.array(t)}


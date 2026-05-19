import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp
from collections import OrderedDict
from einops import rearrange
import os

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux

# DT = 0.5
# TF = 20
MAX = np.iinfo(np.int32).max # maximum int value 

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

class DoublePendulum:
    """Double pendulum problem dataset generator using fixed-step RK solver."""

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
        self.m1, self.m2 = 1.0, 1.0 
        self.l1, self.l2 = 1.0, 1.0
        self.g = 9.81


        # Full trajectories are saved to a single .npy file
        self.data_path = f"datasets/doublependulum_full_{split}.npy"
        self.states = self._load_dataset()

    def __len__(self):
        return len(self.states)

    def dw1(self,s):
        theta1, theta2, w1, w2 = s
        return (
            -self.g * (2*self.m1 + self.m2) * jnp.sin(theta1) - self.m2 * self.g * jnp.sin(theta1 - 2*theta2) -
            2* jnp.sin(theta1-theta2) * self.m2 * (w2**2 * self.l2 + w1**2 * self.l1 * jnp.cos(theta1-theta2))
        ) /  (self.l1 * (2*self.m1 + self.m2 - self.m2 * jnp.cos(2*(theta1-theta2))))
    
    def dw2(self,s):
        theta1, theta2, w1, w2 = s
        return (
            2 * jnp.sin(theta1 - theta2) * (
                w1**2 * self.l1 * (self.m1 + self.m2) +
                self.g * (self.m1 + self.m2) * jnp.cos(theta1) +
                w2**2 * self.l2 * self.m2 * jnp.cos(theta1 - theta2)
            )
        ) / (self.l2 * (2*self.m1 + self.m2 - self.m2 * jnp.cos(2*(theta1 - theta2))))

    def F(self, s, t): 
        """Compute derivatives for double pendulum."""
        _, _, w1, w2 = s
        dtheta1_dt = w1
        dtheta2_dt = w2
        dw1_dt = self.dw1(s)
        dw2_dt = self.dw2(s)
        return jnp.array([dtheta1_dt, dtheta2_dt, dw1_dt, dw2_dt])
    
    def _get_initial_condition(self, seed):
        """Generate random initial conditions based on eccentricity."""
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX // 2 - seed)
        else:  # test
            key = random.PRNGKey(MAX - seed)

        phi = jax.random.uniform(key, shape=(), minval=jnp.pi/4, maxval=3*jnp.pi/4)  # initial angle
        return jnp.array([phi, phi, 0, 0])  # initial angular velocity

    
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

# JAX customate version 
class DampedPendulum():
    parameters = OrderedDict(omega0_square=(2 * jnp.pi / 12) ** 2, alpha=0.2) # T0=12

    def __init__(self, dt, num_steps, path, split, nb_traj, integration_method='RK4', params=None):
        super().__init__()
        self.dt = dt # time step
        self.num_steps = num_steps 
        self.nb_traj = nb_traj  
        self.params = OrderedDict()
        self.integration_method = integration_method     
        if params is None:
            self.params.update(self.parameters)
        else:
            self.params.update(params)
        self.path = path # to save dataset
        self.split = split # train, val or test
        self.data = shelve.open(path) # to store trajectories

    def __len__(self):
        return self.nb_traj

    def F(self, x, t):
        # dX/dt = F(X, t)
        return jnp.array([
            x[1], -self.params['omega0_square'] * jnp.sin(x[0]) - self.params['alpha'] * x[1]
            ])
    
    def _get_initial_condition(self, seed):
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX//2 - seed)
        else: # test 
            key = random.PRNGKey(MAX - seed)
        
        # Generate random numbers
        key, subkey1, subkey2 = random.split(key, 3)
        y0 = jax.random.uniform(subkey1, shape=(2,), minval=-1.0, maxval=1.0)  # Values in range [-1, 1]
        radius = jax.random.uniform(subkey2) + 1.3  # Values in range [1.3, 2.3]
        
        # Normalize y0 and scale by radius
        norm = jnp.sqrt(jnp.sum(y0 ** 2))
        y0 = y0 / norm * radius
        return y0
    
    def __getitem__(self, index): # get one trajectory
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
            states = rearrange(states, 'nc T -> T nc') 
            # save data as numpy array for Dataloader
            self.data[str(index)] = states
            self.data['t'] = t
        else:
            #print("Loading trajectory ", index)
            t = self.data['t']
            states = self.data[str(index)] # get trajectory from shelve
        return {'states': np.array(states), 't': np.array(t)}


    
if __name__ == '__main__':
    ### DampedPendulum dataset
    dt = 0.5
    time_horizon = 20
    # generate training data
    path = 'data/pendulum_data_train.npy'
    train_dataset = DampedPendulum(dt, time_horizon, path, group="train")
    print(len(train_dataset))
    # example
    dico0 = train_dataset[0]
    print(dico0["states"])
    # 50 trajectories for training/validation
    for index in range(50):
        train_dataset[index]
    
    # generate test data
    path = 'data/pendulum_data_test.npy'
    test_dataset = DampedPendulum(dt, time_horizon, path, group="test")
    # 25 trajectories for testing
    for index in range(25):
        test_dataset[index]






    
        






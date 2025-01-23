import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp
#import torch
#from torch.utils.data import Dataset
from collections import OrderedDict

from solvers.runge_kutta import RK_solver_fixed
from scipy.integrate import solve_ivp

# DT = 0.5
# TF = 20
MAX = np.iinfo(np.int32).max # maximum int value 

# Pytorch version from APHYNITY
# class DampedPendulumOriginal(Dataset):
#     __default_params = OrderedDict(omega0_square=(2 * math.pi / 6) ** 2, alpha=0.2)

#     def __init__(self, path, num_seq, time_horizon, dt, params=None, group='train'):
#         super().__init__()
#         self.len = num_seq
#         self.time_horizon = float(time_horizon)  # total time
#         self.dt = float(dt)  # time step
#         self.params = OrderedDict()
#         if params is None:
#             self.params.update(self.__default_params)
#         else:
#             self.params.update(params)
#         self.group = group
#         self.data = shelve.open(path)

#     def _f(self, t, x):  # coords = [q,p]
#         omega0_square, alpha = list(self.params.values())

#         q, p = np.split(x, 2)
#         dqdt = p
#         dpdt = -omega0_square * np.sin(q) - alpha * p
#         return np.concatenate([dqdt, dpdt], axis=-1)

#     def _get_initial_condition(self, seed):
#         np.random.seed(seed if self.group == 'train' else MAX-seed)
#         y0 = np.random.rand(2) * 2.0 - 1
#         radius = np.random.rand() + 1.3
#         y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
#         return y0

#     def __getitem__(self, index):
#         t_eval = torch.from_numpy(np.arange(0, self.time_horizon, self.dt))
#         if self.data.get(str(index)) is None:
#             y0 = self._get_initial_condition(index)
#             states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), y0=y0, method='DOP853', t_eval=t_eval, rtol=1e-10).y
#             self.data[str(index)] = states
#             states = torch.from_numpy(states).float()
#         else:
#             states = torch.from_numpy(self.data[str(index)]).float()
#         return {'states': states, 't': t_eval.float()}

#     def __len__(self):
#         return self.len

# JAX customate version 
class DampedPendulum():
    parameters = OrderedDict(omega0_square=(2 * jnp.pi / 12) ** 2, alpha=0.2) # T0=12

    def __init__(self, dt, time_horizon, path, params=None):
        super().__init__()
        self.dt = dt # time step
        self.time_horizon = time_horizon # final time 
        self.len = int(self.time_horizon / self.dt)  
        self.params = OrderedDict()     
        if params is None:
            self.params.update(self.parameters)
        else:
            self.params.update(params)
        self.path = path # to save dataset
        self.group = 'train' # train or test
        self.data = shelve.open(path) # to store trajectories

    def __len__(self):
        return self.len

    def F(self, x, t):
        # dX/dt = F(X, t)
        print(x.shape)
        return jnp.array([
            x[1], -self.params['omega0_square'] * jnp.sin(x[0]) - self.params['alpha'] * x[1]
            ])
    
    def _get_initial_condition(self, seed):
        key = random.PRNGKey(seed if self.group == 'train' else MAX - seed)
        
        # Generate random numbers
        key, subkey1, subkey2 = random.split(key, 3)
        y0 = jax.random.uniform(subkey1, shape=(2,), minval=-1.0, maxval=1.0)  # Values in range [-1, 1]
        radius = jax.random.uniform(subkey2) + 1.3  
        
        # Normalize y0 and scale by radius
        norm = jnp.sqrt(jnp.sum(y0 ** 2))
        y0 = y0 / norm * radius
        return y0
    
    def __getitem__(self, index): # get one trajectory
        t_eval = jnp.arange(0, self.time_horizon, self.dt) # array of time points
        if self.data.get(str(index)) is None: # if trajectory is not saved
            print("Generating trajectory ", index)
            y0 = self._get_initial_condition(index)
            states, global_err = RK_solver_fixed(
                fun=self.F,
                t_span=(0, self.time_horizon),
                y0=y0,
                method='DOPRI8',
                t_eval=t_eval,
                #rtol=1e-10
                )
            self.data[str(index)] = states
        else:
            print("Loading trajectory ", index)
            states = self.data[str(index)] # get trajectory from shelve
        return {'states': states, 't': t_eval}


    
if __name__ == '__main__':
    ### DampedPendulum dataset
    dt = 0.5
    time_horizon = 20
    path = 'data/damped_pendulum.npy'
    dataset = DampedPendulum(dt, time_horizon, path)
    print(len(dataset))
    dico0 = dataset[0]
    print(dico0["states"])





    
        






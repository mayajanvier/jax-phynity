import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp
from collections import OrderedDict

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux

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

    def __init__(self, dt, time_horizon, path, group, num_seq, method='RK4', params=None):
        super().__init__()
        self.dt = dt # time step
        self.time_horizon = time_horizon # final time 
        self.len = num_seq # number of trajectories 
        self.params = OrderedDict()
        self.method = method     
        if params is None:
            self.params.update(self.parameters)
        else:
            self.params.update(params)
        self.path = path # to save dataset
        self.group = group # train or test
        self.data = shelve.open(path) # to store trajectories

    def __len__(self):
        return self.len

    def F(self, x, t):
        # dX/dt = F(X, t)
        return jnp.array([
            x[1], -self.params['omega0_square'] * jnp.sin(x[0]) - self.params['alpha'] * x[1]
            ])
    
    def _get_initial_condition(self, seed):
        if self.group == 'train':
            key = random.PRNGKey(seed)
        elif self.group == 'val':
            key = random.PRNGKey(MAX/2 - seed)
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
        t_eval = jnp.arange(0, self.time_horizon, self.dt) # array of time points
        if self.data.get(str(index)) is None: # if trajectory is not saved
            #print("Generating trajectory ", index)
            y0 = self._get_initial_condition(index)
            states, global_err, err_list = RK_solver_fixed(
                fun=self.F,
                t_span=(0, self.time_horizon),
                y0=y0,
                t_eval=t_eval,
                tableau = RK_tableaux[self.method],
                #rtol=1e-10
                )
            # save data as numpy array for Dataloader
            states = states.T
            self.data[str(index)] = states
        else:
            #print("Loading trajectory ", index)
            states = self.data[str(index)] # get trajectory from shelve
        return {'states': np.array(states), 't': np.array(t_eval)}


    
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






    
        






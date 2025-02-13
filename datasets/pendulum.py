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






    
        






import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux
MAX = np.iinfo(np.int32).max # maximum int value

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

class TwoBody():
    """ To create original trajectories from which we later sample initial points on to create
    the 2body_init_*.npy files. Stored in data/twobody_curriculum """
    def __init__(self, dt, num_steps, path, split, nb_traj, integration_method='RK4') :
        super().__init__()
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
        x, y, x_prime, y_prime = s
        x_second = -x/ (x**2 + y**2)**(3/2)
        y_second = -y/ (x**2 + y**2)**(3/2)
        return jnp.array([x_prime, y_prime, x_second, y_second])
    
    def _get_initial_condition(self, seed):
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX//2 - seed)
        else: # test
            key = random.PRNGKey(MAX - seed)

        e = jax.random.uniform(key, shape=(), minval=0.5, maxval=0.7)  # eccentricity, ellipse
        return jnp.array([1-e, 0.0, 0.0, jnp.sqrt((1+e)/(1-e))])  

    
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
            self.data[str(index)] = states
            self.data['t'] = t
        else:
            #print("Loading trajectory ", index)
            t = self.data['t']
            states = self.data[str(index)] # get trajectory from shelve
        return {'states': np.array(states), 't': np.array(t)}


class TwoBody_init():

    def __init__(self, dt, num_steps, path, split, nb_traj, integration_method='RK4') :
        super().__init__()
        self.dt = dt # time step
        self.num_steps = num_steps 
        self.nb_traj = nb_traj  
        self.integration_method = integration_method     
        self.path = path # to save dataset
        self.split = split # train, val or test
        self.data = shelve.open(path) # to store trajectories
        self.init_path = f"/Users/mayajanvier/jax-phynity/datasets/2body_init_{split}.npy"
        self.inits = np.load(self.init_path)

    def __len__(self):
        return self.nb_traj

    def F(self, s, t):
        x, y, x_prime, y_prime = s
        x_second = -x/ (x**2 + y**2)**(3/2)
        y_second = -y/ (x**2 + y**2)**(3/2)
        return jnp.array([x_prime, y_prime, x_second, y_second])
    
    def _get_initial_condition(self, seed):
        init_state = self.inits[seed]
        return jnp.array(init_state)  

    
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
            self.data[str(index)] = states
            self.data['t'] = t
        else:
            #print("Loading trajectory ", index)
            t = self.data['t']
            states = self.data[str(index)] # get trajectory from shelve
        return {'states': np.array(states), 't': np.array(t)}
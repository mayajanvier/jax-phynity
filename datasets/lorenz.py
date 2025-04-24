import numpy as np
import math
import shelve
import jax
from jax import random
import jax.numpy as jnp

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux
MAX = np.iinfo(np.int32).max # maximum int value

# beta=8/3, sigma=10, rho=28

class LorenzTrue():

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

        return jax.random.normal(key, 3) * 0.1 + jnp.array([0.0, 0.0, 25.0])
    
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

if __name__ == '__main__':
    ### DampedPendulum dataset
    dt = 0.01
    time_horizon = 10
    num_steps = int(time_horizon / dt)
    # generate training data
    path = 'data/pendulum_data_train.npy'
    train_dataset = LorenzTrue(dt, num_steps, path, split="train", nb_traj=25, integration_method='RK4')
    for i in range(5):
        data = train_dataset[i]
        print(data['states'].shape)
        print(data['t'].shape)

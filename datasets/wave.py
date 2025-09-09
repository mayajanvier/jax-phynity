import numpy as np
import shelve
import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx
from solvers.runge_kutta import RK_solver_fixed, RK_tableaux

### Global parameters (from APHYNITY)
c = 330.0  # wave speed 
dx = 1  # spatial step size
MAX = np.iinfo(np.int32).max # maximum int value

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

class Wave() : 
    def __init__(self, dt, num_steps, path, split, nb_traj, integration_method='RK4') :
        super().__init__()
        self.dt = dt # time step
        self.num_steps = num_steps 
        self.nb_traj = nb_traj  
        self.integration_method = integration_method     
        self.path = path # to save dataset
        self.split = split # train, val or test
        self.data = shelve.open(path) # to store trajectories
        self.c = c
        self.lap = eqx.nn.Conv2d(2, 2, kernel_size=(3,3), padding='same', padding_mode='CIRCULAR', use_bias=False, key=random.PRNGKey(0))
        self.lap.weight.data = jnp.array([[[[0., 1., 0.],
                                            [1., -4, 1.],
                                            [0., 1., 0.]]]], dtype=jax.float)/(dx**2)
        
    def __len__(self):
        return self.nb_traj
    
    def F(self, y, t) : 
         u, v = y
         du = v 
         dv = self.c**2 * self.lap(u[None])[0]
         return jnp.array([du, dv])

    def _get_initial_condition(self, seed):
        if self.split == 'train':
            key = random.PRNGKey(seed)
        elif self.split == 'val':
            key = random.PRNGKey(MAX//2 - seed)
        else: # test
            key = random.PRNGKey(MAX - seed)

        key1, key2 = random.split(key, 2)
        rand_std = jax.random.uniform(key1, minval=10, maxval=100)
        # TODO paper no random mean, code yes
        #rand_mean = jax.random.randint(key2, minval=20, maxval=40, shape=(2,)) 
        condition = jnp.array(self._gaussian_cond(self.mesh[0], self.mesh[1], value=1., std=rand_std)).float().view(1, self.size, self.size)
        y0 = jnp.concatenate([condition, jnp.zeros(1, self.size, self.size)], dim=0)
        return y0
    
    def _gaussian_cond(self, x, y, std, mean=(32, 32), value=1.):
        return value * jnp.exp(-((x - mean[0]) ** 2 + (y - mean[1]) ** 2) / std)

    
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
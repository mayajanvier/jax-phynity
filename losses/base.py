import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from losses.loss import *


### TRAJECTORY LOSS
def MSEjax(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

#@eqx.filter_jit
def loss_trajectory(model, y, epoch_rollout_index):
    #print('loss_trajectory')
    x = y[:,0,:] # y0
    #import ipdb; ipdb.set_trace()
    y_pred = jax.vmap(model)(x)
    y_pred = jax.lax.dynamic_slice(y_pred, tuple([0]*y.ndim), (y.shape[0], epoch_rollout_index, *y.shape[2:])) 
    return MSEjax(y_pred, y), y_pred

class Loss:
    def __init__(self, cfg):
        # all static
        self.reg_loss_name = cfg.train.reg_loss_name
        self.aux_loss_names = cfg.train.aux_loss_names
        self.opt_mode = cfg.train.opt_mode
        self.min_op = cfg.train.min_op
        print(cfg.dataset.name)
        self.dataset_name = cfg.dataset.name
        # TODO update lambda for lagrangian optim
        self.lambda_ = cfg.train.lambda0  # static lambda for now
        self.finite_diff = cfg.train.finite_diff
        self.norm = cfg.train.norm # normalized or not

        self.F = self.F_dynamics
        self.V = self.V_directions
        self.aux_losses_dict = self.build_aux_losses()


    ### methods for supervised losses ###
    def F_dynamics(self):
        return F_REGISTRY[self.dataset_name]

    def V_directions(self):
        key = jax.random.PRNGKey(0)
        if self.dataset_name == "ns_incomp": 
            L_ns = cfg.dataset.size
            V = jax.random.normal(key, (10, L_ns, L_ns))
        else:
            V = jax.random.normal(key, (10, cfg.dataset.size))
            if self.dataset_name == "twobody_forcing":
                V = self.V.at[:,-1].set(0.0) # no time dependence for twobody_forcing
        return V

    
    def build_aux_losses(self):
        aux_losses_dict = {}
        for loss_name in self.aux_loss_names:
            aux_losses_dict[loss_name] = eqx.filter_jit(eqx.Partial(
                LOSS_REGISTRY[loss_name],
                norm=self.norm,
                # supervised only
                F=self.F,
                v=self.V,
                finite_diff=self.finite_diff, # unsupervised only
                min_op=self.min_op, # APHYINITY only
                ))
        return aux_losses_dict

    
    ### Final loss ###
    def loss_fn(self, model, y, epoch_rollout_index):
        loss_traj, y_pred = loss_trajectory(model, y, epoch_rollout_index)

        # build aux losses dict values 
        losses_values_dict = {"loss_traj": loss_traj}
        for key, aux_loss in aux_losses_dict.items():
            losses_values_dict[key] = aux_loss(model, y)

        if self.reg_loss_name=="none":
            return loss_traj, (y_pred, losses_values_dict)
        else:
            reg_loss = losses_values_dict[reg_loss_name]
            # constraint or traj first in lambda optim
            if self.opt_mode == "constraint": 
                total_loss = loss_traj * lambda_ + reg_loss
            elif self.opt_mode == "traj": 
                total_loss = reg_loss  * lambda_ + loss_traj 
            # Convert dict values to arrays for JIT safety
            losses_values_dict = jax.tree.map(lambda x: jnp.array(x), losses_values_dict)
            return loss_val, (y_pred, losses_values_dict)

        
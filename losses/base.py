import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from losses.loss import *
from losses.F_dynamics import F_REGISTRY


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
        self.tau2 = cfg.train.tau2
        self.min_op = cfg.train.min_op
        self.dataset_name = cfg.dataset.name
        self.finite_diff = cfg.train.finite_diff
        self.norm = cfg.train.norm # normalized or not
        self.F = self.F_dynamics
        self.aux_losses_dict = self.build_aux_losses()

    ### methods for supervised losses ###
    def F_dynamics(self, x):
        return F_REGISTRY[self.dataset_name](x)
    
    def build_aux_losses(self):
        aux_losses_dict = {}
        for loss_name in self.aux_loss_names:
            aux_losses_dict[loss_name] = eqx.filter_jit(eqx.Partial(
                LOSS_REGISTRY[loss_name],
                norm=self.norm,
                F=self.F, # supervised only
                finite_diff=self.finite_diff, # unsupervised only
                min_op=self.min_op, # APHYINITY only
                ))
        return aux_losses_dict

    def update_lambda(self, loss_train, lambda_):
        if self.tau2 == 0.0:
            pass
        else:
            if self.opt_mode == "constraint":
                lambda_ = lambda_ + self.tau2 * loss_train['loss_traj'].item()
            elif cfg.train.opt_mode == "traj":
                lambda_ = lambda_ + self.tau2 * loss_train[self.reg_loss_name].item()
                lambda_ = max(0.0, lambda_)  # ensure lambda is non-negative
        return lambda_
    
    ### Final loss ###
    def loss_fn(self, model, y, lambda_, epoch_rollout_index):
        loss_traj, y_pred = loss_trajectory(model, y, epoch_rollout_index)

        # build aux losses dict values 
        losses_values_dict = {"loss_traj": loss_traj}
        for key, aux_loss in self.aux_losses_dict.items():
            losses_values_dict[key] = aux_loss(model, y)

        if self.tau2 == 0.0:
            return loss_traj, (y_pred, losses_values_dict)
        else:
            reg_loss = losses_values_dict[self.reg_loss_name]
            # constraint or traj first in lambda optim
            if self.opt_mode == "constraint": 
                total_loss = loss_traj * lambda_ + reg_loss
            elif self.opt_mode == "traj": 
                total_loss = reg_loss  * lambda_ + loss_traj 
            # Convert dict values to arrays for JIT safety
            losses_values_dict = jax.tree.map(lambda x: jnp.array(x), losses_values_dict)
            return total_loss, (y_pred, losses_values_dict)

        
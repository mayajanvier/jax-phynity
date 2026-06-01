import os
import json
import wandb
import argparse
from omegaconf import OmegaConf

import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from einops import rearrange

from datasets import *
from torch.utils.data import DataLoader 
from forecasters import Forecaster, SNODE
from networks import PendulumParamPDE, MLP, MLPAngular, ConvNetEstimator1D, UNet1D, UNet2D, ConvNetEstimator2D
from utils import init_linear_weight, orthogonal_init, Logger, save, make_basedir, log
from utils import compute_metric, save_loss_local, log_wandb
from loss import loss_fn, init_jit_aux_loss, F_pendulum, F_lorenz, F_twobody
import numpy as np
from solvers.runge_kutta import RK_tableaux

DTYPE = jnp.float32

# Enable 64-bit precision in JAX
#jax.config.update("jax_enable_x64", True)
# switch to gpu if available
#jax.config.update('jax_platform_name', 'gpu')
# JAX_TRACEBACK_FILTERING=off
jax.config.update("jax_traceback_filtering", "off")

g = torch.Generator()
g.manual_seed(0)

def get_datasets(cfg):
    print(f"Loading dataset {cfg.dataset.name} ...")
    # define datasets 
    path = os.path.join(cfg.experiment.path, cfg.dataset.name + str(cfg.dataset.duration))
    num_steps_rollout = int(cfg.dataset.duration/cfg.dataset.dt_num)
    dataset_train = DATASET_REGISTRY[cfg.dataset.name](
        nb_traj=cfg.dataset.nb_traj_train,
        num_steps_rollout=num_steps_rollout, 
        path=path,
        split="train",
        **cfg.dataset)
    dataset_val   = DATASET_REGISTRY[cfg.dataset.name](
        nb_traj=cfg.dataset.nb_traj_val,
        num_steps_rollout=num_steps_rollout, 
        path=path,
        split="val",
        **cfg.dataset)

    # define dataloaders
    dataloader_train_params = {
                'dataset'    : dataset_train,
                'batch_size' : cfg.dataset.batch_size,
                'num_workers': 0,
                'pin_memory' : True,
                'drop_last'  : False,
                'shuffle'    : True,
                #'persistent_workers': True,
                'worker_init_fn': seed_worker,
                'generator':g,
            }
    dataloader_val_params = {
                'dataset'    : dataset_val,
                'batch_size' : cfg.dataset.batch_size,
                'num_workers': 0,
                'pin_memory' : True,
                'drop_last'  : False,
                'shuffle'    : False,
                #'persistent_workers': True,
                'worker_init_fn': seed_worker,
                'generator':g,
        }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_val   = DataLoader(**dataloader_val_params)
    return dataloader_train, dataloader_val

def get_model(cfg, train):
    mkey, ikey = jax.random.split(jax.random.PRNGKey(0))

    if cfg.dataset.name == "pendulum":  
        if cfg.model.phy_option == "true": # true damped pendulum
            model_phy = PendulumParamPDE(is_damped=True, params=train.dataset.params, is_true=True)
        elif cfg.model.phy_option == "complete": # damped pendulum
            model_phy = PendulumParamPDE(is_damped=True)
        else:
            model_phy = PendulumParamPDE(is_damped=False)
        state_c = 2
    elif cfg.dataset.name == "lorenz":
        model_phy, state_c = None, 3
    elif cfg.dataset.name == "twobody":
        model_phy, state_c = None, 4
    elif cfg.dataset.name == "twobody_forcing":
        model_phy, state_c = None, 5
    elif cfg.dataset.name == "doublependulum":
        model_phy, state_c = None, 4
    elif cfg.dataset.name == "rigidbody":
        model_phy, state_c = None, 3
    elif cfg.dataset.name == "ks":
        model_phy, state_c = None, 256
    elif cfg.dataset.name == "burgers":
        model_phy, state_c = None, 1024
    elif cfg.dataset.name == "ns_incomp":
        model_phy, state_c = None, cfg.model.dim_feat
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    if cfg.dataset.name == "doublependulum":
        model_aug = MLPAngular(key=mkey, state_c=state_c, hidden=cfg.model.hidden)
    else:
        if cfg.model.architecture == "mlp":
            model_aug = MLP(key=mkey, state_c=state_c, hidden=cfg.model.hidden)
        elif cfg.model.architecture == "unet":
            model_aug = UNet1D(key=mkey, hidden=cfg.model.hidden) 
        elif cfg.model.architecture == "unet2d":
            model_aug = UNet2D(in_channels=1, out_channels=1, init_features=state_c, key=mkey)
        elif cfg.model.architecture == "convnet":
            model_aug = ConvNetEstimator1D(key=mkey, hidden=cfg.model.hidden)
        elif cfg.model.architecture == "convnet2d":
            model_aug = ConvNetEstimator2D(key=mkey, hidden=cfg.model.hidden)

    model_aug = init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=cfg.model.init_gain)

    if cfg.model.name == "snode":
        net = SNODE(
            model_phy=model_phy,
            model_aug=model_aug,
            is_augmented=cfg.model.aug_option,
            is_phy=cfg.model.phy_option,
            dt=cfg.dataset.dt_factor * train.dataset.dt,
            num_steps=int(train.dataset.num_steps_rollout / cfg.dataset.dt_factor),
            dataset=cfg.dataset.name,
            gamma=cfg.model.gamma,
            integration_method=cfg.model.integration_method,
        )
    else:
        net = Forecaster(
            model_phy=model_phy,
            model_aug=model_aug,
            is_augmented=cfg.model.aug_option,
            is_phy=cfg.model.phy_option,
            dt=cfg.dataset.dt_factor * train.dataset.dt,
            num_steps=int(train.dataset.num_steps_rollout / cfg.dataset.dt_factor),
            integration_method=cfg.model.integration_method,
        )
    return net

def get_optimizer(cfg):
    return optax.adam(learning_rate=cfg.train.lr, b1=0.9, b2=0.999)

def get_Fa_prime_true(cfg, data):
    states = jnp.array(data['states'])[:,::cfg.dataset.dt_factor,:]
    x_in = rearrange(states, 'b T nc -> (b T) nc')
    if cfg.dataset.name == 'lorenz':
        true_deriv = jax.vmap(F_lorenz)(x_in)
    elif cfg.dataset.name == 'pendulum':
        true_deriv = jax.vmap(F_pendulum)(x_in) 
    elif cfg.dataset.name == 'twobody':
        true_deriv = jax.vmap(F_twobody)(x_in)
    true_deriv = rearrange(true_deriv, '(b T) nc -> b T nc', b=states.shape[0])
    F_prime_true = jnp.abs(true_deriv[:,1:,:]-true_deriv[:,:-1,:]) / jnp.abs(states[:,1:,:] - states[:,:-1,:])
    return F_prime_true

class CurriculumScheduler:
    def __init__(self, cfg, train_data):
        self.cfg = cfg
        self.program = cfg.train.curriculum
        self.index_train_min = 10  # 10 steps minimum
        self.dt = cfg.dataset.dt_factor * train_data.dataset.dt
        self.index_train_max = int(cfg.dataset.duration / self.dt)

        # training length heuristic
        if self.program == "curr":
            self.nepoch = int(self.index_train_max / self.index_train_min) * 100 + 400
        else:
            self.nepoch = cfg.train.nepoch #int(self.index_train_max / self.index_train_min) * 100 + 400


        # internal state
        self.epoch_rollout_index = None
        self.reset()

        # TODO
        # old curriculum (smoother)    
        #epoch_rollout_index = min(int(duration/hyperparameters_model["dt"]) + 1, int((duration/hyperparameters_model["dt"])*(epoch/nepoch)) + 2)

    def reset(self):
        """Initialize rollout index at the beginning of training."""
        if self.program == "curr":
            self.epoch_rollout_index = min(self.index_train_min + 1, self.index_train_max) # if rollout smaller than 10 steps 
        else:  # no_curr
            self.epoch_rollout_index = self.index_train_max + 1

    def step(self, epoch: int):
        """Update rollout index given current epoch."""
        if self.program == "curr":
            if (epoch + 1) % 100 == 0:
                self.epoch_rollout_index = min(
                    self.epoch_rollout_index + self.index_train_min,
                    self.index_train_max + 1,
                )
        elif self.program == "no_curr":
            self.epoch_rollout_index = self.index_train_max + 1
        else:
            raise ValueError(f"Unknown curriculum program: {self.program}")

        return self.epoch_rollout_index, self.nepoch

def train(cfg, train_data, val_data, net, optimizer):
    print("Training starting...")
    name_experiment = cfg.model.phy_option+"_"+("aug" if cfg.model.aug_option else "physics")+"_"+str(cfg.dataset.duration)
    # Setup logging
    wandb_run = None
    if cfg.logging.use_wandb:
        wandb_run = wandb.init(
            project=cfg.logging.project,
            name=name_experiment,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_id = wandb.run.id
        # save code in w&b
        wandb.run.log_code("./")
    else:
        wandb_id = "no_wandb"
    
    exp_path = make_basedir(cfg.experiment.path, f"{name_experiment}_{wandb_id}")
    logger = Logger(filename=os.path.join(exp_path, "log.txt"))

    # Save config for reproducibility
    with open(os.path.join(exp_path, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # optimizer initialization
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    # Jitted loss
    print("Initializing loss function...")
    aux_losses_dict = init_jit_aux_loss(cfg.train.aux_loss_names, cfg.train.min_op, cfg.dataset.name, cfg.train.finite_diff, cfg.train.lambda_hutch)
    loss_fn_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(
            eqx.Partial(loss_fn, 
                reg_loss_name=cfg.train.reg_loss_name,
                aux_losses_dict=aux_losses_dict,
                opt_mode=cfg.train.opt_mode
            ),
            has_aux=True
            )
        )
    #loss_fn_grad = eqx.Partial(loss_fn, reg_loss_name=cfg.train.reg_loss_name, aux_losses_dict=aux_losses_dict, opt_mode=cfg.train.opt_mode)
    #loss_fn_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn_grad, has_aux=True))
    # for model selection over val loss
    loss_test_min = None
    # curriculum
    scheduler = CurriculumScheduler(cfg, train_data)
    _lambda = cfg.train.lambda0

    for epoch in range(scheduler.nepoch): 
        print(f"Epoch {epoch+1}/{scheduler.nepoch}")
        epoch_rollout_index, nepoch = scheduler.step(epoch)
        #print(f"Epoch {epoch}, rollout index: {epoch_rollout_index}")

        loss_train = {aug_loss_name: 0.0 for aug_loss_name in cfg.train.aux_loss_names}
        loss_train['loss_traj'] = 0.0             

        #print(f"epoch {epoch} / {nepoch}, rollout index {epoch_rollout_index}")
        for _ in range(cfg.train.niter): # APHYNITY
            for iteration, data in enumerate(train_data, 0):
                print(f"Epoch {epoch+1}/{scheduler.nepoch}, Iteration {iteration+1}/{len(train_data)}", end="\r")
                # --------------------------
                ### TRAIN STEP
                # --------------------------

                # use asarray to avoid creating new data
                states = jnp.asarray(data['states'][:,:epoch_rollout_index*cfg.dataset.dt_factor:cfg.dataset.dt_factor,:], dtype="float32") # bs, time, nc with diffrax, bs, nc, time with RK_solver_fixed
                #states = jax.device_put(states) # move data to GPU if available
                #t = jnp.array(data['t'][0])[::cfg.dataset.dt_factor]
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda), epoch_rollout_index=epoch_rollout_index) #, Fa_prime_true=F_prime_true)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(net, eqx.is_array))
                net = eqx.apply_updates(net, updates)
                #losses_dict_np = jax.tree_map(lambda x: float(x), losses_dict)
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_train[losses_values_dict_key] += losses_values_dict_value
                # pour voir si on train bien
                if cfg.train.log_param_error:
                    metric = compute_metric(net, train_data)
                else:
                    metric = {}

        # average loss over train set
        for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
            loss_train[losses_values_dict_key] /= (iteration + 1) * cfg.train.niter
        
        # update lambda
        if cfg.train.reg_loss_name == "none":
            pass # _lambda stays constant
        else:
            if cfg.train.opt_mode == "constraint":
                _lambda = _lambda + cfg.train.tau2 * loss_train['loss_traj'].item()
            elif cfg.train.opt_mode == "traj":
                _lambda = _lambda + cfg.train.tau2 * loss_train[cfg.train.reg_loss_name].item()
                _lambda = max(0.0, _lambda)  # ensure lambda is non-negative

        ### LOGS 
        total_iteration = epoch * (len(train_data)) + (iteration + 1)
        if total_iteration % cfg.train.nlog == 0:
            log(train_data, epoch, iteration, loss_train | metric, nepoch)
        # log metrics to wandb 
        log_wandb(net, train_data, _lambda, loss_train, 'train', epoch_rollout_index, cfg.train.log_param_error)
        
        # --------------------------
        ### VALIDATION STEP
        # --------------------------
        if total_iteration % cfg.train.nval == 0:
            loss_test = {aug_loss_name: 0.0 for aug_loss_name in cfg.train.aux_loss_names}
            loss_test["loss_traj"] = 0.0 #{"loss_traj": 0.0, "loss_op": 0.0}
            #if cfg.model.phy_option != "none":
            for j, data_test in enumerate(val_data, 0):
                # no backpropagation
                states = jnp.asarray(data_test['states'][:,::cfg.dataset.dt_factor,:], dtype="float32") # bs, time, nc with diffrax
                #states = jax.device_put(states) # move data to GPU if available
                #t = jnp.array(data_test['t'][0])[::cfg.dataset.dt_factor]
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda), epoch_rollout_index=states.shape[1]) #, Fa_prime_true=F_prime_true_val)
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_test[losses_values_dict_key] += losses_values_dict_value

                # for long rollout evaluation
                # states = jnp.array(data_test['states'])[:,::cfg.dataset.dt_factor,:] # bs, time, nc with diffrax
                # #states = jnp.array(val_data['states'])[:,::cfg.dataset.dt_factor,:]
                # # get model for long rollout
                # T = states.shape[1]-1
                # pred = jax.vmap(lambda y0: net.validation_call(y0, T))(states[:,0,:])
                # print(pred.shape, states.shape)
                # loss_test["loss_traj"] = jnp.mean((pred - states)**2)
                # j = 0

                
            # average loss over test set
            for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                loss_test[losses_values_dict_key] /= (j + 1)

            ### LOGS
            print('#' * 80)
            log(train_data, epoch, iteration, loss_test | metric, nepoch)
            print('#' * 80)
            # log metrics to wandb
            log_wandb(net, val_data, _lambda, loss_test, 'val', epoch_rollout_index, cfg.train.log_param_error)
            
            # save model over loss_test
            if loss_test_min == None or loss_test_min > loss_test["loss_traj"].item():
                loss_test_min = loss_test['loss_traj'].item()
                # save model using equinox
                # TODO how to also save optimizer state?
                hyperparameters = {
                    "epoch": epoch,
                    "loss": loss_test_min,
                    "lambda": _lambda,
                    }
                save(exp_path + f'/model_{loss_test_min:.3e}.eqx', hyperparameters, net)

    if wandb_run:
        wandb_run.finish()

def main(cfg):
    train_data, val_data = get_datasets(cfg)
    net = get_model(cfg, train_data)
    optimizer = get_optimizer(cfg)
    train(cfg, train_data, val_data, net, optimizer)

### DEPRECATED
def training_routine(train, test, net, optimizer, min_op, _lambda,tau_1, tau_2, niter, path, device, aux_loss_names=["loss_Fa", "loss_Fa_primeX"], reg_loss_name="none", dt_factor=1, nlog=1, nupdate=1, nepoch=10, name_project="Damped_Pendulum", log_param_error=True, duration=None, dataset_name="pendulum", model_phy_option="none", model_aug_option=False):   
    # Setup to save logs 
    name_experiment = model_phy_option+"_"+("aug" if model_aug_option else "physics")+"_"+str(duration)
    
    # Weights and Biases
    # TODO single config file
    wandb.init(
        project = name_project, # set the wandb project where this run will be logged
        name = name_experiment,     
        config={                    # track hyperparameters and run metadata
        "learning_rate": tau_1,
        "tau2": tau_2,
        "architecture": model_phy_option,
        "epochs": nepoch,
        "batch_size": train.batch_size,
        "Fa_norm": min_op,
        "lambda0": _lambda,    
        "dt_data": train.dataset.dt,
        "dt_train": dt_factor * train.dataset.dt,
        "duration": duration,
        "niter": niter,
        }
        )

    # get w&b id
    wandb_id = wandb.run.id
    exp_path = make_basedir(path, name_experiment+f"_{str(wandb_id)}")
    print(exp_path)
    logger = Logger(filename=os.path.join(exp_path, 'log.txt'))
    # save code in w&b
    wandb.run.log_code("./")


    # save hyperparameters and settings (in case wandb crash)
    # TODO single config file 
    hyperparameters_model = {
        'lambda0': _lambda,
        'tau_1': tau_1,
        'tau_2': tau_2,
        'niter': niter,
        'min_op': min_op,
        'nepoch': nepoch,
        'nlog': nlog,
        'nupdate': nupdate,
        'id': wandb_id,
        'dt': dt_factor * train.dataset.dt,
        'dt_num': train.dataset.dt,
    }
    with open(os.path.join(exp_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters_model, f)

    # optimizer initialization
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    loss_test_min = None
    # in case of wandb crash 
    train_losses = []
    val_losses = []
    # curriculum
    index_train_min = 10 # 10 steps minimum
    index_train_max = int(duration / (dt_factor * train.dataset.dt))
    epoch_rollout_index = index_train_min + 1 # index 0 is y0=x0
    nepoch = int(index_train_max / index_train_min) * 100 + 400 # 400 epochs for the last part of the training at T_train_max
    #nepoch=1000
    # fix permanent variables
    aux_losses_dict = init_jit_aux_loss(aux_loss_names, min_op, dataset_name)
    loss_fn_grad = eqx.Partial(loss_fn, reg_loss_name=reg_loss_name, aux_losses_dict=aux_losses_dict) 
    # for loss_Fa_supervisedYX
    bool_i = 0
    bool_j = 0 
    for epoch in range(nepoch): 
        loss_train = {aug_loss_name: 0.0 for aug_loss_name in aux_loss_names}
        loss_train['loss_traj'] = 0.0 
        # curriculum
        # if (epoch+1) % 100 == 0:
        #     epoch_rollout_index = min(epoch_rollout_index+index_train_min, index_train_max+1)

        # no curriculum
        epoch_rollout_index = index_train_max+1
            
        # old curriculum (smoother)    
        #epoch_rollout_index = min(int(duration/hyperparameters_model["dt"]) + 1, int((duration/hyperparameters_model["dt"])*(epoch/nepoch)) + 2)
        #print(f"epoch {epoch} / {nepoch}, rollout index {epoch_rollout_index}")
        for _ in range(niter): # APHYNITY
            for iteration, data in enumerate(train, 0):
                ### TRAIN STEP
                if bool_i == 0: # get F_prime_true over train only once
                    states = jnp.array(data['states'])[:,:,::dt_factor]
                    #x_in = rearrange(states, 'b nc T -> b (T) nc')
                    x_in = states
                    if dataset_name == 'lorenz':
                        true_deriv = jax.vmap(F_lorenz)(x_in)
                    elif dataset_name == 'pendulum':
                        true_deriv = jax.vmap(F_pendulum)(x_in) # b nc T
                    elif dataset_name == 'twobody':
                        true_deriv = jax.vmap(F_twobody)(x_in)
                    # for loss_Fa_prime_supervisedYX
                    F_prime_true = jnp.abs(true_deriv[:,:,1:]-true_deriv[:,:,:-1]) / jnp.abs(x_in[:,:,1:] - x_in[:,:,:-1])
                    F_prime_true = rearrange(F_prime_true, 'b nc T -> b (T) nc') 
                    bool_i += 1
                states = jnp.array(data['states'])[:,:,:epoch_rollout_index*dt_factor:dt_factor]
                t = jnp.array(data['t'][0])[::dt_factor]
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda), epoch_rollout_index=epoch_rollout_index, Fa_prime_true=F_prime_true)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(net, eqx.is_array))
                net = eqx.apply_updates(net, updates)
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_train[losses_values_dict_key] += losses_values_dict_value
                # pour voir si on train bien
                if log_param_error:
                    metric = compute_metric(net, train)
                else:
                    metric = {}

        # average loss over train set
        for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
            loss_train[losses_values_dict_key] /= (iteration + 1) * niter
        
        # update lambda
        _lambda = _lambda + tau_2 * loss_train['loss_traj'].item()

        ### LOGS 
        total_iteration = epoch * (len(train)) + (iteration + 1)
        if total_iteration % nlog == 0:
            log(train, epoch, iteration, loss_train | metric, nepoch)
        # log metrics to wandb 
        log_wandb(net, train, _lambda, loss_train, 'train', epoch_rollout_index, log_param_error)
        
        ### VALIDATION STEP
        if total_iteration % nupdate == 0:
            loss_test = {aug_loss_name: 0.0 for aug_loss_name in aux_loss_names}
            loss_test["loss_traj"] = 0.0 #{"loss_traj": 0.0, "loss_op": 0.0}
            for j, data_test in enumerate(test, 0):
                if bool_j == 0: # get F_prime_true over val only once
                    states = jnp.array(data['states'])[:,:,::dt_factor]
                    x_in = states
                    if dataset_name == "pendulum":
                        true_deriv = jax.vmap(F_pendulum)(x_in)
                    elif dataset_name == "lorenz":
                        true_deriv = jax.vmap(F_lorenz)(x_in)
                    elif dataset_name == "twobody":
                        true_deriv = jax.vmap(F_twobody)(x_in)
                    F_prime_true_val = jnp.abs(true_deriv[:,:,1:]-true_deriv[:,:,:-1]) / jnp.abs(x_in[:,:,1:] - x_in[:,:, :-1])
                    F_prime_true_val = rearrange(F_prime_true_val, 'b nc T -> b (T) nc') 
                    bool_j += 1
                # no backpropagation
                states = jnp.array(data_test['states'])[:,:,::dt_factor]
                t = jnp.array(data_test['t'][0])[::dt_factor]
                # _lambda should be an array for jit to not recompile when its value changes
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda), epoch_rollout_index=states.shape[2], Fa_prime_true=F_prime_true_val) 
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_test[losses_values_dict_key] += losses_values_dict_value

            # average loss over test set
            for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                loss_test[losses_values_dict_key] /= (j + 1)

            ### LOGS
            print('#' * 80)
            log(train, epoch, iteration, loss_test | metric, nepoch)
            print('#' * 80)
            # log metrics to wandb
            log_wandb(net, test, _lambda, loss_test, 'val', epoch_rollout_index, log_param_error)
            # save epoch losses to csv file
            save_loss_local(val_losses, train_losses, loss_test, loss_train, exp_path)
            
            # save model over loss_test
            if loss_test_min == None or loss_test_min > loss_test["loss_traj"].item():
                loss_test_min = loss_test['loss_traj'].item()
                # save model using equinox
                # TODO how to also save optimizer state?
                hyperparameters = {
                    "epoch": epoch,
                    "loss": loss_test_min,
                    "lambda": _lambda,
                    }
                save(exp_path + f'/model_{loss_test_min:.3e}.eqx', hyperparameters, net)

def train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, integration_method, data_integration_method="RK4", dt_factor=1, dt_num=0.5, duration=20, init_gain=0.2):
    train, val, _ = init_dataloaders(dataset_name, data_integration_method, os.path.join(path, dataset_name+str(duration)), dt_num=dt_num, duration=duration)

    if dataset_name == 'pendulum':
        ### Model definition
        if model_phy_option == 'true': # true damped pendulum
            model_phy = PendulumParamPDE(is_damped=True, params=train.dataset.params, is_true=True)
        elif model_phy_option == 'complete': # damped pendulum
            model_phy = PendulumParamPDE(is_damped=True)
        else: 
            model_phy = PendulumParamPDE(is_damped=False)

        mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
        model_aug = MLP(key=mkey, state_c=2, hidden=200)
        model_aug = init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=init_gain) 
        net = Forecaster(
            model_phy=model_phy,
            model_aug=model_aug,
            is_augmented=model_aug_option,
            is_phy=model_phy_option,
            dt=dt_factor * train.dataset.dt, # enabling comparison with GT for error scheme experiment 
            num_steps=int(train.dataset.num_steps / dt_factor), 
            integration_method=integration_method, # RK2 for error scheme experiment
        )

        ### Training parameters
        name_project ="Damped_Pendulum"
        log_param_error = True
        tau_1 = 1e-3 # 1e-3 dans le git APHYNITY, 1 dans le papier
        niter = 5
        min_op = 'l2'
        nepoch = 1000 # TODO remove 
        nlog = 5
        nupdate = 5

        # TODO: which system for simple usage ?
        if model_phy_option == 'incomplete': # my parameters
            lambda_0 = 10.0
            tau_2 = 100.0
        elif model_phy_option == 'complete':
            lambda_0 = 1000.0
            tau_2 = 100.0
        elif model_phy_option == 'none': # loss_traj only
            reg_loss_name = 'none'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX", "loss_Fa_prime_supervisedX"]
            lambda_0 = 1.0 
            tau_2 = 0.0 
        elif model_phy_option == 'incomplete_no_Fa': # loss_traj only
            lambda_0 = 1.0
            tau_2 = 10.0
        # elif model_phy_option == 'none_Fa': # paper parameters 
        #     lambda_0 = 10.0
        #     tau_2 = 10.0
        elif model_phy_option == 'none_Fa':
            reg_loss_name = 'loss_Fa'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX","loss_Fa_prime_supervisedX"]
            lambda_0 = 100.0
            tau_2 = 0.0
            # lambda_0 = 100.0
            # tau_2 = 0.0
        elif model_phy_option == 'true': # loss_traj only
            lambda_0 = 0.0
            tau_2 = 0.0
        elif model_phy_option == 'none_Fa_prime':
            lambda_0 = 1000.0
            tau_2 = 0.0
        elif model_phy_option == 'none_Fa_prime_supX':
            reg_loss_name = 'loss_Fa_prime_supervisedX'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX", "loss_Fa_prime_supervisedX"]
            lambda_0 = 1.0
            tau_2 = 0.0
        elif model_phy_option == 'incomplete_Fa_prime':
            lambda_0 = 10.0
            tau_2 = 100.0
    
    elif dataset_name == 'lorenz':
        ### Model definition
        model_phy = None # Neural ODE 
        mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
        model_aug = MLP(key=mkey, state_c=3, hidden=200)
        model_aug = init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=init_gain) 
        net = Forecaster(
            model_phy=model_phy,
            model_aug=model_aug,
            is_augmented=model_aug_option,
            is_phy=model_phy_option,
            dt=dt_factor * train.dataset.dt, # enabling comparison with GT for error scheme experiment 
            num_steps=int(train.dataset.num_steps / dt_factor), 
            integration_method=integration_method, # RK2 for error scheme experiment
        )
        
        ### Training parameters
        name_project ="Lorenz"
        log_param_error = False
        tau_1 = 1e-3
        niter = 5 
        nlog = 5
        nupdate = 5
        min_op = 'l2'
        nepoch = 600 # TODO remove
        lambda_0 = 1.0 
        tau_2 = 0.0 

        if model_phy_option == 'none':
            reg_loss_name = 'none'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX", "loss_Fa_prime_supervisedX"]
        elif model_phy_option == 'none_Fa_prime_supX':
            reg_loss_name = 'loss_Fa_prime_supervisedX'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX", "loss_Fa_prime_supervisedX"]
            lambda_0 = 100.0
            tau_2 = 0.0
        elif model_phy_option == "none_Fa_prime_supX_direct":
            reg_loss_name = "loss_Fa_prime_supervisedX_direct"
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX","loss_Fa_prime_supervisedX", "loss_Fa_prime_supervisedX_direct"]
            lambda_0 = 100.0
            tau_2 = 0.0
        elif model_phy_option == 'none_Fa':
            reg_loss_name = 'loss_Fa'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX","loss_Fa_prime_supervisedX"]
            lambda_0 = 10.0
            tau_2 = 10.0
        elif model_phy_option == 'none_Fa_prime_supYX':
            reg_loss_name = 'loss_Fa_prime_supervisedYX'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX","loss_Fa_prime_supervisedYX"]
            lambda_0 = 1000.0
            tau_2 = 0.0
    
    elif dataset_name == 'twobody':
        ### Model definition
        model_phy = None # Neural ODE 
        mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
        model_aug = MLP(key=mkey, state_c=4, hidden=200)
        model_aug = init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=init_gain) 
        net = Forecaster(
            model_phy=model_phy,
            model_aug=model_aug,
            is_augmented=model_aug_option,
            is_phy=model_phy_option,
            dt=dt_factor * train.dataset.dt, # enabling comparison with GT for error scheme experiment 
            num_steps=int(train.dataset.num_steps / dt_factor), 
            integration_method=integration_method, # RK2 for error scheme experiment
        )
        
        ### Training parameters
        name_project ="TwoBodyProblem"
        log_param_error = False
        tau_1 = 1e-3
        niter = 5 
        nlog = 5
        nupdate = 5
        min_op = 'l2'
        nepoch = 600 # TODO remove
        lambda_0 = 1.0 
        tau_2 = 0.0 

        if model_phy_option == 'none':
            reg_loss_name = 'none'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX", "loss_Fa_prime_supervisedX"]
        elif model_phy_option == 'none_Fa_prime_supX':
            reg_loss_name = 'loss_Fa_prime_supervisedX'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX", "loss_Fa_prime_supervisedX", "loss_Fa_prime_supervisedYX"]
            lambda_0 = 10.0
            tau_2 = 1000.0
        elif model_phy_option == "none_Fa_prime_supX_direct":
            reg_loss_name = "loss_Fa_prime_supervisedX_direct"
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX","loss_Fa_prime_supervisedX", "loss_Fa_prime_supervisedX_direct"]
            lambda_0 = 1.0
            tau_2 = 0.0
        elif model_phy_option == 'none_Fa':
            reg_loss_name = 'loss_Fa'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX","loss_Fa_prime_supervisedX", "loss_Fa_prime_supervisedYX"]
            lambda_0 = 10.0
            tau_2 = 10.0
        elif model_phy_option == 'none_Fa_prime_supYX':
            reg_loss_name = 'loss_Fa_prime_supervisedYX'
            aux_loss_names = ["loss_Fa", "loss_Fa_primeX","loss_Fa_prime_supervisedYX"]
            lambda_0 = 1000.0
            tau_2 = 0.0
    # don't think we need a seed for optimizer initialization
    optimizer = optax.adam(learning_rate=tau_1, b1=0.9, b2=0.999)
    training_routine(train, val, net, optimizer, min_op, lambda_0,tau_1, tau_2, niter, path, device,
                    aux_loss_names=aux_loss_names,
                    reg_loss_name=reg_loss_name,
                    dt_factor=dt_factor,
                    nlog=nlog,
                    nupdate=nupdate,
                    nepoch=nepoch,
                    name_project=name_project,
                    log_param_error=log_param_error,
                    duration=duration,
                    dataset_name=dataset_name,
                    model_phy_option=model_phy_option,
                    model_aug_option=model_aug_option,)

if __name__ == '__main__':
    print(jax.devices())
    wandb.login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    # parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Override config values (e.g. dataset.name=lorenz)")
    # args = parser.parse_args()
    # base_cfg = OmegaConf.load(args.config)
    # cli_cfg = OmegaConf.from_dotlist(args.overrides)
    # cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # main(cfg)

    args, unknown = parser.parse_known_args()
    base_cfg = OmegaConf.load(args.config)
    # remove leading "--" from wandb args
    dotlist = [arg.lstrip("--") for arg in unknown]
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    #main(cfg)
    train_data, val_data = get_datasets(cfg)
    for iteration, data in enumerate(train_data, 0):
        print(data["states"].shape)
        break
    for iteration, data in enumerate(val_data, 0):
        print(data["states"].shape)
        break
    



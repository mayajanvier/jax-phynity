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

# do not change order of import, otherwise breaks cuda with diffusion env
from datasets import *
from forecasters import *
from networks import *
from losses.base import Loss
from utils import init_linear_weight, orthogonal_init, Logger, save, make_basedir, log
from utils import compute_metric, save_loss_local, log_wandb

from solvers.runge_kutta import RK_tableaux
from torch.utils.data import DataLoader 

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
    # model definition
    if cfg.model.name == "hybrid":
        # TODO: finish later for hybrid experiments compatibility
        model_phy = PHYSIC_MODEL_REGISTRY[cfg.dataset.name](cfg.model.phy_params)
        is_augmented = cfg.model.is_augmented
    else:
        model_phy, is_augmented = None, None
    model_aug = AUG_MODEL_REGISTRY[cfg.model.architecture](
        key=mkey,
        dim_state=cfg.model.dim_state,
        hidden=cfg.model.hidden,
        )
    model_aug = init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=cfg.model.init_gain)
    # forecasting method
    net = FORECASTER_REGISTRY[cfg.model.name](
        model_aug=model_aug,
        dt=cfg.dataset.dt_factor * train.dataset.dt,
        num_steps=int(train.dataset.num_steps_rollout / cfg.dataset.dt_factor),
        integration_method=cfg.model.integration_method,
        dataset=cfg.dataset.name,
        gamma=cfg.model.gamma, 
        model_phy=model_phy,
        is_augmented=is_augmented
    )
    return net 

def get_optimizer(cfg):
    return optax.adam(learning_rate=cfg.train.lr, b1=0.9, b2=0.999)

class CurriculumScheduler:
    """ Two curriculum schedulers:
        - "curr": minimum of 10 steps, add 10 steps every 100 epochs
        - "no_curr": take full trajectories 
    """
    def __init__(self, cfg, train_data):
        self.cfg = cfg
        self.program = cfg.train.curriculum
        self.index_train_min = 10  # 10 steps minimum
        self.dt = cfg.dataset.dt_factor * train_data.dataset.dt
        self.index_train_max = int(cfg.dataset.duration / self.dt)

        # training length heuristic
        if self.program == "curr":
            self.nepoch = int(self.index_train_max / self.index_train_min) * 100 + 400
        else: # no_curr
            self.nepoch = cfg.train.nepoch

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
    name_experiment = cfg.model.name+"_"+cfg.train.reg_loss_name+"_"+str(cfg.dataset.duration)
    # Setup logging
    wandb_run = None
    if cfg.logging.use_wandb:
        wandb_run = wandb.init(
            project=cfg.logging.project,
            name=name_experiment,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_id = wandb.run.id
        wandb.run.log_code("./") # save code in w&b
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
    loss = Loss(cfg)
    loss_fn_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(
            eqx.Partial(loss.loss_fn, 
            ),
            has_aux=True
            )
        )
    # curriculum
    scheduler = CurriculumScheduler(cfg, train_data)
    lambda_ = cfg.train.lambda0
    # for model selection over val loss
    loss_val_min = None

    for epoch in range(scheduler.nepoch): 
        print(f"Epoch {epoch+1}/{scheduler.nepoch}")
        epoch_rollout_index, nepoch = scheduler.step(epoch)
        loss_train = {aug_loss_name: 0.0 for aug_loss_name in cfg.train.aux_loss_names}
        loss_train['loss_traj'] = 0.0             
        for _ in range(cfg.train.niter): # APHYNITY
            for iteration, data in enumerate(train_data, 0):
                print(f"Epoch {epoch+1}/{scheduler.nepoch}, Iteration {iteration+1}/{len(train_data)}", end="\r")
                # --------------------------
                ### TRAIN STEP
                # --------------------------
                # use asarray to avoid creating new data
                states = jnp.asarray(data['states'][:,:epoch_rollout_index*cfg.dataset.dt_factor:cfg.dataset.dt_factor,:], dtype="float32") 
                #t = jnp.array(data['t'][0])[::cfg.dataset.dt_factor]
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(
                    net,
                    states,
                    jnp.asarray(lambda_),
                    epoch_rollout_index=epoch_rollout_index
                    ) 
                updates, opt_state = optimizer.update(
                    grads,
                    opt_state,
                    eqx.filter(net, eqx.is_array)
                    )
                net = eqx.apply_updates(net, updates)
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_train[losses_values_dict_key] += losses_values_dict_value
                # logging
                if cfg.train.log_param_error:
                    metric = compute_metric(net, train_data)
                else:
                    metric = {}

        # average loss over train set
        for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
            loss_train[losses_values_dict_key] /= (iteration + 1) * cfg.train.niter
        
        # update lambda according to chosen curriculum
        lambda_ = loss.update_lambda(loss_train, lambda_)

        ### LOGS 
        total_iteration = epoch * (len(train_data)) + (iteration + 1)
        if total_iteration % cfg.train.nlog == 0:
            log(train_data, epoch, iteration, loss_train | metric, nepoch)
        # log metrics to wandb 
        log_wandb(net, train_data, lambda_, loss_train,loss_val_min, 'train', epoch_rollout_index, cfg.train.log_param_error)
        
        # --------------------------
        ### VALIDATION STEP
        # --------------------------
        if total_iteration % cfg.train.nval == 0:
            loss_val = {aug_loss_name: 0.0 for aug_loss_name in cfg.train.aux_loss_names}
            loss_val["loss_traj"] = 0.0 
            for j, dt_val in enumerate(val_data, 0):
                states = jnp.asarray(dt_val['states'][:,::cfg.dataset.dt_factor,:], dtype="float32") # bs, time, nc with diffrax
                #t = jnp.array(dt_val['t'][0])[::cfg.dataset.dt_factor]
                # no backpropagation
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(
                    net,
                    states,
                    lambda_ = jnp.asarray(lambda_),
                    epoch_rollout_index=states.shape[1]) 
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_val[losses_values_dict_key] += losses_values_dict_value
            # average loss over val set
            for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                loss_val[losses_values_dict_key] /= (j + 1)
            
            # save model over loss_val
            if loss_val_min == None or loss_val_min > loss_val["loss_traj"].item():
                loss_val_min = loss_val['loss_traj'].item()
                # save model using equinox
                # TODO how to also save optimizer state?
                hyperparameters = {
                    "epoch": epoch,
                    "loss": loss_val_min,
                    "lambda": lambda_,
                    }
                save(exp_path + f'/model_{loss_val_min:.3e}.eqx', hyperparameters, net)
            
            ### LOGS
            print('#' * 80)
            log(train_data, epoch, iteration, loss_val | metric, nepoch)
            print('#' * 80)
            # log metrics to wandb
            log_wandb(net, val_data, lambda_, loss_val, loss_val_min, 'val', epoch_rollout_index, cfg.train.log_param_error)

    if wandb_run:
        wandb_run.finish()

def main(cfg):
    train_data, val_data = get_datasets(cfg)
    net = get_model(cfg, train_data)
    optimizer = get_optimizer(cfg)
    train(cfg, train_data, val_data, net, optimizer)

if __name__ == '__main__':
    print(jax.devices())
    wandb.login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args, unknown = parser.parse_known_args()
    base_cfg = OmegaConf.load(args.config)
    # remove leading "--" from wandb args
    dotlist = [arg.lstrip("--") for arg in unknown]
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    main(cfg)

    # train_data, val_data = get_datasets(cfg)
    # model = get_model(cfg, train_data)
    # print(model)
    # loss_fn = Loss(cfg)
    # loss_fn_grad = eqx.filter_jit(
    #     eqx.filter_value_and_grad(
    #         eqx.Partial(loss_fn, 
    #         ),
    #         has_aux=True
    #         )
    #     )
    # loss_fn.update_lambda()
    # print(loss_fn.lambda_)
    



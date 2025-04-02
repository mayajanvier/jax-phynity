import optax 
import os
import json
import pandas as pd
import wandb
import statistics
from experiments import APHYNITYExperiment
from networks import *
from forecasters import *
from utils import init_linear_weight, orthogonal_init
from datasets import init_dataloaders
from utils import Logger, save, make_basedir, log
import torch
import numpy as np

# Pytorch seed
#torch.manual_seed(1)

def compute_metric(net, train_data):
    metrics = {}
    metrics['param_error'] = statistics.mean(abs(v1-float(v2))/v1 for v1, v2 in zip(train_data.dataset.params.values(), net.get_pde_params().values()))
    metrics.update(net.get_pde_params())
    metrics.update({f'{k}_real': v for k, v in train_data.dataset.params.items() if k in metrics})
    return metrics

def log_wandb(net, dataloader, _lambda, loss_dict, split):
    metric = compute_metric(net, dataloader)
    omega_error = abs(metric["omega0_square"] - metric["omega0_square_real"]) / metric["omega0_square_real"]
    alpha_error = abs(metric["alpha"] - metric["alpha_real"]) / metric['alpha_real']
    if split == 'train':
        wandb.log({"Train loss": loss_dict["loss_traj"], "Lambda": _lambda, "Loss_Fa": loss_dict["loss_op"],
                    "Param error": metric["param_error"], "Omega error":omega_error, "Alpha error":alpha_error,})
    elif split == 'val':
        wandb.log({"Test loss": loss_dict["loss_traj"], "Param error test": metric["param_error"]})

def save_loss_local(val_losses, train_losses, l_test, l_train, exp_path):
    val_losses.append(l_test['loss_traj'].item())
    train_losses.append(l_train['loss_traj'].item())
    L = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    L.to_csv(exp_path+'/loss.csv', index=False)

# Losses
def MSEjax(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

@eqx.filter_jit
def loss_trajectory(model, y):
    print('loss_trajectory')
    x = y[:,:,0] # y0
    y_pred = jax.vmap(model)(x)
    return MSEjax(y_pred, y), y_pred

@eqx.filter_jit
def loss_Fa(model, y, min_op):
    print('loss_Fa')
    # TODO find better idea to deal, maybe with jax 
    y_in = rearrange(y, 'b nc T -> (b T) nc')
    #aug_deriv = jax.vmap(model.derivative_estimator.model_aug)(y_in)
    aug_deriv = jax.vmap(model.model_aug)(y_in) 
    aug_deriv = rearrange(aug_deriv, '(b T) nc -> b nc T', b=y.shape[0])
    if min_op == 'l2_normalized':
        loss_op = ((jnp.linalg.norm(aug_deriv, ord=2, axis=1) / (jnp.linalg.norm(y, ord=2, axis=1) + 1e-5)) ** 2).mean()
    elif min_op == 'l2':
        loss_op = (jnp.linalg.norm(aug_deriv, ord=2, axis=1) ** 2).mean()
    else:
        loss_op = jnp.array(0.0)  # Default to zero if min_op is not recognized
    return loss_op

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, min_op, lambda_, model_phy_option: str, model_aug_option: bool):
    print('loss_fn')
    lossT, y_pred = loss_trajectory(model, y)
    if model_phy_option == "none": # none_aug
        return lossT, (lossT, jnp.array(0.0), y_pred)
    elif model_phy_option == 'incomplete_no_Fa':
        loss_op = loss_Fa(model, y, min_op)
        return lossT, (lossT, loss_op, y_pred)
    else:
        if model_aug_option: # complete_aug or incomplete_aug
            loss_op = loss_Fa(model, y, min_op)
            return lossT * lambda_ + loss_op, (lossT,loss_op, y_pred)
        else: # complete_physics or incomplete_physics
            return lossT, (lossT, jnp.array(0.0), y_pred)
          
# Routine
def training_routine(train, test, net, optimizer, min_op, _lambda,tau_1, tau_2, niter, path, device, dt_factor=1, nlog=1, nupdate=1, nepoch=10):   
    # Setup to save logs 
    name_experiment = model_phy_option+"_"+("aug" if model_aug_option else "physics")
    
    # Weights and Biases
    wandb.init(
        project = "Damped_Pendulum", # set the wandb project where this run will be logged
        name = name_experiment,     #
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
    hyperparameters = {
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
    }
    with open(os.path.join(exp_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)

    # optimizer initialization
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    loss_test_min = None
    # in case of wandb crash 
    train_losses = []
    val_losses = []
    loss_fn_grad = eqx.Partial(loss_fn, min_op=min_op, model_phy_option=model_phy_option, model_aug_option=model_aug_option) #, dt_factor=dt_factor)
    for epoch in range(nepoch): 
        loss_train = {'loss_traj': 0.0, 'loss_op': 0.0}
        for _ in range(niter): # APHYNITY
            for iteration, data in enumerate(train, 0):
                ### TRAIN STEP
                states = jnp.array(data['states'])[:,:,::dt_factor]
                t = jnp.array(data['t'][0])[::dt_factor]
                (loss_total, (loss_val, loss_op, pred)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda))
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(net, eqx.is_array))
                net = eqx.apply_updates(net, updates)
                # accumulate loss
                loss_train['loss_traj'] += loss_val
                loss_train['loss_op'] += loss_op
                # pour voir si on train bien
                metric = compute_metric(net, train)
                print(metric)

        # average loss over train set
        loss_train['loss_traj'] /= (iteration + 1) * niter
        loss_train['loss_op'] /= (iteration + 1) * niter
        
        # update lambda
        _lambda = _lambda + tau_2 * loss_train['loss_traj'].item()

        ### LOGS 
        total_iteration = epoch * (len(train)) + (iteration + 1)
        if total_iteration % nlog == 0:
            log(train, epoch, iteration, loss_train | metric, nepoch)
        # log metrics to wandb 
        log_wandb(net, train, _lambda, loss_train, 'train')
        
        ### VALIDATION STEP
        if total_iteration % nupdate == 0:
            loss_test = {"loss_traj": 0.0, "loss_op": 0.0}
            for j, data_test in enumerate(test, 0):
                # no backpropagation
                states = jnp.array(data_test['states'])[:,:,::dt_factor]
                t = jnp.array(data_test['t'][0])[::dt_factor]
                # _lambda should be an array for jit to not recompile when its value changes
                (loss_total, (loss_val, loss_op, pred)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda)) 
                # accumulate loss
                loss_test['loss_traj'] += loss_val
                loss_test['loss_op'] += loss_op
                
            # average loss over test set
            loss_test['loss_traj'] /= j + 1
            loss_test['loss_op'] /= j + 1

            ### LOGS
            print('#' * 80)
            log(train, epoch, iteration, loss_test | metric, nepoch)
            print('#' * 80)
            # log metrics to wandb
            log_wandb(net, test, _lambda, loss_test, 'val')
            # save epoch losses to csv file
            save_loss_local(val_losses, train_losses, loss_test, loss_train, exp_path)
            
            # save model
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

                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': self.net.state_dict(),
                #     'optimizer_state_dict': self.optimizer.state_dict(),
                #     'loss': loss_test_min, 
                # }, self.exp_path + f'/model_{loss_test_min:.3e}.pt')

      


# Main
def train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, integration_method, data_integration_method="RK4", dt_factor=1, dt_num=0.5, duration=20):
    train, val, _ = init_dataloaders(dataset_name, data_integration_method, os.path.join(path, dataset_name+str(duration)), dt_num=dt_num, duration=duration)

    if dataset_name == 'pendulum':
        if model_phy_option == 'incomplete':
            model_phy = PendulumParamPDE(is_damped=False)
        elif model_phy_option == 'complete':
            model_phy = PendulumParamPDE(is_damped=True)
        elif model_phy_option == 'true':
            model_phy = PendulumParamPDE(is_damped=True, params=train.dataset.params)
        # SC3
        elif model_phy_option == 'none':
            model_phy = PendulumParamPDE(is_damped=False) # mock model not trained 
        # SC2.2
        elif model_phy_option == 'incomplete_no_Fa':
            model_phy = PendulumParamPDE(is_damped=False)
        # SC4
        elif model_phy_option == 'none_Fa':
            model_phy = PendulumParamPDE(is_damped=False) # mock model not trained
        
        mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
        model_aug = MLP(key=mkey, state_c=2, hidden=200)
        init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=0.2) 
        net = Forecaster(
            model_phy=model_phy,
            model_aug=model_aug,
            is_augmented=model_aug_option,
            is_phy=model_phy_option,
            dt=dt_factor * train.dataset.dt, # enabling comparison with GT for error scheme experiment 
            num_steps=int(train.dataset.num_steps / dt_factor), 
            integration_method=integration_method, # RK2 for error scheme experiment
        )
        
        tau_1 = 1e-3 # 1e-3 dans le git APHYNITY, 1 dans le papier
        niter = 5
        min_op = 'l2'
        if model_phy_option == 'incomplete':
            lambda_0 = 10.0
            tau_2 = 100.0
        elif model_phy_option == 'complete':
            lambda_0 = 1000.0
            tau_2 = 100.0
        elif model_phy_option == 'none': # loss_traj only
            lambda_0 = 0.0 
            tau_2 = 0.0 
            min_op = 'none' # loss_op=0, quicker evaluation
        elif model_phy_option == 'incomplete_no_Fa': # loss_traj only
            lambda_0 = 1.0
            tau_2 = 10.0
        elif model_phy_option == 'none_Fa':
            lambda_0 = 1.0
            tau_2 = 10.0
        
        
        nepoch = 400
        nlog = 5
        nupdate = 5
    
    # don't think we need a seed for optimizer initialization
    optimizer = optax.adam(learning_rate=tau_1, b1=0.9, b2=0.999)
    training_routine(train, val, net, optimizer, min_op, lambda_0,tau_1, tau_2, niter, path, device, dt_factor, nlog, nupdate, nepoch)

if __name__ == '__main__':
    wandb.login()

    ### SC1 - Train a model with complete physics
    # method = 'RK4' 
    # dataset_name = 'pendulum'
    # model_phy_option = 'complete'
    # model_aug_option = False 
    # path = 'data/sanity_checks2'
    # device = 'cpu'
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method)

    ### SC2 - Train a model with incomplete physics and augmentation
    # method = 'RK4' 
    # dataset_name = 'pendulum'
    # model_phy_option = 'incomplete'
    # model_aug_option = True
    # path = 'data/sanity_checks2'
    # device = 'cpu'
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method)

    ### SC2.2 - Train a model with incomplete physics and augmentation, only loss_traj
    # method = 'RK4' 
    # dataset_name = 'pendulum'
    # model_phy_option = 'incomplete_no_Fa'
    # model_aug_option = True
    # path = 'data/sanity_checks2'
    # device = 'cpu'
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method)

    ## SC3 - Neural ODE 
    # method = 'RK4' 
    # dataset_name = 'pendulum'
    # model_phy_option = 'none'
    # model_aug_option = True
    # path = 'data/sanity_checks2'
    # device = 'cpu'
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method)

    ## SC4 - Neural ODE + penalisation Fa
    # method = 'RK4' 
    # dataset_name = 'pendulum'
    # model_phy_option = 'none_Fa'
    # model_aug_option = True
    # path = 'data/lipschitz'
    # device = 'cpu'
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, dt_factor = 1) #, duration=duration)

    ### debug
    # method = 'RK4' 
    # dataset_name = 'pendulum'
    # model_phy_option = 'complete'
    # model_aug_option = True
    # path = 'data/tests'
    # device = 'cpu'
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method)

    ### Error scheme 1
    # for dt_factor in [2,5,8,10,16,20,25]:
    #     method = 'RK2' 
    #     dataset_name = 'pendulum'
    #     model_phy_option = 'complete'
    #     model_aug_option = False 
    #     path = 'data/error_scheme'
    #     device = 'cpu'
    #     train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, data_integration_method="RK4", dt_factor=dt_factor, dt_num=0.05)

    # for dt_factor in [2,8,16]:
    #     method = 'RK2' 
    #     dataset_name = 'pendulum'
    #     model_phy_option = 'complete'
    #     model_aug_option = False 
    #     path = 'data/error_scheme2'
    #     device = 'cpu'
    #     train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, data_integration_method="RK4", dt_factor=dt_factor, dt_num=0.05)

    ### Lipschitz
    method = 'RK4' 
    dataset_name = 'pendulum'
    model_phy_option = 'incomplete'
    model_aug_option = True
    path = 'data/lipschitz'
    device = 'cpu'
    duration = 20
    train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, dt_factor = 1, duration=duration)


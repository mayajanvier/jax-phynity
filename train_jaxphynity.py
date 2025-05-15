import optax 
import os
import json
import wandb
from networks import *
from forecasters import *
from utils import init_linear_weight, orthogonal_init, compute_metric, save_loss_local, log_wandb
from datasets import init_dataloaders
from utils import Logger, save, make_basedir, log
from loss import loss_fn, F, init_jit_aux_loss
          
# Routine
def training_routine(train, test, net, optimizer, min_op, _lambda,tau_1, tau_2, niter, path, device, dt_factor=1, nlog=1, nupdate=1, nepoch=10, name_project="Damped_Pendulum", log_param_error=True, duration=None):   
    # Setup to save logs 
    name_experiment = model_phy_option+"_"+("aug" if model_aug_option else "physics")+"_"+str(duration)
    
    # Weights and Biases
    wandb.init(
        project = name_project, # set the wandb project where this run will be logged
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
    }
    with open(os.path.join(exp_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters_model, f)

    # optimizer initialization
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    loss_test_min = None
    # in case of wandb crash 
    train_losses = []
    val_losses = []
    # fix permanent variables
    aux_loss_names = ['loss_Fa', 'loss_Fa_primeX', 'loss_Fa_prime_supervisedX', 'loss_Fa_prime_supervisedYX']
    reg_loss_name = 'loss_Fa_prime_supervisedYX' # loss_Fa_prime_supervisedYX
    aux_losses_dict = init_jit_aux_loss(aux_loss_names, min_op)
    loss_fn_grad = eqx.Partial(loss_fn, reg_loss_name=reg_loss_name, aux_losses_dict=aux_losses_dict) 
    # for loss_Fa_supervisedYX
    bool_i = 0
    bool_j = 0 
    for epoch in range(nepoch): 
        loss_train = {aug_loss_name: 0.0 for aug_loss_name in aux_loss_names}
        loss_train['loss_traj'] = 0.0 
        # curriculum
        epoch_rollout_index = min(int(duration/hyperparameters_model["dt"]) + 1, int((duration/hyperparameters_model["dt"])*(epoch/nepoch)) + 2)
        #print(f"epoch {epoch} / {nepoch}, rollout index {epoch_rollout_index}")
        for _ in range(niter): # APHYNITY
            for iteration, data in enumerate(train, 0):
                ### TRAIN STEP
                if bool_i == 0: # get F_prime_true over train only once
                    states = jnp.array(data['states'])[:,:,::dt_factor]
                    y_theta_in = rearrange(states, 'b nc T -> b (T) nc')
                    true_deriv = jax.vmap(F)(y_theta_in)
                    F_prime_true = jnp.abs(true_deriv[:,1:,:]-true_deriv[:,:-1,:]) / jnp.abs(y_theta_in[:,1:,:] - y_theta_in[:, :-1,:])
                    bool_i += 1
                states = jnp.array(data['states'])[:,:,:epoch_rollout_index:dt_factor]
                t = jnp.array(data['t'][0])[::dt_factor]
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda), epoch_rollout_index=epoch_rollout_index, Fa_prime_true=F_prime_true)
                #loss_prime = loss_Fa_prime(net, states)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(net, eqx.is_array))
                net = eqx.apply_updates(net, updates)
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_train[losses_values_dict_key] += losses_values_dict_value
                # loss_train['loss_traj'] += loss_val
                # loss_train['loss_op'] += loss_op
                # loss_train['loss_Fa_prime'] += loss_prime
                # pour voir si on train bien
                if log_param_error:
                    metric = compute_metric(net, train)
                else:
                    metric = {}

        # average loss over train set
        for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
            loss_train[losses_values_dict_key] /= (iteration + 1) * niter
        # loss_train['loss_traj'] /= (iteration + 1) * niter
        # loss_train['loss_op'] /= (iteration + 1) * niter
        # loss_train['loss_Fa_prime'] /= (iteration + 1) * niter
        
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
                    y_theta_in = rearrange(states, 'b nc T -> b (T) nc')
                    true_deriv = jax.vmap(F)(y_theta_in)
                    F_prime_true_val = jnp.abs(true_deriv[:,1:,:]-true_deriv[:,:-1,:]) / jnp.abs(y_theta_in[:,1:,:] - y_theta_in[:, :-1,:])
                    bool_j += 1
                # no backpropagation
                states = jnp.array(data_test['states'])[:,:,::dt_factor]
                t = jnp.array(data_test['t'][0])[::dt_factor]
                # _lambda should be an array for jit to not recompile when its value changes
                (loss_total, (pred, losses_values_dict)), grads = loss_fn_grad(net, states, lambda_ = jnp.array(_lambda), epoch_rollout_index=states.shape[2], Fa_prime_true=F_prime_true_val) 
                # accumulate loss
                for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                    loss_test[losses_values_dict_key] += losses_values_dict_value
                # loss_test['loss_traj'] += loss_val
                # loss_test['loss_op'] += loss_op
                
            # average loss over test set
            for losses_values_dict_key, losses_values_dict_value in losses_values_dict.items():
                loss_test[losses_values_dict_key] /= (j + 1)
            # loss_test['loss_traj'] /= j + 1
            # loss_test['loss_op'] /= j + 1

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

      


# Main
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
        nepoch = 1000
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
            lambda_0 = 0.0 
            tau_2 = 0.0 
        elif model_phy_option == 'incomplete_no_Fa': # loss_traj only
            lambda_0 = 1.0
            tau_2 = 10.0
        elif model_phy_option == 'none_Fa': # paper parameters 
            lambda_0 = 10.0
            tau_2 = 10.0
        elif model_phy_option == 'true': # loss_traj only
            lambda_0 = 0.0
            tau_2 = 0.0
        elif model_phy_option == 'none_Fa_prime':
            lambda_0 = 1000.0
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
        nepoch = 600
        lambda_0 = 10.0 
        tau_2 = 100.0 
    
    # don't think we need a seed for optimizer initialization
    optimizer = optax.adam(learning_rate=tau_1, b1=0.9, b2=0.999)
    training_routine(train, val, net, optimizer, min_op, lambda_0,tau_1, tau_2, niter, path, device, dt_factor, nlog, nupdate, nepoch, name_project=name_project, log_param_error=log_param_error, duration=duration)

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
    # method = 'RK4' 
    # dataset_name = 'pendulum'
    # model_phy_option = 'none_Fa_prime'
    # model_aug_option = True
    # path = 'data/lipschitz_init'
    # device = 'cpu'
    # duration = 20
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, dt_factor = 1, duration=duration, init_gain=0.2)

    ### Lipschitz curriculum
    method = 'RK4' 
    dataset_name = 'pendulum'
    model_phy_option = 'none_Fa_prime'
    model_aug_option = True
    path = 'data/debug'
    device = 'cpu'
    duration = 20
    train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, dt_factor = 1, duration=duration, init_gain=0.2)


    ### Correct numerical errors 
    # for dt_factor in [2,5]:
    #     method = 'RK2'
    #     dataset_name = 'pendulum'
    #     model_phy_option = 'true'
    #     model_aug_option = True
    #     path = 'data/correct_num_err'
    #     device = 'cpu'
    #     duration = 20
    #     train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, dt_factor = dt_factor, duration=duration, dt_num=0.05)


    ### Lorenz
    # method = 'RK4' 
    # dataset_name = 'lorenz'
    # model_phy_option = "none_Fa_prime"
    # model_aug_option = True
    # path = 'data/lorenz'
    # device = 'cpu'
    # duration = 1.0
    # dt_num = 0.01
    # train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device, method, dt_factor = 1, duration=duration, init_gain=1., dt_num=dt_num)



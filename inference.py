import os
import jax
import jax.numpy as jnp
import equinox as eqx
import json
import numpy as np
from networks import *
from forecasters import *
from datasets import init_dataloaders
from loss import loss_trajectory, loss_Fa #loss_fn

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, t, min_op, lambda_, model_aug_option=False):
    lossT, y_pred = loss_trajectory(model, y, y.shape[2])
    if model_aug_option:
        loss_op = loss_Fa(model, y, min_op)
        return lossT + lambda_ * loss_op, (lossT,loss_op, y_pred)
    else:
        return lossT, (lossT, jnp.array(0.0), y_pred)

### Inference methods to reproduce APHYNITY (set test batch_size=1 to use)
def inference(model_name,exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.5, duration=20):
    """Compute trajectory losses and Fa loss for a given model, on its duration of training data."""
    # load test data 
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)), dt_num=dt_num, duration=duration)

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
        hyperparameters_dict = json.load(f)
    min_op = hyperparameters_dict['min_op']
    dt = hyperparameters_dict["dt"]
    dt_factor = int(dt/test.dataset.dt)
    print(dt, dt_factor)
    if dataset_name == 'pendulum':
        if model_phy_option == 'true': # true damped pendulum
            model_phy = PendulumParamPDE(is_damped=True, params=test.dataset.params, is_true=True)
        elif model_phy_option == 'complete': # damped pendulum
            model_phy = PendulumParamPDE(is_damped=True)
        else: 
            model_phy = PendulumParamPDE(is_damped=False)
    
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey = jax.random.PRNGKey(0)
            model_aug = MLP(key=mkey, state_c=2, hidden=200)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt,
                num_steps=int(test.dataset.num_steps/dt_factor),
                integration_method=integration_method, # error scheme exp
            )
            model = eqx.tree_deserialise_leaves(f, net)

    elif dataset_name == "lorenz":
        model_phy = None # Neural ODE 
        with open(model_path, "rb") as f:
            mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
            model_aug = MLP(key=mkey, state_c=3, hidden=200)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt_factor * test.dataset.dt, # enabling comparison with GT for error scheme experiment 
                num_steps=int(test.dataset.num_steps / dt_factor), 
                integration_method=integration_method, # RK2 for error scheme experiment
            )
            model = eqx.tree_deserialise_leaves(f, net)

    _lambda = hyperparams['lambda']
    
    print(min_op, dt_factor)

    print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )
    # TODO: fix bug in omega, alpha 
    # alpha = model.model_phy.alpha
    # omega = model.model_phy.omega0_square
    # if type(alpha) == jnp.ndarray:
    #     alpha = float(alpha.item())
    # if type(omega) == jnp.ndarray:
    #     omega = float(omega.item())
    # # save omega and alpha in folder
    # with open(os.path.join(data_path, f'{exp_name}/omega_alpha.json'), 'w') as f:
    #     json.dump({"omega": omega.item(), "alpha": alpha.item()}, f)

    # inference
    results = {}
    tot_states = []
    for i, data in enumerate(test):
        states = jnp.array(data['states'])[:,:,::dt_factor]
        t = jnp.array(data['t'][0])[::dt_factor]
        (loss_total, (loss_val, loss_op, pred)), _ = loss_fn(model, states, t, min_op, _lambda) 
        pred_i = {
            'states': np.array(states[0]).tolist(),
            'pred': np.array(pred[0]).tolist(),
            'loss_traj': loss_val.item(),
            'loss_op': loss_op.item(),
        }
        results[i] = pred_i
        print(f'Trajectory: {i}, loss_val: {loss_val}, loss_op: {loss_op}')
        tot_states.append(states[0])
        # write json file line after line
    with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}.json'), 'a') as f:
        f.write(json.dumps(results) + '\n')
    
    (loss_total, (loss_val, loss_op, pred)), _ = loss_fn(model, jnp.array(tot_states), t, min_op, _lambda) 
    print(f'Total loss_val: {loss_val}, loss_op: {loss_op}')

def Fa_behaviour(model_name,exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method):
    # load test data 
    _, _, test = init_dataloaders(dataset_name, integration_method, os.path.join(data_path, dataset_name))

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    if dataset_name == 'pendulum':
        if model_phy_option == 'incomplete':
            model_phy = PendulumParamPDE(is_damped=False)
        elif model_phy_option == 'complete':
            model_phy = PendulumParamPDE(is_damped=True)
        elif model_phy_option == 'true':
            model_phy = PendulumParamPDE(is_damped=True, params=test.dataset.params)
        elif model_phy_option == 'none':
            model_phy = PendulumParamPDE(is_damped=False) # mock model for eqx compatibility
        elif model_phy_option == 'incomplete_no_Fa':
            model_phy = PendulumParamPDE(is_damped=False)
        elif model_phy_option == 'none_Fa':
            model_phy = PendulumParamPDE(is_damped=False)
        elif model_phy_option == 'none_Fa_prime':
            model_phy = PendulumParamPDE(is_damped=False)
    
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey = jax.random.PRNGKey(0)
            model_aug = MLP(key=mkey, state_c=2, hidden=200)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=test.dataset.dt,
                num_steps=test.dataset.num_steps,
                integration_method=integration_method,
            )
            model = eqx.tree_deserialise_leaves(f, net)

    with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
        min_op = json.load(f)['min_op']
    _lambda = hyperparams['lambda']
    print(min_op)

    print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )

    # inference
    results = {}
    tot_states = []
    for i, data in enumerate(test):
        states = jnp.array(data['states'])
        y_in = rearrange(states, 'b nc T -> (b T) nc')
        Fa_out = jax.vmap(model.model_aug)(y_in) 
        Fp_out = jax.vmap(model.model_phy)(y_in)
        pred_i = {
            'states': np.array(y_in).tolist(),
            'Fa': np.array(Fa_out).tolist(),
            'Fp': np.array(Fp_out).tolist(),
        }
        results[i] = pred_i
        # write json file line after line
    with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_Fa.json'), 'a') as f:
        f.write(json.dumps(results) + '\n')

def inference_longrun(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.5, duration=200):
    """Compute trajectory losses and Fa loss for a given model, on long-term simulations."""
    # load test data 
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)), dt_num=dt_num, duration=duration)

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
        hyperparameters_dict = json.load(f)
    min_op = hyperparameters_dict['min_op']
    dt = hyperparameters_dict["dt"]
    dt_factor = int(dt/test.dataset.dt)
    print(dt,test.dataset.dt, dt_factor)
    if dataset_name == 'pendulum':
        if model_phy_option == 'true': # true damped pendulum
            model_phy = PendulumParamPDE(is_damped=True, params=test.dataset.params, is_true=True)
        elif model_phy_option == 'complete': # damped pendulum
            model_phy = PendulumParamPDE(is_damped=True)
        else: 
            model_phy = PendulumParamPDE(is_damped=False)
    
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey = jax.random.PRNGKey(0)
            model_aug = MLP(key=mkey, state_c=2, hidden=200)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt,
                num_steps=int(test.dataset.num_steps/dt_factor),
                integration_method=integration_method, # error scheme exp
            )
            model = eqx.tree_deserialise_leaves(f, net)

    elif dataset_name == "lorenz":
        model_phy = None # Neural ODE 
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
            model_aug = MLP(key=mkey, state_c=3, hidden=200)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt_factor * test.dataset.dt, # enabling comparison with GT for error scheme experiment 
                num_steps=int(test.dataset.num_steps / dt_factor), 
                integration_method=integration_method, # RK2 for error scheme experiment
            )
            model = eqx.tree_deserialise_leaves(f, net)
    
    elif dataset_name == "twobody":
        model_phy = None # Neural ODE 
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
            model_aug = MLP(key=mkey, state_c=4, hidden=200)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt_factor * test.dataset.dt, # enabling comparison with GT for error scheme experiment 
                num_steps=int(test.dataset.num_steps / dt_factor), 
                integration_method=integration_method, # RK2 for error scheme experiment
            )
            model = eqx.tree_deserialise_leaves(f, net)

    _lambda = hyperparams['lambda']
    
    print(min_op, dt_factor)
    if dataset_name == 'pendulum':
        print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )

    # inference
    results = {}
    tot_states = []
    for i, data in enumerate(test):
        states = jnp.array(data['states'])[:,:,::dt_factor]
        t = jnp.array(data['t'][0])[::dt_factor]
        pred = jax.vmap(model)(states[:,:,0]) # states[:,:,0] is the initial condition for the trajectory
        pred_i = {
            'y_true': np.array(states[0]).tolist(),
            'y_pred': np.array(pred[0]).tolist(),
        }
        results[i] = pred_i
        tot_states.append(states[0])
        # write json file line after line
    with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_{duration}.json'), 'a') as f:
        f.write(json.dumps(results) + '\n')           

### Faster inference (test batch_size=all trajectories), default
def inference_longrun_dt(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, data_integration_method="RK4", dt_num=0.5, duration=200):
    """Compute test trajectories ground truth (time step=dt_num) and predictions of a trained model (time step=dt), 
    for a certain duration and integration method."""
    # load test data 
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)+"_"+str(dt_num)), dt_num=dt_num, duration=duration)

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
        hyperparameters_dict = json.load(f)
    dt_factor = int(dt/test.dataset.dt)
    print({"dt":dt, "test data dt":test.dataset.dt, "dt factor":dt_factor})

    if dataset_name == 'pendulum':
        if model_phy_option == 'true': # true damped pendulum
            model_phy = PendulumParamPDE(is_damped=True, params=test.dataset.params, is_true=True)
        elif model_phy_option == 'complete': # damped pendulum
            model_phy = PendulumParamPDE(is_damped=True)
        elif model_phy_option == 'none': # neural ODE
            model_phy = None 
        else: 
            model_phy = PendulumParamPDE(is_damped=False)
        input_size, hidden_size = 2, 200

    elif dataset_name == 'lorenz':
        model_phy = None # neural ODE
        input_size, hidden_size = 3, 200

    elif dataset_name == 'twobody':
        model_phy = None # neural ODE
        input_size, hidden_size = 4, 200
    
    with open(model_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        mkey = jax.random.PRNGKey(0)
        model_aug = MLP(key=mkey, state_c=input_size, hidden=hidden_size)
        net = Forecaster(
            model_phy=model_phy,
            model_aug=model_aug,
            is_augmented=model_aug_option,
            is_phy=model_phy_option,
            dt=dt,
            num_steps=int(test.dataset.num_steps/dt_factor),
            integration_method=integration_method, # error scheme exp
        )
        model = eqx.tree_deserialise_leaves(f, net)

    _lambda = hyperparams['lambda'] 
    if dataset_name == 'pendulum':
        print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )

    # inference
    results = {}
    for i, data in enumerate(test):
        states = jnp.array(data['states'], dtype=jnp.float64)[:,:,::dt_factor] # float 64 for dt precision
        print(states.shape)
        t = jnp.array(data['t'][0])[::dt_factor]
        pred = jax.vmap(model)(states[:,:,0]) # states[:,:,0] is the initial condition for the trajectory
    for k in range(pred.shape[0]):
        results[k] = {
                        'y_true': np.array(states[k]).tolist(),
                        'y_pred': np.array(pred[k]).tolist(),
                    }
    with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_dt_{duration}_{dt}.json'), 'a') as f:
        f.write(json.dumps(results) + '\n')

def get_best_model(experiment_path):
    """ Get best model (lowest mse) in a folder of models. """
    # get list of .eqx files in folder
    models_list = [f for f in os.listdir(experiment_path) if f.endswith(".eqx")]
    mse_list = [float(model[6:-4]) for model in models_list]  # extract mse from filenames
    idx_best_mse = np.argmin(mse_list) # index of best mse
    best_model_name = models_list[idx_best_mse]
    return best_model_name

def run_inference_longrun_dt_bestmodel(dt, experiment_path, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.01, duration=100):
    data_path, experiment_name = experiment_path.rsplit('/', 1)
    model_name = get_best_model(experiment_path)
    # model aug option
    if "aug" in experiment_name: # augmented model
        model_aug_option = True
        # split before _aug
        model_phy_option = experiment_name.split("_aug")[0]
    else: # physical model
        model_aug_option = False
        # split before _physics
        model_phy_option = experiment_name.split("_physics")[0]
    
    # perform inference
    inference_longrun_dt(
        model_name,
        experiment_name,
        data_path,
        model_phy_option,
        model_aug_option,
        dataset_name,
        integration_method,
        dt,
        data_integration_method,
        dt_num,
        duration)

if __name__ == '__main__':
    dt = 0.1
    experiment_path = "/Users/mayajanvier/jax-phynity/data/test_float64/none_Fa_prime_supX_aug_1.0_7_y9m718id"
    dataset_name = "twobody"
    integration_method = "RK4"
    data_integration_method = "RK4"
    dt_num = 0.01
    duration = 100
    run_inference_longrun_dt_bestmodel(dt, experiment_path, dataset_name, integration_method, data_integration_method, dt_num, duration)


import os
import jax
import jax.numpy as jnp
import equinox as eqx
import json
import numpy as np
from networks import *
from forecasters import *
from datasets import init_dataloaders
from loss import loss_trajectory, loss_Fa 
import argparse
from omegaconf import OmegaConf
import h5py


@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, t, min_op, lambda_, model_aug_option=False):
    lossT, y_pred = loss_trajectory(model, y, y.shape[2])
    if model_aug_option:
        loss_op = loss_Fa(model, y, min_op)
        return lossT + lambda_ * loss_op, (lossT,loss_op, y_pred)
    else:
        return lossT, (lossT, jnp.array(0.0), y_pred)


def inference_longrun_dt(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, data_integration_method="RK4", dt_num=0.5, duration=200):
    """Compute test trajectories ground truth (time step=dt_num) and predictions of a trained model (time step=dt), 
    for a certain duration and integration method."""
    # load test data 
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)+"_"+str(dt_num)), dt_num=dt_num, duration=duration, split="test")
    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    #with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
    #    hyperparameters_dict = json.load(f)
    dt_factor = int(dt/test.dataset.dt)
    print({"dt":dt, "test data dt":test.dataset.dt, "dt factor":dt_factor})

    if os.path.isfile(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_dt_{duration}_{dt}.json')): 
        print("File already exists, skipping inference.")
    else:
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
        
        elif dataset_name == "rigidbody":
            model_phy = None
            input_size, hidden_size = 3, 200
        
        elif dataset_name == "doublependulum":
            model_phy = None
            input_size, hidden_size = 4, 200
        
        elif dataset_name == "ks":
            model_phy = None
            input_size, hidden_size = 256, 200
        
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey = jax.random.PRNGKey(0)
            if dataset_name == "doublependulum":
                model_aug = MLPAngular(key=mkey, state_c=input_size, hidden=hidden_size)
            else:
                model_aug = MLP(key=mkey, state_c=input_size, hidden=hidden_size)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt,
                num_steps=int(test.dataset.num_steps_rollout/dt_factor),
                integration_method=integration_method, # error scheme exp
            )
            model = eqx.tree_deserialise_leaves(f, net)

        _lambda = hyperparams['lambda'] 
        if dataset_name == 'pendulum':
            print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )

        # inference
        results = {}
        for i, data in enumerate(test):
            states = jnp.array(data['states'])[:,::dt_factor,:] 
            print(states.shape)
            #t = jnp.array(data['t'][0])[::dt_factor]
            pred = jax.vmap(model)(states[:,0,:]) # states[:,:,0] is the initial condition for the trajectory
        for k in range(pred.shape[0]):
            if model_phy_option == 'none':
                results[k] = {
                                'y_true': np.array(states[k]).tolist(),
                                'y_pred': np.array(pred[k]).tolist(),
                            }
            else: # economize space
                results[k] = {
                    'y_pred': np.array(pred[k]).tolist(),
                }
        with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_dt_{duration}_{dt}.json'), 'a') as f:
            f.write(json.dumps(results) + '\n')

def inference_longrun_dt_val(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, data_integration_method="RK4", dt_num=0.5, duration=200):
    """Compute test trajectories ground truth (time step=dt_num) and predictions of a trained model (time step=dt), 
    for a certain duration and integration method."""
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)+"_"+str(dt_num)), dt_num=dt_num, duration=duration, split="test")
    # load val data 
    if dataset_name == "twobody":
        if "_0.02_" in exp_name:
            dt_path = "data_exp/twobody_dt001_8s/twobody0.02_val.npy"
        elif "_0.05_" in exp_name:
            dt_path = "data_exp/twobody_dt001_8s/twobody0.05_val.npy"
        elif "_0.1_" in exp_name:
            dt_path = "data_exp/twobody_dt001_8s/twobody0.1_val.npy"
        elif "_0.2_" in exp_name:
            dt_path = "data_exp/twobody_dt001_8s/twobody0.2_val.npy"
        elif "_0.01_" in exp_name:
            dt_path = "data_exp/twobody_dt001_8s/twobody0.01_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()
    elif dataset_name == "rigidbody":
        if "_0.5_" in exp_name:
            dt_path = "data_exp/rigidbody_gridsearch/rigidbody0.5_val.npy"
        elif "_1.0_" in exp_name:
            dt_path = "data_exp/rigidbody_gridsearch/rigidbody1.0_val.npy"
        elif "_0.1_" in exp_name:
            dt_path = "data_exp/rigidbody_gridsearch/rigidbody0.1_val.npy"
        elif "_0.2_" in exp_name:
            dt_path = "data_exp/rigidbody_gridsearch/rigidbody0.2_val.npy"
        elif "_2.0_" in exp_name:
            dt_path = "data_exp/rigidbody_gridsearch/rigidbody2.0_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()
    elif dataset_name == "ks":
        dt_path = "data_exp/KS_gridsearch/ks0.4_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    dt_factor = int(dt/test.dataset.dt)
    print({"dt":dt, "test data dt":test.dataset.dt, "dt factor":dt_factor})

    if os.path.isfile(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_dt_{duration}_{dt}_val.npy')): 
        print("File already exists, skipping inference.")
    else:
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
        
        elif dataset_name == "rigidbody":
            model_phy = None
            input_size, hidden_size = 3, 200
        
        elif dataset_name == "doublependulum":
            model_phy = None
            input_size, hidden_size = 4, 200
        
        elif dataset_name == "ks":
            model_phy = None
            input_size, hidden_size = 256, 200
        
        print(int(data["states"].shape[1]/dt_factor))
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey = jax.random.PRNGKey(0)
            if dataset_name == "doublependulum":
                model_aug = MLPAngular(key=mkey, state_c=input_size, hidden=hidden_size)
            else:
                model_aug = MLP(key=mkey, state_c=input_size, hidden=hidden_size)
            net = Forecaster(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt,
                num_steps= int(data["states"].shape[1]/dt_factor), #int(test.dataset.num_steps_rollout/dt_factor),
                integration_method=integration_method, # error scheme exp
            )
            model = eqx.tree_deserialise_leaves(f, net)

        _lambda = hyperparams['lambda'] 
        if dataset_name == 'pendulum':
            print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )

        # inference
        results = {}
        #for i, data in enumerate(test):
        states = jnp.array(data['states'])[:,::dt_factor,:] 
        print(states.shape)
        #t = jnp.array(data['t'][0])[::dt_factor]
        pred = jax.vmap(model)(states[:,0,:]) # states[:,:,0] is the initial condition for the trajectory
        pred = np.array(pred)
        f = f"{data_path}/{exp_name}/{model_name[:-4]}_longrun_dt_{duration}_{dt}_val.npy"
        np.save(f, pred)

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

def run_inference_longrun_dt_bestmodel_val(dt, experiment_path, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.01, duration=100):
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
    inference_longrun_dt_val(
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

def main(cfg):
    data_folder = cfg.data_folder
    dt = cfg.dt
    integration_method = cfg.integration_method
    duration = cfg.dataset.duration
    dataset_name = cfg.dataset.name
    dt_num = cfg.dataset.dt_num
    data_integration_method = cfg.dataset.integration_method

    experiments = os.listdir(data_folder)
    for exp in experiments:
        experiment_path = os.path.join(data_folder, exp)
        if os.path.isdir(experiment_path):
            print(f"Processing experiment: {exp}")
            run_inference_longrun_dt_bestmodel(dt, experiment_path, dataset_name, integration_method, data_integration_method, dt_num, duration)

def main_val(cfg):
    data_folder = cfg.data_folder
    dt = cfg.dt
    integration_method = cfg.integration_method
    duration = cfg.dataset.duration
    dataset_name = cfg.dataset.name
    dt_num = cfg.dataset.dt_num
    data_integration_method = cfg.dataset.integration_method

    experiments = os.listdir(data_folder)
    for exp in experiments:
        experiment_path = os.path.join(data_folder, exp)
        if os.path.isdir(experiment_path):
            print(f"Processing experiment: {exp}")
            run_inference_longrun_dt_bestmodel_val(dt, experiment_path, dataset_name, integration_method, data_integration_method, dt_num, duration)

def get_states_test(data_path, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.5, duration=200):
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)+"_"+str(dt_num)), dt_num=dt_num, duration=duration, split="test")

def get_states_train(data_path, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.5, duration=200):
    _,val, _ = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)+"_"+str(dt_num)), dt_num=dt_num, duration=duration, split="train")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Override config values (e.g. dataset.name=lorenz)")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    main(cfg)


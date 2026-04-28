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
from metrics import *
from loss import F_dict
from einops import rearrange

jax.config.update("jax_enable_x64", False)

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(model, y, t, min_op, lambda_, model_aug_option=False):
    lossT, y_pred = loss_trajectory(model, y, y.shape[2])
    if model_aug_option:
        loss_op = loss_Fa(model, y, min_op)
        return lossT + lambda_ * loss_op, (lossT,loss_op, y_pred)
    else:
        return lossT, (lossT, jnp.array(0.0), None) # too heavy for high dim


def inference_longrun_dt(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, data_integration_method="RK4", dt_num=0.5, duration=200, model_architecture='mlp', name="node", gamma=0.0):
    """Compute test trajectories ground truth (time step=dt_num) and predictions of a trained model (time step=dt), 
    for a certain duration and integration method."""
    # load test data 
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)+"_"+str(dt_num)), dt_num=dt_num, duration=duration, split="test")
    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path, name, gamma)
    #with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
    #    hyperparameters_dict = json.load(f)
    #dt_factor = int(dt/test.dataset.dt)
    dt_factor = int(dt/dt_num)
    print({"dt":dt, "test data dt":dt_num, "dt factor":dt_factor})

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
        
        elif dataset_name == "twobody_forcing":
            model_phy = None
            input_size, hidden_size = 5, 200
        
        elif dataset_name == "rigidbody":
            model_phy = None
            input_size, hidden_size = 3, 200
        
        elif dataset_name == "doublependulum":
            model_phy = None
            input_size, hidden_size = 4, 200
        
        elif dataset_name == "ks":
            model_phy = None
            input_size, hidden_size = 256, 200
        
        elif dataset_name == "burgers":
            model_phy = None
            input_size, hidden_size = 1024, 2048
        
        elif dataset_name == "ns_incomp":
            model_phy = None
            init_features = 16
        
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey = jax.random.PRNGKey(0)
            if dataset_name == "doublependulum":
                model_aug = MLPAngular(key=mkey, state_c=input_size, hidden=hidden_size)
            else:
                if model_architecture == "mlp":
                    model_aug = MLP(key=mkey, state_c=input_size, hidden=hidden_size)
                elif model_architecture == "unet":
                    model_aug = UNet1D(key=mkey, hidden=hidden_size) 
                elif model_architecture == "unet2d":
                    model_aug = UNet2D(in_channels=1, out_channels=1,key=mkey, init_features=init_features)
                elif model_architecture == "convnet":
                    model_aug = ConvNetEstimator1D(key=mkey, hidden=64)
                elif model_architecture == "convnet2d":
                    model_aug = ConvNetEstimator2D(key=mkey, hidden=16)

            
            if name == "snode":
                net = SNODE(
                    model_phy=model_phy,
                    model_aug=model_aug,
                    is_augmented=model_aug_option,
                    is_phy=model_phy_option,
                    dt=dt,
                    num_steps=int(test.dataset.num_steps_rollout/dt_factor),
                    integration_method=integration_method, # error scheme exp
                    dataset=dataset_name,
                    gamma=gamma,
                )

            else:
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
            states = jnp.asarray(data['states'][:,::dt_factor,:], dtype=jnp.float32) 
            print(states.shape)
            #t = jnp.array(data['t'][0])[::dt_factor]
            pred = jax.vmap(model)(states[:,0,:]) # states[:,:,0] is the initial condition for the trajectory
        for k in range(pred.shape[0]):
            if model_phy_option == 'none':
                results[k] = {
                                #'y_true': np.array(states[k]).tolist(),
                                'y_pred': np.array(pred[k]).tolist(),
                            }
            else: # economize space
                results[k] = {
                    'y_pred': np.array(pred[k]).tolist(),
                }
        with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_dt_{duration}_{dt}.json'), 'a') as f:
            f.write(json.dumps(results) + '\n')


def inference_metrics(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, data_integration_method="RK4", dt_num=0.5, duration=200, model_architecture='mlp', name="node", gamma=0.0):
    """Compute test trajectories ground truth (time step=dt_num) and predictions of a trained model (time step=dt), 
    for a certain duration and integration method."""
    # load test data 
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)+"_"+str(dt_num)), dt_num=dt_num, duration=duration, split="test")
    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path, name, gamma)
    dt_factor = int(dt/dt_num)
    print({"dt":dt, "test data dt":dt_num, "dt factor":dt_factor})

    # if os.path.isfile(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_stats_{duration}_{dt}.json')): 
    #     print("File already exists, skipping inference.")
    # else:
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
    
    elif dataset_name == "twobody_forcing":
        model_phy = None
        input_size, hidden_size = 5, 200
    
    elif dataset_name == "rigidbody":
        model_phy = None
        input_size, hidden_size = 3, 200
    
    elif dataset_name == "doublependulum":
        model_phy = None
        input_size, hidden_size = 4, 200
    
    elif dataset_name == "ks":
        model_phy = None
        input_size, hidden_size = 256, 200
    
    elif dataset_name == "burgers":
        model_phy = None
        input_size, hidden_size = 1024, 2048
    
    elif dataset_name == "ns_incomp":
        model_phy = None
        init_features = 16
    
    with open(model_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        mkey = jax.random.PRNGKey(0)
        if dataset_name == "doublependulum":
            model_aug = MLPAngular(key=mkey, state_c=input_size, hidden=hidden_size)
        else:
            if model_architecture == "mlp":
                model_aug = MLP(key=mkey, state_c=input_size, hidden=hidden_size)
            elif model_architecture == "unet":
                model_aug = UNet1D(key=mkey, hidden=hidden_size) 
            elif model_architecture == "unet2d":
                model_aug = UNet2D(in_channels=1, out_channels=1,key=mkey, init_features=init_features)
            elif model_architecture == "convnet":
                model_aug = ConvNetEstimator1D(key=mkey, hidden=64)
            elif model_architecture == "convnet2d":
                model_aug = ConvNetEstimator2D(key=mkey, hidden=16)

        
        if name == "snode":
            net = SNODE(
                model_phy=model_phy,
                model_aug=model_aug,
                is_augmented=model_aug_option,
                is_phy=model_phy_option,
                dt=dt,
                num_steps=int(test.dataset.num_steps_rollout/dt_factor),
                integration_method=integration_method, # error scheme exp
                dataset=dataset_name,
                gamma=gamma,
            )

        else:
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


        # TIMESERIES data
        # if file already exists, skip
        results = {}
        for i, data in enumerate(test):
            states = jnp.asarray(data['states'][:,::dt_factor,:], dtype=jnp.float32) 
            print(states.shape)
            #t = jnp.array(data['t'][0])[::dt_factor]
            pred = jax.vmap(model)(states[:,0,:]) # states[:,:,0] is the initial condition for the trajectory
        if not os.path.isfile(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_stats_{duration}_{dt}.json')):
            for k in range(pred.shape[0]):
                results[k] = {
                                'y_true': np.array(states[k]).tolist(),
                                'y_pred': np.array(pred[k]).tolist(),
                            }
            data = pd.DataFrame(results).T
            data = compute_metrics_timeseries(data, dataset_name=dataset_name)
            # save data["L2_over_time_relative"] and data["cons_over_time"] to json
            with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_stats_{duration}_{dt}.json'), 'a') as f:
                f.write(data[["L2_over_time_relative", "cons_over_time"]].to_json(orient="records") + '\n')
        
        # MODEL data
        jac_data = {}
        y = rearrange(states, 'b T nc -> (b T) nc') 
        y_pred = rearrange(pred,'b T nc -> (b T) nc') 
        trueF = F_dict[dataset_name]
        jac_data["jacF"] = compute_jacobian_test(trueF, y, bool_true=True)
        #jac_data["eigenF"] = get_eigenvalues_jacobian(trueF, y)
        lip_trueF = jnp.max(jac_data["jacF"])
        jac_data["jacFtheta"] = compute_jacobian_test(model, y, bool_true=False)
        #jac_data["eigenFtheta"] = get_eigenvalues_jacobian(model.model_aug, y)
        lip_model_x = jnp.max(jac_data["jacFtheta"])
        jac_data["jacFtheta_pred"] = compute_jacobian_test(model, y_pred, bool_true=False)
        #jac_data["eigenFtheta_pred"] = get_eigenvalues_jacobian_pred(model.model_aug, y_pred)
        lip_model_xpred = jnp.max(jac_data["jacFtheta_pred"])
        jac_data = pd.DataFrame(jac_data)
        if not os.path.isfile(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_jacobian_{duration}_{dt}.json')):
            with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_jacobian_{duration}_{dt}.json'), 'a') as f:
                f.write(jac_data.to_json(orient="records") + '\n')

        if not os.path.isfile(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_stats_model_{duration}_{dt}.json')):
            model_stats = {}
            model_stats["lip_trueF"] = lip_trueF.item()
            model_stats["lip_model_x"] = lip_model_x.item()
            model_stats["lip_model_xpred"] = lip_model_xpred.item()
            model_stats["offline_error"] = offline_error(model, trueF, y).item()
            if dataset_name in ["twobody", "rigidbody"]:
                model_stats["J_error"] = J_error(model, trueF, y).item()
            B, dim = y.shape
            master_key = jax.random.PRNGKey(42)
            keys = jax.random.split(master_key, B)
            model_stats["J_error_hutch"] = (J_error_hutch(model, trueF, y, keys) / dim**2).item()
            print(model_stats)
            with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_stats_model_{duration}_{dt}.json'), 'a') as f:
                f.write(json.dumps(model_stats) + '\n')


def inference_longrun_dt_val(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, data_integration_method="RK4", dt_num=0.5, duration=200, model_architecture='mlp', name="node", gamma=0.0):
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

        # for long val 
        # dt_path = "datasets/2body_full_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()
    elif dataset_name == "rigidbody":
        if "_0.5_" in exp_name:
            dt_path = "data_exp/rigidbody/rigidbody0.5_val.npy"
        elif "_1.0_" in exp_name:
            dt_path = "data_exp/rigidbody/rigidbody1.0_val.npy"
        elif "_0.1_" in exp_name:
            dt_path = "data_exp/rigidbody/rigidbody0.1_val.npy"
        elif "_0.2_" in exp_name:
            dt_path = "data_exp/rigidbody/rigidbody0.2_val.npy"
        elif "_2.0_" in exp_name:
            dt_path = "data_exp/rigidbody/rigidbody2.0_val.npy"

        # for long val 
        #dt_path = "datasets/rigidbody_full_val.npy"
        
        data = np.load(dt_path, allow_pickle=True).item()
    elif dataset_name == "ks":
        dt_path = "data_exp/KS_gridsearch/ks0.4_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()

        # long val
        # dt_path = "data/KS_val.h5" 
        # with h5py.File(dt_path, "r") as f:
        #     states = f["valid"]["pde_140-256"][:]
        # data = {"states": states}
    elif dataset_name == "burgers":
        dt_path = "data_exp/burgers/burgers0.02_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()
    elif dataset_name == "twobody_forcing":
        dt_path = "data_exp/twobody_forcing/twobody_forcing0.1_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()
    
    elif dataset_name == "ns_incomp":
        if "_2_" in exp_name:
            dt_path = "data_exp/ns_incomp/ns_incomp2_val.npy"
        elif "_5_" in exp_name:
            dt_path = "data_exp/ns_incomp/ns_incomp5_val.npy"
        elif "_10_" in exp_name:
            dt_path = "data_exp/ns_incomp/ns_incomp10_val.npy"
        data = np.load(dt_path, allow_pickle=True).item()

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    dt_factor = int(dt/test.dataset.dt)
    print({"dt":dt, "test data dt":test.dataset.dt, "dt factor":dt_factor})

    # val long momentarily
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
            
        elif dataset_name == "twobody_forcing":
            model_phy = None
            input_size, hidden_size = 5, 200
        
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
                if model_architecture == "mlp":
                    model_aug = MLP(key=mkey, state_c=input_size, hidden=hidden_size)
                elif model_architecture == "unet":
                    model_aug = UNet1D(key=mkey, hidden=hidden_size)
                elif model_architecture == "convnet":
                    model_aug = ConvNetEstimator1D(key=mkey, hidden=16)

            if name == "snode":
                net = SNODE(
                    model_phy=model_phy,
                    model_aug=model_aug,
                    is_augmented=model_aug_option,
                    is_phy=model_phy_option,
                    dt=dt,
                    num_steps=int(data["states"].shape[1]/dt_factor),
                    integration_method=integration_method, # error scheme exp
                    dataset=dataset_name,
                    gamma=gamma,
                )
            else: 
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

def run_inference_longrun_dt_bestmodel(dt, experiment_path, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.01, duration=100, type="timeseries"):
    data_path, experiment_name = experiment_path.rsplit('/', 1)
    model_name = get_best_model(experiment_path)
    cfg = OmegaConf.load(os.path.join(data_path, f'{experiment_name}/config.yaml'))
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
    if type == "timeseries":
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
            duration,
            cfg.model.architecture,
            cfg.model.name,
            cfg.model.gamma,)
    elif type == "metrics":
        inference_metrics(
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
            duration,
            cfg.model.architecture,
            cfg.model.name,
            cfg.model.gamma,)

def run_inference_longrun_dt_bestmodel_val(dt, experiment_path, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.01, duration=100):
    data_path, experiment_name = experiment_path.rsplit('/', 1)
    model_name = get_best_model(experiment_path)
    cfg = OmegaConf.load(os.path.join(data_path, f'{experiment_name}/config.yaml'))
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
        duration, 
        cfg.model.architecture,
        cfg.model.name,
        cfg.model.gamma)

def main(cfg, type="timeseries"):
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
            run_inference_longrun_dt_bestmodel(dt, experiment_path, dataset_name, integration_method, data_integration_method, dt_num, duration, type)

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

    main(cfg, "metrics")


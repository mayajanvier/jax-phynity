import os
import jax
import jax.numpy as jnp
import equinox as eqx
import json
import argparse
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
import pandas as pd

# do not change order of import, otherwise breaks cuda with diffusion env
from datasets import DATASET_REGISTRY, seed_worker
from forecasters import *
from networks import *
from losses.F_dynamics import F_REGISTRY
from metrics import * 
from utils import init_linear_weight, orthogonal_init
from train_jaxphynity import get_model
import torch
from torch.utils.data import DataLoader

#jax.config.update("jax_enable_x64", False)
g = torch.Generator()
g.manual_seed(0)

def get_test_dataset(cfg_inf):
    print(f"Loading dataset {cfg_inf.dataset.name} ...")
    # define datasets 
    path = os.path.join(cfg_inf.data_folder, cfg_inf.dataset.name + str(cfg_inf.dataset.duration)+"_"+str(cfg_inf.dataset.dt_num))
    num_steps_rollout = int(cfg_inf.dataset.duration/cfg_inf.dataset.dt_num)
    dataset_test = DATASET_REGISTRY[cfg_inf.dataset.name](
        nb_traj=cfg_inf.dataset.nb_traj_test,
        num_steps_rollout=num_steps_rollout, 
        num_steps_max=num_steps_rollout,
        path=path,
        split="test",
        **cfg_inf.dataset)
    # define dataloader
    dataloader_test_params = {
                'dataset'    : dataset_test,
                'batch_size' : cfg_inf.dataset.nb_traj_test,
                'num_workers': 0,
                'pin_memory' : True,
                'drop_last'  : False,
                'shuffle'    : False,
                #'persistent_workers': True,
                'worker_init_fn': seed_worker,
                'generator':g,
        }
    dataloader_test   = DataLoader(**dataloader_test_params)
    return dataloader_test

def get_model(cfg_exp, cfg_inf, test):
    mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
    # model definition from experiment parameters
    if cfg_exp.model.name == "hybrid":
        # TODO: finish later for hybrid experiments compatibility
        model_phy = PHYSIC_MODEL_REGISTRY[cfg_exp.dataset.name](cfg_exp.model.phy_params)
        is_augmented = cfg_exp.model.is_augmented
    else:
        model_phy, is_augmented = None, None
    model_aug = AUG_MODEL_REGISTRY[cfg_exp.model.architecture](
        key=mkey,
        dim_state=cfg_exp.model.dim_state,
        hidden=cfg_exp.model.hidden,
        )
    model_aug = init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=cfg_exp.model.init_gain)
    # forecasting method
    net = FORECASTER_REGISTRY[cfg_exp.model.name](
        model_aug=model_aug,
        model_phy=model_phy,
        is_augmented=is_augmented,
        dt=cfg_inf.dt, # we get to choose the dt
        integration_method=cfg_inf.integration_method, # we get to choose the integration method
        num_steps=int(test.dataset.num_steps_rollout / cfg_exp.dataset.dt_factor),
        dataset=cfg_exp.dataset.name,
        gamma=cfg_exp.model.gamma, 
    )
    return net 

def get_best_model_name(experiment_folder):
    """ Get best model (lowest mse) in a folder of models. """
    # get list of .eqx files in folder
    models_list = [f for f in os.listdir(experiment_folder) if f.endswith(".eqx")]
    mse_list = [float(model[6:-4]) for model in models_list]  # extract mse from filenames
    idx_best_mse = np.argmin(mse_list) # index of best mse
    best_model_name = models_list[idx_best_mse]
    return best_model_name

def inference_metrics(experiment_folder, best_model_name, cfg_inf, cfg_exp):
    """Compute test trajectories ground truth (time step=dt_num) and predictions of a trained model (time step=dt), 
    for a certain duration and integration method."""
    model_path = os.path.join(experiment_folder, f"{best_model_name}")
    inference_filepath = f'{model_path[:-4]}_stats_{cfg_inf.dataset.duration}_{cfg_inf.dt}.json'
    inference_model_filepath = f'{model_path[:-4]}_stats_model_{cfg_inf.dataset.duration}_{cfg_inf.dt}.json'
    inference_jac_filepath = f'{model_path[:-4]}_jacobian_{cfg_inf.dataset.duration}_{cfg_inf.dt}.json'

    # load test data 
    test = get_test_dataset(cfg_inf)
    # load model
    net = get_model(cfg_exp, cfg_inf, test)
    with open(model_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = eqx.tree_deserialise_leaves(f, net)

    # if file already exists, skip
    if os.path.isfile(inference_filepath): 
        print("Stats file already exists, skipping inference.")
    else:
        # TIMESERIES data
        results = {}
        for i, data in enumerate(test):
            states = jnp.asarray(data['states'][:,::cfg_exp.dataset.dt_factor,:], dtype=jnp.float32) 
            print(states.shape)
            #t = jnp.array(data['t'][0])[::dt_factor]
            pred = jax.vmap(model)(states[:,0,:]) # states[:,:,0] is the initial condition for the trajectory
        if not os.path.isfile(inference_filepath):
            for k in range(pred.shape[0]):
                results[k] = {
                                'y_true': np.array(states[k]).tolist(),
                                'y_pred': np.array(pred[k]).tolist(),
                            }
            data = pd.DataFrame(results).T
            data = compute_metrics_timeseries(data, dataset_name=cfg_inf.dataset.name)
            # save data["L2_over_time_relative"] and data["cons_over_time"] to json
            with open(inference_filepath, 'a') as f:
                f.write(data[["L2_over_time_relative", "cons_over_time"]].to_json(orient="records") + '\n')
        
    # MODEL data
    for i, data in enumerate(test):
        states = jnp.asarray(data['states'][:,::cfg_exp.dataset.dt_factor,:], dtype=jnp.float32)
        pred = jax.vmap(model)(states[:,0,:]) # states[:,:,0] is the initial condition for the trajectory

        y = rearrange(states, 'b T nc -> (b T) nc') 
        y_pred = rearrange(pred,'b T nc -> (b T) nc') 
        trueF = F_REGISTRY[cfg_inf.dataset.name]

        if not os.path.isfile(inference_model_filepath):
            model_stats = {}
            model_stats["offline_error"] = offline_error(model, trueF, y).item()
            if cfg.dataset.name in ["twobody", "rigidbody"]:
                model_stats["J_error"] = J_error(model, trueF, y).item()
            B, dim = y.shape
            master_key = jax.random.PRNGKey(42)
            keys = jax.random.split(master_key, B)
            model_stats["J_error_hutch"] = (J_error_hutch(model, trueF, y, keys) / dim**2).item()
            with open(inference_model_filepath, 'a') as f:
                f.write(json.dumps(model_stats) + '\n')

        # TODO: adapt for KS, too expensive for now
        if cfg_inf.dataset.name != "ks": 
            if not os.path.isfile(inference_jac_filepath):
                jac_data = {}
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
                with open(inference_jac_filepath, 'a') as f:
                    f.write(jac_data.to_json(orient="records") + '\n')            
            
def inference_timeseries(experiment_folder, best_model_name, cfg_inf, cfg_exp):
    """Compute test trajectories ground truth (time step=dt_num) and predictions of a trained model (time step=dt), 
    for a certain duration and integration method."""
    model_path = os.path.join(experiment_folder, f"{best_model_name}")
    inference_filepath = f'{model_path[:-4]}_timeseries_{cfg_inf.dataset.duration}_{cfg_inf.dt}.json'

    if os.path.isfile(inference_filepath): 
        print("File already exists, skipping inference.")
    else:
        # load test data 
        test = get_test_dataset(cfg_inf)
        # load model
        net = get_model(cfg_exp, cfg_inf, test)
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = eqx.tree_deserialise_leaves(f, net)
            if cfg_inf.dataset.name == 'pendulum':
                print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )

        # inference
        results = {}
        for i, data in enumerate(test):
            states = jnp.asarray(data['states'][:,::cfg_exp.dataset.dt_factor,:], dtype=jnp.float32) 
            print(states.shape)
            #t = jnp.array(data['t'][0])[::dt_factor]
            pred = jax.vmap(model)(states[:,0]) # states[:,:,0] is the initial condition for the trajectory
        for k in range(pred.shape[0]):
            results[k] = {
                            'y_true': np.array(states[k]).tolist(),
                            'y_pred': np.array(pred[k]).tolist(),
                        }
        with open(inference_filepath, 'a') as f:
            f.write(json.dumps(results) + '\n')

def run_inference(cfg_inf, experiment_folder, type="timeseries"):
    best_model_name = get_best_model_name(experiment_folder)
    cfg_exp = OmegaConf.load(os.path.join(experiment_folder, "config.yaml"))
    
    # perform inference
    if type == "timeseries":
        inference_timeseries(
            experiment_folder,
            best_model_name,
            cfg_inf,
            cfg_exp
            )
    elif type == "metrics":
        inference_metrics(
            experiment_folder,
            best_model_name,
            cfg_inf,
            cfg_exp
            )

def main(cfg_inf, type="timeseries"):
    """ for all experiment folder in data_folder (inference folder),
    find best model in terms of val MSE and return:
    - if type="timeseries": full time series on test trajectories
    - elif type="metrics": general summary of mse and conservation relative errors, 
    jacobian, offline errors... 
    """
    data_folder = cfg_inf.data_folder
    experiments = os.listdir(data_folder)
    for exp in experiments: 
        experiment_folder = os.path.join(data_folder, exp)
        if os.path.isdir(experiment_folder):
            print(f"Processing experiment: {exp}")
            run_inference(cfg_inf, experiment_folder, type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Override config values (e.g. dataset.name=lorenz)")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    main(cfg, "metrics")


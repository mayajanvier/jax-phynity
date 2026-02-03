import numpy as np
import os
import json
import pandas as pd
from scipy.stats import entropy
import jax
import jax.numpy as jnp
from networks import * 
from forecasters import Forecaster   
from inference import *
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


### PERFORMANCE METRICS
def compute_metrics_lorenz(data, data_true=None):
    if data_true is not None:
        data["y_true"] = data_true.values
    data["y_true"] = data["y_true"].apply(lambda x: np.array(x))
    data["y_pred"] = data["y_pred"].apply(lambda x: np.array(x))
    data["L2"] = data.apply(lambda x: np.mean(np.array((x["y_true"] - x["y_pred"])**2)), axis=1)
    data["L2_5s"] = data.apply(lambda x: np.mean(np.array((x["y_true"][:,:501] - x["y_pred"][:,:501])**2)), axis=1)
    data["L2_0.5s"] = data.apply(lambda x: np.mean(np.array((x["y_true"][:,:51] - x["y_pred"][:,:51])**2)), axis=1)
    data["L2_0.1s"] = data.apply(lambda x: np.mean(np.array((x["y_true"][:,:11] - x["y_pred"][:,:11])**2)), axis=1)

    # bins from true 
    for i, key in enumerate(["x_min", "y_min", "z_min"]):
        data[key] = data.apply(lambda x: np.min(x["y_true"][i]), axis=1)
    for i, key in enumerate(["x_max", "y_max", "z_max"]):
        data[key] = data.apply(lambda x: np.max(x["y_true"][i]), axis=1)

    # compute pdf
    data["x_pdf_true"] = data.apply(lambda x: np.histogram(x["y_true"][0], bins=50, range=(x["x_min"], x["x_max"]), density=True)[0], axis=1)
    data["x_pdf_pred"] = data.apply(lambda x: np.histogram(x["y_pred"][0], bins=50, range=(x["x_min"], x["x_max"]), density=True)[0], axis=1)
    data["y_pdf_true"] = data.apply(lambda x: np.histogram(x["y_true"][1], bins=50, range=(x["y_min"], x["y_max"]), density=True)[0], axis=1)
    data["y_pdf_pred"] = data.apply(lambda x: np.histogram(x["y_pred"][1], bins=50, range=(x["y_min"], x["y_max"]), density=True)[0], axis=1)
    data["z_pdf_true"] = data.apply(lambda x: np.histogram(x["y_true"][2], bins=50, range=(x["z_min"], x["z_max"]), density=True)[0], axis=1)
    data["z_pdf_pred"] = data.apply(lambda x: np.histogram(x["y_pred"][2], bins=50, range=(x["z_min"], x["z_max"]), density=True)[0], axis=1)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    for key in ["x_pdf_true", "x_pdf_pred", "y_pdf_true", "y_pdf_pred", "z_pdf_true", "z_pdf_pred"]:
        data[key] = data[key].apply(lambda x: x + epsilon)

    # KL divergence
    data["KL_x"] = data.apply(lambda x: entropy(x["x_pdf_true"], x["x_pdf_pred"]), axis=1)
    data["KL_y"] = data.apply(lambda x: entropy(x["y_pdf_true"], x["y_pdf_pred"]), axis=1)
    data["KL_z"] = data.apply(lambda x: entropy(x["z_pdf_true"], x["z_pdf_pred"]), axis=1)
    return data

def compute_metrics(data, data_true=None):
    if data_true is not None:
        data["y_true"] = data_true.values
    data["y_true"] = data["y_true"].apply(lambda x: np.array(x))
    data["y_pred"] = data["y_pred"].apply(lambda x: np.array(x))
    data["L2"] = data.apply(lambda x: np.mean(np.array((x["y_true"] - x["y_pred"])**2)), axis=1)
    data["L2_over_time"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2).mean(axis=1), axis=1)
    data["L2_over_time_relative"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2)/np.array(x["y_true"]**2).mean(axis=1), axis=1)
    data["L2_state_time"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2)[:,:2].mean(axis=1), axis=1)
    data["L2_momentum_time"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2)[:,2:].mean(axis=1), axis=1)

    data["momentum_pred"] = data.apply(lambda x: np.array(x["y_pred"][:,0]*x["y_pred"][:,3] - x["y_pred"][:,1]*x["y_pred"][:,2]), axis=1)
    data["momentum_true"] = data.apply(lambda x: np.array(x["y_true"][:,0]*x["y_true"][:,3] - x["y_true"][:,1]*x["y_true"][:,2]), axis=1)
    data["momentum_error"] = data.apply(lambda x: np.array((x["momentum_true"] - x["momentum_pred"])**2), axis=1)
    data["momentum_error_relative"] = data.apply(lambda x: np.array(np.abs(x["momentum_true"] - x["momentum_pred"])/np.abs(x["momentum_true"])), axis=1)
    data["momentum_error_relative_mean"] = data.apply(lambda x: np.mean(x["momentum_error_relative"]), axis=1)
    data["momentum_error_relative_max"] = data.apply(lambda x: np.max(x["momentum_error_relative"]), axis=1)
    data["final_momentum_error"] = data.apply(lambda x: x["momentum_error_relative"][-1] , axis=1)


    # bins from true 
    for i, key in enumerate(["x_min", "y_min"]):
        data[key] = data.apply(lambda x: np.min(x["y_true"][i]), axis=1)
    for i, key in enumerate(["x_max", "y_max"]):
        data[key] = data.apply(lambda x: np.max(x["y_true"][i]), axis=1)

    # compute pdf
    data["x_pdf_true"] = data.apply(lambda x: np.histogram(x["y_true"][0], bins=50, range=(x["x_min"], x["x_max"]), density=True)[0], axis=1)
    data["x_pdf_pred"] = data.apply(lambda x: np.histogram(x["y_pred"][0], bins=50, range=(x["x_min"], x["x_max"]), density=True)[0], axis=1)
    data["y_pdf_true"] = data.apply(lambda x: np.histogram(x["y_true"][1], bins=50, range=(x["y_min"], x["y_max"]), density=True)[0], axis=1)
    data["y_pdf_pred"] = data.apply(lambda x: np.histogram(x["y_pred"][1], bins=50, range=(x["y_min"], x["y_max"]), density=True)[0], axis=1)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    for key in ["x_pdf_true", "x_pdf_pred", "y_pdf_true", "y_pdf_pred"]:
        data[key] = data[key].apply(lambda x: x + epsilon)

    # KL divergence
    data["KL_x"] = data.apply(lambda x: entropy(x["x_pdf_true"], x["x_pdf_pred"]), axis=1)
    data["KL_y"] = data.apply(lambda x: entropy(x["y_pdf_true"], x["y_pdf_pred"]), axis=1)
    return data

def compute_metrics_rigidbody(data, data_true=None):
    if data_true is not None:
        data["y_true"] = data_true.values
    data["y_true"] = data["y_true"].apply(lambda x: np.array(x))
    true_shape = data["y_true"][0].shape
    data["y_pred"] = data["y_pred"].apply(lambda x: np.array(x)[:true_shape[0], :true_shape[1]])
    data["L2"] = data.apply(lambda x: np.mean(np.array((x["y_true"] - x["y_pred"])**2)), axis=1)
    data["L2_600s"] = data.apply(lambda x: np.mean(np.array((x["y_true"][:6001,:] - x["y_pred"][:6001,:])**2)), axis=1)
    data["L2_over_time"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2).mean(axis=1), axis=1)

    data["holo_pred"] = data.apply(lambda x: np.array(x["y_pred"][:,0]**2+ x["y_pred"][:,1]**2 + x["y_pred"][:,2]**2), axis=1)
    data["holo_true"] = data.apply(lambda x: np.array(x["y_true"][:,0]**2+ x["y_true"][:,1]**2 + x["y_true"][:,2]**2), axis=1)
    data["holo_error"] = data.apply(lambda x: np.array((x["holo_true"] - x["holo_pred"])**2), axis=1)
    data["holo_error_relative"] = data.apply(lambda x: np.array(np.abs(x["holo_true"] - x["holo_pred"])/np.abs(x["holo_true"])), axis=1)
    data["holo_error_relative_mean"] = data.apply(lambda x: x["holo_error_relative"].mean(), axis=1)
    data["holo_error_relative_max"] = data.apply(lambda x: x["holo_error_relative"].max(), axis=1)
    data["final_holo_error"] = data.apply(lambda x: x["holo_error_relative"][-1], axis=1)
    return data

def compute_metrics_ks(data, data_true=None):
    if data_true is not None:
        data["y_true"] = data_true.values
    data["y_true"] = data["y_true"].apply(lambda x: np.array(x))
    data["y_pred"] = data["y_pred"].apply(lambda x: np.array(x)[:-1])
    data["L2"] = data.apply(lambda x: np.mean(np.array((x["y_true"] - x["y_pred"])**2)), axis=1)
    data["L2_over_time"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2).mean(axis=1), axis=1)

    # conservation
    data["sum_true"] = data.apply(lambda x: np.sum(x["y_true"], axis=1), axis=1)
    data["sum_pred"] = data.apply(lambda x: np.sum(x["y_pred"], axis=1), axis=1)
    data["sum_error"] = data.apply(lambda x: np.abs(x["sum_true"] - x["sum_pred"]), axis=1) # true very close to 0 initially
    data["sum_error_mean"] = data.apply(lambda x: np.mean(x["sum_error"]), axis=1)

    # correlation for each time step
    data["correlation_over_time"] = data.apply(lambda x: np.array([pearsonr(x["y_true"][t], x["y_pred"][t])[0] for t in range(x["y_true"].shape[0])]), axis=1)

    data["idx_corr_09_threshold"] = data.apply(lambda x: np.where(x["correlation_over_time"] < 0.9)[0][0] if np.any(x["correlation_over_time"] < 0.9) else x["correlation_over_time"].shape[0], axis=1)
    data["idx_corr_08_threshold"] = data.apply(lambda x: np.where(x["correlation_over_time"] < 0.8)[0][0] if np.any(x["correlation_over_time"] < 0.8) else x["correlation_over_time"].shape[0], axis=1)
    return data

def compute_metrics_pendulum(data, data_true=None):
    m1, m2, l1, l2 = 1.0, 1.0, 1.0, 1.0
    g = 9.81
    if data_true is not None:
        data["y_true"] = data_true.values
    data["y_true"] = data["y_true"].apply(lambda x: np.array(x))
    data["y_pred"] = data["y_pred"].apply(lambda x: np.array(x))
    time_shape = data["y_true"][0].shape[0]
    data["y_pred"] = data.apply(lambda x: x["y_pred"][:time_shape,:], axis=1)
    data["L2"] = data.apply(lambda x: np.mean(np.array((x["y_true"] - x["y_pred"])**2)), axis=1)
    data["L2_state_time"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2)[:,:2].mean(axis=1), axis=1)
    data["L2_momentum_time"] = data.apply(lambda x: np.array((x["y_true"] - x["y_pred"])**2)[:,2:].mean(axis=1), axis=1)
    data["E_pred"] = data.apply(
        lambda x: np.array(
            1/2 * m1 * (l1**2) * (x["y_pred"][:,2]**2) +
            1/2 * m2 * (l1**2 * (x["y_pred"][:,2]**2) + l2**2 * (x["y_pred"][:,3]**2) + 2 * l1 * l2 * x["y_pred"][:,2] * x["y_pred"][:,3] * np.cos(x["y_pred"][:,0] - x["y_pred"][:,1])) +
            (-(m1 + m2) * g * l1 * np.cos(x["y_pred"][:,0]) - m2 * g * l2 * np.cos(x["y_pred"][:,1])
            )), axis=1)
    data["E_true"] = data.apply(
        lambda x: np.array(
            1/2 * m1 * (l1**2) * (x["y_true"][:,2]**2) +
            1/2 * m2 * (l1**2 * (x["y_true"][:,2]**2) + l2**2 * (x["y_true"][:,3]**2) + 2 * l1 * l2 * x["y_true"][:,2] * x["y_true"][:,3] * np.cos(x["y_true"][:,0] - x["y_true"][:,1])) +
            (-(m1 + m2) * g * l1 * np.cos(x["y_true"][:,0]) - m2 * g * l2 * np.cos(x["y_true"][:,1])
        )), axis=1)
    
    data["E_error"] = data.apply(lambda x: np.array((x["E_true"] - x["E_pred"])**2), axis=1)
    data["E_error_relative"] = data.apply(lambda x: np.array(np.abs(x["E_true"] - x["E_pred"]))/np.abs(x["E_true"]), axis=1)
    data["final_E_error"] = data.apply(lambda x: ((x["E_true"][-1] - x["E_pred"][-1])**2)/(x["E_true"][-1])**2, axis=1)

    # KL on theta1, theta2
    # keep same bins for true and pred to be comparable, true as reference
    for i, key in enumerate(["theta1_min", "theta2_min"]):
        data[key] = data.apply(lambda x: np.min(x["y_true"][:,i]), axis=1)
    for i, key in enumerate(["theta1_max", "theta2_max"]):
        data[key] = data.apply(lambda x: np.max(x["y_true"][:,i]), axis=1)

    # compute pdf
    data["theta1_pdf_true"] = data.apply(lambda x: np.histogram(x["y_true"][:,0], bins=50, range=(x["theta1_min"], x["theta1_max"]), density=True)[0], axis=1)
    data["theta1_pdf_pred"] = data.apply(lambda x: np.histogram(x["y_pred"][:,0], bins=50, range=(x["theta1_min"], x["theta1_max"]), density=True)[0],axis=1)
    data["theta2_pdf_true"] = data.apply(lambda x: np.histogram(x["y_true"][:,1], bins=50, range=(x["theta2_min"], x["theta2_max"]), density=True)[0], axis=1)
    data["theta2_pdf_pred"] = data.apply(lambda x: np.histogram(x["y_pred"][:,1], bins=50, range=(x["theta2_min"], x["theta2_max"]), density=True)[0],axis=1)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    for key in ["theta1_pdf_true", "theta1_pdf_pred", "theta2_pdf_true", "theta2_pdf_pred"]:
        data[key] = data[key].apply(lambda x: x + epsilon)
    # KL divergence
    data["KL_theta1"] = data.apply(lambda x: entropy(x["theta1_pdf_true"], x["theta1_pdf_pred"]), axis=1)
    data["KL_theta2"] = data.apply(lambda x: entropy(x["theta2_pdf_true"], x["theta2_pdf_pred"]), axis=1)
    return data


### JACOBIAN METRICS

def load_model_dt(model_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, dt_num, duration):
    experiment_path, model_name = model_path.rsplit('/', 1)
    data_path, exp_name = experiment_path.rsplit('/', 1)
    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    dt_factor = int(dt/dt_num)
    num_steps = int(duration/dt_num)
    #print({"dt":dt, "test data dt":dt_num, "dt factor":dt_factor})

    if dataset_name == 'pendulum':
        if model_phy_option == 'true': # true damped pendulum
            model_phy = PendulumParamPDE(is_damped=True, params={"omega0_square": (2 * jnp.pi / 12) ** 2, "alpha":0.2}, is_true=True)
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

    elif dataset_name == 'rigidbody':
        model_phy = None
        input_size, hidden_size = 3, 200
    
    elif dataset_name == 'doublependulum':
        model_phy = None
        input_size, hidden_size = 4, 200
    
    elif dataset_name == 'ks':
        model_phy = None
        input_size, hidden_size = 256, 200
    
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
            num_steps=int(num_steps/dt_factor),
            integration_method=integration_method, # error scheme exp
        )
        model = eqx.tree_deserialise_leaves(f, net)
    return model

def load_best_model_dt(dt, experiment_path, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.01, duration=100):
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
    
    model_path = os.path.join(data_path, f"{experiment_name}/{model_name}")
    model = load_model_dt(
        model_path,
        model_phy_option,
        model_aug_option,
        dataset_name,
        integration_method,
        dt,
        dt_num,
        duration
        )
    return model

def load_model_dict_dt(data_folder, dt, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.01, duration=100):
    model_dict_Fa = defaultdict(lambda: defaultdict(dict))
    model_dict_Fa_prime_supX = defaultdict(lambda: defaultdict(dict))
    for experiment_name in os.listdir(data_folder):
        experiment_path = os.path.join(data_folder, experiment_name)
        if os.path.isdir(experiment_path):
            try:
                with open(os.path.join(experiment_path, "config.yaml"), "r") as f:
                    cfg = yaml.safe_load(f)
                _lambda = cfg["train"]["lambda0"]
                tau2 = cfg["train"]["tau2"]
                model = load_best_model_dt(dt, experiment_path, dataset_name, integration_method, data_integration_method, dt_num, duration)
                if "Fa_prime_supX" in experiment_name:
                    model_dict_Fa_prime_supX[_lambda][tau2] = model
                else:
                    model_dict_Fa[_lambda][tau2] = model
            except Exception as e:
                print(f"Could not load model for experiment {experiment_name}: {e}")
    return model_dict_Fa, model_dict_Fa_prime_supX

def make_jacobian_norm_fn(model):
    """Returns a batched, JIT-compiled function to compute Jacobian norms."""
    # JIT compile jacobian computation
    jac_fn = jax.jit(jax.jacfwd(model))

    def jacobian_norm(y0):
        return jnp.linalg.norm(jac_fn(y0))

    return jax.vmap(jacobian_norm)  # vectorized over batch of y0

def compute_jacobian_test(model, y, bool_true=False):
    if bool_true:
        jacobian_norms_fn = make_jacobian_norm_fn(model)
    else:
        jacobian_norms_fn = make_jacobian_norm_fn(model.model_aug)
    return jacobian_norms_fn(y)  # y shape: (N, features)
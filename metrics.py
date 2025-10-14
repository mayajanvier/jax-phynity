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

### JACOBIAN METRICS

def load_model_dt(model_path, model_phy_option, model_aug_option, dataset_name, integration_method, dt, dt_num, duration):
    experiment_path, model_name = model_path.rsplit('/', 1)
    data_path, exp_name = experiment_path.rsplit('/', 1)
    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    dt_factor = int(dt/dt_num)
    num_steps = int(duration/dt_num)
    print({"dt":dt, "test data dt":dt_num, "dt factor":dt_factor})

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
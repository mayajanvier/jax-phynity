import optax 
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

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, t, min_op, lambda_, model_aug_option=False):
    lossT, y_pred = loss_trajectory(model, y, y.shape[2])
    if model_aug_option:
        loss_op = loss_Fa(model, y, min_op)
        return lossT + lambda_ * loss_op, (lossT,loss_op, y_pred)
    else:
        return lossT, (lossT, jnp.array(0.0), y_pred)


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

def inference_longrun(model_name,exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.5, duration=200):
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
        pred = jax.vmap(model)(states[:,:,0]) # states[:,:,0] is the initial condition for the trajectory
        #(loss_total, (loss_val, loss_op, pred)), _ = loss_fn(model, states, t, min_op, _lambda, model_aug_option) 
        pred_i = {
            'y_true': np.array(states[0]).tolist(),
            'y_pred': np.array(pred[0]).tolist(),
            #'loss_traj': loss_val.item(),
            #'loss_op': loss_op.item(),
        }
        results[i] = pred_i
        #print(f'Trajectory: {i}')
        #print(f'Trajectory: {i}, loss_val: {loss_val}, loss_op: {loss_op}')
        tot_states.append(states[0])
        # write json file line after line
    with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_{duration}.json'), 'a') as f:
        f.write(json.dumps(results) + '\n')
    
    #(loss_total, (loss_val, loss_op, pred)), _ = loss_fn(model, jnp.array(tot_states), t, min_op, _lambda) 
    #print(f'Total loss_val: {loss_val}, loss_op: {loss_op}')

def inference_longrun_lorenz(model_name,exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.5, duration=200):
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name+str(duration)), dt_num=dt_num, duration=duration)
    print("dataset loaded")
    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
        hyperparameters_dict = json.load(f)
    min_op = hyperparameters_dict['min_op']
    dt = hyperparameters_dict["dt"]
    dt_factor = int(dt/test.dataset.dt)
    print(dt,test.dataset.dt, dt_factor)
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
    print("model loaded")
    # inference
    results = {}
    tot_states = []
    for i, data in enumerate(test):
        #if i !=7: # skip trajectory 7, it is too long
        states = jnp.array(data['states'])[:,:,::dt_factor]
        y0 = states[:,:,0]
        #t = jnp.array(data['t'][0])[::dt_factor]
        pred = model(y0)
        #(loss_total, (loss_val, loss_op, pred)), _ = loss_fn(model, states, t, min_op, _lambda) 
        pred_i = {
            'y_true': np.array(states[0]).tolist(),
            'y_pred': np.array(pred[0]).tolist(),
            #'loss_traj': loss_val.item(),
            #'loss_op': loss_op.item(),
        }
        results[i] = pred_i
        print(f'Trajectory: {i}, pred shape: {pred.shape}')
        #print(f'Trajectory: {i}, loss_val: {loss_val}, loss_op: {loss_op}')
        tot_states.append(states[0])
        # write json file line after line
        with open(os.path.join(data_path, f'{exp_name}/{model_name[:-4]}_longrun_{duration}.json'), 'a') as f:
            f.write(json.dumps(results) + '\n')


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

if __name__ == '__main__':
    # SC1
    # print("SC1")
    # exp_name = 'complete_physics_26_f6dwvliv'
    # model_name = 'model_2.107e-08.eqx'
    # data_path = 'data/sanity_checks2'
    # model_phy_option = 'complete'
    # model_aug_option = False
    # dataset_name = 'pendulum'
    # inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')
    # Fa_behaviour(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # SC2
    # print("SC2")
    # exp_name = 'incomplete_aug_34_lx8y2wmb'
    # model_name = 'model_4.318e-03.eqx'
    # data_path = 'data/sanity_checks2'
    # model_phy_option = 'incomplete'
    # model_aug_option = True
    # dataset_name = 'pendulum'
    # inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')
    # Fa_behaviour(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # SC2.2   
    # print("SC2.2") 
    # exp_name = 'incomplete_no_Fa_aug_20_ps3fe9eb'
    # model_name = 'model_4.466e-03.eqx'
    # data_path = 'data/lipschitz'
    # model_phy_option = 'incomplete_no_Fa'
    # model_aug_option = True
    # dataset_name = 'pendulum'
    # inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')
    # Fa_behaviour(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # SC3
    # print("SC3")
    # exp_name = 'none_aug_19_yt8wqw57'
    # model_name = 'model_3.270e-02.eqx'
    # data_path = 'data/lipschitz'
    # model_phy_option = 'none'
    # model_aug_option = True
    # dataset_name = 'pendulum'
    # inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # SC4
    # print("SC4")
    # exp_name = 'none_Fa_aug_33_e6ynty14'
    # model_name = 'model_1.030e-01.eqx'
    # data_path = 'data/lipschitz'
    # model_phy_option = 'none_Fa'
    # model_aug_option = True
    # dataset_name = 'pendulum'
    # inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # Error scheme
    # print("ES1")
    # model_list = ['model_1.425e-07.eqx','model_5.492e-06.eqx', 'model_3.518e-05.eqx', 'model_8.506e-05.eqx',  'model_5.444e-04.eqx']
    # for k in range(2,7):
    #     exp_name = f'complete_physics_{k}_3tfct1zi' 
    #     model_name = model_list[k-2]
    #     data_path = 'data/error_scheme'
    #     model_phy_option = 'complete'
    #     model_aug_option = False
    #     dataset_name = 'pendulum'
    #     inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK2', dt_num=0.05)

    # Error scheme
    # print("ES1, variant")
    # model_list = ['model_1.401e-07.eqx','model_3.491e-05.eqx', 'model_5.444e-04.eqx']
    # for k in range(1,4):
    #     exp_name = f'complete_physics_{k}_ms726d1v' 
    #     model_name = model_list[k-1]
    #     data_path = 'data/error_scheme2'
    #     model_phy_option = 'complete'
    #     model_aug_option = False
    #     dataset_name = 'pendulum'
    #     inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK2', dt_num=0.05)

    # Lipschitz
    # model_list = [["model_6.485e-05.eqx", "model_1.202e-03.eqx", "model_4.321e-04.eqx", "model_6.669e+00.eqx"][2]]
    # id_list = [["eiuqzzae", "3eqegq6e","srefavep","aawyc6as"][2]]
    # for k, duration in enumerate([[5,10,20,40][2]]):
    #     exp_name = f'incomplete_aug_{duration}_{id_list[k]}'
    #     model_name = model_list[k]
    #     data_path = 'data/lipschitz'
    #     model_phy_option = 'incomplete'
    #     model_aug_option = True
    #     dataset_name = 'pendulum'
    #     inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')
    #     Fa_behaviour(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # Fa prime 
    # data_path = "data/lipschitz_init"
    # model_list = ["model_5.308e-01.eqx", "model_1.219e-02.eqx",  "model_1.364e-02.eqx"]
    # model_phy_options = ["none", "none_Fa", "none_Fa_prime"]
    # id_list = ["7_g5si31tr", "8_2ifdmo1z", "9_hsr8f7u5"]
    # model_aug_options = [True, True, True]
    # # model_phy_options = ["incomplete_Fa_prime", "incomplete"]
    # # model_list = ["model_1.006e-03.eqx","model_6.908e-04.eqx"]
    # # id_list = ["7_cs85eg71","9_xez0or7d"]
    # dataset_name = 'pendulum'
    # duration = 40
    
    # for k in range(3):
    #     exp_name = f'{model_phy_options[k]}_aug_{duration}_{id_list[k]}'
    #     model = model_list[k]
    #     model_phy_option = model_phy_options[k]
    #     model_aug_option = model_aug_options[k]
    #     inference(model, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4', duration=duration)
        #Fa_behaviour(model, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')


    # PENDULUM CURRICULUM
    data_path = "data/pendulum_curriculum"
    #dataset_name = 'pendulum'
    # 10,15,20,25,30,35,40,50
    #model_list = ["model_4.474e-03.eqx", "model_8.605e-03.eqx", "model_7.232e-03.eqx", "model_5.600e-03.eqx", "model_4.379e-03.eqx", "model_3.710e-03.eqx", "model_3.301e-03.eqx", "model_2.637e-03.eqx"]
    #exp_names = ["none_aug_10_7_wni7d7vw", "none_aug_15_11_tuagzu2h", "none_aug_20_3_7r3pt5i6", "none_aug_25_15_mwuh5fu7", "none_aug_30_19_1px44q94", "none_aug_35_23_dvo9pd8k", "none_aug_40_30_2mrv9dnr", "none_aug_50_34_26unnadg"]
    # 10s modesl
    #model_list = ["model_4.911e-03.eqx"]
    #exp_names = ["none_Fa_prime_supX_aug_10_42_tuhrn6ch"]
    # 20s models
    #model_list = ["model_6.262e-03.eqx"]#,"model_3.211e-03.eqx"] #,"model_6.073e-03.eqx"]#,"model_3.826e-03.eqx"]#,"model_9.422e-03.eqx"]
    #exp_names = ["none_Fa_prime_supX_aug_20_40_mji7rtrw"]#,"none_Fa_prime_supX_aug_20_38_e32f6kk3"]#,"none_Fa_prime_supX_aug_20_37_l71dirtf"] #,"none_Fa_prime_supX_aug_20_36_ddvssrzw"]#,"none_Fa_prime_supX_aug_20_35_stgfld1q"]
    #model_list = ["model_3.839e-02.eqx","model_1.341e-02.eqx"]
    #exp_names = ["none_Fa_aug_20_45_y2asmpno","none_Fa_aug_20_44_x5syrqwn"]
    # 25s models
    # model_list=["model_3.125e-03.eqx"]
    # exp_names=["none_Fa_prime_supX_aug_25_43_l64do47n"]
    # 30s models
    #model_list =["model_2.338e-03.eqx"]
    #exp_names = ["none_Fa_prime_supX_aug_30_41_azh5fjat"]
    # 40s models
    # model_list=["model_1.920e-02.eqx","model_4.090e-03.eqx","model_1.552e-03.eqx","model_4.450e-03.eqx","model_1.758e-03.eqx"]
    # exp_names=["none_Fa_aug_40_49_vfcnhnan","none_Fa_prime_supX_aug_20_48_y401ldbf","none_Fa_prime_supX_aug_40_47_wuwh8czs","none_Fa_aug_40_46_msaprmil","none_Fa_prime_supX_aug_40_39_p089jjoy"]
    # model_phy_option = "none_Fa" 
    # model_aug_option = True
    # dataset_name = 'pendulum'
    #durations = [10,15,20,25,30,35,40,50]
    # durations = [40]
    # model_list = ["model_3.524e-02.eqx",'model_1.509e-03.eqx']
    # exp_names = ["none_Fa_prime_supX_aug_5_55_p17tot2d",'none_aug_5_54_e0f4gs2q']
    # model_phy_option = ["none_Fa_prime_supX","none"]
    # model_aug_option = True
    # durations = [5,5]
    # for i, model_name in enumerate(model_list):
    #     exp_name = exp_names[i]
    #     duration = durations[i]
    #     #inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4', duration=duration)
    #     inference_longrun(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')
    #     break

    # LORENZ CURRICULUM
    # data_path = "data/lorenz_curriculum"
    # model_list = ["model_6.805e+01.eqx","model_3.684e+01.eqx","model_4.397e+01.eqx", "model_4.449e+01.eqx", "model_4.073e+01.eqx"]
    # exp_names = ["none_Fa_prime_supX_aug_0.5_38_6ip39mbt","none_Fa_aug_1.0_37_ynthb5uu","none_aug_1.0_1_lpirex33", "none_Fa_prime_supX_aug_1.0_20_ifl1kpia", "none_Fa_prime_supX_aug_1.0_36_u46i7ois"]
    # model_phy_option = ["none_Fa_prime_supX","none_Fa", "none", "none_Fa_prime_supX", "none_Fa_prime_supX"]
    # #model_list = ["/model_1.236e+02.eqx","model_7.111e+01.eqx","model_1.423e+02.eqx","model_6.016e+01.eqx","model_8.037e+01.eqx", "model_6.919e+01.eqx"]
    # #exp_names = ["none_Fa_prime_supX_direct_aug_0.5_35_307wy485","none_Fa_prime_supX_direct_aug_0.5_33_twgwtlpy","none_Fa_prime_supX_direct_aug_0.5_31_xur5ddcr","none_Fa_prime_supX_aug_0.5_30_h4az6l6s","none_Fa_prime_supX_aug_0.5_23_bcaporpk", "none_Fa_aug_0.5_29_0aonngt2"]
    # #model_phy_option = ["none_Fa_prime_supX_direct","none_Fa_prime_supX_direct","none_Fa_prime_supX_direct","none_Fa_prime_supX","none_Fa_prime_supX", "none_Fa"]
    # #model_list = ["model_8.569e+01.eqx","model_7.108e+01.eqx","model_7.176e+01.eqx","model_8.037e+01.eqx","model_8.008e+01.eqx","model_4.768e+01.eqx","model_7.108e+01.eqx", "model_6.919e+01.eqx"]#"model_4.449e+01.eqx","model_4.611e+01.eqx",,"model_4.397e+01.eqx"]
    # #exp_names = ["none_aug_0.5_12_h8nc71jn","none_aug_0.5_28_tf2kusqq","none_Fa_prime_supX_aug_0.5_27_16mx8mqy","none_Fa_prime_supX_aug_0.5_23_bcaporpk","none_Fa_aug_0.5_22_il3xfk60","none_Fa_prime_supX_aug_0.5_18_vzkua19c","none_aug_0.5_12_d7x29ou8", "none_Fa_aug_0.5_29_0aonngt2"] #"none_Fa_prime_supX_aug_1.0_20_ifl1kpia","none_aug_0.1_16_y4dgk7so",,"none_aug_1.0_1_lpirex33"]
    # #model_phy_option = ["none","none","none_Fa_prime_supX", "none_Fa_prime_supX", "none_Fa", "none_Fa_prime_supX", "none","none_Fa"] #"none_Fa_prime_supX", "none", ,"none"]
    # model_aug_option = True
    # dataset_name = 'lorenz'
    # #durations = [1.0]
    # for i, model_name in enumerate(model_list):
    #     exp_name = exp_names[i]
    #     #duration = durations[i]
    #     #inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4', duration=duration)
    #     inference_longrun(model_name, exp_name, data_path, model_phy_option[i], model_aug_option, dataset_name, 'RK4', dt_num=0.01, duration=100)
    #     break

    ### TWO BODY CURRICULUM
    data_path = "data/twobody_curriculum"
    #model_list = ["model_8.838e-03.eqx","model_7.667e-04.eqx","model_8.666e-06.eqx", "model_1.873e-04.eqx"]
    #exp_names = ["none_Fa_prime_supX_aug_5_13_eqe5cguq","none_aug_10_12_vfm4hxed","none_aug_1_5_sqjjkugz", "none_aug_5_9_nn7jwd9d"]
    model_list = ["model_1.410e-05.eqx"]
    exp_names = ["none_aug_1.0_1_gn2n5qyl"]
    model_phy_option = ["none"]
    model_aug_option = True
    dataset_name = 'twobody'
    durations = [1.0]
    #durations = [5.0,10.0,1.0, 5.0]
    dt_num = 0.01
    for i, model_name in enumerate(model_list):
        exp_name = exp_names[i]
        inference_longrun(model_name, exp_name, data_path, model_phy_option[i], model_aug_option, dataset_name, 'RK4', dt_num=dt_num, duration=100)


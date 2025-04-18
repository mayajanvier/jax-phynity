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
from train_jaxphynity import loss_trajectory, loss_Fa #loss_fn

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, t, min_op, lambda_):
    lossT, y_pred = loss_trajectory(model, y)
    if model_aug_option:
        loss_op = loss_Fa(model, y, min_op)
        return lossT + lambda_ * loss_op, (lossT,loss_op, y_pred)
    else:
        return lossT, (lossT, jnp.array(0.0), y_pred)


def inference(model_name,exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method, data_integration_method="RK4", dt_num=0.5):
    # load test data 
    _, _, test = init_dataloaders(dataset_name, data_integration_method, os.path.join(data_path, dataset_name), dt_num)

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
        if model_phy_option == 'incomplete':
            model_phy = PendulumParamPDE(is_damped=False)
        elif model_phy_option == 'complete':
            model_phy = PendulumParamPDE(is_damped=True)
        elif model_phy_option == 'true':
            model_phy = PendulumParamPDE(is_damped=True, params=test.dataset.params)
        elif model_phy_option == 'none':
            model_phy = PendulumParamPDE(is_damped=False) # mock model for eqx compatibility
        elif model_phy_option == 'none_Fa':
            model_phy = PendulumParamPDE(is_damped=False) # mock model for eqx compatibility
        elif model_phy_option == 'incomplete_no_Fa':
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

    _lambda = hyperparams['lambda']
    
    print(min_op, dt_factor)

    print(f"Final omega: {model.model_phy.omega0_square}, Final alpha: {model.model_phy.alpha}" )
    alpha = model.model_phy.alpha
    omega = model.model_phy.omega0_square
    print(type(alpha), type(omega))
    if type(alpha) == jnp.ndarray:
        alpha = float(alpha.item())
    if type(omega) == jnp.ndarray:
        omega = float(omega.item())
    # save omega and alpha in folder
    with open(os.path.join(data_path, f'{exp_name}/omega_alpha.json'), 'w') as f:
        json.dump({"omega": omega.item(), "alpha": alpha.item()}, f)

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
    print("ES1, variant")
    model_list = ['model_1.401e-07.eqx','model_3.491e-05.eqx', 'model_5.444e-04.eqx']
    for k in range(1,4):
        exp_name = f'complete_physics_{k}_ms726d1v' 
        model_name = model_list[k-1]
        data_path = 'data/error_scheme2'
        model_phy_option = 'complete'
        model_aug_option = False
        dataset_name = 'pendulum'
        inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK2', dt_num=0.05)

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

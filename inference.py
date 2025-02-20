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


def inference(model_name,exp_name, data_path, model_phy_option, model_aug_option, dataset_name, integration_method):
    # load test data 
    _, _, test = init_dataloaders(dataset_name, integration_method, os.path.join(data_path, dataset_name))

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    print(model_path)
    if dataset_name == 'pendulum':
        if model_phy_option == 'incomplete':
            model_phy = PendulumParamPDE(is_damped=False)
        elif model_phy_option == 'complete':
            model_phy = PendulumParamPDE(is_damped=True)
        elif model_phy_option == 'true':
            model_phy = PendulumParamPDE(is_damped=True, params=test.dataset.params)
        elif model_phy_option == 'none':
            model_phy = PendulumParamPDE(is_damped=False) # mock model for eqx compatibility
    
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
        t = jnp.array(data['t'][0]) 
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


if __name__ == '__main__':
    # SC1
    # exp_name = 'complete_aug_17_m9bz4n8c'
    # model_name = 'model_2.790e+00.eqx'
    # data_path = 'data/sanity_checks2'
    # model_phy_option = 'complete'
    # model_aug_option = False
    # dataset_name = 'pendulum'
    # inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # SC2
    exp_name = 'incomplete_aug_19_korcnphf'
    model_name = 'model_2.400e+00.eqx'
    data_path = 'data/sanity_checks2'
    model_phy_option = 'incomplete'
    model_aug_option = True
    dataset_name = 'pendulum'
    inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

    # SC3
    # exp_name = 'none_aug_17_m9bz4n8c'
    # model_name = 'model_2.790e+00.eqx'
    # data_path = 'data/sanity_checks2'
    # model_phy_option = 'none'
    # model_aug_option = True
    # dataset_name = 'pendulum'
    # inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name, 'RK4')

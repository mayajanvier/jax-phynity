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
from train_jaxphynity import loss_fn


def inference(model_name,exp_name, data_path, model_phy_option, model_aug_option, dataset_name):
    # load test data 
    _, _, test = init_dataloaders(dataset_name, os.path.join(data_path, dataset_name))

    # load model
    model_path = os.path.join(data_path, f"{exp_name}/{model_name}")
    if dataset_name == 'pendulum':
        if model_phy_option == 'incomplete':
            model_phy = DampedPendulumParamPDE(is_complete=False, real_params=None)
        elif model_phy_option == 'complete':
            model_phy = DampedPendulumParamPDE(is_complete=True, real_params=None)
        elif model_phy_option == 'true':
            model_phy = DampedPendulumParamPDE(is_complete=True, real_params=test.dataset.params)
    
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            mkey = jax.random.PRNGKey(0)
            model_aug = MLP(key=mkey, state_c=2, hidden=200)
            net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=model_aug_option)
            model = eqx.tree_deserialise_leaves(f, net)

    with open(os.path.join(data_path, f'{exp_name}/hyperparameters.json'), 'r') as f:
        min_op = json.load(f)['min_op']
    _lambda = hyperparams['lambda']

    # inference
    results = {}
    tot_states = []
    for i, data in enumerate(test):
        states = jnp.permute_dims(jnp.array(data['states']), (0, 2, 1))
        t = jnp.array(data['t'][0]) 
        (loss_total, (loss_val, loss_op, pred)), _ = loss_fn(model, states, t, min_op, _lambda) 
        pred_i = {
            'states': np.array(states[0]).tolist(),
            'pred': np.array(pred[0]).tolist(),
            'loss_val': loss_val.item(),
            'loss_op': loss_op.item(),
        }
        results[i] = pred_i
        print(f'Trajectory: {i}, loss_val: {loss_val}, loss_op: {loss_op}')
        tot_states.append(states[0])
        # write json file line after line
    with open(os.path.join(data_path, f'{exp_name}/{model_name[:-3]}.json'), 'a') as f:
        f.write(json.dumps(results) + '\n')
    
    (loss_total, (loss_val, loss_op, pred)), _ = loss_fn(model, jnp.array(tot_states), t, min_op, _lambda) 
    print(f'Total loss_val: {loss_val}, loss_op: {loss_op}')


if __name__ == '__main__':
    exp_name = 'complete_aug2'
    model_name = 'model_7.662e+00.eqx'
    data_path = 'data/damped_pendulum_complete'
    model_phy_option = 'complete'
    model_aug_option = True
    dataset_name = 'pendulum'
    inference(model_name, exp_name, data_path, model_phy_option, model_aug_option, dataset_name)

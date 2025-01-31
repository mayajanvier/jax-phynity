import optax 
import os

from experiments import APHYNITYExperiment
from networks import *
from forecasters import *
from utils import init_linear_weight, orthogonal_init
from datasets import init_dataloaders


def train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device):
    train, test = init_dataloaders(dataset_name, os.path.join(path, dataset_name))

    if dataset_name == 'pendulum':
        if model_phy_option == 'incomplete':
            model_phy = DampedPendulumParamPDE(is_complete=False, real_params=None)
        elif model_phy_option == 'complete':
            model_phy = DampedPendulumParamPDE(is_complete=True, real_params=None)
        elif model_phy_option == 'true':
            model_phy = DampedPendulumParamPDE(is_complete=True, real_params=train.dataset.params)
        
        mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
        model_aug = MLP(key=mkey, state_c=2, hidden=200)
        init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=0.2) 
        net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=model_aug_option)

        lambda_0 = 1.0
        tau_1 = 1e-3
        tau_2 = 1
        niter = 5
        min_op = 'l2_normalized'
    
    optimizer = optax.adam(learning_rate=tau_1, b1=0.9, b2=0.999)
    experiment = APHYNITYExperiment(
            train=train, test=test, net=net, optimizer=optimizer, 
            min_op=min_op, lambda_0=lambda_0, tau_2=tau_2, niter=niter, nlog=10,
            nupdate=100, nepoch=50000, path=path, device=device
        )
    experiment.run()

if __name__ == '__main__':
    dataset_name = 'pendulum'
    model_phy_option = 'complete'
    model_aug_option = True
    path = 'data/damped_pendulum'
    device = 'cpu'
    train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device)
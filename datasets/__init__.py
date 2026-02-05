from .pendulum import DampedPendulum, DoublePendulum
from .lorenz import LorenzTrue
from .twobody import TwoBody
from .wave import Wave
from .rigidbody import RigidBodyTrue
from .ks import KSTrue
from torch.utils.data import DataLoader 
import torch
import numpy as np
import random

# fix torch seed for reproducibility
#torch.manual_seed(1)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Build our dataloaders

def param_dataset(buffer_filepath, integration_method, dataset_name="pendulum", batch_size=25, dt_num=0.5, duration=20, nb_traj=25, num_steps_max=100, split="train", val_type="short"):
    g = torch.Generator()
    g.manual_seed(0)

    dataset_train_params = {
        'nb_traj': nb_traj, 
        'num_steps_max': num_steps_max,
        'num_steps_rollout': int(duration/dt_num), 
        'dt': dt_num, # 0.05 for error scheme experiments
        'split': 'train',
        'path': buffer_filepath+'_train',
        'integration_method': integration_method,
    }

    dataset_val_params = dict()
    dataset_val_params.update(dataset_train_params) # shared parameters across train and val
    #dataset_val_params['nb_traj'] = nb_traj
    dataset_val_params['split'] = 'val'
    dataset_val_params['path'] = buffer_filepath+'_val'

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    #dataset_test_params['nb_traj'] = nb_traj
    dataset_test_params['split'] = 'test'
    dataset_test_params['path'] = buffer_filepath+'_test'

    if dataset_name == "pendulum":
        if split == "train":
            dataset_train = DampedPendulum(**dataset_train_params)
            dataset_val   = DampedPendulum(**dataset_val_params)
        elif split == "test":
            dataset_test  = DampedPendulum(**dataset_test_params)
    elif dataset_name == "lorenz":
        if split == "train":
            dataset_train = LorenzTrue(**dataset_train_params)
            dataset_val   = LorenzTrue(**dataset_val_params)
        elif split == "test":
            dataset_test_params['nb_traj'] = 200
            dataset_test  = LorenzTrue(**dataset_test_params)
    elif dataset_name == "twobody":
        if split == "train":
            dataset_train = TwoBody(**dataset_train_params)
            dataset_val   = TwoBody(**dataset_val_params)
        elif split == "test":
            dataset_test_params['nb_traj'] = 100
            dataset_test  = TwoBody(**dataset_test_params)
    elif dataset_name == "wave":
        if split == "train":
            dataset_val_params['nb_traj'] = 50
            dataset_train = Wave(**dataset_train_params)
            dataset_val   = Wave(**dataset_val_params)
        elif split == "test":
            dataset_test_params['nb_traj'] = 50
            dataset_test  = Wave(**dataset_test_params)
    elif dataset_name == "rigidbody":
        if split == "train":
            dataset_train = RigidBodyTrue(**dataset_train_params)
            dataset_val   = RigidBodyTrue(**dataset_val_params)
        elif split == "test":
            dataset_test_params['nb_traj'] = 100
            print(dataset_test_params)
            dataset_test  = RigidBodyTrue(**dataset_test_params)
    elif dataset_name == "ks":
        if split == "train":
            dataset_val_params['nb_traj'] = 128
            dataset_train = KSTrue(**dataset_train_params)
            dataset_val   = KSTrue(**dataset_val_params)
        elif split == "test":
            dataset_test_params['nb_traj'] = 128
            dataset_test  = KSTrue(**dataset_test_params)
    elif dataset_name == "doublependulum":
        if split == "train":
            dataset_train = DoublePendulum(**dataset_train_params)
            dataset_val   = DoublePendulum(**dataset_val_params)
        elif split == "test":
            dataset_test_params['nb_traj'] = 100
            dataset_test  = DoublePendulum(**dataset_test_params)

    if split == "train":
        dataloader_train_params = {
                'dataset'    : dataset_train,
                'batch_size' : batch_size,
                'num_workers': 0,
                'pin_memory' : True,
                'drop_last'  : False,
                'shuffle'    : True,
                'worker_init_fn': seed_worker,
                'generator':g,
            }
        dataloader_val_params = {
            'dataset'    : dataset_val,
            'batch_size' : batch_size,
            'num_workers': 0,
            'pin_memory' : True,
            'drop_last'  : False,
            'shuffle'    : False,
            'worker_init_fn': seed_worker,
            'generator':g,
        }
        dataloader_train = DataLoader(**dataloader_train_params)
        dataloader_val   = DataLoader(**dataloader_val_params)
        return dataloader_train, dataloader_val, None

    elif split == "test":
        dataloader_test_params = {
            'dataset'    : dataset_test,
            'batch_size' : dataset_test_params['nb_traj'], # all test data in one batch for jax.vmap
            'num_workers': 0,
            'pin_memory' : True,
            'drop_last'  : False,
            'shuffle'    : False,
            'worker_init_fn': seed_worker,
            'generator':g,
        }
        dataloader_test  = DataLoader(**dataloader_test_params)
        return None, None, dataloader_test         


def init_dataloaders(dataset, integration_method, buffer_filepath=None, dt_num=0.5, duration=20, split="train", val_type="short"):
    assert buffer_filepath is not None
    if dataset == 'pendulum': # from Yin paper
        batch_size = 25
        nb_traj = 25
    elif dataset == 'lorenz': 
        batch_size = 40
        nb_traj = 40
        num_steps_max = 1000
    elif dataset == 'twobody': # from White paper
        ## Less traj setting
        # batch_size = 100
        # nb_traj = 10
        # num_steps_max = 1000 # in dt_num units 10s
        ## 8s or 5s setting 
        batch_size = 40
        nb_traj = 40
        num_steps_max = 800 # 8s
        #num_steps_max = 500 # 5s
        
    elif dataset == 'wave': # from Yin paper 
        batch_size = 64
        nb_traj = 200
    elif dataset == 'rigidbody': #dt_num=0.01
        batch_size = 40
        nb_traj = 40
        num_steps_max = 1500 # 15 seconds / dt_num
    elif dataset == 'doublependulum': # dt_num=0.01 
        batch_size = 40
        nb_traj = 40
        num_steps_max = 1000 # 10 seconds / dt_num
    elif dataset == "ks":
        batch_size = 128
        nb_traj = 512 # train, 128 val, 128 test
        num_steps_max = 140 # 28s / dt_num = 0.2

    return param_dataset(
        buffer_filepath,
        integration_method,
        dataset_name=dataset,
        batch_size=batch_size,
        dt_num=dt_num,
        num_steps_max=num_steps_max,
        duration=duration,
        nb_traj=nb_traj,
        split=split,
        val_type=val_type,)


if __name__ == '__main__':
    buffer_filepath = 'data/wave'
    dataloader_train, dataloader_val, dataloader_test = init_dataloaders(
        dataset='wave',
        dt_num=0.001,
        duration=0.001*25, # 25 time steps
        integration_method="RK4",
        buffer_filepath=buffer_filepath)

from .pendulum import DampedPendulum
from torch.utils.data import DataLoader 
import torch

# fix torch seed for reproducibility
torch.manual_seed(1)

# Build our dataloaders

def param_pendulum(buffer_filepath, integration_method, batch_size=25):
    dataset_train_params = {
        'nb_traj': 25, 
        'num_steps': 40,
        'dt': 0.5, 
        'split': 'train',
        'path': buffer_filepath+'_train',
        'integration_method': integration_method,
    }

    dataset_val_params = dict()
    dataset_val_params.update(dataset_train_params) # shared parameters across train and val
    dataset_val_params['nb_traj'] = 25
    dataset_val_params['split'] = 'val'
    dataset_val_params['path'] = buffer_filepath+'_val'

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['nb_traj'] = 25
    dataset_test_params['split'] = 'test'
    dataset_test_params['path'] = buffer_filepath+'_test'

    dataset_train = DampedPendulum(**dataset_train_params)
    dataset_val   = DampedPendulum(**dataset_val_params)
    dataset_test  = DampedPendulum(**dataset_test_params)

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : True,
    }

    dataloader_val_params = {
        'dataset'    : dataset_val,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }

    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : 1,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_val   = DataLoader(**dataloader_val_params)
    dataloader_test  = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_val, dataloader_test         


def init_dataloaders(dataset, integration_method, buffer_filepath=None):
    assert buffer_filepath is not None
    if dataset == 'pendulum':
        return param_pendulum(buffer_filepath, integration_method)


if __name__ == '__main__':
    buffer_filepath = 'data/tests'
    dataloader_train, dataloader_val, dataloader_test = init_dataloaders('pendulum', integration_method="RK4", buffer_filepath=buffer_filepath)

from .pendulum import DampedPendulum
from torch.utils.data import DataLoader 

# Build our dataloaders

def param_pendulum(buffer_filepath, batch_size=25):
    dataset_train_params = {
        'num_seq': 25, 
        'time_horizon': 20,
        'dt': 0.5, 
        'group': 'train',
        'path': buffer_filepath+'_train',
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['num_seq'] = 25
    dataset_test_params['group'] = 'test'
    dataset_test_params['path'] = buffer_filepath+'_test'

    dataset_train = DampedPendulum(**dataset_train_params)
    dataset_test  = DampedPendulum(**dataset_test_params)

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : True,
    }

    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test  = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test         


def init_dataloaders(dataset, buffer_filepath=None):
    assert buffer_filepath is not None
    if dataset == 'pendulum':
        return param_pendulum(buffer_filepath)


if __name__ == '__main__':
    buffer_filepath = 'data/damped_pendulum'
    dataloader_train, dataloader_test = init_dataloaders('pendulum', buffer_filepath)

# import torch
# import torch.nn as nn
import statistics
import jax 
import jax.numpy as jnp
import optax # https://github.com/deepmind/optax
import equinox as eqx

from utils import make_basedir # fix seed 
from utils import Logger, save

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams["figure.dpi"] = 100

_EPSILON = 1e-5

import os, sys
import collections

from equinox import Module
import optax
from torch.utils.data import DataLoader

def MSEjax(y_pred, y_true):
    return optax.squared_error(y_pred, y_true).mean()

class BaseExperiment(object):
    def __init__(self, device, path='./exp', seed=None):
        self.device = device
        self.path = make_basedir(path)

        # TODO what's the use of seed here?
        if seed is not None:
            key = jax.random.PRNGKey(seed)
            
        # if seed is not None: # fix torch seed, in jax ? 
        #     fix_seed(seed)

    def training(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def evaluating(self):
        self.training(mode=False)

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()        

    def to(self, device):
        for m in self.modules():
            jax.device_put(m) 
            #m.to(device)
        return self

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self):
        for name, module in self._modules.items():
            yield name, module

    def datasets(self):
        for name, dataset in self.named_datasets():
            yield dataset

    def named_datasets(self):
        for name, dataset in self._datasets.items():
            yield name, dataset

    def optimizers(self):
        for name, optimizer in self.named_optimizers():
            yield optimizer

    def named_optimizers(self):
        for name, optimizer in self._optimizers.items():
            yield name, optimizer

    def __setattr__(self, name, value):
        if isinstance(value, Module): # from equinox
            if not hasattr(self, '_modules'):
                self._modules = collections.OrderedDict()
            self._modules[name] = value
        elif isinstance(value, DataLoader):
            if not hasattr(self, '_datasets'):
                self._datasets = collections.OrderedDict()
            self._datasets[name] = value
        elif isinstance(value, optax.GradientTransformation): # from optax 
            if not hasattr(self, '_optimizers'):
                self._optimizers = collections.OrderedDict()
            self._optimizers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_datasets' in self.__dict__:
            datasets = self.__dict__['_datasets']
            if name in datasets:
                return datasets[name]
        if '_optimizers' in self.__dict__:
            optimizers = self.__dict__['_optimizers']
            if name in optimizers:
                return optimizers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        elif name in self._datasets:
            del self._datasets[name]
        elif name in self._optimizers:
            del self._optimizers[name]
        else:
            object.__delattr__(self, name)

class LoopExperiment(BaseExperiment):
    def __init__(
        self, train, test=None, root=None, nepoch=10, **kwargs):
        super().__init__(**kwargs)
        self.train = train
        self.test = test
        self.nepoch = nepoch
        self.logger = Logger(filename=os.path.join(self.path, 'log.txt'))
        print(' '.join(sys.argv))

    def train_step(self, batch, val=False):
        self.training()
        batch = jax.device_put(batch) #convert_tensor(batch, self.device)
        loss, output = self.step(batch)
        metric = self.metric(**output, **batch)
        return batch, output, loss, metric

    def val_step(self, batch, val=False):
        self.evaluating()
        batch = jax.device_put(batch)
        loss, output = self.step(batch, backward=False)
        metric = self.metric(**output, **batch)
        return batch, output, loss, metric

    def log(self, epoch, iteration, metrics):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(self.train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=self.nepoch,
            i=iteration+1,
            max_i=len(self.train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=name, value=float(value))
            
        print(message)

    def step(self, **kwargs):
        raise NotImplementedError

class APHYNITYExperiment(LoopExperiment):
    def __init__(self, net, optimizer, min_op, lambda_0, tau_2, niter=1, nupdate=100, nlog=10, **kwargs):
        super().__init__(**kwargs)

        self.traj_loss = MSEjax
        self.net = net #.to(self.device) 
        self.optimizer = optimizer
        # It only makes sense to train the arrays in our model,
        # so filter out everything else.
        self.opt_state = self.optimizer.init(eqx.filter(net, eqx.is_array)) 

        self.min_op = min_op
        self.tau_2 = tau_2
        self.niter = niter
        self._lambda = lambda_0

        self.nlog = nlog
        self.nupdate = nupdate

    
    def lambda_update(self, loss):
        self._lambda = self._lambda + self.tau_2 * loss
    
    def _forward(self, states, t, backward):
        target = states
        y0 = states[:, :, 0]
        pred = self.net(y0, t)

        def loss_fn(net): # one loss function for backpropagation
            pred = net(y0, t)
            loss_value = self.traj_loss(pred, target)
            aug_deriv = net.model_aug.get_derivatives(states)

            if self.min_op == 'l2_normalized':
                loss_op = ((jnp.linalg.norm(aug_deriv, ord=2, axis=1) / (jnp.linalg.norm(states, ord=2, axis=1) + 1e-8)) ** 2).mean()
            elif self.min_op == 'l2':
                loss_op = (jnp.linalg.norm(aug_deriv, ord=2, axis=1) ** 2).mean()
            else:
                loss_op = 0.0  # Default to zero if min_op is not recognized
            
            loss_total = loss_value * self._lambda + loss_op
            return loss_total, (loss_value, loss_op)
        
        
        (loss_total, (loss, loss_op)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(self.net) # has_aux=True to return loss_value and loss_op
        # loss, grads = eqx.filter_value_and_grad(self.traj_loss)(self.net, pred, target)
        # aug_deriv = self.net.model_aug.get_derivatives(states)
        # if self.min_op == 'l2_normalized':
        #     loss_op = ((jnp.linalg.norm(aug_deriv, ord=2, axis=1) / (jnp.linalg.norm(states, ord=2, axis=1) + _EPSILON)) ** 2).mean()
        # elif self.min_op == 'l2':
        #     loss_op = (jnp.linalg.norm(aug_deriv, ord=2, axis=1) ** 2).mean()

        if backward: 
            updates, self.opt_state = self.optimizer.update(
                grads, self.opt_state, eqx.filter(self.net, eqx.is_array))
            self.net = eqx.apply_updates(self.net, updates)
            # loss_total.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()

        loss = {
            'loss': loss,
            'loss_op': loss_op,
        }

        output = {
            'states_pred': pred,
        }
        return loss, output

    def step(self, batch, backward=True):
        states = batch['states']
        t = batch['t'][0]
        loss, output = self._forward(states, t, backward)
        return loss, output

    def metric(self, states, states_pred, **kwargs):
        metrics = {}
        metrics['param_error'] = statistics.mean(abs(v1-float(v2))/v1 for v1, v2 in zip(self.train.dataset.params.values(), self.net.get_pde_params().values()))
        metrics.update(self.net.get_pde_params())
        metrics.update({f'{k}_real': v for k, v in self.train.dataset.params.items() if k in metrics})
        return metrics

    def run(self):
        loss_test_min = None
        for epoch in range(self.nepoch): 
            # for now batch_size = dataset_size 
            for iteration, data in enumerate(self.train.dataset, 0):

            #for iteration, data in zip(range(self.train.__len__()), self.train):
            #for iteration, data in enumerate(self.train, 0):
                
                for _ in range(self.niter):
                    _, _, loss, metric = self.train_step(data)

                total_iteration = epoch * (len(self.train)) + (iteration + 1)
                loss_train = loss['loss'].item()
                self.lambda_update(loss_train)

                if total_iteration % self.nlog == 0:
                    self.log(epoch, iteration, loss | metric)

                if total_iteration % self.nupdate == 0:
                    loss_test = 0.
                    for j, data_test in enumerate(self.test, 0):
                        _, _, loss, metric = self.val_step(data_test)
                        loss_test += loss['loss'].item()
                        
                    loss_test /= j + 1

                    if loss_test_min == None or loss_test_min > loss_test:
                        loss_test_min = loss_test
                        # save model using equinox
                        # TODO how to also save optimizer state?
                        hyperparameters = {
                            "epoch": epoch,
                            "loss": loss_test_min,
                            }
                        save(self.net, self.path + f'/model_{loss_test_min:.3e}.eqx', hyperparameters)

                        # torch.save({
                        #     'epoch': epoch,
                        #     'model_state_dict': self.net.state_dict(),
                        #     'optimizer_state_dict': self.optimizer.state_dict(),
                        #     'loss': loss_test_min, 
                        # }, self.path + f'/model_{loss_test_min:.3e}.pt')

                    loss_test = {
                        'loss_test': loss_test,
                    }
                    print('#' * 80)
                    self.log(epoch, iteration, loss_test | metric)
                    print(f'lambda: {self._lambda}')
                    print('#' * 80)

                    # for debugging
                    break
                break
            break



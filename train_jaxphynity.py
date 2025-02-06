import optax 
import os

from experiments import APHYNITYExperiment
from networks import *
from forecasters import *
from utils import init_linear_weight, orthogonal_init
from datasets import init_dataloaders
from utils import Logger, save, make_basedir

def log(train, epoch, iteration, metrics, nepoch):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=nepoch,
            i=iteration+1,
            max_i=len(train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=name, value=float(value))
            
        print(message)

# def metric(net, states, states_pred, **kwargs):
#     metrics = {}
#     metrics['param_error'] = statistics.mean(abs(v1-float(v2))/v1 for v1, v2 in zip(train.dataset.params.values(), net.get_pde_params().values()))
#     metrics.update(self.net.get_pde_params())
#     metrics.update({f'{k}_real': v for k, v in train.dataset.params.items() if k in metrics})
#     return metrics
# Losses
def MSEjax(y_pred, y_true):
    return optax.squared_error(y_pred, y_true).mean()

@eqx.filter_jit
def loss_trajectory(model, y, t):
    x = y[:,:,0] # y0
    y_pred = model(x,t)
    return MSEjax(y_pred, y), y_pred

@eqx.filter_jit
def loss_Fa(model, y, min_op):
    aug_deriv = model.model_aug.get_derivatives(y)
    if min_op == 'l2_normalized':
        loss_op = ((jnp.linalg.norm(aug_deriv, ord=2, axis=1) / (jnp.linalg.norm(y, ord=2, axis=1) + 1e-8)) ** 2).mean()
    elif min_op == 'l2':
        loss_op = (jnp.linalg.norm(aug_deriv, ord=2, axis=1) ** 2).mean()
    else:
        loss_op = 0.0  # Default to zero if min_op is not recognized
    return loss_op

@eqx.filter_value_and_grad(has_aux=True)
@eqx.filter_jit
def loss_fn(model, y, t, min_op, lambda_):
    lossT, y_pred = loss_trajectory(model, y, t)
    loss_op = loss_Fa(model, y, min_op)
    return lossT + lambda_ * loss_op, (lossT,loss_op, y_pred)

# Routine
def training_routine(train, test, net, optimizer, min_op, _lambda, tau_2, niter, path, device, nlog=1, nupdate=1, nepoch=10):
    path = make_basedir(path)
    logger = Logger(filename=os.path.join(path, 'log.txt'))
    # write hyperparameters and settings
    logger.write("lambda_0: {}, tau_2: {}, niter: {}, min_op: {}".format(_lambda, tau_2, niter, min_op))
    logger.write("\n")
    logger.write("nepoch: {}, nlog: {}, nupdate: {}".format(nepoch, nlog, nupdate))
    logger.write("\n")

    # optimizer initialization
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    loss_test_min = None
    for epoch in range(nepoch): 
        for iteration, data in enumerate(train, 0):  
            for _ in range(niter): 
                ### TRAIN STEP
                states = jnp.permute_dims(jnp.array(data['states']), (0, 2, 1))
                t = jnp.array(data['t'][0])
                (loss_total, (loss_val, loss_op, pred)), grads = loss_fn(net, states, t, min_op, _lambda) 
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(net, eqx.is_array))
                net = eqx.apply_updates(net, updates)
                loss = {
                    'loss': loss_val,
                    'loss_op': loss_op,
                }
                output = {'states_pred': pred,}  
                # TODO compute metrics : param error, params 
                metric = ""

            total_iteration = epoch * (len(train)) + (iteration + 1)
            loss_train = loss['loss'].item()
            _lambda = _lambda + tau_2 * loss_train
            print(f'lambda: {_lambda}')
            if total_iteration % nlog == 0:
                log(train, epoch, iteration, loss, nepoch) # | metric)

            
            ### VALIDATION STEP
            if total_iteration % nupdate == 0:
                loss_test = 0.
                for j, data_test in enumerate(test, 0):
                    # no backpropagation
                    states = jnp.permute_dims(jnp.array(data_test['states']), (0, 2, 1))
                    t = jnp.array(data_test['t'][0])
                    (loss_total, (loss_val, loss_op, pred)), grads = loss_fn(net, states, t, min_op, _lambda) 
                    loss = {
                        'loss': loss_val,
                        'loss_op': loss_op,
                    }

                    output = {'states_pred': pred,}
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
                    save(path + f'/model_{loss_test_min:.3e}.eqx', hyperparameters, net)

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
                log(train, epoch, iteration, loss_test, nepoch) # | metric)
                print(f'lambda: {_lambda}')
                print('#' * 80)       


# Main
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
        niter = 1 #5
        min_op = 'l2_normalized'
    
    optimizer = optax.adam(learning_rate=tau_1, b1=0.9, b2=0.999)
    training_routine(train, test, net, optimizer, min_op, lambda_0, tau_2, niter, path, device)
    # experiment = APHYNITYExperiment(
    #         train=train, test=test, net=net, optimizer=optimizer, 
    #         min_op=min_op, lambda_0=lambda_0, tau_2=tau_2, niter=niter, nlog=10,
    #         nupdate=100, nepoch=50000, path=path, device=device
    #     )
    # experiment.run()

if __name__ == '__main__':
    dataset_name = 'pendulum'
    model_phy_option = 'complete'
    model_aug_option = True
    path = 'data/damped_pendulum'
    device = 'cpu'
    train_aphynity(dataset_name, model_phy_option, model_aug_option, path, device)
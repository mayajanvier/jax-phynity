from networks import *
from solvers.runge_kutta import RK_solver_fixed, RK_tableaux, ButcherTableau
import jax.numpy as jnp
import equinox as eqx 
from einops import rearrange

class DerivativeEstimator(eqx.Module):
    """ Returns the model with the augmented model
    if is_augmented is True """
    model_phy: eqx.Module
    model_aug: eqx.Module
    is_augmented: bool

    def __init__(self, model_phy, model_aug, is_augmented):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented

    def __call__(self, state, t):
        res_phy = self.model_phy(state)
        if self.is_augmented:
            res_aug = jax.vmap(self.model_aug)(state)
            return res_phy + res_aug
        else:
            return res_phy

class Forecaster(eqx.Module):
    """ Integrates a trajectory using int_ method """
    model_phy: eqx.Module
    model_aug: eqx.Module
    derivative_estimator: eqx.Module
    method: str
    int_: callable 
    tableau: ButcherTableau

    def __init__(self, model_phy, model_aug, is_augmented, method='RK4'):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug
        # our true trainable model
        self.derivative_estimator = DerivativeEstimator(self.model_phy, self.model_aug, is_augmented=is_augmented)
        self.method = method
        self.int_ = RK_solver_fixed #odeint 
        self.tableau = RK_tableaux[self.method]
        
    def __call__(self, y, t):
        y0 = y[:,:,0]
        t_span = t[-1] - t[0]
        res, _, _ = self.int_(self.derivative_estimator, t_span=t_span, y0=y0, t_eval=t, tableau=self.tableau) 
        # res: T x batch_size x n_c (x h x w)
        return rearrange(res, 'T b nc -> b nc T') # batch_size x n_c x T (x h x w)
    
    def get_pde_params(self):
        params = {
            "omega0_square": self.derivative_estimator.model_phy.omega0_square,
            "alpha": self.derivative_estimator.model_phy.alpha,
        }
        return params
    
if __name__ == '__main__':
    from utils import init_linear_weight, orthogonal_init

    mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
    model_phy = DampedPendulumParamPDE(is_complete=True, real_params=None)
    model_aug = MLP(key=mkey, state_c=2, hidden=200)
    init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=0.2) 
    net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=True)

    y0 = jnp.ones((5,2))
    t = jnp.linspace(0, 10, 10)
    print(t.shape, y0.shape)
    print(t)
    y = net(y0, t)
    print(y.shape)
    print(net.get_pde_params())
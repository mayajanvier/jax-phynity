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
    is_phy: str

    def __init__(self, model_phy, model_aug, is_augmented, is_phy):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented
        self.is_phy = is_phy

    def __call__(self, state, t):
        if self.is_phy == "none":
            res_aug = self.model_aug(state)
            return res_aug
        else:
            res_phy = self.model_phy(state)
            if self.is_augmented:
                res_aug = self.model_aug(state)
                return res_phy + res_aug
            else:
                return res_phy

# TODO join avec derivative estimator
class Forecaster(eqx.Module):
    """ Integrates a trajectory using int_ method """
    model_phy: eqx.Module
    model_aug: eqx.Module
    dt: float
    num_steps: int
    integration_method: str

    ### if we use external class
    #derivative_estimator: eqx.Module

    ### if we use internal function
    is_phy: str
    is_augmented: bool

    int_: callable 
    # TODO le fait de déclarer ButcherTableau en argument dans la classe crée une erreur avec eqx.filter_jit
    #tableau: ButcherTableau

    def __init__(self, model_phy, model_aug, is_augmented, is_phy, dt, num_steps, integration_method='RK4'):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug
        # our true trainable model
        #self.derivative_estimator = DerivativeEstimator(self.model_phy, self.model_aug, is_augmented=is_augmented, is_phy=is_phy) 
        # or 
        self.is_augmented = is_augmented
        self.is_phy = is_phy  
        # solver
        self.dt = dt
        self.num_steps = num_steps
        self.integration_method = integration_method
        self.int_ = RK_solver_fixed # on definit dt et le tableau là #odeint 
        #self.tableau = RK_tableaux[self.method]
        
    def __call__(self, y0):
        # y0:   (n_c,)
        # res:  (n_c, T) 
        res, _, _, _ = self.int_(self.derivative_estimator, y0=y0, dt=self.dt, num_steps=self.num_steps, tableau=RK_tableaux[self.integration_method]) 
        return res 
    
    def get_pde_params(self):
        params = {
            "omega0_square": self.model_phy.omega0_square,
            "alpha": self.model_phy.alpha,
        }
        return params
    
    def derivative_estimator(self, state, t):
        if self.is_phy == "none":
            res_aug = self.model_aug(state)
            return res_aug
        else:
            res_phy = self.model_phy(state)
            if self.is_augmented:
                res_aug = self.model_aug(state)
                return res_phy + res_aug
            else:
                return res_phy

    
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
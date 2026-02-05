from networks import *
from solvers.runge_kutta import RK_solver_fixed, RK_tableaux, ButcherTableau
import jax
import jax.numpy as jnp
import equinox as eqx 
import diffrax
from einops import rearrange
from diffrax import diffeqsolve, ODETerm

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

class ForecasterDiffrax(eqx.Module):
    model_phy: eqx.Module
    model_aug: eqx.Module
    t: jax.Array
    dt: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    integration_method: str = eqx.field(static=True)
    term: diffrax.ODETerm = eqx.field(static=True)
    solver: diffrax.AbstractSolver = eqx.field(static=True)

    is_phy: str = eqx.field(static=True)
    is_augmented: bool = eqx.field(static=True)
    int_: callable = eqx.field(static=True)

    def __init__(self, model_phy, model_aug, is_augmented, is_phy, dt, num_steps, integration_method='DOPRI5'):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented
        self.is_phy = is_phy
        self.dt = dt
        self.num_steps = num_steps
        self.integration_method = integration_method
        self.int_ = diffrax.diffeqsolve
        self.t = jnp.linspace(0, dt * num_steps, num_steps + 1)
        self.term = diffrax.ODETerm(Forecaster.derivative_estimator)
        self.solver = RK_tableaux_diffrax[self.integration_method]

    @eqx.filter_jit
    def __call__(self, y0):
        res = self.int_(
            self.term,
            solver=self.solver,
            t0=self.t[0],
            t1=self.t[-1],
            dt0=self.dt,
            y0=y0,
            args=(self.model_phy, self.model_aug, self.is_phy, self.is_augmented),  
            saveat=diffrax.SaveAt(ts=self.t),
        )
        #res = rearrange(res.ys, 't nc -> nc t')  # (n_c, T)
        return res.ys

    @staticmethod
    def derivative_estimator(t, y, args):
        model_phy, model_aug, is_phy, is_augmented = args
        if is_phy in [
            "none", "none_Fa", "none_Fa_prime", "none_Fa_prime_supX",
            "none_Fa_prime_supX_norm", "none_Fa_prime_supX_l2",
            "none_Fa_prime_supX_direct", "none_Fa_prime_supYX",
        ]:
            return model_aug(y)
        else:
            res_phy = model_phy(y)
            if is_augmented:
                res_aug = model_aug(y)
                return res_phy + res_aug
            else:
                return res_phy
            

class Forecaster(eqx.Module):
    """ Integrates a trajectory using int_ method (RK_solver_fixed I wrote) """
    model_phy: eqx.Module
    model_aug: eqx.Module
    dt: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    integration_method: str = eqx.field(static=True)

    ### if we use external class
    #derivative_estimator: eqx.Module

    ### if we use internal function
    is_phy: str = eqx.field(static=True)
    is_augmented: bool = eqx.field(static=True)

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

        # derivative_estimator: si on le définit ici ca va vite mais 
        # ne s'entraine pas 
        # if self.is_phy == "none":
        #     self.derivative_estimator = lambda state, t: self.model_aug(state)
        # else:
        #     if self.is_augmented:
        #         self.derivative_estimator = lambda state, t: self.model_phy(state) + self.model_aug(state)
        #     else:
        #         self.derivative_estimator = lambda state, t: self.model_phy(state)
        
    def __call__(self, y0):
        # y0:   (n_c,)
        # res:  (n_c, T) 
        res, _, _, _ = self.int_(self.derivative_estimator, y0=y0, dt=self.dt, num_steps=self.num_steps, tableau=RK_tableaux[self.integration_method]) 
        res = rearrange(res, 'nc T -> T nc')  # (T, nc)
        return res 
    
    def validation_call(self, y0, num_steps):
        # y0:   (n_c,)
        # res:  (n_c, T) 
        res, _, _, _ = self.int_(self.derivative_estimator, y0=y0, dt=self.dt, num_steps=num_steps, tableau=RK_tableaux[self.integration_method]) 
        res = rearrange(res, 'nc T -> T nc')
        return res
    
    def get_pde_params(self):
        params = {
            "omega0_square": self.model_phy.omega0_square,
            "alpha": self.model_phy.alpha,
        }
        return params
    
    def derivative_estimator(self, state, t):
        # state of shape (nc,)
        if self.is_phy in["none",
                          "none_AD_sup",
                          "none_FD_unsup",
                          "none_Fa",
                          "none_Fa_prime",
                          "none_Fa_prime_supX",
                          "none_Fa_prime_supX_norm",
                          "none_Fa_prime_supX_l2",
                          "none_Fa_prime_supX_direct",
                          "none_Fa_prime_supYX"]:
            res_aug = self.model_aug(state)
            return res_aug
        else:
            res_phy = self.model_phy(state)
            if self.is_augmented:
                res_aug = self.model_aug(state)
                return res_phy + res_aug
            else:
                return res_phy

        # use lax.cond to switch between models
        # conditions lax imbriquées trop lourd avec vmap
        # return jax.lax.cond(
        #     self.is_phy == "none",
        #     lambda s: self.model_aug(s), # true branch
        #     lambda s: jax.lax.cond(
        #         self.is_augmented, 
        #         lambda s: self.model_phy(s) + self.model_aug(s), # true branch
        #         lambda s: self.model_phy(s), # false branch
        #         s),
        #     state)

    
if __name__ == '__main__':
    from utils import init_linear_weight, orthogonal_init

    mkey, ikey = jax.random.split(jax.random.PRNGKey(0))
    model_phy = PendulumParamPDE(is_complete=True, real_params=None)
    model_aug = MLP(key=mkey, state_c=2, hidden=200)
    model_aug = init_linear_weight(model_aug, orthogonal_init, key=ikey, init_gain=0.2) 
    net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=True)

    y0 = jnp.ones((5,2))
    t = jnp.linspace(0, 10, 10)
    print(t.shape, y0.shape)
    print(t)
    y = net(y0, t)
    print(y.shape)
    print(net.get_pde_params())
import jax
import jax.numpy as jnp
import numpy as np 
from dataclasses import dataclass, field # dataclass is a decorator that is used to create classes with attributes, __init__ method, __repr__ method, and __eq__ method all in one go.
from typing import Optional, Tuple
import equinox as eqx

### diffrax Butcher tableau class 
@dataclass(frozen=True)
class ButcherTableau:
    """The Butcher tableau for an explicit or diagonal Runge--Kutta method."""

    # Explicit RK methods
    c: jnp.ndarray
    b_sol: jnp.ndarray
    b_error: jnp.ndarray
    a_lower: Tuple[jnp.ndarray, ...]

    # Properties implied by the above tableaus, e.g. used to define fast-paths.
    # field is used to define default values for the attributes of the class
    ssal: bool = field(init=False) # (Same As Last): Indicates whether the final solution equals the last stage.
    fsal: bool = field(init=False) # (First Same As Last): Indicates whether the first stage is equivalent to the last stage. 
    implicit: bool = field(init=False)
    num_stages: int = field(init=False) # number of "k_i" stages in the Runge-Kutta method

     # Implicit RK methods
    a_diagonal: Optional[jnp.ndarray] = None
    a_predictor: Optional[Tuple[jnp.ndarray, ...]] = None 
    c1: float = 0.0 # first stage coefficient

    def __post_init__(self):
        assert self.c.ndim == 1
        for a_i in self.a_lower:
            assert a_i.ndim == 1
        assert self.b_sol.ndim == 1
        assert self.b_error.ndim == 1
        assert self.c.shape[0] == len(self.a_lower)
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.a_lower))
        assert self.c.shape[0] + 1 == self.b_sol.shape[0]
        assert self.c.shape[0] + 1 == self.b_error.shape[0]
        for i, (a_i, c_i) in enumerate(zip(self.a_lower, self.c)):
            diagonal = 0 if self.a_diagonal is None else self.a_diagonal[i + 1]
            assert jnp.allclose(sum(a_i) + diagonal, c_i)
        assert jnp.allclose(sum(self.b_sol), 1.0)
        assert jnp.allclose(sum(self.b_error), 0.0, atol=1e-6) # nn.allclose returns True but no jnp.allclose 

        if self.a_diagonal is None:
            assert self.a_predictor is None
        else:
            assert self.a_predictor is not None
            assert self.a_diagonal.ndim == 1
            assert self.c.shape[0] + 1 == self.a_diagonal.shape[0]
            assert len(self.a_lower) == len(self.a_predictor)
            for a_lower_i, a_predictor_i in zip(self.a_lower, self.a_predictor):
                assert a_lower_i.shape == a_predictor_i.shape
                assert jnp.allclose(sum(a_predictor_i), 1.0)

        lower_b_sol_equal = (self.b_sol[:-1] == self.a_lower[-1]).all().item()
        last_diagonal = 0 if self.a_diagonal is None else self.a_diagonal[-1]
        diagonal_b_sol_equal = (self.b_sol[-1] == last_diagonal).item()
        explicit_first_stage = (
            self.a_diagonal is None or (self.a_diagonal[0] == 0).item()
        )
        explicit_last_stage = (
            self.a_diagonal is None or (self.a_diagonal[-1] == 0).item()
        )
        # (vector field)-control product `k1` is the same across first/last stages.
        object.__setattr__(
            self,
            "fsal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_first_stage,
        )
        # Solution `y1` is the same as the last stage
        object.__setattr__(
            self,
            "ssal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_last_stage,
        )
        object.__setattr__(self, "implicit", self.a_diagonal is not None)
        object.__setattr__(self, "num_stages", len(self.b_sol))

### Butcher tableaux for Runge-Kutta methods
# 3/8 rule in odeint used in APHYNITY (torchdiffeq)
RK4_tableau = ButcherTableau(
    a_lower=(
        jnp.array([1 / 3]),
        jnp.array([-1 / 3, 1]),
        jnp.array([1, -1, 1]),
    ),
    b_sol=jnp.array([1 / 8, 3 / 8, 3 / 8, 1 / 8]),
    # TODO comment calculer une erreur pour RK4 ? 
    b_error=jnp.array([0, 0, 0, 0]),
    c =jnp.array([1 / 3, 2 / 3, 1]),
)

dopri5_tableau = ButcherTableau(
    a_lower=(
        jnp.array([1 / 5]),
        jnp.array([3 / 40, 9 / 40]),
        jnp.array([44 / 45, -56 / 15, 32 / 9]),
        jnp.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
        jnp.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
        jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
    ),
    # 5th order weights 
    b_sol=jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]), 
    # b_error embeds the 4th order error estimate: bi(5) - bi(4)
    b_error=jnp.array(
        [
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1.0 / 60.0,
        ]
    ), 
    c=jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]),
)

dopri8_tableau = ButcherTableau(
    a_lower=(
        jnp.array([1 / 18]),
        jnp.array([1 / 48, 1 / 16]),
        jnp.array([1 / 32, 0, 3 / 32]),
        jnp.array([5 / 16, 0, -75 / 64, 75 / 64]),
        jnp.array([3 / 80, 0, 0, 3 / 16, 3 / 20]),
        jnp.array(
            [
                29443841 / 614563906,
                0,
                0,
                77736538 / 692538347,
                -28693883 / 1125000000,
                23124283 / 1800000000,
            ]
        ),
        jnp.array(
            [
                16016141 / 946692911,
                0,
                0,
                61564180 / 158732637,
                22789713 / 633445777,
                545815736 / 2771057229,
                -180193667 / 1043307555,
            ]
        ),
        jnp.array(
            [
                39632708 / 573591083,
                0,
                0,
                -433636366 / 683701615,
                -421739975 / 2616292301,
                100302831 / 723423059,
                790204164 / 839813087,
                800635310 / 3783071287,
            ]
        ),
        jnp.array(
            [
                246121993 / 1340847787,
                0,
                0,
                -37695042795 / 15268766246,
                -309121744 / 1061227803,
                -12992083 / 490766935,
                6005943493 / 2108947869,
                393006217 / 1396673457,
                123872331 / 1001029789,
            ]
        ),
        jnp.array(
            [
                -1028468189 / 846180014,
                0,
                0,
                8478235783 / 508512852,
                1311729495 / 1432422823,
                -10304129995 / 1701304382,
                -48777925059 / 3047939560,
                15336726248 / 1032824649,
                -45442868181 / 3398467696,
                3065993473 / 597172653,
            ]
        ),
        jnp.array(
            [
                185892177 / 718116043,
                0,
                0,
                -3185094517 / 667107341,
                -477755414 / 1098053517,
                -703635378 / 230739211,
                5731566787 / 1027545527,
                5232866602 / 850066563,
                -4093664535 / 808688257,
                3962137247 / 1805957418,
                65686358 / 487910083,
            ]
        ),
        jnp.array(
            [
                403863854 / 491063109,
                0,
                0,
                -5068492393 / 434740067,
                -411421997 / 543043805,
                652783627 / 914296604,
                11173962825 / 925320556,
                -13158990841 / 6184727034,
                3936647629 / 1978049680,
                -160528059 / 685178525,
                248638103 / 1413531060,
                0,
            ]
        ),
        jnp.array(
            [
                14005451 / 335480064,
                0,
                0,
                0,
                0,
                -59238493 / 1068277825,
                181606767 / 758867731,
                561292985 / 797845732,
                -1041891430 / 1371343529,
                760417239 / 1151165299,
                118820643 / 751138087,
                -528747749 / 2220607170,
                1 / 4,
            ]
        ),
    ),
    b_sol=jnp.array(
        [
            14005451 / 335480064,
            0,
            0,
            0,
            0,
            -59238493 / 1068277825,
            181606767 / 758867731,
            561292985 / 797845732,
            -1041891430 / 1371343529,
            760417239 / 1151165299,
            118820643 / 751138087,
            -528747749 / 2220607170,
            1 / 4,
            0,
        ]
    ),
    b_error=jnp.array(
        [
            14005451 / 335480064 - 13451932 / 455176623,
            0,
            0,
            0,
            0,
            -59238493 / 1068277825 - -808719846 / 976000145,
            181606767 / 758867731 - 1757004468 / 5645159321,
            561292985 / 797845732 - 656045339 / 265891186,
            -1041891430 / 1371343529 - -3867574721 / 1518517206,
            760417239 / 1151165299 - 465885868 / 322736535,
            118820643 / 751138087 - 53011238 / 667516719,
            -528747749 / 2220607170 - 2 / 45,
            1 / 4,
            0,
        ]
    ),
    c=jnp.array(
        [
            1 / 18,
            1 / 12,
            1 / 8,
            5 / 16,
            3 / 8,
            59 / 400,
            93 / 200,
            5490023248 / 9719169821,
            13 / 20,
            1201146811 / 1299019798,
            1,
            1,
            1,
        ]
    ),
)

RK_tableaux = {
    "RK4": RK4_tableau,
    "DOPRI5": dopri5_tableau,
    "DOPRI8": dopri8_tableau,
}

# Runge-Kutta step
def runge_kutta_step(
    f,
    y: jnp.ndarray,
    t: jnp.ndarray,
    dt: jnp.ndarray,
    tableau: ButcherTableau,
    k_first: Optional[jnp.ndarray] = None,
):
    """Perform a single step of a Runge--Kutta method.

    Args:
        f: The vector field function.
        y: The current state, (nc,)
        t: The current time, Array[floay]
        dt: The step size, float
        tableau: The Butcher tableau of the Runge--Kutta method.
        k_first: First stage derivative (optional, used in FSAL).

    Returns:
        y_next: Next state after the step.
        error: Error estimate for adaptive methods (optional).
        k_last: Last stage derivative (for FSAL-enabled methods).
    """
    num_stages = tableau.num_stages

    # jax.lax.scan function for building k 
    # def step(carry, i):
    #     k, y = carry
    #     ti = t + tableau.c[i] * dt
    #     yi = y + dt * jnp.tensordot(tableau.a_lower[i], k[:i+1], axes=1)
    #     k_next = f(yi, ti)
    #     k = k.at[i+1].set(k_next)
    #     return (k,y), k_next
    
    k = jnp.zeros((num_stages,) + y.shape, dtype=y.dtype) # shape (num_stages, y.shape)
    # explicit Runge-Kutta methods
    if not tableau.implicit: # a_diagonal is None
        if tableau.fsal and k_first is not None: # FSAL (First Same As Last) optimization
            k = k.at[0].set(k_first) # k1 = k_first(n_step) = k_last(n_step-1)
        else: 
            k = k.at[0].set(f(y, t+ dt * tableau.c1)) # k1 = f(y, t + c1 * dt)

        # loop over stages
        # (final_k, _),_ = jax.lax.scan(step, (k,y), jnp.arange(num_stages-1))
        # y_next = y + dt * jnp.tensordot(tableau.b_sol, final_k, axes=1)
        # error = jnp.tensordot(tableau.b_error, final_k, axes=1)

        # avoid loop
        #tiVec = t + tableau.c * dt
        
        for i in range(tableau.num_stages-1):
            ti = t + tableau.c[i] * dt
            yi = y + dt * jnp.tensordot(tableau.a_lower[i], k[:i+1], axes=1)
            k = k.at[i+1].set(f(yi, ti))    
        y_next =  y + dt * jnp.tensordot(tableau.b_sol, k, axes=1)
        error = jnp.tensordot(tableau.b_error, k, axes=1)

        if tableau.fsal:
            return y_next, error, k[-1] # return the last stage for FSAL
        else:
            return y_next, error 
        
    # implicit Runge-Kutta methods
    else:
        raise NotImplementedError("Implicit Runge-Kutta methods are not supported yet.")  

### Runge-Kutta methods
def RK_solver_fixed(fun, y0, dt, num_steps, tableau):
    """Solve an initial value problem using the Dormand--Prince 5 method with JAX scan.
    Args:
        fun: The vector field function.
        y0: The initial state, (nc,)
        dt: The step size, float
        num_steps: The number of steps to take, int
        tableau: The Butcher tableau of the Runge--Kutta method, ButcherTableau
    
    Returns:
        t_eval: time points of evaluation, (num_steps+1,)
        y: solution evaluated on t_eval points, (num_steps+1, y0.shape)
        global_error: global error of the method, float
        errors: list of errors at each time step, (num_steps,)"""
    
    def step(carry, t):
        y, k_first = carry
        if tableau.fsal and k_first is not None:
            y_next, error, k_first = runge_kutta_step(fun, y, t, dt, tableau, k_first)
        else:
            y_next, error = runge_kutta_step(fun, y, t, dt, tableau)
            k_first = None  # Not needed if FSAL is disabled
        return (y_next, k_first), (y_next, error)
    
    t_eval = jnp.arange(0, (num_steps+1) * dt, dt) # array of time points to evaluate
    y_init = (y0, None)  # Initial carry (state, k_first)
    
    (final_state, _), (y_sol, errors) = jax.lax.scan(step, y_init, t_eval[:-1])
    
    # Prepend initial condition
    y_sol = jnp.vstack([y0[None, :], y_sol])
    global_error = jnp.sum(errors)
    
    return y_sol.T, t_eval, global_error, errors


# fixed step size
# def RK_solver_fixed(fun, y0, dt, num_steps, tableau):
#     """Solve an initial value problem using the Dormand--Prince 5 method.
    
#     Args:
#         fun: The vector field function.
#         y0: The initial state.
#         dt: The step size.
#         num_steps: The number of steps to take.
#         tableau: The Butcher tableau of the Runge--Kutta method.
    
#     Returns:
#         t_eval: time points of evaluation, (num_steps+1,)
#         y: solution evaluated on t_eval points, (num_steps+1, y0.shape)
#         global_error: global error of the method, float
#         errors: list of errors at each time step, (num_steps,)
#     """
#     # initialize
#     t_eval = jnp.arange(0, (num_steps+1) * dt, dt) # array of time points to evaluate
#     y = jnp.zeros(y0.shape + (len(t_eval),), dtype=y0.dtype) 
#     y = y.at[:,0].set(y0)
#     global_error = 0.
#     errors = []
#     n_step = 0
#     k_first = None

#     for t_current in t_eval:
#         if tableau.fsal:
#             y_next, error, k_first = runge_kutta_step(fun, y[:,n_step], t_current, dt, tableau, k_first)
#         else:
#             y_next, error = runge_kutta_step(fun, y[:,n_step], t_current, dt, tableau)
#         n_step += 1 
#         y = y.at[:,n_step].set(y_next)
#         global_error += error
#         errors.append(error)
#     return y, t_eval, global_error, errors


### equinox Integrator class
# TODO: voir si c'est mieux pour jiter mais pour l'instant ne fonctionne pas 
class IntegratorRK(eqx.Module):
    tableau: ButcherTableau

    def __init__(self, method):
        self.tableau = RK_tableaux[method]

    def __call__(self, f, t_span, y0, t_eval):
        return self.RK_solver_fixed(f, t_span, y0, t_eval, self.tableau)
    
    def RK_solver_fixed(self, fun, t_span, y0, t_eval, tableau, rtol=1e-10, n_step_max=1000):
        """Solve an initial value problem using the Dormand--Prince 5 method."""
        # initialize
        #tableau = RK_tableaux[method]
        y = jnp.zeros((len(t_eval),) + y0.shape, dtype=y0.dtype)
        y = y.at[0].set(y0)
        #t, tf = t_span # needed when adaptive step size is used
        dt = t_eval[1] - t_eval[0]
        global_error = 0.
        errors = []
        n_step = 0
        k_first = None

        for tc in t_eval:
            if tableau.fsal:
                y_next, error, k_first = self.runge_kutta_step(fun, y[n_step], tc, dt, tableau, k_first)
            else:
                y_next, error = self.runge_kutta_step(fun, y[n_step], tc, dt, tableau)
            n_step += 1 
            y = y.at[n_step].set(y_next)
            global_error += error
            errors.append(error)
        return y, global_error, errors

    def runge_kutta_step(self,
        f,
        y: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        tableau: ButcherTableau,
        k_first: Optional[jnp.ndarray] = None,
    ):
        """Perform a single step of a Runge--Kutta method.

        Args:
            f: The vector field function.
            y: The current state.
            t: The current time.
            dt: The step size.
            tableau: The Butcher tableau of the Runge--Kutta method.
            k_first: First stage derivative (optional, used in FSAL).

        Returns:
            - Next state after the step.
            - Error estimate for adaptive methods (optional).
            - Last stage derivative (for FSAL-enabled methods).
        """
        num_stages = tableau.num_stages
        k = jnp.zeros((num_stages,) + y.shape, dtype=y.dtype) # shape (num_stages, y.shape)
        
        # explicit Runge-Kutta methods
        if not tableau.implicit: # a_diagonal is None
            if tableau.fsal and k_first is not None: # FSAL (First Same As Last) optimization
                k = k.at[0].set(k_first) # k1 = k_first(n_step) = k_last(n_step-1)
            else: 
                k = k.at[0].set(f(y, t+ dt * tableau.c1)) # k1 = f(y, t + c1 * dt)

            for i in range(tableau.num_stages-1):
                ti = t + tableau.c[i] * dt
                yi = y + dt * jnp.tensordot(tableau.a_lower[i], k[:i+1], axes=1)
                #yi = y + dt * jnp.dot(tableau.a_lower[i], k[:i+1])
                k = k.at[i+1].set(f(yi, ti))    
            y_next =  y + dt * jnp.tensordot(tableau.b_sol, k, axes=1)
            error = jnp.tensordot(tableau.b_error, k, axes=1)

            if tableau.fsal:
                return y_next, error, k[-1] # return the last stage for FSAL
            else:
                return y_next, error 
            
        # implicit Runge-Kutta methods
        else:
            raise NotImplementedError("Implicit Runge-Kutta methods are not supported yet.") 
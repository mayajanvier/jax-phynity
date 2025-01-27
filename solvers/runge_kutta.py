import jax
import jax.numpy as jnp
import numpy as np 
from dataclasses import dataclass, field # dataclass is a decorator that is used to create classes with attributes, __init__ method, __repr__ method, and __eq__ method all in one go.
from typing import Optional

# diffrax Butcher tableau class 
@dataclass(frozen=True)
class ButcherTableau:
    """The Butcher tableau for an explicit or diagonal Runge--Kutta method."""

    # Explicit RK methods
    c: np.ndarray
    b_sol: np.ndarray
    b_error: np.ndarray
    a_lower: tuple[np.ndarray, ...]

    # Implicit RK methods
    a_diagonal: Optional[np.ndarray] = None
    a_predictor: Optional[tuple[np.ndarray, ...]] = None 
    c1: float = 0.0

    # Properties implied by the above tableaus, e.g. used to define fast-paths.
    # field is used to define default values for the attributes of the class
    ssal: bool = field(init=False) # (Same As Last): Indicates whether the final solution equals the last stage.
    fsal: bool = field(init=False) # (First Same As Last): Indicates whether the first stage is equivalent to the last stage. 
    implicit: bool = field(init=False)
    num_stages: int = field(init=False) # number of "k_i" stages in the Runge-Kutta method

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
            assert np.allclose(sum(a_i) + diagonal, c_i)
        assert np.allclose(sum(self.b_sol), 1.0)
        assert np.allclose(sum(self.b_error), 0.0)

        if self.a_diagonal is None:
            assert self.a_predictor is None
        else:
            assert self.a_predictor is not None
            assert self.a_diagonal.ndim == 1
            assert self.c.shape[0] + 1 == self.a_diagonal.shape[0]
            assert len(self.a_lower) == len(self.a_predictor)
            for a_lower_i, a_predictor_i in zip(self.a_lower, self.a_predictor):
                assert a_lower_i.shape == a_predictor_i.shape
                assert np.allclose(sum(a_predictor_i), 1.0)

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
            yi = y + dt * jnp.dot(tableau.a_lower[i], k[:i+1])
            k = k.at[i+1].set(f(yi, ti))    
        y_next =  y + dt * jnp.dot(tableau.b_sol, k)
        error = jnp.dot(tableau.b_error, k)

        if tableau.fsal:
            return y_next, error, k[-1] # return the last stage for FSAL
        else:
            return y_next, error 
        
    # implicit Runge-Kutta methods
    else:
        raise NotImplementedError("Implicit Runge-Kutta methods are not supported yet.")  

### Butcher tableaux for Runge-Kutta methods

dopri5_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 5]),
        np.array([3 / 40, 9 / 40]),
        np.array([44 / 45, -56 / 15, 32 / 9]),
        np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
        np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
        np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
    ),
    # 5th order weights 
    b_sol=np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]), 
    # b_error embeds the 4th order error estimate: bi(5) - bi(4)
    b_error=np.array(
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
    c=np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]),
)


dopri8_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 18]),
        np.array([1 / 48, 1 / 16]),
        np.array([1 / 32, 0, 3 / 32]),
        np.array([5 / 16, 0, -75 / 64, 75 / 64]),
        np.array([3 / 80, 0, 0, 3 / 16, 3 / 20]),
        np.array(
            [
                29443841 / 614563906,
                0,
                0,
                77736538 / 692538347,
                -28693883 / 1125000000,
                23124283 / 1800000000,
            ]
        ),
        np.array(
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
        np.array(
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
        np.array(
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
        np.array(
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
        np.array(
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
        np.array(
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
        np.array(
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
    b_sol=np.array(
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
    b_error=np.array(
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
    c=np.array(
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
    "DOPRI5": dopri5_tableau,
    "DOPRI8": dopri8_tableau,
}

### Runge-Kutta methods

# fixed step size
def RK_solver_fixed(fun, t_span, y0, t_eval, method, rtol=1e-10, n_step_max=1000):
    """Solve an initial value problem using the Dormand--Prince 5 method."""
    # initialize
    tableau = RK_tableaux[method]
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
            y_next, error, k_first = runge_kutta_step(fun, y[n_step], tc, dt, tableau, k_first)
        else:
            y_next, error = runge_kutta_step(fun, y[n_step], tc, dt, tableau)
        n_step += 1 
        y = y.at[n_step].set(y_next)
        global_error += error
        errors.append(error)
    return y, global_error, errors


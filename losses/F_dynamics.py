import jax
import jax.numpy as jnp
from utils import fft_diff_jax, fft_diff_jax_fast
import numpy as np


### CONSTANTS
beta = 8/3
sigma = 10.
rho = 28.
I = jnp.array([1.6, 1.0, 2 / 3]) # rigid body
m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81 # double pendulum
L_ks = 64 # KS domain length
VISC = 1e-3 # viscosity for Navier-Stokes
L_ns = 64 # domain size for Navier-Stokes
tt = jnp.linspace(0, 1, L_ns + 1)[0:-1]
X, Y = jnp.meshgrid(tt, tt)
f_ns = 0.1 * (jnp.sin(2 * jnp.pi * (X + Y)) + jnp.cos(2 * jnp.pi * (X + Y))) # forcing for Navier-Stokes

def F_pendulum(x):
    return jnp.array([x[1], -((2*jnp.pi/12)**2)* jnp.sin(x[0]) - 0.2*x[1]])

def F_lorenz(s):
    x, y, z = s
    dxdt = sigma * (y - x)
    dydt = rho * x - y - x * z
    dzdt = x * y - beta * z
    return jnp.array([dxdt, dydt, dzdt])

def F_twobody(s):
    x, y, x_prime, y_prime = s
    x_second = -x / (x**2 + y**2)**(3/2)
    y_second = -y / (x**2 + y**2)**(3/2)
    return jnp.array([x_prime, y_prime, x_second, y_second])

def F_twobody_forcing(s): 
    x, y, x_prime, y_prime, t = s
    forcing_magnitude = 0.01 * jnp.sin(2 * jnp.pi * t / 10)
    r = (x**2 + y**2) # Fixed direction along x-axis
    x_second = -x / r**(3/2) - forcing_magnitude * y/r
    y_second = -y / r**(3/2) + forcing_magnitude * x/r
    return jnp.array([x_prime, y_prime, x_second, y_second, 1.0])

def F_rigidbody(s): 
    y1, y2, y3 = s
    mat = jnp.array([
        [0, -y3, y2],
        [y3, 0, -y1],
        [-y2, y1, 0]])
    vect = jnp.array([y1/I[0], y2/I[1], y3/I[2]])
    dydt = mat @ vect
    return dydt

### double pendulum
def dw1(s):
    theta1, theta2, w1, w2 = s
    return (
        -g * (2*m1 + m2) * jnp.sin(theta1) - m2 * g * jnp.sin(theta1 - 2*theta2) -
        2* jnp.sin(theta1-theta2) * m2 * (w2**2 * l2 + w1**2 * l1 * jnp.cos(theta1-theta2))
    ) /  (l1 * (2*m1 + m2 - m2 * jnp.cos(2*(theta1-theta2))))
    
def dw2(s):
    theta1, theta2, w1, w2 = s
    return (
        2 * jnp.sin(theta1 - theta2) * (
            w1**2 * l1 * (m1 + m2) +
            g * (m1 + m2) * jnp.cos(theta1) +
            w2**2 * l2 * m2 * jnp.cos(theta1 - theta2)
        )
    ) / (l2 * (2*m1 + m2 - m2 * jnp.cos(2*(theta1 - theta2))))

def F_doublependulum(s): 
    """Compute derivatives for double pendulum."""
    _, _, w1, w2 = s
    dtheta1_dt = w1
    dtheta2_dt = w2
    dw1_dt = dw1(s)
    dw2_dt = dw2(s)
    return jnp.array([dtheta1_dt, dtheta2_dt, dw1_dt, dw2_dt])


### KS 1D uses pseudospectral reconstruction in Brandsetter generated data 
def F_KS_fast(u):
    # Compute the x derivatives using the pseudo-spectral method.
    ux = fft_diff_jax_fast(u, period=L_ks)
    uxx = fft_diff_jax_fast(u, period=L_ks, order=2)
    uxxxx = fft_diff_jax_fast(u, period=L_ks, order=4)
    # Compute du/dt.
    dudt = - u*ux - uxx - uxxxx
    return dudt

# def F_BG(u):
#     f = flux(u)
#     return -dx_inv * (f[1:nx+1] - f[0:nx])

def F_navier_stokes_2d(w0, f=f_ns, visc=VISC):
    # Grid size - must be power of 2
    N = w0.shape[-1]
    # Maximum frequency
    k_max = jnp.floor(N / 2.0)
    # Initial vorticity to Fourier space
    w_h = jnp.fft.fftn(w0, (N, N))
    # Forcing to Fourier space
    f_h = jnp.fft.fftn(f, (N, N))
    # If same forcing for the whole batch
    if f_h.ndim < w_h.ndim:
        f_h = jnp.expand_dims(f_h, 0)
    # Wavenumbers in y-direction
    k_max = N // 2
    k_range = jnp.concatenate([
        jnp.arange(0, k_max),
        jnp.arange(-k_max, 0)
    ])

    k_y = jnp.tile(k_range[None, :], (N, 1))
    k_x = k_y.T
    
    # Negative Laplacian in Fourier space
    lap = 4 * (jnp.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap = lap.at[0,0].set(1.0)
    # Dealiasing mask
    dealias = jnp.expand_dims(
        jnp.logical_and(
            jnp.abs(k_y) <= (2.0 / 3.0) * k_max,
            jnp.abs(k_x) <= (2.0 / 3.0) * k_max
        ),
        axis=0
    )
    # Stream function in Fourier space: solve Poisson equation
    psi_h = w_h / lap
    # Velocity field in x-direction = psi_y
    q = (1j * 2 * jnp.pi * k_y) * psi_h
    q = jnp.fft.ifftn(q, (N, N))
    # Velocity field in y-direction = -psi_x
    v = (1j * 2 * jnp.pi * k_x) * psi_h
    v = jnp.fft.ifftn(v, (N, N))
    # Partial x of vorticity
    w_x = (1j * 2 * jnp.pi * k_x) * w_h
    w_x = jnp.fft.ifftn(w_x, (N, N))
    # Partial y of vorticity
    w_y = (1j * 2 * jnp.pi * k_y) * w_h
    w_y = jnp.fft.ifftn(w_y, (N, N))
    # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
    F_h = jnp.fft.fftn(q * w_x + v * w_y, (N, N))
    # Dealias
    F_h = dealias * F_h
    # Rhs
    rhs_h = - F_h + f_h - visc * lap * w_h
    rhs = jnp.fft.ifftn(rhs_h, (N, N))
    return jnp.real(rhs[0])


F_REGISTRY = {
    "lorenz": F_lorenz,
    "twobody": F_twobody,
    "twobody_forcing": F_twobody_forcing,
    "pendulum": F_pendulum,
    "doublependulum": F_doublependulum,
    "rigidbody": F_rigidbody,
    "ks": F_KS_fast,
    "ns_incomp": F_navier_stokes_2d,
    }
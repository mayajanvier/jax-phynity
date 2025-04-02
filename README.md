# jax-phynity

Implementation of _Augmenting Physical Models with Deep Networks for Complex Dynamics Forecasting_ (Yin et al., 2021) (APHYNITY) damped pendulum in JAX.

## Organisation of repository
- `Reports`: reports of implementation and tests (`Issues.ipynb`), slides of presentation to ANGE team (01/04/2025)
- `datasets`: pendulum class for generating data (derived in jax from APHYNITY)
- `solvers`: custom Runge-Kutta solvers based on Butcher Tableaux (derived from diffrax)
- `utils.py`: loggers, init weights of neural network in jax
- `networks.py`: physical and MLP models 
- `forecasters.py`: class to combine models for hybridation and trajectory prediction (derived from APHYNITY, simplified)
- `train_jaxphynity.py`: training routines for sanity checks, error scheme and Lipschitz experiments
- `inference.py`: inference for all experiments

Notebooks:
- `visualize_results`: intermediate trials, final sanity checks and final results and figures (last part is the most interesting)
- `error_scheme`: error scheme experiment visualisation
- `lipschitz`: Lipschitz experiments visualisation
- `tipping_points`: robustness of models for out-of-domain initial conditions experiments 

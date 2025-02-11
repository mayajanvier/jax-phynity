# Issues and how I solved them

Sanity checks experiments 
## $\alpha$ and $\omega_0$ are not updated during training
RK4 for generation and integration during training. 
### H1: Bad gestion with auxiliary in loss_trajectory: 
Test: On effectue le calcul dans loss_fn directement: Non

### H2: is it because of y0 instead of y ? 
Il faut peut être qu'on différencie aussi par rapport à y0 ? 
Ou alors donner y et prendre y0 dans le forecaster ? : on met y c'est plus propre mais ne devrait pas changer

### H3: On travaille avec "loss_trajectory" uniquement 
On a pas "jit" le solver, et Butcher tableau était en numpy. Solver était en "callable" donc static et pas trainable -> faire une classe Integrator(eqx.Module). 
Je sépare RK_solver pour génération de données et celui en méthode dans Integrator pour débuguer, on verra si on peut unir les 2 ensuite.

CCL: en fait les vrais modèles entraînés sont ceux dans derivative_estimator. On doit donc regarder les paramètres de derivative_estimator.model_phy, et pour la loss Fa derivative_estimator.model_aug ! 

## SC1: Can we recover true $\alpha$ and $\omega_0$ ? 
- GT: Complete + RK4
- Train: ParamODE($\omega_0$,$\alpha$) + RK4

Criteria:
- Loss_traj = 0
- error_param = 0 



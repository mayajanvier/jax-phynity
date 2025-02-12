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
rmq: le fait de déclarer ButcherTableau en argument dans la classe crée une erreur avec eqx.filter_jit

CCL: en fait les vrais modèles entraînés sont ceux dans derivative_estimator. On doit donc regarder les paramètres de derivative_estimator.model_phy, et pour la loss Fa derivative_estimator.model_aug ! 

## SC1: Can we recover true $\alpha$ and $\omega_0$ ? 
- GT: Complete | RK4
- Train: ParamODE($\omega_0$,$\alpha$) | RK4 | loss=loss_traj

Criteria:
- Loss_traj = 0
- error_param = 0 

Trials
- First try 20 epochs: s4qjf2yv, complete_physics1
- Run plus longtemps pour atteindre 0 totalement: 09ki0cct, complete_physics4: omega0 optimisé d'abord, puis augmente pendant optimisation de alpha0, et ensuite redimimue. 

## SC2: Can we correctly complete the frictionless pendulum (ParamODE($\omega_0$))?
- GT: Complete | RK4
- Train: ParamODE($\omega_0$)+ aug | RK4 | loss = loss_traj + $\lambda$ lossFa

Criteria:
- Loss_traj = 0

Trials:
- 20 epochs: incomplet_aug3 - eqm0ku80: ressemble pas mal à alpha=0_complete_aug2 qui a l'equation complète mais avec alpha qui commence à zero: normal, les paramètres évoluent doucement.


## SC3: Neural ODE
- GT: Complete | RK4
- Train: aug | RK4 | loss=loss_traj

Criteria:
- loss_traj tend vers 0

Trials:
- 50 epochs - sc2nap9g - none_aug5: very fast run, goes to zero
- 150 epochs - idwrvp49 - none_aug6: unstable 

model_phy is the slow part -> not entirely in eqx, modify

## SC surprise: o4jfm4i8
- GT: Complete | RK4
- Train: ParamODE($\omega_0$,$\alpha$) + aug | RK4 et $alpha_0=0$

Entrainement en mode APH mais avec mauvais départ: alpha=0_complete_aug2




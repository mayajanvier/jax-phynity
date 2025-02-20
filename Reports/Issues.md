# Issues and how I solved them

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

# Sanity checks
## SC1: Can we recover true $\alpha$ and $\omega_0$ ? 
- GT: Complete | RK4
- Train: ParamODE($\omega_0$,$\alpha$) | RK4 | loss=loss_traj | $\tau_1$=1e-3, $\tau_2$=100, $\lambda_0$=1000, $N_{iter}$=5

Criteria:
- Loss_traj = 0
- error_param = 0 

Trials 03/02-12/02 (`data/sanity_checks`)
- First try 20 epochs: s4qjf2yv, complete_physics1
- Run plus longtemps pour atteindre 0 totalement: 09ki0cct, complete_physics4: omega0 optimisé d'abord, puis augmente pendant optimisation de alpha0, et ensuite redimimue. 

Trials 17/02 (`data/sanity_checks2`)
- complete_physics_10_p6zoywbr: n_iter après batch (APH), run 2m44s pour 150 epochs et retrouve bien les paramètres // pas bonne indentation des boucles 
- complete_physics_13_9p67kp2y: APH corrected, 2min45s (150 epochs)
- complete_physics_14_d7l5vx50: 200 epochs, 4min30s

JUSQUE LÀ ERREUR $\tau_1$=1e-3. Recommence avec $\tau_1$=1 (papier)
- complete_physics_18_1d0dmkyn: lr too high, unstable -> $\tau_1$=1e-3 better

## SC2.1: Can we correctly complete the frictionless pendulum (ParamODE($\omega_0$)) like APHYNITY?
- GT: Complete | RK4
- Train: ParamODE($\omega_0$)+ aug | RK4 | loss = $\lambda$ loss_traj + lossFa | $\tau_1$=1e-3, $\tau_2$=10, $\lambda_0$=1, $N_{iter}$=5

Criteria:
- Loss_traj = 0
- NEW 20/02: Fa devrait représenter les frottements ie ne varier que selon $p$ et en pente $-\alpha$ 

Trials 03/02-12/02 (`data/sanity_checks`)
- 20 epochs: incomplet_aug3 - eqm0ku80: ressemble pas mal à alpha=0_complete_aug2 qui a l'equation complète mais avec alpha qui commence à zero: normal, les paramètres évoluent doucement.

Trials 17/02-19/02 (`data/sanity_checks2`) (loss_traj only)
- incomplete_aug_12_xsolpxby: 150 epochs, mauvaise indentation des boucles. loss_traj stagne et Fa ne réaugmente pas 
- incomplete_aug_17_m9bz4n8c: 200 epochs, 6min46s. Loss_traj starts to decrease but very slowly once omega0 is very close to truth / Error loss (lambda devant Fa au lieu de loss_traj)
- incomplete_aug_19_korcnphf: 200 epochs, 11min, loss pas assez basse. Si on regarde les données train/test: ceux qui ont loss_traj élevée ont peu d'exemples d'entraînement -> manque de généralisation . 400 epochs ne change rien

Trials 20/02 (`data/sanity_checks2`) (loss_traj and Fa(p) vs p)
- incomplete_aug_19_korcnphf: quel Fa(p) ? 


## SC2.2: Can we correctly complete the frictionless pendulum (ParamODE($\omega_0$)) without Fa constrain ?
- GT: Complete | RK4
- Train: ParamODE($\omega_0$)+ aug | RK4 | loss =loss_traj  | $\tau_1$=1e-3, $\tau_2$=10, $\lambda_0$=1, $N_{iter}$=5

Criteria:
- Loss_traj = 0
- Fa devrait représenter les frottements ie ne varier que selon $p$ et en pente $-\alpha$ 

## SC3: Neural ODE
- GT: Complete | RK4
- Train: aug | RK4 | loss=loss_traj | $\tau_1$=1e-3, $\tau_2$=0, $\lambda_0$=0, $N_{iter}$=5

Criteria:
- loss_traj tend vers 0

Trials 03/02-12/02 (`data/sanity_checks`)
- 50 epochs - sc2nap9g - none_aug5: very fast run, goes to zero
- 150 epochs - idwrvp49 - none_aug6: unstable 

model_phy is the slow part -> not entirely in eqx, modify

Trials 17/02-19/02 (`data/sanity_checks2`)
- none_aug_1_9i0h01av: very fast, now reproducible, loss_traj goes to zero 
- none_aug_11_kstaf7ub: n_iter APH, run fast et atteint zero // pas bonne indentation des boucles 
- none_aug_15_k2yhglal: bon ordre des boucles, 200 epochs, tend vers 0 mais explose step 126
- none_aug_16_1m63z5th: for reproducibility, same as non_aug_15: same behaviour. Run duration: 1min52s

### Loss explosion investigation
Goal: visualize 1. $\frac{\partial Fa}{\partial \theta}$, and  2. $\frac{\partial Fa}{\partial q}$,  $\frac{\partial Fa}{\partial p}$ to understand source of explosion (if not bug):
- If 1. is big just big step of Adam. 
- If 2. is small, our system is chaotic and small perturbations can induce large changes. 

Trials 20/02 (`data/sanity_checks2`)


# Error scheme experiments

## ES1: SC1 + change of scheme (// Norbert)
- GT: ParamODE($\omega_0$,$\alpha$)| RK4 | **$\Delta t_{num}=0$**
- Train: ParamODE($\omega_0$,$\alpha$) | **RK2** | loss=loss_traj | $\tau_1$=1e-3, $\tau_2$=100, $\lambda_0$=1000, $N_{iter}$=5 **with different $\Delta t$**

Criteria:
- error_param tend vers 0 ?
- Loss_traj_test($\Delta t$) vs $\Delta t$ : should see order $\Delta t^{2}$ (RK4 - RK2)


## ES2: ES1 augmented, balance $\Delta t$ and NN quality
- GT: Complete | RK4 | **$\Delta t_{num}=0$**
- Train: ParamODE($\omega_0$,$\alpha$) **+ aug** | **RK2** | loss=$\lambda$ loss_traj + loss_Fa | $\tau_1$=1e-3, $\tau_2$=100, $\lambda_0$=1000, $N_{iter}$=5 **with different $\Delta t$**

Criteria:
- error_param tend vers 0 ?
- Loss_traj_test($\Delta t$) vs $\Delta t$ : should see order $\Delta t^{2}$ + $\epsilon_{NN}$ (mais inégalité)
- Compute some $\Delta t^*$

## Convergence: Acceleration de Richardson
Avec ces modèles $\Delta t, \Delta t/2$... entrainés, on peut les combiner et observer accélération de la convergence 

## Convergence 2: change $\Delta t$ at inference
- $\Delta t \leq \Delta t_{train}$ (ce qui nous intéresse), RK2 train, RK4 (dt=0) data ? 
- on peut regarder pour nos modèles entraînés avec différents dt:
    - ParamODE($\omega_0$,$\alpha$) ? 
    - ParamODE($\omega_0$,$\alpha$) + aug 

Criteria:
- voir si entrainer en RK2(dt) puis réduire dt/2 permet de meilleurs resultats que inference avec dt ?  d'aussi bon résultats qu'entraîner en RK4(dt) ?  


# Deprecated
## SC surprise: o4jfm4i8 (03/02)
- GT: Complete | RK4
- Train: ParamODE($\omega_0$,$\alpha$) + aug | RK4 et $alpha_0=0$

Entrainement en mode APH mais avec mauvais départ: alpha=0_complete_aug2

Mauvaise boucle, mauvais tout... 
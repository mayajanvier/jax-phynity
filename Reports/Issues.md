# Records for jax-phynity project
Maya Janvier, 01/25-03/25

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

## loss_fn is rejitted each epoch
- loss_Fa and loss_traj are not so not here
- not the fault of y_pred (if loss_fn=loss_traj no problem)
- if lambda is argument: problem ! but when not here, it's okay 
--> lambda is traced somehow and forces recompilation of jit 
Pas réussi à forcer lambda en statique, mais on a peut être pas besoin de jiter loss_fn si on met en partial les autres arguments (min_op etc) et si loss_Fa et loss_traj sont déjà jitées.
- incomplete_aug_21_xteapxnx: 150 epochs, sans loss_fn jit, duration=2min, mais instabilités différentes d'avant...

- est ce que c'est les nouveaux arguments dans la fonction (model_aug etc): non
- origine: 
    - decorateurs jit sur PDE et forecaster pour incomplete_aug -> si je les remets je retrouve mes courbes d'avant
    - none_aug instable: jit sur PDE et forecaster
    - none_aug sans ces décorateurs: instable mais pas au même endroit 
    - stable avec mauvaise boucle (n_iter en 2e)
-> pas mettre de filter jit dans une classe, pas comme ça dans les exemples equinox
- none aug sans décorateurs sans jit de loss_fn: toujours pareil, plus rapide
-> on jit Loss_Fa et Loss_traj uniquement, loss_fn n'est qu'un encapsulement des 2. eqx.partial pour min_op, model_aug_option et model_phy_option accélère aussi la compilation


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

Trials 24/02
-  complete_physics_26_f6dwvliv: run for 400 epochs with good jit compilation, 1min30 ! error relative 7e-5

### SC1: VALID -> CLOSED

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
- incomplete_aug_25_wixz2qcg: jit corrigé, 4min30s
- incomplete_aug_30_q5w7dej1: jit remis avec lambda en array + float 0 fort networks, duration= 4min47s

Trial 24/02
- incomplete_aug_32_346dtoyv: test avec eps=1e-5 dans l2_normalized, on atteint  1.506 de test loss
- incomplete_aug_33_2gh57448: min_op=l2, on atteint 7.6e-01 de test loss et on se rapproche beaucoup plus d'oméga 0, 16% d'erreur. Mais entrainement toujours très instable 
- incomplete_aug_34_lx8y2wmb: min_op=l2, lambda0 =10, tau2=100, loss test = 4.318e-03 !!!, oméga0=0.256 (vs 0.27 truth) -> suffit de tuner un peu pour s'en sortir. On récupère bien les dynamiques en -alpha pour Fa (en réalité très dépendant de notre approximation de omega0)

### SC2 VALID -> CLOSED

## SC2.2: Can we correctly complete the frictionless pendulum (ParamODE($\omega_0$)) without Fa constrain ?
- GT: Complete | RK4
- Train: ParamODE($\omega_0$)+ aug | RK4 | loss =loss_traj  | $\tau_1$=1e-3, $\tau_2$=10, $\lambda_0$=1, $N_{iter}$=5

Criteria:
- Loss_traj = 0
- Fa devrait représenter les frottements ie ne varier que selon $p$ et en pente $-\alpha$

Trials 24/04
- incomplete_no_Fa_aug_28_riymbky6:  loss_traj va bien vers 0 ! loss_Fa reste autour de 1e-3. Mais on a une séparation sous optimale des modèles (on ne retrouve pas oméga 0 (0.2 au lieu de 0.27) et donc Fa ne retrouve pas exactement alpha). test loss un peu moins meilleure que incomplete_aug, peut être avec un lr plus petit on pourrait mieux approximer le omega 

### SC2.2 VALID -> CLOSED

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

Trials 24/02 (`data/sanity_checks2`)
- none_aug_24_of0iieil: 400 epochs run with good jit, 4min15s

## SC4: Neural ODE + loss Fa

Trials 28/03 (`data/lipschitz`)
- none_Fa_aug_31_92x54p4c: $\tau_1=1e-3, \tau_2=10, \lambda_0=100$ (params that works for my SC2): logMSE = -3.573 +- 0.86
- none_Fa_aug_32_583z1zwy: params paper (explodes)
- none_Fa_aug_33 : $\tau_1=1e-3, \tau_2=1, \lambda_0=10$ (mix): logMSE= -3.129 +- 0.84

In the paper: -2.84 +- 0.7


# Complete dynamics experiments (03/25)
The final experiments that I used for my plots. 

**Setup APHYNITY ($N_{iter}$=5, T=20s):**
- GT: Complete | RK4
- NeuralODE-traj: $NN_\theta$ | RK4 | loss=loss_traj | $\tau_1$=1e-3 
- NeuralODE-aph: $NN_\theta$ | RK4 | loss=loss_traj + Fa | $\tau_1$=1e-3, $\tau_2$=1, $\lambda_0$=10 and $\tau_2$=10, $\lambda_0$=100
- ParamODE($\omega_0$)+$NN_\theta$-traj: incomplete+aug | RK4 | loss =loss_traj  | $\tau_1$=1e-3 
- ParamODE($\omega_0$)+$NN_\theta$-aph: incomplete+aug | RK4 | loss =loss_traj + Fa | $\tau_1$=1e-3, $\tau_2$=10, $\lambda_0$=1


**Results in `data/lipschitz`:**  logMSE ($\downarrow$)
- NeuralODE-traj: none_aug_19_yt8wqw57 $\rightarrow$ -4.2633 +-0.792
- NeuralODE-aph: 
    - none_Fa_aug_33_e6ynty14, $\tau_2$=1, $\lambda_0$=10 $\rightarrow$ logMSE= -3.129 +- 0.84
    - none_Fa_aug_31_92x54p4c, $\tau_2$=10, $\lambda_0$=100 $\rightarrow$ logMSE = -3.573 +- 0.86
    - paper: -2.84 +- 0.7

- ParamODE($\omega_0$)+$NN_\theta$-traj: incomplete_no_Fa_aug_20_ps3fe9eb $\rightarrow$ -7.055 +- 0.6767

- ParamODE($\omega_0$)+$NN_\theta$-aph: 
    - incomplete_aug_20_srefavep $\rightarrow$ -7.80 +-0.7098
    - paper: -7.86 +- 0.6

# Lipschitz constant investigation (03/25)
Same setups than before, with different training durations T=[5,10,20,40]s, one 60s for none_aug: explosion of loss to NaN  

Data in `data/lipschitz`: `{name_exp}_{duration}_{id_wb}`

Figures in `lipschitz.ipynb`





# Error scheme experiments

## ES1: SC1 + change of scheme (// Norbert) (`error_scheme`, `error_scheme2`)
- GT: ParamODE($\omega_0$,$\alpha$)| RK4 | **$\Delta t_{num}=0$**
- Train: ParamODE($\omega_0$,$\alpha$) | **RK2** | loss=loss_traj | $\tau_1$=1e-3, $\tau_2$=100, $\lambda_0$=1000, $N_{iter}$=5 **with different $\Delta t$**

Criteria:
- error_param tend vers 0 ?
- Loss_traj_test($\Delta t$) vs $\Delta t$ : should see order $\Delta t^{2}$ (RK2)
- |theta - theta_pred| ordre 2 aussi

`error_scheme`
- dt_factors = [2,5,8,10,16], dt_num=0.05
- on évalue tous les points disponibles pour le facteur (::dt_factor)
- loss est MSE donc on observe ordre $2^2=4$ (!!)
- ordre 2 pour l'erreur sur les paramètres

`error_scheme2`
- dt_factors = [2,8,16], dt_num=0.05
- on évalue aux mêmes points (::16)
- loss MSE ordre $2^2=4$ + verif: model fait bien RK2
- ordre 2 pour l'erreur sur les paramètres

### ES1: VALID -> CLOSED


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
- m'inspirer des setups (train et test) de DINo et space and time continuous pde (Wolf) ? pour généralisation en fait 


# Deprecated
## SC surprise: o4jfm4i8 (03/02)
- GT: Complete | RK4
- Train: ParamODE($\omega_0$,$\alpha$) + aug | RK4 et $\alpha_0=0$

Entrainement en mode APH mais avec mauvais départ: alpha=0_complete_aug2

Mauvaise boucle, mauvais tout... 
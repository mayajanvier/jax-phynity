from .pendulum import DampedPendulum, DoublePendulum
from .lorenz import Lorenz
from .twobody import TwoBody, TwoBodyForcing
from .wave import Wave
from .rigidbody import RigidBody
from .ks import KS
from .burgers import Burgers
from .ns_incomp import NavierStokes
import torch
import numpy as np
import random

# fix torch seed for reproducibility
#torch.manual_seed(1)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

DATASET_REGISTRY = {
    "doublependulum": DoublePendulum,
    "lorenz": Lorenz,
    "burgers": Burgers,
    "twobody": TwoBody,
    "twobody_forcing": TwoBodyForcing,
    "rigidbody": RigidBody,
    "ks": KS,
    "ns_incomp": NavierStokes
    }
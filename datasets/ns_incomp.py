import os
import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
import h5py
import torch 
import math

# Enable 64-bit precision in JAX
#jax.config.update("jax_enable_x64", True)

MAX = np.iinfo(np.int32).max # maximum int value

def get_mgrid(sidelen, vmin=-1, vmax=1, dim=2):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    if isinstance(sidelen, int):
        tensors = tuple(dim * [torch.linspace(vmin, vmax, steps=sidelen)])
    elif isinstance(sidelen, (list, tuple)):
        if isinstance(vmin, (list, tuple)) and isinstance(vmax, (list, tuple)):
            tensors = tuple([torch.linspace(mi, ma, steps=l) for mi, ma, l in zip(vmin, vmax, sidelen)])
        else:
            tensors = tuple([torch.linspace(vmin, vmax, steps=l) for l in sidelen])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid


def get_mgrid_from_tensors(tensors):
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None):
        self.dim = dim
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1), torch.arange(start=-k_max, end=0, step=1)), 0)
            self.sqrt_eig = size * math.sqrt(2.0) * sigma * ((4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.
        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, 1)
            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers
            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, size, 1)
            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)
            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self):
        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        return u


class NavierStokesDataset:
    def __init__(self, dt, num_steps_max, num_steps_rollout, path, split, nb_traj, size, integration_method='RK4', *args, **kwargs):
        super().__init__()
        self.size = size
        self.sampler = GaussianRF(2, self.size, alpha=2.5, tau=7)
        self.dt_num = 1e-3
        self.dt = dt
        self.visc = 1e-3
        self.coords = get_mgrid(self.size, vmin=0, vmax=0.5, dim=2)
        self.coord_dim = self.coords.shape[-1]

        tt = torch.linspace(0, 1, self.size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        self.f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

        self.num_steps_max = num_steps_max
        self.num_steps_rollout = num_steps_rollout
        self.nb_traj = nb_traj
        self.split = split
        self.path = path

        # Full trajectories are saved to a single .npy file
        self.data_path = f"datasets/ns_incomp_full_{split}.npy"
        self.states = self._load_dataset()

    def navier_stokes_2d(self, w0, f, visc, T, delta_t, record_steps):
        # Grid size - must be power of 2
        N = w0.size()[-1]
        # Maximum frequency
        k_max = math.floor(N / 2.0)
        # Number of steps to final time
        steps = math.ceil(T / delta_t)
        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, (N, N))
        # Forcing to Fourier space
        f_h = torch.fft.fftn(f, (N, N))
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)
        # Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1),
                         torch.arange(start=-k_max, end=0, step=1)), 0).repeat(N, 1)
        # Wavenumbers in x-direction
        k_x = k_y.transpose(0, 1)
        # Negative Laplacian in Fourier space
        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0
        # Dealiasing mask
        dealias = torch.unsqueeze(
            torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, 1, dtype=torch.float)
        sol_t = torch.zeros(record_steps)
        # Record counter
        c = 0
        # Physical time
        t = 0.0
        for j in range(steps):
            if j % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, (N, N))
                # Record solution and time
                sol[..., c, 0] = w.real
                # sol[...,c,1] = w.imag
                sol_t[c] = t
                c += 1
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h.clone()
            psi_h = psi_h / lap
            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * temp
            q = torch.fft.ifftn(q, (N, N))
            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * temp
            v = torch.fft.ifftn(v, (N, N))
            # Partial x of vorticity
            w_x = w_h.clone()
            temp = w_x.real.clone()
            w_x.real = -2 * math.pi * k_x * w_x.imag
            w_x.imag = 2 * math.pi * k_x * temp
            w_x = torch.fft.ifftn(w_x, (N, N))
            # Partial y of vorticity
            w_y = w_h.clone()
            temp = w_y.real.clone()
            w_y.real = -2 * math.pi * k_y * w_y.imag
            w_y.imag = 2 * math.pi * k_y * temp
            w_y = torch.fft.ifftn(w_y, (N, N))
            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y, (N, N))
            # Dealias
            F_h = dealias * F_h
            # Cranck-Nicholson update
            w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / \
                  (1.0 + 0.5 * delta_t * visc * lap)
            # Update real time (used only for recording)
            t += delta_t

        return sol, sol_t


    def _get_init_cond(self, index):
        if self.split == "train":
            torch.manual_seed(index)
        elif self.split == "val":
            torch.manual_seed(MAX // 2 - index)
        else:  # test
            torch.manual_seed(MAX - index)

        w0 = self.sampler.sample() # (size, size)
        state, _ = self.navier_stokes_2d(
            w0.unsqueeze(0), # add batch dimension 
            f=self.f,
            visc=self.visc,
            T=30,
            delta_t=self.dt_num,
            record_steps=20
            )
        
        init_cond = state[:, :, :, -1, 0] # cut off the transient part and only keep the last frame as the initial condition for training
        return init_cond

    def _generate_and_save_dataset(self):
        """Generate full trajectories and save them in one .npy file."""
        all_states = []
        all_t = None
        num_steps = max(self.num_steps_rollout, self.num_steps_max) 
        for idx in range(self.nb_traj):
            with torch.no_grad():
                w0 = self._get_init_cond(idx)
                states, _ = self.navier_stokes_2d(w0, f=self.f, visc=self.visc,
                                                T= (num_steps *1)+1, # truc chelou de dt=1 mais dt=1e-3 dans le code ? 
                                                 delta_t=self.dt_num, record_steps=num_steps+1)
                
            states = rearrange(states[0,:,:,:,0], 's1 s2 T -> T s1 s2') #.permute(0, 4, 3, 1, 2)
            all_states.append(np.array(states))

        all_states = np.stack(all_states)  # shape (nb_traj, T, s1, s2)
        print(all_states.shape)
        np.save(self.data_path, dict(states=all_states, t=all_t))
        print(f"Saved {self.nb_traj} trajectories to {self.data_path}")

    
    def __len__(self):
        return len(self.states)

    def _chunk_and_save_dataset(self):
        """Get full trajectories and save new dataset of chunked trajectories of num_steps_rollout steps."""
        # Load full trajectories
        if not os.path.exists(self.data_path):
            print(f"Generating {self.split} dataset...")
            self._generate_and_save_dataset()
        data = np.load(self.data_path, allow_pickle=True).item()
        print(data["states"].shape, data["states"][0].shape)  # (nb traj, time, 64, 64)
        
        if self.num_steps_rollout < self.num_steps_max: # needs chunking 
            # Create chunked dataset
            all_chunks = []
            T = data["states"][0].shape[0]
            print(T)
            for k in range(data['states'].shape[0]): 
                traj = data['states'][k]
                start, end = 0, self.num_steps_rollout+1
                while end < T+1:
                    print(start, end)
                    chunk = traj[start:end]
                    all_chunks.append(chunk)
                    start += self.num_steps_rollout
                    end = start + (self.num_steps_rollout+1)
            print(all_chunks[0].shape) # (num_steps_rollout+1, nc)
            all_chunks = np.stack(all_chunks)  # shape (num_chunks, num_steps_rollout, 64, 64)
            print(all_chunks.shape)
            np.save(self.path, dict(states=all_chunks))
            print(f"Saved chunked dataset to {self.path}")
        else: # chunk=full
            np.save(self.path, dict(states=data['states']))
            print(f"Copied full dataset to {self.path}")

    def _load_dataset(self):
        """Load chunked dataset, generate if not existing."""
        if not os.path.exists(self.path+".npy"):
            print(f"Chunked dataset not found at {self.path}. Generating...")
            self._chunk_and_save_dataset()
        data = np.load(f"{self.path}.npy", allow_pickle=True).item()
        return data['states']

    def __getitem__(self, index):
        """Get one trajectory."""
        return {"states": self.states[index]}


import os
import abc
import numpy as np
import jax  

from solvers.runge_kutta import RK_solver_fixed, RK_tableaux
jax.config.update("jax_enable_x64", True)


class BaseDataset(abc.ABC):
    """Base dataset for all types of datasets:
        - load dataset
        - chunking
        - getitem
        - length
    """
    VALID_SPLITS = ('train', 'val', 'test')

    def __init__(self, dt_num, num_steps_max, num_steps_rollout,
                 path, split, nb_traj, integration_method='RK4', *args, **kwargs):
        """
        Args:
            dt: float, time step
            num_steps_max: int, number of total steps per trajectory
            num_steps_rollout: int, number of steps per rollout
            path: str, directory path to save chunked trajectories (into data_exp)
            split: str, 'train', 'val', or 'test'
            nb_traj: int, number of trajectories to generate
            integration_method: str, e.g. 'RK4', 'DOPRI5'
        """
        assert split in self.VALID_SPLITS, f"split must be one of {self.VALID_SPLITS}"
        self.dt = dt_num
        self.num_steps_max = num_steps_max
        self.num_steps_rollout = num_steps_rollout
        self.nb_traj = nb_traj
        self.integration_method = integration_method
        self.split = split
        self.chunk_path = path+'_'+split
        self.states = self._load_dataset()

    @abc.abstractmethod
    def _chunk_source_dataset(self) -> np.ndarray:
        """Return full trajectories array of shape (nb_traj, T, nc)."""

    def _chunk_trajectories(self, states: np.ndarray) -> np.ndarray:
        if self.num_steps_rollout >= self.num_steps_max:
            return states
        T = states.shape[1]  # (nb_traj, T, nc) — use axis 1
        all_chunks = []
        for k in range(states.shape[0]): 
            traj = states[k]
            start, end = 0, self.num_steps_rollout+1
            while end < T+1:
                chunk = traj[start:end]
                all_chunks.append(chunk)
                start += self.num_steps_rollout
                end = start + (self.num_steps_rollout+1)
        all_chunks = np.stack(all_chunks)  # shape (num_chunks, num_steps_rollout, nc)
        return np.stack(all_chunks)

    def _chunk_and_save_dataset(self):
        states = self._chunk_source_dataset()
        chunks = self._chunk_trajectories(states)
        np.save(self.chunk_path, dict(states=chunks))
        print(f"Saved chunked dataset to {self.chunk_path}")

    def _load_dataset(self):
        if not os.path.exists(self.chunk_path + ".npy"):
            self._chunk_and_save_dataset()
        data = np.load(f"{self.chunk_path}.npy", allow_pickle=True).item()
        assert 'states' in data, "Corrupt cache: missing 'states' key"
        return data['states']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return {"states": self.states[index]}


class ODEDataset(BaseDataset):
    """ Datasets when trajectories are generated """
    def __init__(self, name, *args, **kwargs):
        # full trajectories 
        self.data_path = f"data/{name}_full_{kwargs['split']}.npy"
        super().__init__(*args,**kwargs)

    def F(self, s, t): 
        raise NotImplementedError
    
    def _get_initial_condition(self, seed):
        raise NotImplementedError

    def _generate_and_save_dataset(self):
        """Generate full trajectories and save them in one .npy file."""
        all_states = []
        num_steps = max(self.num_steps_rollout, self.num_steps_max) 
        for idx in range(self.nb_traj):
            y0 = self._get_initial_condition(idx)
            states, t, _, _ = RK_solver_fixed(
                fun=self.F,
                y0=y0,
                dt=self.dt,
                num_steps=num_steps,
                tableau=RK_tableaux[self.integration_method],
            )
            all_states.append(np.array(states))
        all_states = np.stack(all_states)  # shape (nb_traj, T, nc)
        np.save(self.data_path, dict(states=all_states))
        print(f"Saved {self.nb_traj} trajectories to {self.data_path}")
    
    def _chunk_source_dataset(self):
        if not os.path.exists(self.data_path):
            print(f"Generating {self.split} dataset...")
            self._generate_and_save_dataset()
        data = np.load(self.data_path, allow_pickle=True).item()
        return data['states']




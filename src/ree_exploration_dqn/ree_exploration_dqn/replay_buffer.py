"""
Thread-safe experience replay buffer with float16 storage and gzip persistence.

Storage arrays (all pre-allocated at construction, pointer wraps around):
    _obs:       (capacity, H, W, C)  float16
    _next_obs:  (capacity, H, W, C)  float16
    _pos:       (capacity, 2)        float32
    _next_pos:  (capacity, 2)        float32
    _actions:   (capacity,)          int8
    _rewards:   (capacity,)          float32
    _dones:     (capacity,)          bool

Float16 halves memory vs float32. Precision loss is negligible for RL because
rewards and Q-values have far lower precision requirements than, e.g., vision.
Sampling casts obs/next_obs back to float32 before returning.
"""

import gzip
import os
import pickle
import threading
from typing import Optional, Tuple

import numpy as np


class ReplayBuffer:
    """Circular float16 replay buffer with gzip persistence and thread safety."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]) -> None:
        self._capacity = capacity
        self._obs_shape = obs_shape
        self._lock = threading.Lock()

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float16)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float16)
        self._pos = np.zeros((capacity, 2), dtype=np.float32)
        self._next_pos = np.zeros((capacity, 2), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int8)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        self._ptr: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------ #
    #  WRITE                                                               #
    # ------------------------------------------------------------------ #

    def push(
        self,
        obs: np.ndarray,
        pos: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_pos: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition. Thread-safe."""
        with self._lock:
            self._obs[self._ptr] = obs.astype(np.float16)
            self._next_obs[self._ptr] = next_obs.astype(np.float16)
            self._pos[self._ptr] = pos
            self._next_pos[self._ptr] = next_pos
            self._actions[self._ptr] = int(action)
            self._rewards[self._ptr] = float(reward)
            self._dones[self._ptr] = bool(done)
            self._ptr = (self._ptr + 1) % self._capacity
            self._size = min(self._size + 1, self._capacity)

    # ------------------------------------------------------------------ #
    #  READ                                                                #
    # ------------------------------------------------------------------ #

    def sample(
        self, batch_size: int
    ) -> Optional[Tuple[np.ndarray, ...]]:
        """Return a random batch without replacement; None if too few samples.

        Returns (obs_f32, pos, actions, rewards, next_obs_f32, next_pos, dones)
        where obs arrays are float32 (cast from float16 storage).
        """
        with self._lock:
            if self._size < batch_size:
                return None
            idx = np.random.choice(self._size, batch_size, replace=False)
            obs = self._obs[idx].astype(np.float32)
            next_obs = self._next_obs[idx].astype(np.float32)
            pos = self._pos[idx].copy()
            next_pos = self._next_pos[idx].copy()
            actions = self._actions[idx].astype(np.int64)
            rewards = self._rewards[idx].copy()
            dones = self._dones[idx].copy()

        assert obs.shape == (batch_size, *self._obs_shape), (
            f"obs shape mismatch: {obs.shape} != {(batch_size, *self._obs_shape)}"
        )
        return obs, pos, actions, rewards, next_obs, next_pos, dones

    # ------------------------------------------------------------------ #
    #  PERSISTENCE                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Atomically save the buffer to a gzip-compressed pickle file."""
        with self._lock:
            payload = {
                'obs': self._obs[: self._size],
                'next_obs': self._next_obs[: self._size],
                'pos': self._pos[: self._size],
                'next_pos': self._next_pos[: self._size],
                'actions': self._actions[: self._size],
                'rewards': self._rewards[: self._size],
                'dones': self._dones[: self._size],
                'ptr': self._ptr,
                'size': self._size,
                'capacity': self._capacity,
                'obs_shape': self._obs_shape,
            }
        os.makedirs(
            os.path.dirname(os.path.abspath(path)), exist_ok=True
        )
        tmp = path + '.tmp'
        with gzip.open(tmp, 'wb') as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Load buffer from a gzip pickle file. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            with gzip.open(path, 'rb') as fh:
                p = pickle.load(fh)
            n = int(p['size'])
            with self._lock:
                self._obs[:n] = p['obs']
                self._next_obs[:n] = p['next_obs']
                self._pos[:n] = p['pos']
                self._next_pos[:n] = p['next_pos']
                self._actions[:n] = p['actions']
                self._rewards[:n] = p['rewards']
                self._dones[:n] = p['dones']
                self._ptr = int(p['ptr']) % self._capacity
                self._size = n
            return True
        except Exception as exc:
            print(f"[Buffer] [WARN] load failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    #  DUNDER                                                              #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size}/{self._capacity}, "
            f"obs={self._obs_shape}, dtype=float16)"
        )

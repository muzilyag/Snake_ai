import numpy as np
from typing import List

class RolloutBuffer:
    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.global_obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(self, obs: np.ndarray, global_obs: np.ndarray, action: int, log_prob: float, reward: float, value: float, done: bool) -> None:
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.obs.clear()
        self.global_obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        
    def __len__(self) -> int:
        return len(self.obs)
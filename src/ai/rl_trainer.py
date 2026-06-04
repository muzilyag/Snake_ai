import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import copy
from typing import Tuple, List
from .model import SnakeNet

class RLTrainer:
    def __init__(self, model: SnakeNet, role: str, lr: float = 0.001, gamma: float = 0.9, batch_size: int = 64, tau: float = 0.005, memory_capacity: int = 100000):
        self.model: SnakeNet = model
        self.target_model: SnakeNet = copy.deepcopy(model)
        self.target_model.eval()
        
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        
        self.role: str = role
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.train_freq: int = 10
        self.step_counter: int = 0
        self.tau: float = tau
        
        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion: nn.Module = nn.SmoothL1Loss()
        self.memory: deque = deque(maxlen=memory_capacity)

    def train_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        self.memory.append((state, action, reward, next_state, done))
        self.step_counter += 1
        
        if len(self.memory) < self.batch_size or self.step_counter % self.train_freq != 0:
            return 0.0
            
        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t: torch.Tensor = torch.tensor(np.array(states), dtype=torch.float, device=self.device)
        next_states_t: torch.Tensor = torch.tensor(np.array(next_states), dtype=torch.float, device=self.device)
        actions_t: torch.Tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t: torch.Tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones_t: torch.Tensor = torch.tensor(dones, dtype=torch.float, device=self.device)
        
        preds: torch.Tensor = self.model(states_t, self.role)
        current_q: torch.Tensor = preds.gather(1, actions_t)
        
        with torch.no_grad():
            next_preds: torch.Tensor = self.target_model(next_states_t, self.role)
            max_next: torch.Tensor = torch.max(next_preds, dim=1)[0]
            target_q: torch.Tensor = rewards_t + self.gamma * max_next * (1 - dones_t)
        
        self.optimizer.zero_grad()
        loss: torch.Tensor = self.criterion(current_q, target_q.unsqueeze(1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self._soft_update_target_network()
        
        return loss.item()

    def _soft_update_target_network(self) -> None:
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * model_param.data + (1.0 - self.tau) * target_param.data)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class RLTrainer:
    def __init__(self, model: nn.Module, lr: float = 0.001, gamma: float = 0.9, batch_size: int = 64) -> None:
        self.model = model
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = 10
        self.step_counter = 0
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=100_000)

    def train_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        self.memory.append((state, action, reward, next_state, done))
        self.step_counter += 1
        
        if len(self.memory) < self.batch_size or self.step_counter % self.train_freq != 0:
            return 0.0
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.tensor(np.array(states), dtype=torch.float)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        dones_t = torch.tensor(dones, dtype=torch.float)
        
        preds = self.model(states_t)
        target = preds.clone()
        
        with torch.no_grad():
            next_preds = self.model(next_states_t)
            max_next = torch.max(next_preds, dim=1)[0]
            
        q_new = rewards_t + self.gamma * max_next * (1 - dones_t)
        target.scatter_(1, actions_t, q_new.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss = self.criterion(preds, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
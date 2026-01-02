import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class RLTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=100_000)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        
        pred = self.model(state)
        target = pred.clone()
        
        new_q = reward
        if not done:
            new_q = reward + self.gamma * torch.max(self.model(next_state))
            
        target[0][action] = new_q
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
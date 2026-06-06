import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict
import numpy as np

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNet(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 64, output_size: int = 4, num_roles: int = 3, embedding_dim: int = 4):
        super().__init__()
        self.role_embedding = nn.Embedding(num_roles, embedding_dim)
        
        self.linear1 = layer_init(nn.Linear(input_size + embedding_dim, hidden_size))
        self.linear2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.action_head = layer_init(nn.Linear(hidden_size, output_size), std=0.01)
        
        self.role_to_idx: Dict[str, int] = {
            "Harvester": 0, 
            "Hunter": 1, 
            "Defender": 2
        }

    def forward(self, x: torch.Tensor, role_name: str) -> Categorical:
        device = x.device
        role_idx: int = self.role_to_idx.get(role_name, 0)
        role_tensor: torch.Tensor = torch.tensor([role_idx] * x.size(0), dtype=torch.long, device=device)
        
        role_emb: torch.Tensor = self.role_embedding(role_tensor)
        x = torch.cat([x, role_emb], dim=1)
        
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        action_logits = self.action_head(x)
        
        return Categorical(logits=action_logits)

class CriticNet(nn.Module):
    def __init__(self, global_state_size: int, hidden_size: int = 64):
        super().__init__()
        
        self.linear1 = layer_init(nn.Linear(global_state_size, hidden_size))
        self.linear2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, global_x: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.linear1(global_x))
        x = F.tanh(self.linear2(x))
        return self.value_head(x)
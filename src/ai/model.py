import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class SnakeNet(nn.Module):
    def __init__(self, input_size: int = 33, hidden_size: int = 256, output_size: int = 3, num_roles: int = 3, embedding_dim: int = 4):
        super().__init__()
        self.role_embedding = nn.Embedding(num_roles, embedding_dim)
        
        self.linear1 = nn.Linear(input_size + embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        self.role_to_idx: Dict[str, int] = {
            "Harvester": 0, 
            "Hunter": 1, 
            "Defender": 2
        }

    def forward(self, x: torch.Tensor, role_name: str) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        device = x.device
        role_idx: int = self.role_to_idx.get(role_name, 0)
        role_tensor: torch.Tensor = torch.tensor([role_idx] * x.size(0), dtype=torch.long, device=device)
        
        role_emb: torch.Tensor = self.role_embedding(role_tensor)
        x = torch.cat([x, role_emb], dim=1)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)

    def save(self, file_name: str = 'model.pth') -> None:
        model_folder_path: str = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        path: str = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), path)

    def load(self, file_name: str = 'model.pth') -> bool:
        path: str = os.path.join('./model', file_name)
        if not os.path.exists(path):
            return False
        try:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.eval()
            return True
        except RuntimeError:
            return False
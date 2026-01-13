import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SnakeNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), path)
        print(f"--- Model saved: {path} ---")

    def load(self, file_name='model.pth'):
        path = os.path.join('./model', file_name)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            print(f"--- Model loaded: {path} ---")
            return True
        return False
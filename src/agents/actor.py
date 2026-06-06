import torch
import numpy as np
from typing import Tuple
from src.agents.mappo_models import ActorNet, CriticNet
from src.agents.rollout import RolloutBuffer

class MAPPOAgent:
    def __init__(self, agent_id: str, role: str, global_state_size: int, device: torch.device):
        self.agent_id: str = agent_id
        self.role: str = role
        self.device: torch.device = device
        
        self.actor = ActorNet().to(self.device)
        self.critic = CriticNet(global_state_size=global_state_size).to(self.device)
        self.buffer = RolloutBuffer()

    def act(self, obs: np.ndarray, global_obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
        global_obs_t = torch.tensor(global_obs, dtype=torch.float, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            dist = self.actor(obs_t, self.role)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(global_obs_t)
            
        return action.item(), log_prob.item(), value.item()

    def act_batched(self, obs_batch: np.ndarray, global_obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.tensor(obs_batch, dtype=torch.float, device=self.device)
        global_obs_t = torch.tensor(global_obs_batch, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            dist = self.actor(obs_t, self.role)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = self.critic(global_obs_t).flatten()
            
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def save(self, directory: str) -> None:
        torch.save(self.actor.state_dict(), f"{directory}/{self.agent_id}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{self.agent_id}_critic.pth")

    def load(self, directory: str) -> None:
        self.actor.load_state_dict(torch.load(f"{directory}/{self.agent_id}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{directory}/{self.agent_id}_critic.pth", map_location=self.device))
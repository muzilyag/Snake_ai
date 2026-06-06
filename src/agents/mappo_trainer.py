import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from src.agents.actor import MAPPOAgent

class MAPPOTrainer:
    def __init__(self, agents: Dict[str, MAPPOAgent], device: torch.device, lr_actor: float = 3e-4, lr_critic: float = 1e-3, gamma: float = 0.99, clip_ratio: float = 0.2, entropy_coef: float = 0.01, gae_lambda: float = 0.95):
        self.agents: Dict[str, MAPPOAgent] = agents
        self.device: torch.device = device
        self.gamma: float = gamma
        self.clip_ratio: float = clip_ratio
        self.entropy_coef: float = entropy_coef
        self.gae_lambda: float = gae_lambda
        
        self.actor_optimizers: Dict[str, optim.Optimizer] = {
            a_id: optim.Adam(agent.actor.parameters(), lr=lr_actor) for a_id, agent in agents.items()
        }
        self.critic_optimizers: Dict[str, optim.Optimizer] = {
            a_id: optim.Adam(agent.critic.parameters(), lr=lr_critic) for a_id, agent in agents.items()
        }

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_values: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Теперь сюда приходят матрицы [num_steps, num_envs]
        num_steps, num_envs = rewards.shape
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = np.zeros(num_envs)
        
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_val = next_values
            else:
                next_val = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
            
        # Расплющиваем обратно в 1D для нейросети
        return torch.tensor(returns.flatten(), dtype=torch.float, device=self.device), torch.tensor(advantages.flatten(), dtype=torch.float, device=self.device)

    def train_agent(self, agent_id: str, next_global_obs: List[np.ndarray], epochs: int = 4) -> None:
        agent = self.agents[agent_id]
        buffer = agent.buffer
        
        if len(buffer) == 0:
            return

        num_envs = len(next_global_obs)
        num_steps = len(buffer.rewards) // num_envs

        # Конвертируем плоские списки в правильные 2D-матрицы
        np_rewards = np.array(buffer.rewards, dtype=np.float32).reshape(num_steps, num_envs)
        np_values = np.array(buffer.values, dtype=np.float32).reshape(num_steps, num_envs)
        np_dones = np.array(buffer.dones, dtype=np.float32).reshape(num_steps, num_envs)

        next_global_obs_t = torch.tensor(np.array(next_global_obs), dtype=torch.float, device=self.device)
        with torch.no_grad():
            next_values = agent.critic(next_global_obs_t).squeeze(-1).cpu().numpy()

        returns, advantages = self.compute_gae(np_rewards, np_values, np_dones, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_obs = torch.tensor(np.array(buffer.obs), dtype=torch.float, device=self.device)
        old_global_obs = torch.tensor(np.array(buffer.global_obs), dtype=torch.float, device=self.device)
        old_actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float, device=self.device)

        for _ in range(epochs):
            dist = agent.actor(old_obs, agent.role)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            values = agent.critic(old_global_obs).squeeze(-1)
            critic_loss = nn.MSELoss()(values, returns)

            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            self.actor_optimizers[agent_id].step()

            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            self.critic_optimizers[agent_id].step()

        buffer.clear()
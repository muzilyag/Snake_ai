import numpy as np
import sys
import os
from typing import Any, Tuple, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import snake_cpp
from src.env.entity import DeathReason

class DummyPoint:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

class DummySnake:
    def __init__(self, agent_id, team_name, role, color):
        self.agent_id = agent_id
        self.team_name = team_name
        self.role = role
        self.color = color
        self.is_alive = True
        self.hp = 100.0
        self.max_hp = 100.0
        self.score = 0
        self.head = DummyPoint(0, 0)
        self.body = []
        self.direction = None

class CppVecEnv:
    def __init__(self, num_envs: int, config: Any):
        self.num_envs = num_envs
        self.config = config
        
        self.agent_ids = []
        self.snakes = {}
        for team in config.teams:
            team_color = getattr(team, 'color', None)
            if team_color is None:
                if "Red" in team.name:
                    team_color = (255, 50, 50)
                elif "Green" in team.name:
                    team_color = (50, 255, 50)
                elif "Blue" in team.name:
                    team_color = (50, 50, 255)
                else:
                    team_color = (200, 200, 200)

            for i, role in enumerate(team.agent_roles):
                a_id = f"{team.name}_{role}_{i}"
                self.agent_ids.append(a_id)
                self.snakes[a_id] = DummySnake(a_id, team.name, role, team_color)
        
        self.num_snakes = len(self.agent_ids)
        self.engine = snake_cpp.VecSnakeEngine(
            num_envs, self.num_snakes, config.grid_width, config.grid_height, config.block_size, config.food_count
        )
        
        self.last_global_states = None
        self.main_env = self
        self.iteration = 0
        self.foods = []

    def _sync_render_state(self, render_np=None):
        if self.last_global_states is None: 
            return
        state = self.last_global_states[0]
        
        for s, agent_id in enumerate(self.agent_ids):
            idx = s * 4
            snake = self.snakes[agent_id]
            snake.is_alive = bool(state[idx] > 0.5)
            if snake.is_alive:
                snake.head.x = int(state[idx + 1])
                snake.head.y = int(state[idx + 2])
                snake.hp = float(state[idx + 3])
                
                if render_np is not None:
                    snake.body.clear()
                    body_matrix = render_np[0][s]
                    for i in range(len(body_matrix)):
                        bx, by = int(body_matrix[i][0]), int(body_matrix[i][1])
                        if bx == -1000: 
                            break
                        snake.body.append(DummyPoint(bx, by))
            else:
                snake.head.x = -1000
                snake.head.y = -1000
                snake.body.clear()

        food_offset = self.num_snakes * 4
        self.foods.clear()
        for f in range(self.config.food_count):
            fx = int(state[food_offset + (f * 2)])
            fy = int(state[food_offset + (f * 2) + 1])
            self.foods.append(DummyPoint(fx, fy))

    def reset(self) -> List[Dict[str, np.ndarray]]:
        self.engine.reset_all()
        self.iteration = 0
        actions_zeros = np.zeros(self.num_envs * self.num_snakes, dtype=np.int32)
        
        obs_np, global_state_np, _, dones_np, render_np, scores_np, events_np, killers_np = self.engine.step(actions_zeros)
        self.last_global_states = global_state_np
        
        self._sync_render_state(render_np)
        
        obs_list = []
        for e in range(self.num_envs):
            obs_dict = {a_id: obs_np[e * self.num_snakes + s] for s, a_id in enumerate(self.agent_ids)}
            obs_list.append(obs_dict)
        return obs_list

    def step(self, actions_list: List[Dict[str, int]], render: bool = False):
        self.iteration += 1
        actions_flat = np.zeros(self.num_envs * self.num_snakes, dtype=np.int32)
        for e in range(self.num_envs):
            for s, agent_id in enumerate(self.agent_ids):
                actions_flat[e * self.num_snakes + s] = actions_list[e].get(agent_id, 0)
                
        obs_np, global_state_np, _, dones_np, render_np, scores_np, events_np, killers_np = self.engine.step(actions_flat)
        self.last_global_states = global_state_np
        
        if render:
            self._sync_render_state(render_np)
            
        event_map = {0: 'move', 1: 'food', 2: 'death', 3: 'starve', 4: 'wall', 5: 'self'}
        
        obs_all, rew_all, don_all, inf_all = [], [], [], []
        for e in range(self.num_envs):
            obs_dict = {}
            rew_dict = {a_id: 0.0 for a_id in self.agent_ids}
            don_dict = {}
            inf_dict = {}
            
            for s, agent_id in enumerate(self.agent_ids):
                idx = e * self.num_snakes + s
                obs_dict[agent_id] = obs_np[idx]
                don_dict[agent_id] = bool(dones_np[idx])
                
                event_code = int(events_np[idx])
                event_str = event_map.get(event_code, 'move')
                hp_val = float(global_state_np[e][s * 4 + 3])
                
                role = self.snakes[agent_id].role
                team_name = self.snakes[agent_id].team_name
                presets = self.config.reward_presets.get(role, {})
                
                if event_code == 1:
                    food_reward = presets.get('food', 10.0)
                    rew_dict[agent_id] += food_reward
                    
                    for ally_id in self.agent_ids:
                        if ally_id != agent_id and self.snakes[ally_id].team_name == team_name:
                            rew_dict[ally_id] += food_reward * 0.5  
                            
                elif event_code == 2:
                    rew_dict[agent_id] += presets.get('death', -50.0)
                    killer_idx = int(killers_np[idx])
                    if killer_idx >= 0 and killer_idx != s:
                        killer_agent_id = self.agent_ids[killer_idx]
                        killer_role = self.snakes[killer_agent_id].role
                        killer_team = self.snakes[killer_agent_id].team_name
                        killer_presets = self.config.reward_presets.get(killer_role, {})
                        
                        reward_key = f"kill_{role.lower()}"
                        kill_reward = killer_presets.get(reward_key, 0.0)
                        rew_dict[killer_agent_id] += kill_reward
                        
                        for ally_id in self.agent_ids:
                            if ally_id != killer_agent_id and self.snakes[ally_id].team_name == killer_team:
                                rew_dict[ally_id] += kill_reward * 0.5 
                                
                elif event_code == 3:
                    rew_dict[agent_id] += presets.get('starve', -50.0)
                elif event_code == 4:
                    rew_dict[agent_id] += presets.get('death', -50.0) + presets.get('wall_penalty', -1.0)
                elif event_code == 5:
                    rew_dict[agent_id] += presets.get('death', -50.0)
                else:
                    rew_dict[agent_id] += presets.get('idle_penalty', -0.01)
                    
                role_max_hp = 100.0
                if role == "Hunter":
                    role_max_hp = 150.0
                elif role == "Defender":
                    role_max_hp = 200.0

                inf_dict[agent_id] = {
                    'score': int(scores_np[idx]),
                    'hp': hp_val,
                    'max_hp': role_max_hp,
                    'event': event_str
                }
                
                if event_code == 2:
                    inf_dict[agent_id]['death_reason'] = DeathReason.ENEMY_COLLISION
                elif event_code == 3:
                    inf_dict[agent_id]['death_reason'] = DeathReason.STARVATION
                elif event_code == 4:
                    inf_dict[agent_id]['death_reason'] = DeathReason.WALL_COLLISION
                elif event_code == 5:
                    inf_dict[agent_id]['death_reason'] = DeathReason.SELF_COLLISION
            
            obs_all.append(obs_dict)
            rew_all.append(rew_dict)
            don_all.append(don_dict)
            inf_all.append(inf_dict)
            
        return obs_all, rew_all, don_all, inf_all

    def get_global_states(self) -> List[np.ndarray]:
        return [self.last_global_states[e] for e in range(self.num_envs)]
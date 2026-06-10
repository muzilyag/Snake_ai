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
                if "Red" in team.name: team_color = (255, 50, 50)
                elif "Green" in team.name: team_color = (50, 255, 50)
                elif "Blue" in team.name: team_color = (50, 50, 255)
                else: team_color = (200, 200, 200)

            for i, role in enumerate(team.agent_roles):
                a_id = f"{team.name}_{role}_{i}"
                self.agent_ids.append(a_id)
                self.snakes[a_id] = DummySnake(a_id, team.name, role, team_color)
        
        self.num_snakes = len(self.agent_ids)
        
        flat_cfg = []
        roles = ["Harvester", "Hunter", "Defender"]
        for r in roles:
            p = config.reward_presets.get(r, {})
            flat_cfg.extend([
                p.get('food', 0.0), p.get('death', 0.0), p.get('starve', 0.0),
                p.get('wall_penalty', 0.0), p.get('idle_penalty', 0.0),
                p.get('kill_harvester', 0.0), p.get('kill_hunter', 0.0), p.get('kill_defender', 0.0),
                p.get('friendly_fire', 0.0)
            ])

        total_snakes = self.num_envs * self.num_snakes
        obs_size = 28
        global_state_size = (self.num_snakes * 4) + (self.config.food_count * 2)
        max_body_length = 100

        self.obs_np = np.zeros((total_snakes, obs_size), dtype=np.float32)
        self.global_state_np = np.zeros((self.num_envs, global_state_size), dtype=np.float32)
        self.rewards_np = np.zeros(total_snakes, dtype=np.float32)
        self.dones_np = np.zeros(total_snakes, dtype=np.int32)
        self.render_np = np.full((self.num_envs, self.num_snakes, max_body_length, 2), -1000, dtype=np.int32)
        self.scores_np = np.zeros(total_snakes, dtype=np.int32)
        self.events_np = np.zeros(total_snakes, dtype=np.int32)
        self.killers_np = np.full(total_snakes, -1, dtype=np.int32)
            
        self.engine = snake_cpp.VecSnakeEngine(
            num_envs, self.num_snakes, config.grid_width, config.grid_height, 
            config.block_size, config.food_count, flat_cfg,
            self.obs_np, self.global_state_np, self.rewards_np,
            self.dones_np, self.render_np, self.scores_np,
            self.events_np, self.killers_np
        )
        
        self.last_global_states = None
        self.main_env = self
        self.iteration = 0
        self.foods = []
        self.actions_flat = np.zeros(total_snakes, dtype=np.int32)

    def _sync_render_state(self, render_np=None):
        if self.last_global_states is None: return
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
                        if bx == -1000: break
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
        self.actions_flat.fill(0)
        
        self.engine.step(self.actions_flat)
        self.last_global_states = self.global_state_np
        self._sync_render_state(self.render_np)
        
        obs_list = []
        for e in range(self.num_envs):
            obs_dict = {a_id: self.obs_np[e * self.num_snakes + s].copy() for s, a_id in enumerate(self.agent_ids)}
            obs_list.append(obs_dict)
        return obs_list

    def step(self, actions_list: List[Dict[str, int]], render: bool = False):
        self.iteration += 1
        for e in range(self.num_envs):
            for s, agent_id in enumerate(self.agent_ids):
                self.actions_flat[e * self.num_snakes + s] = actions_list[e].get(agent_id, 0)
                
        self.engine.step(self.actions_flat)
        self.last_global_states = self.global_state_np
        
        if render:
            self._sync_render_state(self.render_np)
            
        obs_all, rew_all, don_all, inf_all = [], [], [], []
        ev_map = {0: 'move', 1: 'food', 2: 'death', 3: 'starve', 4: 'wall', 5: 'self'}
        
        for e in range(self.num_envs):
            obs_dict, rew_dict, don_dict, inf_dict = {}, {}, {}, {}
            for s, agent_id in enumerate(self.agent_ids):
                idx = e * self.num_snakes + s
                obs_dict[agent_id] = self.obs_np[idx].copy()
                rew_dict[agent_id] = float(self.rewards_np[idx])
                don_dict[agent_id] = bool(self.dones_np[idx])
                
                ev_code = int(self.events_np[idx])
                role = self.snakes[agent_id].role
                inf_dict[agent_id] = {
                    'score': int(self.scores_np[idx]),
                    'hp': float(self.global_state_np[e][s * 4 + 3]),
                    'max_hp': 150.0 if role == "Hunter" else (200.0 if role == "Defender" else 100.0),
                    'event': ev_map.get(ev_code, 'move')
                }
                
                if ev_code == 2: inf_dict[agent_id]['death_reason'] = DeathReason.ENEMY_COLLISION
                elif ev_code == 3: inf_dict[agent_id]['death_reason'] = DeathReason.STARVATION
                elif ev_code == 4: inf_dict[agent_id]['death_reason'] = DeathReason.WALL_COLLISION
                elif ev_code == 5: inf_dict[agent_id]['death_reason'] = DeathReason.SELF_COLLISION
            
            obs_all.append(obs_dict)
            rew_all.append(rew_dict)
            don_all.append(don_dict)
            inf_all.append(inf_dict)
            
        return obs_all, rew_all, don_all, inf_all

    def get_global_states(self) -> List[np.ndarray]:
        return [self.last_global_states[e].copy() for e in range(self.num_envs)]
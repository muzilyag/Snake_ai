import torch
import numpy as np
from src.core.types import Direction, Point

class MultiAgentStrategy:
    def __init__(self, config):
        self.config = config

    def get_action(self, model, snake, state_dto):
        sensors = self._get_sensors(snake, state_dto)
        state_tensor = torch.tensor(sensors, dtype=torch.float).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(state_tensor)
            action_idx = torch.argmax(prediction).item()
            
        move = self._transform_action(snake, action_idx)
        return move, action_idx, sensors

    def _get_sensors(self, snake, state_dto):
        head = snake.head
        bs = self.config.block_size
        
        point_l = Point(head.x - bs, head.y)
        point_r = Point(head.x + bs, head.y)
        point_u = Point(head.x, head.y - bs)
        point_d = Point(head.x, head.y + bs)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        def is_obstacle(pt):
            if pt.x < 0 or pt.x >= self.config.map_width_px or \
               pt.y < 0 or pt.y >= self.config.map_height_px:
                return True
            
            for s in state_dto.snakes:
                if s.is_alive and s.team_name == snake.team_name:
                    if pt in s.body:
                        return True
            return False

        def is_enemy(pt):
            for s in state_dto.snakes:
                if s.is_alive and s.team_name != snake.team_name:
                    if pt in s.body:
                        return True
            return False

        obs_l = is_obstacle(point_l)
        obs_r = is_obstacle(point_r)
        obs_u = is_obstacle(point_u)
        obs_d = is_obstacle(point_d)

        enm_l = is_enemy(point_l)
        enm_r = is_enemy(point_r)
        enm_u = is_enemy(point_u)
        enm_d = is_enemy(point_d)

        state = [
            (dir_r and obs_r) or (dir_l and obs_l) or (dir_u and obs_u) or (dir_d and obs_d),
            (dir_u and obs_r) or (dir_d and obs_l) or (dir_l and obs_u) or (dir_r and obs_d),
            (dir_d and obs_r) or (dir_u and obs_l) or (dir_r and obs_u) or (dir_l and obs_d),

            (dir_r and enm_r) or (dir_l and enm_l) or (dir_u and enm_u) or (dir_d and enm_d),
            (dir_u and enm_r) or (dir_d and enm_l) or (dir_l and enm_u) or (dir_r and enm_d),
            (dir_d and enm_r) or (dir_u and enm_l) or (dir_r and enm_u) or (dir_l and enm_d),

            dir_l,
            dir_r,
            dir_u,
            dir_d,
        ]
        
        food = self._get_closest_food(snake, state_dto.foods)
        state.append(food.x < head.x)
        state.append(food.x > head.x)
        state.append(food.y < head.y)
        state.append(food.y > head.y)
        
        return np.array(state, dtype=float)

    def _get_closest_food(self, snake, foods):
        if not foods: return Point(-1, -1)
        closest = min(foods, key=lambda f: (snake.head.x - f.x)**2 + (snake.head.y - f.y)**2)
        return closest

    def _transform_action(self, snake, action_idx):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake.direction)

        if action_idx == 0:
            return clock_wise[idx]
        elif action_idx == 1:
            return clock_wise[(idx + 1) % 4]
        else:
            return clock_wise[(idx - 1) % 4]
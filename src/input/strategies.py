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
        
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        state = [
            (dir_r and self._is_collision(point_r, state_dto)) or 
            (dir_l and self._is_collision(point_l, state_dto)) or 
            (dir_u and self._is_collision(point_u, state_dto)) or 
            (dir_d and self._is_collision(point_d, state_dto)),

            (dir_u and self._is_collision(point_r, state_dto)) or 
            (dir_d and self._is_collision(point_l, state_dto)) or 
            (dir_l and self._is_collision(point_u, state_dto)) or 
            (dir_r and self._is_collision(point_d, state_dto)),

            (dir_d and self._is_collision(point_r, state_dto)) or 
            (dir_u and self._is_collision(point_l, state_dto)) or 
            (dir_r and self._is_collision(point_u, state_dto)) or 
            (dir_l and self._is_collision(point_d, state_dto)),

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

    def _is_collision(self, pt, state):
        if pt.x < 0 or pt.x >= self.config.map_width_px or \
           pt.y < 0 or pt.y >= self.config.map_height_px:
            return True
        
        for s in state.snakes:
            if s.is_alive and pt in s.body:
                return True
        return False

    def _transform_action(self, snake, action_idx):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake.direction)

        if action_idx == 0: 
            return clock_wise[idx]
        elif action_idx == 1: 
            return clock_wise[(idx + 1) % 4]
        else: 
            return clock_wise[(idx - 1) % 4]
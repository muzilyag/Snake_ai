import torch
import numpy as np
from src.core.interfaces import MoveStrategy
from src.core.types import Direction, Point

class AIStrategy(MoveStrategy):
    def __init__(self, model):
        self.model = model

    def get_next_move(self, state_dto):
        sensors = self._get_sensors(state_dto)
        with torch.no_grad():
            prediction = self.model(sensors)
            action_idx = torch.argmax(prediction).item()
        
        direction = self._transform_action(state_dto, action_idx)
        return direction, action_idx

    def _get_sensors(self, state):
        head = state.snake_body[0]
        cur_dir = self._get_current_direction(state)
        
        def get_point(d):
            return Point(head.x + d[0], head.y + d[1])

        dir_map = {
            Direction.UP:    (0, -20),
            Direction.DOWN:  (0, 20),
            Direction.LEFT:  (-20, 0),
            Direction.RIGHT: (20, 0)
        }
        
        cw = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = cw.index(cur_dir)
        
        d_straight = dir_map[cur_dir]
        d_right = dir_map[cw[(idx + 1) % 4]]
        d_left = dir_map[cw[(idx - 1) % 4]]

        def is_unsafe(p):
            if p.x < 0 or p.x >= 640 or p.y < 0 or p.y >= 480:
                return 1
            if p in state.snake_body:
                return 1
            return 0

        sensors = [
            is_unsafe(get_point(d_straight)),
            is_unsafe(get_point(d_right)),
            is_unsafe(get_point(d_left)),
            cur_dir == Direction.LEFT, cur_dir == Direction.RIGHT,
            cur_dir == Direction.UP, cur_dir == Direction.DOWN,
            state.food.x < head.x, state.food.x > head.x,
            state.food.y < head.y, state.food.y > head.y
        ]
        return np.array(sensors, dtype=float)

    def _get_current_direction(self, state):
        if len(state.snake_body) < 2: return Direction.RIGHT
        h, n = state.snake_body[0], state.snake_body[1]
        if h.x > n.x: return Direction.RIGHT
        if h.x < n.x: return Direction.LEFT
        if h.y > n.y: return Direction.DOWN
        return Direction.UP

    def _transform_action(self, state, action_idx):
        cur_dir = self._get_current_direction(state)
        cw = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = cw.index(cur_dir)
        if action_idx == 0: return cur_dir
        if action_idx == 1: return cw[(idx + 1) % 4]
        return cw[(idx - 1) % 4]
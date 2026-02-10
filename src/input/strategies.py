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

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake.direction)
        
        dir_straight = clock_wise[idx]
        dir_right = clock_wise[(idx + 1) % 4]
        dir_left = clock_wise[(idx - 1) % 4]
        
        def get_point_in_dir(d):
            if d == Direction.LEFT: return point_l
            if d == Direction.RIGHT: return point_r
            if d == Direction.UP: return point_u
            return point_d

        p_straight = get_point_in_dir(dir_straight)
        p_right = get_point_in_dir(dir_right)
        p_left = get_point_in_dir(dir_left)

        points_to_check = [p_straight, p_right, p_left]
        
        surroundings = []
        
        for pt in points_to_check:
            is_wall = (pt.x < 0 or pt.x >= self.config.map_width_px or 
                       pt.y < 0 or pt.y >= self.config.map_height_px)
            is_friend = False
            enemy_role = None
            
            if not is_wall:
                for s in state_dto.snakes:
                    if s.is_alive and pt in s.body:
                        if s.team_name == snake.team_name:
                            is_friend = True
                        else:
                            enemy_role = s.role
                        break
            
            surroundings.append(float(is_wall or is_friend))
            
            surroundings.append(float(enemy_role == "Harvester"))
            surroundings.append(float(enemy_role == "Hunter"))
            surroundings.append(float(enemy_role == "Defender"))

        dir_inputs = [
            float(snake.direction == Direction.LEFT),
            float(snake.direction == Direction.RIGHT),
            float(snake.direction == Direction.UP),
            float(snake.direction == Direction.DOWN)
        ]
        
        food = self._get_closest_food(snake, state_dto.foods)
        food_inputs = [
            float(food.x < head.x),
            float(food.x > head.x),
            float(food.y < head.y),
            float(food.y > head.y)
        ]
        
        ally = self._get_closest_ally(snake, state_dto.snakes)
        ally_inputs = [0.0, 0.0, 0.0] # [Ally Straight, Ally Right, Ally Left] - вектор
        
        if ally:
             dx = ally.head.x - head.x
             dy = ally.head.y - head.y
             
             dist = (dx**2 + dy**2)**0.5
             is_close = 1.0 if dist < (bs * 4) else 0.0
             ally_inputs = [is_close, 0.0, 0.0]
             
             ally_inputs = [
                 float(ally.head.x < head.x),
                 float(ally.head.x > head.x),
                 float(ally.head.y < head.y)
             ]

        ally_dirs = [0.0, 0.0, 0.0]
        if ally:
            ally_dirs[0] = float(ally.head.x < head.x)
            ally_dirs[1] = float(ally.head.x > head.x)
            ally_dirs[2] = float(ally.head.y < head.y)

        state = np.concatenate([
            surroundings, 
            dir_inputs, 
            food_inputs, 
            ally_dirs
        ])
        
        return np.array(state, dtype=float)

    def _get_closest_food(self, snake, foods):
        if not foods: return Point(-1, -1)
        closest = min(foods, key=lambda f: (snake.head.x - f.x)**2 + (snake.head.y - f.y)**2)
        return closest

    def _get_closest_ally(self, snake, snakes):
        closest = None
        min_dist = float('inf')
        for s in snakes:
            if s is not snake and s.is_alive and s.team_name == snake.team_name:
                dist = (snake.head.x - s.head.x)**2 + (snake.head.y - s.head.y)**2
                if dist < min_dist:
                    min_dist = dist
                    closest = s
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
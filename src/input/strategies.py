import torch
import math
import numpy as np
from typing import Any
from src.core.types import Direction, Point, GameStateDTO
from src.core.snake import Snake

class MultiAgentStrategy:
    config: Any

    def __init__(self, config: Any) -> None:
        self.config = config

    def get_action(self, model: Any, snake: Snake, state_dto: GameStateDTO) -> tuple[Direction, int, np.ndarray]:
        sensors: np.ndarray = self._get_sensors(snake, state_dto)
        state_tensor: torch.Tensor = torch.tensor(sensors, dtype=torch.float).unsqueeze(0)
        
        with torch.no_grad():
            prediction: torch.Tensor = model(state_tensor)
            action_idx: int = int(torch.argmax(prediction).item())
            
        move: Direction = self._transform_action(snake, action_idx)
        return move, action_idx, sensors

    def _get_sensors(self, snake: Snake, state_dto: GameStateDTO) -> np.ndarray:
        head: Point = snake.head
        bs: int = self.config.block_size
        
        clock_wise: list[Direction] = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx: int = clock_wise.index(snake.direction)
        
        dirs_to_check: list[Direction] = [
            clock_wise[idx],
            clock_wise[(idx + 1) % 4],
            clock_wise[(idx - 1) % 4]
        ]
        
        points_to_check: list[Point] = []
        for d in dirs_to_check:
            if d == Direction.LEFT: 
                points_to_check.append(Point(head.x - bs, head.y))
            elif d == Direction.RIGHT: 
                points_to_check.append(Point(head.x + bs, head.y))
            elif d == Direction.UP: 
                points_to_check.append(Point(head.x, head.y - bs))
            else: 
                points_to_check.append(Point(head.x, head.y + bs))
        
        surroundings: list[float] = []
        grid: list[list[Any]] | None = getattr(state_dto, 'grid', None)
        
        for pt in points_to_check:
            gx: int = pt.x // bs
            gy: int = pt.y // bs
            
            is_wall: bool = (gx < 0 or gx >= self.config.grid_width or 
                             gy < 0 or gy >= self.config.grid_height)
            is_friend: bool = False
            enemy_role: str | None = None
            
            if not is_wall and grid:
                cell_obj: Any = grid[gx][gy]
                if isinstance(cell_obj, Snake): 
                    if cell_obj.team_name == snake.team_name:
                        is_friend = True
                    else:
                        enemy_role = cell_obj.role
            
            surroundings.extend([
                float(is_wall or is_friend),
                float(enemy_role == "Harvester"),
                float(enemy_role == "Hunter"),
                float(enemy_role == "Defender")
            ])

        dir_inputs: list[float] = [
            float(snake.direction == Direction.LEFT),
            float(snake.direction == Direction.RIGHT),
            float(snake.direction == Direction.UP),
            float(snake.direction == Direction.DOWN)
        ]
        
        food: Point | None = self._get_closest_food(snake, state_dto.foods)
        food_inputs: list[float] = [
            float(food.x < head.x) if food else 0.0,
            float(food.x > head.x) if food else 0.0,
            float(food.y < head.y) if food else 0.0,
            float(food.y > head.y) if food else 0.0
        ]
        
        ally: Snake | None = self._get_closest_ally(snake, state_dto.snakes)
        ally_dirs: list[float] = [
            float(ally.head.x < head.x) if ally else 0.0,
            float(ally.head.x > head.x) if ally else 0.0,
            float(ally.head.y < head.y) if ally else 0.0,
            float(ally.head.y > head.y) if ally else 0.0
        ]

        enemy: Snake | None = self._get_closest_enemy(snake, state_dto.snakes)
        enemy_dirs: list[float] = [
            float(enemy.head.x < head.x) if enemy else 0.0,
            float(enemy.head.x > head.x) if enemy else 0.0,
            float(enemy.head.y < head.y) if enemy else 0.0,
            float(enemy.head.y > head.y) if enemy else 0.0
        ]

        state: list[float] = surroundings + dir_inputs + food_inputs + ally_dirs + enemy_dirs
        return np.array(state, dtype=float)

    def _get_closest_food(self, snake: Snake, foods: list[Point]) -> Point | None:
        if not foods: 
            return None
        
        closest: Point | None = None
        min_dist: float = float('inf')
        
        for f in foods:
            d: float = math.hypot(snake.head.x - f.x, snake.head.y - f.y)
            if d < min_dist:
                min_dist = d
                closest = f
                
        return closest

    def _get_closest_ally(self, snake: Snake, snakes: list[Snake]) -> Snake | None:
        closest: Snake | None = None
        min_dist: float = float('inf')
        
        for s in snakes:
            if s is not snake and s.is_alive and s.team_name == snake.team_name:
                d: float = math.hypot(snake.head.x - s.head.x, snake.head.y - s.head.y)
                if d < min_dist:
                    min_dist = d
                    closest = s
                    
        return closest

    def _get_closest_enemy(self, snake: Snake, snakes: list[Snake]) -> Snake | None:
        closest: Snake | None = None
        min_dist: float = float('inf')
        
        for s in snakes:
            if s is not snake and s.is_alive and s.team_name != snake.team_name:
                d: float = math.hypot(snake.head.x - s.head.x, snake.head.y - s.head.y)
                if d < min_dist:
                    min_dist = d
                    closest = s
                    
        return closest

    def _transform_action(self, snake: Snake, action_idx: int) -> Direction:
        clock_wise: list[Direction] = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx: int = clock_wise.index(snake.direction)
        
        if action_idx == 0:
            return clock_wise[idx]
        if action_idx == 1:
            return clock_wise[(idx + 1) % 4]
        return clock_wise[(idx - 1) % 4]
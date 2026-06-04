import torch
import math
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from src.core.types import Direction, Point, GameStateDTO
from src.core.snake import Snake

if TYPE_CHECKING:
    from src.config import GameConfig

class MultiAgentStrategy:
    def __init__(self, config: 'GameConfig') -> None:
        self.config: 'GameConfig' = config

    def get_action(self, model: torch.nn.Module, snake: Snake, state_dto: GameStateDTO) -> Tuple[Direction, int, np.ndarray]:
        sensors: np.ndarray = self._get_sensors(snake, state_dto)
        
        device = next(model.parameters()).device
        state_tensor: torch.Tensor = torch.tensor(sensors, dtype=torch.float, device=device).unsqueeze(0)
        
        with torch.no_grad():
            prediction: torch.Tensor = model(state_tensor, snake.role)
            action_idx: int = int(torch.argmax(prediction).item())
            
        move: Direction = self._transform_action(snake, action_idx)
        return move, action_idx, sensors

    def _get_sensors(self, snake: Snake, state_dto: GameStateDTO) -> np.ndarray:
        head: Point = snake.head
        bs: int = self.config.block_size
        
        clock_wise: List[int] = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx: int = clock_wise.index(snake.direction)
        
        dirs_to_check: List[int] = [
            clock_wise[idx],
            clock_wise[(idx + 1) % 4],
            clock_wise[(idx - 1) % 4]
        ]
        
        points_to_check: List[Point] = []
        for d in dirs_to_check:
            if d == Direction.LEFT: 
                points_to_check.append(Point(head.x - bs, head.y))
            elif d == Direction.RIGHT: 
                points_to_check.append(Point(head.x + bs, head.y))
            elif d == Direction.UP: 
                points_to_check.append(Point(head.x, head.y - bs))
            else: 
                points_to_check.append(Point(head.x, head.y + bs))
        
        surroundings: List[float] = []
        grid: Optional[List[List[Optional[str | Snake]]]] = getattr(state_dto, 'grid', None)
        
        for pt in points_to_check:
            gx: int = pt.x // bs
            gy: int = pt.y // bs
            
            is_wall: bool = (gx < 0 or gx >= self.config.grid_width or 
                             gy < 0 or gy >= self.config.grid_height)
            is_friend: bool = False
            enemy_role: Optional[str] = None
            cell_obj: Optional[str | Snake] = grid[gx][gy] if not is_wall and grid else None

            if isinstance(cell_obj, Snake): 
                is_friend = (cell_obj.team_name == snake.team_name)
                enemy_role = None if is_friend else cell_obj.role
            
            surroundings.extend([
                float(is_wall or is_friend),
                float(enemy_role == "Harvester"),
                float(enemy_role == "Hunter"),
                float(enemy_role == "Defender")
            ])

        dir_inputs: List[float] = [
            float(snake.direction == Direction.LEFT),
            float(snake.direction == Direction.RIGHT),
            float(snake.direction == Direction.UP),
            float(snake.direction == Direction.DOWN)
        ]
        
        food: Optional[Point] = self._get_closest_food(snake, state_dto.foods)
        food_inputs: List[float] = [
            float(food.x < head.x) if food else 0.0,
            float(food.x > head.x) if food else 0.0,
            float(food.y < head.y) if food else 0.0,
            float(food.y > head.y) if food else 0.0
        ]
        
        ally: Optional[Snake] = self._get_closest_ally(snake, state_dto.snakes)
        ally_dirs: List[float] = [
            float(ally.head.x < head.x) if ally else 0.0,
            float(ally.head.x > head.x) if ally else 0.0,
            float(ally.head.y < head.y) if ally else 0.0,
            float(ally.head.y > head.y) if ally else 0.0
        ]

        enemy: Optional[Snake] = self._get_closest_enemy(snake, state_dto.snakes)
        enemy_dirs: List[float] = [
            float(enemy.head.x < head.x) if enemy else 0.0,
            float(enemy.head.x > head.x) if enemy else 0.0,
            float(enemy.head.y < head.y) if enemy else 0.0,
            float(enemy.head.y > head.y) if enemy else 0.0
        ]

        state: List[float] = surroundings + dir_inputs + food_inputs + ally_dirs + enemy_dirs
        return np.array(state, dtype=float)

    def _get_closest_food(self, snake: Snake, foods: List[Point]) -> Optional[Point]:
        if not foods: 
            return None
        
        closest: Optional[Point] = None
        min_dist: float = float('inf')
        
        for f in foods:
            d: float = math.hypot(snake.head.x - f.x, snake.head.y - f.y)
            if d < min_dist:
                min_dist = d
                closest = f
                
        return closest

    def _get_closest_ally(self, snake: Snake, snakes: List[Snake]) -> Optional[Snake]:
        closest: Optional[Snake] = None
        min_dist: float = float('inf')
        
        for s in snakes:
            if s is snake or not s.is_alive or s.team_name != snake.team_name:
                continue
            d: float = math.hypot(snake.head.x - s.head.x, snake.head.y - s.head.y)
            if d < min_dist:
                min_dist = d
                closest = s
                    
        return closest

    def _get_closest_enemy(self, snake: Snake, snakes: List[Snake]) -> Optional[Snake]:
        closest: Optional[Snake] = None
        min_dist: float = float('inf')
        
        for s in snakes:
            if s is snake or not s.is_alive or s.team_name == snake.team_name:
                continue
            d: float = math.hypot(snake.head.x - s.head.x, snake.head.y - s.head.y)
            if d < min_dist:
                min_dist = d
                closest = s
                    
        return closest

    def _transform_action(self, snake: Snake, action_idx: int) -> int:
        clock_wise: List[int] = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx: int = clock_wise.index(snake.direction)
        
        if action_idx == 0:
            return clock_wise[idx]
        elif action_idx == 1:
            return clock_wise[(idx + 1) % 4]
        return clock_wise[(idx - 1) % 4]
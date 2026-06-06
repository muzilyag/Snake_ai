from collections import namedtuple
from enum import IntEnum
from typing import Any

Point = namedtuple('Point', 'x, y')

class Direction(IntEnum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class DeathReason(IntEnum):
    ALIVE = 0
    WALL_COLLISION = 1
    SELF_COLLISION = 2
    ENEMY_COLLISION = 3
    STARVATION = 4

class Snake:
    def __init__(self, agent_id: str, x: int, y: int, team_config: Any, role: str, config: Any) -> None:
        self.id: str = agent_id
        self.head: Point = Point(x, y)
        self.direction: Direction = Direction.RIGHT
        self.team_name: str = team_config.name
        self.color: tuple[int, int, int] = team_config.color
        self.role: str = role
        
        role_cfg: Any = config.role_settings.get(role, config.role_settings["Harvester"])
        init_len: int = getattr(role_cfg, 'initial_length', config.initial_snake_length)
        
        self.body: list[Point] = [Point(x - (i * config.block_size), y) for i in range(init_len)]
        self.max_hp: float = role_cfg.max_hp
        self.hp: float = role_cfg.start_hp
        self.damage_dealt: float = role_cfg.damage_dealt
        self.victim_return_damage: float = role_cfg.victim_return_damage
        self.self_damage: float = role_cfg.self_damage
        self.collision_survivable: bool = role_cfg.collision_survivable
        
        self.is_alive: bool = True
        self.score: int = 0
        self.steps_alive: int = 0
        self.deaths: int = 0
        
        self.brain_type: str = team_config.brain_type
        self.pending_reward: float = 0.0

    def set_action(self, action_idx: int) -> None:
        if not self.is_alive:
            return
            
        clock_wise: list[int] = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx: int = clock_wise.index(self.direction)
        
        if action_idx == 0:
            self.direction = Direction(clock_wise[idx])
        elif action_idx == 1:
            self.direction = Direction(clock_wise[(idx + 1) % 4])
        elif action_idx == 2:
            self.direction = Direction(clock_wise[(idx - 1) % 4])

    def move_head_prediction(self, block_size: int) -> Point:
        x: int = self.head.x
        y: int = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += block_size
        elif self.direction == Direction.LEFT:
            x -= block_size
        elif self.direction == Direction.DOWN:
            y += block_size
        elif self.direction == Direction.UP:
            y -= block_size
            
        return Point(x, y)

    def commit_move(self, new_head: Point) -> None:
        self.head = new_head
        self.steps_alive += 1

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)
        if self.hp <= 0.0:
            self.is_alive = False

    def heal(self, amount: float) -> None:
        if not self.is_alive:
            return
        self.hp = min(self.max_hp, self.hp + amount)
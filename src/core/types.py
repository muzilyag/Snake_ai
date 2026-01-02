from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass
class GameStateDTO:
    """Слепок состояния игры для отправки в UI или AI"""
    snake_body: List[Point]
    food: Point
    score: int
    is_game_over: bool
    iteration: int
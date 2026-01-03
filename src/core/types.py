from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict

Point = namedtuple('Point', 'x, y')

class Direction:
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

@dataclass
class TeamStats:
    record: int = 0
    deaths: int = 0
    current_score: int = 0
    generation: int = 1

@dataclass
class GlobalStats:
    total_iterations: int
    total_time: float
    total_deaths: int

@dataclass
class GameStateDTO:
    snakes: List
    foods: List[Point]
    global_stats: GlobalStats
    team_stats: Dict[str, TeamStats]
    is_game_over: bool
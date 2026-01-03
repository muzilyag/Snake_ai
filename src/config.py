from dataclasses import dataclass, field
from typing import List

@dataclass
class Color:
    BACKGROUND = (255, 255, 255)
    GRID = (230, 230, 230)
    TEXT = (0, 0, 0)
    SIDEBAR_BG = (245, 245, 245)
    FOOD = (200, 0, 0)

@dataclass
class TeamConfig:
    name: str
    count: int
    color: tuple
    brain_type: str = "RL"
    reward_mode: str = "linear"

@dataclass
class GameConfig:
    grid_width: int = 40
    grid_height: int = 40
    block_size: int = 20
    sidebar_width: int = 300
    fps_train: int = 0
    fps_watch: int = 15
    food_count: int = 6
    initial_snake_length: int = 3
    
    dynamic_death_penalty: float = -10.0
    dynamic_food_reward: float = 15.0
    
    teams: List[TeamConfig] = field(default_factory=lambda: [
        TeamConfig("Green Linear", 2, (0, 180, 0), "RL", "linear"),
        TeamConfig("Blue Dynamic", 2, (0, 0, 180), "RL", "dynamic")
    ])
    
    colors: Color = field(default_factory=Color)

    def __post_init__(self):
        self.map_width_px = self.grid_width * self.block_size
        self.map_height_px = self.grid_height * self.block_size
        self.window_width = self.map_width_px + self.sidebar_width
        self.window_height = self.map_height_px

SETTINGS = GameConfig()
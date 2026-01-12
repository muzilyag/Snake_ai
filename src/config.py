from dataclasses import dataclass, field
from typing import List

@dataclass
class RewardConfig:
    starvation_penalty: float = -100.0
    food_reward_base: float = 20.0
    death_penalty_base: float = -20.0
    
    # ! Linear params
    step_closer: float = 1.0
    step_farther: float = -1.5
    
    # ! Dynamic params
    dynamic_velocity_multiplier: float = 2.0 
    wall_proximity_penalty: float = -0.5
    
    # ! New params for smarter AI
    danger_sensing_penalty: float = -0.5
    survival_bonus: float = 0.01

@dataclass
class Color:
    BACKGROUND = (255, 255, 255)
    GRID = (230, 230, 230)
    TEXT = (0, 0, 0)
    SIDEBAR_BG = (245, 245, 245)
    FOOD = (200, 0, 0)

    # ? Dark Mode
    # BACKGROUND = (16, 16, 30)
    # GRID = (30, 30, 30)
    # TEXT = (200, 200, 200)
    # SIDEBAR_BG = (0, 0, 0)

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
    max_steps_without_food: int = 200
    
    # ! Analytics
    stats_interval: int = 1000
    
    rewards: RewardConfig = field(default_factory=RewardConfig)
    teams: List[TeamConfig] = field(default_factory=lambda: [
        TeamConfig("Green Linear", 2, (0, 180, 0), "RL", "linear"),
        TeamConfig("Blue Dynamic", 2, (0, 0, 180), "RL", "dynamic"),
    ])
    colors: Color = field(default_factory=Color)

    def __post_init__(self):
        self.map_width_px = self.grid_width * self.block_size
        self.map_height_px = self.grid_height * self.block_size
        self.window_width = self.map_width_px + self.sidebar_width
        self.window_height = self.map_height_px

SETTINGS = GameConfig()
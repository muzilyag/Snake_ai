from dataclasses import dataclass, field
from typing import List, Dict

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
    agent_roles: List[str] = field(default_factory=list)

@dataclass
class GameConfig:
    grid_width: int = 40
    grid_height: int = 40
    block_size: int = 20
    sidebar_width: int = 300
    fps_train: int = 0
    fps_watch: int = 15
    food_count: int = 6
    initial_snake_length: int = 1
    max_steps_without_food: int = 200
    
    stats_interval: int = 1000
    
    reward_presets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Harvester": {
            "food": 20.0,
            "starve": -100.0,
            "death": -20.0,
            "step_closer_food": 1.0,
            "step_farther_food": -1.5,
            "step_closer_enemy": 0.0,
            "step_farther_enemy": 0.0,
            "wall_penalty": -0.5,
            "idle_penalty": -0.05
        },
        "Hunter": {
            "food": 5.0,
            "starve": -50.0,
            "death": -50.0,
            "step_closer_food": 0.1,
            "step_farther_food": -0.1,
            "step_closer_enemy": 7.5,
            "step_farther_enemy": -5.0,
            "wall_penalty": -0.5,
            "idle_penalty": -0.05
        }
    })

    teams: List[TeamConfig] = field(default_factory=lambda: [
        TeamConfig(
            name="Green Squad", 
            count=2, 
            color=(0, 180, 0), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Harvester", "Harvester"]
        ),
        TeamConfig(
            name="Blue Squad", 
            count=2, 
            color=(0, 0, 180), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Harvester", "Hunter"]
        ),
    ])
    colors: Color = field(default_factory=Color)

    def __post_init__(self):
        self.map_width_px = self.grid_width * self.block_size
        self.map_height_px = self.grid_height * self.block_size
        self.window_width = self.map_width_px + self.sidebar_width
        self.window_height = self.map_height_px
        
        for team in self.teams:
            if not team.agent_roles:
                team.agent_roles = ["Harvester"] * team.count
            elif len(team.agent_roles) < team.count:
                team.agent_roles.extend(["Harvester"] * (team.count - len(team.agent_roles)))

SETTINGS = GameConfig()
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
class RoleConfig:
    max_hp: float = 100.0
    start_hp: float = 100.0
    damage_dealt: float = 10.0      # Урон врагу при ударе
    self_damage: float = 100.0      # Урон себе при ударе (100 = смерть)
    collision_survivable: bool = False

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
    
    # HP System
    hp_decay_per_step: float = 0.5
    food_heal_amount: float = 30.0
    
    stats_interval: int = 1000
    
    role_settings: Dict[str, RoleConfig] = field(default_factory=lambda: {
        "Harvester": RoleConfig(
            max_hp=100.0,
            start_hp=100.0,
            damage_dealt=10.0,
            self_damage=100.0,
            collision_survivable=False
        ),
        "Hunter": RoleConfig(
            max_hp=150.0,
            start_hp=150.0,
            damage_dealt=50.0,
            self_damage=15.0,
            collision_survivable=True
        )
    })
    
    reward_presets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Harvester": {
            "food": 20.0,
            "starve": -50.0,
            "death": -20.0,
            "kill_reward": 5.0,
            "friendly_fire": -50.0,
            "step_closer_food": 1.0,
            "step_farther_food": -1.5,
            "step_closer_enemy": 0.0,
            "step_farther_enemy": 0.0,
            "wall_penalty": -0.5,
            "idle_penalty": -0.05
        },
        "Hunter": {
            "food": 5.0,
            "starve": -20.0,
            "death": -50.0,
            "kill_reward": 50.0,
            "damage_dealt_reward": 1.0,
            "friendly_fire": -100.0,
            "step_closer_food": 0.1,
            "step_farther_food": -0.1,
            "step_closer_enemy": 2.5,
            "step_farther_enemy": -3.0,
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
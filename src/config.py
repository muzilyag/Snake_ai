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
    damage_dealt: float = 10.0
    victim_return_damage: float = 0.0
    self_damage: float = 100.0
    collision_survivable: bool = False
    initial_length: int = 1

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
    
    hp_decay_per_step: float = 0.5
    food_heal_amount: float = 30.0
    
    stats_interval: int = 1000
    
    role_settings: Dict[str, RoleConfig] = field(default_factory=lambda: {
        "Harvester": RoleConfig(
            max_hp=100.0,
            start_hp=100.0,
            damage_dealt=10.0,
            victim_return_damage=0.0,
            self_damage=100.0,
            collision_survivable=False,
            initial_length=1
        ),
        "Hunter": RoleConfig(
            max_hp=150.0,
            start_hp=150.0,
            damage_dealt=50.0,
            victim_return_damage=10.0,
            self_damage=15.0,
            collision_survivable=True,
            initial_length=1
        ),
        "Defender": RoleConfig(
            max_hp=200.0,
            start_hp=200.0,
            damage_dealt=20.0,
            victim_return_damage=60.0,
            self_damage=100.0,
            collision_survivable=False,
            initial_length=6
        )
    })
    
    reward_presets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Harvester": {
            "food": 30.0,
            "starve": -50.0,
            "death": -20.0,
            "kill_generic": 5.0, 
            "kill_harvester": 5.0,
            "kill_hunter": 10.0,
            "kill_defender": 10.0,
            "friendly_fire": -50.0,
            "step_closer_food": 2.0,
            "step_farther_food": -2.5,
            "wall_penalty": -1.0,
            "idle_penalty": -0.1
        },
        "Hunter": {
            "food": 15.0,           
            "starve": -40.0,
            "death": -50.0,
            "kill_harvester": 150.0,
            "kill_hunter": 100.0,   
            "kill_defender": -50.0, 
            "damage_dealt_reward": 5.0,
            "friendly_fire": -100.0,
            "step_closer_food": 0.5, 
            "step_farther_food": -0.2,
            "step_closer_enemy_harvester": 4.0, 
            "step_closer_enemy_hunter": 2.0,
            "step_closer_enemy_defender": -2.0,
            "wall_penalty": -0.5,
            "idle_penalty": -0.1
        },
        "Defender": {
            "food": 10.0,
            "starve": -50.0,
            "death": -30.0,
            "kill_harvester": 30.0,
            "kill_hunter": 80.0, 
            "kill_defender": 20.0,
            "damage_dealt_reward": 15.0,
            "friendly_fire": -50.0,
            "step_closer_food": 0.5,
            "step_closer_enemy_hunter": 3.0,
            "step_closer_enemy_harvester": 0.0,
            "step_closer_team": 1.5, 
            "step_farther_team": -1.5,
            "wall_penalty": -0.2,
            "idle_penalty": 0.0
        }
    })

    teams: List[TeamConfig] = field(default_factory=lambda: [
        # TeamConfig(
        #     name="Green Squad", 
        #     count=3, 
        #     color=(0, 180, 0), 
        #     brain_type="RL", 
        #     reward_mode="linear",
        #     agent_roles=["Harvester", "Harvester", "Defender"]
        # ),
        # TeamConfig(
        #     name="Blue Squad", 
        #     count=3, 
        #     color=(0, 0, 180), 
        #     brain_type="RL", 
        #     reward_mode="linear",
        #     agent_roles=["Harvester", "Harvester", "Hunter"]
        # ),
        # TeamConfig(
        #     name="Red Squad", 
        #     count=3, 
        #     color=(180, 0, 0), 
        #     brain_type="RL", 
        #     reward_mode="linear",
        #     agent_roles=["Harvester", "Harvester", "Harvester"]
        # )
        TeamConfig(
            name="Green Squad", 
            count=2, 
            color=(0, 180, 0), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Harvester", "Defender"]
        ),
        TeamConfig(
            name="Blue Squad", 
            count=2, 
            color=(0, 0, 180), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Harvester", "Hunter"]
        ),
        TeamConfig(
            name="Red Squad", 
            count=2, 
            color=(180, 0, 0), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Harvester", "Harvester"]
        ),
        TeamConfig(
            name="Orange Squad", 
            count=2, 
            color=(255, 150, 0), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Defender", "Defender"]
        ),
        TeamConfig(
            name="Purple Squad", 
            count=2, 
            color=(128, 0, 128), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Hunter", "Hunter"]
        ),
        TeamConfig(
            name="Yellow Squad", 
            count=2, 
            color=(247, 215, 32), 
            brain_type="RL", 
            reward_mode="linear",
            agent_roles=["Hunter", "Defender"]
        )
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
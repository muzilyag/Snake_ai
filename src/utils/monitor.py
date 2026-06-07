import time
from typing import Any, Dict, Optional
from src.env.entity import DeathReason
import time

class GameMonitor:
    def __init__(self, config: Any) -> None:
        self.config: Any = config
        self.start_time: float = time.time()
        self.global_deaths: int = 0
        self.global_record: int = 0
        self.iteration: int = 0
        
        self.team_stats: Dict[str, Dict[str, int]] = {
            team.name: {'record': 0, 'deaths': 0, 'current_score': 0, 'apples': 0}
            for team in config.teams
        }
        
        self.agent_states: Dict[str, Dict[str, Any]] = {}

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def update(self, iteration: int, infos: Dict[str, Any]) -> None:
        self.iteration = iteration
        
        for t_name in self.team_stats:
            self.team_stats[t_name]['current_score'] = 0

        for agent_id, info in infos.items():
            team_name: str = agent_id.split('_')[0]
            
            self.agent_states[agent_id] = {
                'hp': info.get('hp', 0),
                'max_hp': info.get('max_hp', 100),
                'score': info.get('score', 0)
            }
            
            self.team_stats[team_name]['current_score'] += int(info.get('score', 0))

            event: Optional[str] = info.get('event')
            if event == 'food':
                self.team_stats[team_name]['apples'] += 1
                
            if 'death_reason' in info:
                self.global_deaths += 1
                self.team_stats[team_name]['deaths'] += 1

        for stats in self.team_stats.values():
            if stats['current_score'] > stats['record']:
                stats['record'] = stats['current_score']
            if stats['record'] > self.global_record:
                self.global_record = stats['record']

    def get_state(self) -> dict:
        return {
            "iteration": getattr(self, 'iteration', 0),
            "global_deaths": getattr(self, 'global_deaths', 0),
            "global_record": getattr(self, 'global_record', 0),
            "elapsed_time": getattr(self, 'elapsed_time', 0.0),
            "team_stats": getattr(self, 'team_stats', {})
        }

    def load_state(self, state: dict) -> None:
        self.iteration = state.get("iteration", 0)
        self.global_deaths = state.get("global_deaths", 0)
        self.global_record = state.get("global_record", 0)
        self.team_stats = state.get("team_stats", getattr(self, 'team_stats', {}))
        
        saved_elapsed = state.get("elapsed_time", 0.0)
        if hasattr(self, 'start_time'):
            self.start_time = time.time() - saved_elapsed
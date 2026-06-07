import csv
import os, pandas as pd, csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from src.env.entity import DeathReason
from src.utils.monitor import GameMonitor

class MetricsLogger:
    def __init__(self, config: Any) -> None:
        self.config: Any = config
        self.stats_dir: Path = Path("stats")
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename: Path = self.stats_dir / f"session_{timestamp}.csv"
        self.current_csv_path: str = str(self.csv_filename)
        self.interval_causes: Dict[str, Dict[DeathReason, int]] = self._init_interval_causes()
        self.interval_apples: Dict[str, int] = {team.name: 0 for team in self.config.teams}
        self.interval_deaths: Dict[str, int] = {team.name: 0 for team in self.config.teams}
        self._init_csv()

    def get_current_filename(self) -> str:
        return str(self.csv_filename)

    def _init_csv(self) -> None:
        headers: List[str] = [
            "Iteration", "Team", "Apples", "Deaths", "Ratio_AD",
            "Cause_Wall", "Cause_Self", "Cause_Enemy", "Cause_Starve"
        ]
        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)

    def _init_interval_causes(self) -> Dict[str, Dict[DeathReason, int]]:
        return {
            team.name: {
                DeathReason.WALL_COLLISION: 0,
                DeathReason.SELF_COLLISION: 0,
                DeathReason.ENEMY_COLLISION: 0,
                DeathReason.STARVATION: 0
            } for team in self.config.teams
        }

    def log_step(self, infos: Dict[str, Any], monitor: GameMonitor) -> None:
        for agent_id, info in infos.items():
            team_name: str = agent_id.split('_')[0]
            if 'death_reason' in info:
                reason: DeathReason = info['death_reason']
                if reason in self.interval_causes[team_name]:
                    self.interval_causes[team_name][reason] += 1
                self.interval_deaths[team_name] += 1
            if info.get('event') == 'food':
                self.interval_apples[team_name] += 1

        if monitor.iteration > 0 and monitor.iteration % self.config.stats_interval == 0:
            self._finalize_interval(monitor.iteration)

    def _finalize_interval(self, iteration: int) -> None:
        rows: List[List[Any]] = []
        print(f"\n[ANALYTICS] Iteration {iteration} Summary:")
        
        for team_name in self.interval_apples.keys():
            apples: int = self.interval_apples[team_name]
            deaths: int = self.interval_deaths[team_name]
            ratio: float = apples / deaths if deaths > 0 else float(apples)
            causes: Dict[DeathReason, int] = self.interval_causes[team_name]
            
            print(f"  > Team {team_name}: Apples={apples}, Deaths={deaths}, A/D Ratio={ratio:.2f}")
            
            rows.append([
                iteration, team_name, apples, deaths, round(ratio, 4),
                causes[DeathReason.WALL_COLLISION], causes[DeathReason.SELF_COLLISION],
                causes[DeathReason.ENEMY_COLLISION], causes[DeathReason.STARVATION]
            ])
            
        with open(self.csv_filename, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)
            
        self.interval_causes = self._init_interval_causes()
        self.interval_apples = {team.name: 0 for team in self.config.teams}
        self.interval_deaths = {team.name: 0 for team in self.config.teams}

    def set_csv_file(self, file_path: str, current_iteration: int) -> None:        
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
            
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = df[df['Iteration'] <= current_iteration]
            df.to_csv(file_path, index=False)
            
        self.current_csv_path = file_path
        
        self.file = open(file_path, 'a', newline='')
        if hasattr(self, 'headers'):
            self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
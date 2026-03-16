import csv
import os
from datetime import datetime
from typing import Any
from src.core.types import DeathReason

class AnalyticsEngine:
    def __init__(self, config: Any) -> None:
        self.config: Any = config
        self.history: list[Any] = []
        self.teams: list[str] = [t.name for t in config.teams]
        self.current_interval_stats: dict[str, dict[str, Any]] = self._init_interval_stats()
        
        self.stats_dir: str = "stats"
        if not os.path.exists(self.stats_dir): os.makedirs(self.stats_dir)
            
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename: str = os.path.join(self.stats_dir, f"session_{timestamp}.csv")
        self._init_csv()

    def get_current_filename(self) -> str:
        return self.csv_filename

    def _init_csv(self) -> None:
        headers: list[str] = [
            "Iteration", 
            "Team", 
            "Apples", 
            "Deaths", 
            "Ratio_AD",
            "Cause_Wall", 
            "Cause_Self", 
            "Cause_Enemy", 
            "Cause_Starve"
        ]
        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _init_interval_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for team in self.config.teams:
            stats[team.name] = {
                'apples': 0,
                'deaths': 0,
                'causes': {
                    DeathReason.WALL: 0,
                    DeathReason.SELF_COLLISION: 0,
                    DeathReason.ENEMY_COLLISION: 0,
                    DeathReason.STARVATION: 0
                }
            }
        return stats

    def log_food(self, team_name: str) -> None:
        if team_name in self.current_interval_stats:
            self.current_interval_stats[team_name]['apples'] += 1

    def log_death(self, team_name: str, reason: DeathReason) -> None:
        if team_name in self.current_interval_stats:
            self.current_interval_stats[team_name]['deaths'] += 1
            if reason in self.current_interval_stats[team_name]['causes']:
                self.current_interval_stats[team_name]['causes'][reason] += 1

    def update(self, current_iteration: int) -> None:
        if current_iteration > 0 and current_iteration % self.config.stats_interval == 0:
            self._finalize_interval(current_iteration)

    def _finalize_interval(self, iteration: int) -> None:
        self._write_to_csv(iteration)
        
        print(f"\n[ANALYTICS] Iteration {iteration} Summary:")
        for team_name, stats in self.current_interval_stats.items():
            apples: int = stats['apples']
            deaths: int = stats['deaths']
            ratio: float = apples / deaths if deaths > 0 else float(apples)
            print(f"  > Team {team_name}: Apples={apples}, Deaths={deaths}, A/D Ratio={ratio:.2f}")

        self.current_interval_stats = self._init_interval_stats()

    def _write_to_csv(self, iteration: int) -> None:
        rows: list[list[Any]] = []
        for team_name, stats in self.current_interval_stats.items():
            apples: int = stats['apples']
            deaths: int = stats['deaths']
            ratio: float = apples / deaths if deaths > 0 else float(apples)
            causes: dict[DeathReason, int] = stats['causes']
            
            row: list[Any] = [
                iteration,
                team_name,
                apples,
                deaths,
                round(ratio, 4),
                causes[DeathReason.WALL],
                causes[DeathReason.SELF_COLLISION],
                causes[DeathReason.ENEMY_COLLISION],
                causes[DeathReason.STARVATION]
            ]
            rows.append(row)
            
        with open(self.csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
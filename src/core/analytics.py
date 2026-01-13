import csv
import os
from datetime import datetime
from src.core.types import DeathReason

class AnalyticsEngine:
    def __init__(self, config):
        self.config = config
        self.history = []
        self.teams = [t.name for t in config.teams]
        self.current_interval_stats = self._init_interval_stats()
        
        self.stats_dir = "stats"
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.stats_dir, f"session_{timestamp}.csv")
        self._init_csv()

    def get_current_filename(self):
        return self.csv_filename

    def _init_csv(self):
        headers = [
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
        with open(self.csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _init_interval_stats(self):
        stats = {}
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

    def log_food(self, team_name: str):
        if team_name in self.current_interval_stats:
            self.current_interval_stats[team_name]['apples'] += 1

    def log_death(self, team_name: str, reason: int):
        if team_name in self.current_interval_stats:
            self.current_interval_stats[team_name]['deaths'] += 1
            if reason in self.current_interval_stats[team_name]['causes']:
                self.current_interval_stats[team_name]['causes'][reason] += 1

    def update(self, current_iteration: int):
        if current_iteration > 0 and current_iteration % self.config.stats_interval == 0:
            self._finalize_interval(current_iteration)

    def _finalize_interval(self, iteration: int):
        self._write_to_csv(iteration)
        
        print(f"\n[ANALYTICS] Iteration {iteration} Summary:")
        for team_name, stats in self.current_interval_stats.items():
            apples = stats['apples']
            deaths = stats['deaths']
            ratio = apples / deaths if deaths > 0 else apples
            print(f"  > Team {team_name}: Apples={apples}, Deaths={deaths}, A/D Ratio={ratio:.2f}")

        self.current_interval_stats = self._init_interval_stats()

    def _write_to_csv(self, iteration):
        rows = []
        for team_name, stats in self.current_interval_stats.items():
            apples = stats['apples']
            deaths = stats['deaths']
            ratio = apples / deaths if deaths > 0 else apples
            causes = stats['causes']
            
            row = [
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
            
        with open(self.csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
import csv
import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons, TextBox, Button
from typing import Any

try:
    from src.config import SETTINGS
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.config import SETTINGS

class SnakePlotter:
    filename: str
    data: dict[str, dict[str, list[tuple[int, float]]]]
    teams: list[str]
    metrics: list[str]
    selected_teams: dict[str, bool]
    selected_metric: str
    show_raw: bool
    show_trend: bool
    min_iter: int
    max_iter: float
    fig: plt.Figure
    ax: plt.Axes

    def __init__(self, csv_filename: str) -> None:
        self.filename = csv_filename
        self.data = {}
        self.teams = []
        self.metrics = []
        
        self._load_data()
        
        self.selected_teams = {t: True for t in self.teams}
        self.selected_metric = self.metrics[0] if self.metrics else "Apples"
        self.show_raw = True
        self.show_trend = True
        
        self.min_iter = 0
        self.max_iter = float('inf')
        
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(left=0.25, bottom=0.2, right=0.95, top=0.9)
        
        self._init_widgets()
        self._draw_plot()
        
        plt.show()

    def _load_data(self) -> None:
        if not os.path.exists(self.filename):
            return

        with open(self.filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return
                
            self.metrics = [field for field in reader.fieldnames if field not in ['Iteration', 'Team']]
            
            for row in reader:
                try:
                    iteration: int = int(row['Iteration'])
                    team: str = row['Team']
                    
                    if team not in self.teams:
                        self.teams.append(team)
                        self.data[team] = {m: [] for m in self.metrics}
                        
                    for m in self.metrics:
                        val: float = float(row[m])
                        self.data[team][m].append((iteration, val))
                except (ValueError, KeyError):
                    continue

    def _get_team_color(self, team_name: str) -> tuple[float, float, float]:
        team_conf: Any = next((t for t in SETTINGS.teams if t.name == team_name), None)
        if team_conf:
            return tuple(c / 255.0 for c in team_conf.color)
        return tuple(np.random.rand(3).tolist())

    def _calculate_trend(self, ys: np.ndarray, window_fraction: float = 0.1) -> tuple[np.ndarray, int]:
        if len(ys) < 5:
            return ys, 1
        
        window: int = max(3, int(len(ys) * window_fraction))
        weights: np.ndarray = np.ones(window) / window
        trend: np.ndarray = np.convolve(ys, weights, mode='valid')
        return trend, window

    def _draw_plot(self) -> None:
        self.ax.clear()
        self.ax.set_title(f"Metric: {self.selected_metric}")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, linestyle='--', alpha=0.4)

        has_data: bool = False
        
        for team in self.teams:
            if not self.selected_teams.get(team, False):
                continue
            
            points: list[tuple[int, float]] = self.data.get(team, {}).get(self.selected_metric, [])
            filtered: list[tuple[int, float]] = [(i, v) for i, v in points if self.min_iter <= i <= self.max_iter]
            
            if not filtered:
                continue
                
            xs_tuple, ys_tuple = zip(*filtered)
            xs: np.ndarray = np.array(xs_tuple)
            ys: np.ndarray = np.array(ys_tuple)
            
            color: tuple[float, float, float] = self._get_team_color(team)
            
            if self.show_raw:
                alpha: float = 0.3 if self.show_trend else 1.0
                self.ax.plot(xs, ys, marker='o', markersize=3, linestyle='-', 
                             linewidth=1, color=color, alpha=alpha, label=f"{team} (Raw)")
            
            if self.show_trend and len(ys) > 1:
                trend_ys, window = self._calculate_trend(ys)
                start_idx: int = (window - 1) // 2
                end_idx: int = start_idx + len(trend_ys)
                trend_xs: np.ndarray = xs[start_idx:end_idx]
                
                if len(trend_xs) > len(trend_ys): 
                    trend_xs = trend_xs[:len(trend_ys)]
                
                self.ax.plot(trend_xs, trend_ys, linestyle='--', linewidth=2.5, 
                             color=color, label=f"{team} (Trend)")

            has_data = True
        
        if has_data:
            self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        self.fig.canvas.draw_idle()

    def _on_team_toggle(self, label: str) -> None:
        self.selected_teams[label] = not self.selected_teams[label]
        self._draw_plot()

    def _on_view_toggle(self, label: str) -> None:
        if label == "Show Raw":
            self.show_raw = not self.show_raw
        elif label == "Show Trend":
            self.show_trend = not self.show_trend
        self._draw_plot()

    def _on_metric_change(self, label: str) -> None:
        self.selected_metric = label
        self._draw_plot()

    def _on_range_update(self, event: Any) -> None:
        try:
            self.min_iter = int(self.box_min.text)
        except ValueError:
            self.min_iter = 0
            
        try:
            if self.box_max.text.lower() == "max":
                self.max_iter = float('inf')
            else:
                self.max_iter = float(self.box_max.text)
        except ValueError:
            self.max_iter = float('inf')
            
        self._draw_plot()

    def _init_widgets(self) -> None:
        ax_teams: plt.Axes = plt.axes([0.02, 0.55, 0.15, 0.3])
        self.check_teams = CheckButtons(ax_teams, self.teams, [True] * len(self.teams))
        self.check_teams.on_clicked(self._on_team_toggle)
        ax_teams.set_title("Teams", fontsize=10)

        ax_view: plt.Axes = plt.axes([0.02, 0.40, 0.15, 0.1])
        self.check_view = CheckButtons(ax_view, ["Show Raw", "Show Trend"], [True, True])
        self.check_view.on_clicked(self._on_view_toggle)
        ax_view.set_title("View Options", fontsize=10)

        ax_metrics: plt.Axes = plt.axes([0.02, 0.15, 0.15, 0.2])
        self.radio_metrics = RadioButtons(ax_metrics, self.metrics, active=0)
        self.radio_metrics.on_clicked(self._on_metric_change)
        ax_metrics.set_title("Metrics", fontsize=10)

        ax_min: plt.Axes = plt.axes([0.30, 0.05, 0.1, 0.04])
        ax_max: plt.Axes = plt.axes([0.45, 0.05, 0.1, 0.04])
        ax_btn: plt.Axes = plt.axes([0.60, 0.05, 0.1, 0.04])

        self.box_min = TextBox(ax_min, "Min: ", initial="0")
        self.box_max = TextBox(ax_max, "Max: ", initial="Max")
        self.btn_update = Button(ax_btn, "Update Range")
        self.btn_update.on_clicked(self._on_range_update)

def run_standalone() -> None:
    stats_dir: str = "stats"
    if os.path.exists(stats_dir):
        return None
    if os.path.exists(os.path.join("..", "stats")):
            stats_dir = os.path.join("..", "stats")
    else:
        return

    files: list[str] = glob.glob(os.path.join(stats_dir, "*.csv"))
    files.sort(key=os.path.getmtime, reverse=True)
    
    if not files:
        return

    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    
    try:
        choice: str = input(f"Select file [0-{len(files)-1}]: ")
        idx: int = int(choice)
        filename: str = files[idx]
        SnakePlotter(filename)
    except (ValueError, IndexError):
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        SnakePlotter(sys.argv[1])
    else:
        run_standalone()
import csv
import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons, TextBox, Button

try:
    from src.config import SETTINGS
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.config import SETTINGS

class SnakePlotter:
    def __init__(self, csv_filename):
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
        self.max_iter = 1000000000
        
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(left=0.25, bottom=0.2, right=0.95, top=0.9)
        
        self._init_widgets()
        self._draw_plot()
        
        plt.show()

    def _load_data(self):
        if not os.path.exists(self.filename):
            print(f"File not found: {self.filename}")
            return

        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            self.metrics = [field for field in reader.fieldnames if field not in ['Iteration', 'Team']]
            
            for row in reader:
                try:
                    iteration = int(row['Iteration'])
                    team = row['Team']
                    
                    if team not in self.teams:
                        self.teams.append(team)
                        self.data[team] = {m: [] for m in self.metrics}
                        
                    for m in self.metrics:
                        val = float(row[m])
                        self.data[team][m].append((iteration, val))
                except ValueError:
                    continue

    def _get_team_color(self, team_name):
        team_conf = next((t for t in SETTINGS.teams if t.name == team_name), None)
        if team_conf:
            return tuple(c / 255.0 for c in team_conf.color)
        return np.random.rand(3,)

    def _calculate_trend(self, ys, window_fraction=0.1):
        if len(ys) < 5:
            return ys
        
        window = max(3, int(len(ys) * window_fraction))
        
        weights = np.ones(window) / window
        trend = np.convolve(ys, weights, mode='valid')
        return trend, window

    def _draw_plot(self):
        self.ax.clear()
        self.ax.set_title(f"Metric: {self.selected_metric}")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, linestyle='--', alpha=0.4)

        has_data = False
        
        for team in self.teams:
            if not self.selected_teams.get(team, False):
                continue
            
            points = self.data.get(team, {}).get(self.selected_metric, [])
            # Фильтрация по диапазону
            filtered = [(i, v) for i, v in points if self.min_iter <= i <= self.max_iter]
            
            if not filtered:
                continue
                
            xs, ys = zip(*filtered)
            xs = np.array(xs)
            ys = np.array(ys)
            
            color = self._get_team_color(team)
            
            if self.show_raw:
                alpha = 0.3 if self.show_trend else 1.0 # Если включен тренд, делаем точки прозрачнее
                self.ax.plot(xs, ys, marker='o', markersize=3, linestyle='-', 
                             linewidth=1, color=color, alpha=alpha, label=f"{team} (Raw)")
            
            if self.show_trend and len(ys) > 1:
                trend_ys, window = self._calculate_trend(ys)
                start_idx = (window - 1) // 2
                end_idx = start_idx + len(trend_ys)
                trend_xs = xs[start_idx:end_idx]
                
                if len(trend_xs) > len(trend_ys): trend_xs = trend_xs[:len(trend_ys)]
                
                self.ax.plot(trend_xs, trend_ys, linestyle='--', linewidth=2.5, 
                             color=color, label=f"{team} (Trend)")

            has_data = True
        
        if has_data:
            self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        self.fig.canvas.draw_idle()

    def _init_widgets(self):
        ax_teams = plt.axes([0.02, 0.55, 0.15, 0.3])
        self.check_teams = CheckButtons(ax_teams, self.teams, [True]*len(self.teams))
        def toggle_team(label):
            self.selected_teams[label] = not self.selected_teams[label]
            self._draw_plot()
        self.check_teams.on_clicked(toggle_team)
        ax_teams.set_title("Teams", fontsize=10)

        ax_view = plt.axes([0.02, 0.40, 0.15, 0.1])
        self.check_view = CheckButtons(ax_view, ["Show Raw", "Show Trend"], [True, True])
        def toggle_view(label):
            if label == "Show Raw": self.show_raw = not self.show_raw
            if label == "Show Trend": self.show_trend = not self.show_trend
            self._draw_plot()
        self.check_view.on_clicked(toggle_view)
        ax_view.set_title("View Options", fontsize=10)

        ax_metrics = plt.axes([0.02, 0.15, 0.15, 0.2])
        self.radio_metrics = RadioButtons(ax_metrics, self.metrics, active=0)
        def change_metric(label):
            self.selected_metric = label
            self._draw_plot()
        self.radio_metrics.on_clicked(change_metric)
        ax_metrics.set_title("Metrics", fontsize=10)

        ax_min = plt.axes([0.30, 0.05, 0.1, 0.04])
        ax_max = plt.axes([0.45, 0.05, 0.1, 0.04])
        ax_btn = plt.axes([0.60, 0.05, 0.1, 0.04])

        self.box_min = TextBox(ax_min, "Min: ", initial="0")
        self.box_max = TextBox(ax_max, "Max: ", initial="Max")
        self.btn_update = Button(ax_btn, "Update Range")
        
        def update_range(event):
            try:
                self.min_iter = int(self.box_min.text)
            except:
                self.min_iter = 0
            try:
                if self.box_max.text.lower() == "max":
                    self.max_iter = 1000000000
                else:
                    self.max_iter = int(self.box_max.text)
            except:
                self.max_iter = 1000000000
            self._draw_plot()
            
        self.btn_update.on_clicked(update_range)


def run_standalone():
    stats_dir = "stats"
    if not os.path.exists(stats_dir):
        if os.path.exists(os.path.join("..", "stats")):
            stats_dir = os.path.join("..", "stats")
        else:
            print("No stats directory found.")
            return

    files = glob.glob(os.path.join(stats_dir, "*.csv"))
    files.sort(key=os.path.getmtime, reverse=True)
    
    if not files:
        print("No CSV files found.")
        return

    print("--- Available Stats Files ---")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    
    try:
        choice = input(f"Select file [0-{len(files)-1}]: ")
        idx = int(choice)
        filename = files[idx]
        print(f"Opening {filename}...")
        SnakePlotter(filename)
    except (ValueError, IndexError):
        print("Invalid selection.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        SnakePlotter(sys.argv[1])
    else:
        run_standalone()
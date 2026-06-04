import pygame
from typing import Any

class PygameRenderer:
    def __init__(self, config: Any) -> None:
        self.config: Any = config
        pygame.init()
        self.font: pygame.font.Font = pygame.font.SysFont('arial', 16)
        self.font_bold: pygame.font.Font = pygame.font.SysFont('arial', 16, bold=True)
        self.display: pygame.Surface = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption('Multi-Brain Snake AI (HP & Roles)')
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def render(self, state: Any) -> None:
        self._handle_events()
        self.display.fill(self.config.colors.BACKGROUND)
        self._draw_grid()
        self._draw_world(state)
        self._draw_sidebar(state)
        pygame.display.flip()
        
    def _handle_events(self) -> None:
        pygame.event.pump()

    def _draw_grid(self) -> None:
        w: int = self.config.map_width_px
        h: int = self.config.map_height_px
        bs: int = self.config.block_size
        color: tuple[int, int, int] = self.config.colors.GRID
        
        for x in range(0, w + 1, bs):
            pygame.draw.line(self.display, color, (x, 0), (x, h))
        for y in range(0, h + 1, bs):
            pygame.draw.line(self.display, color, (0, y), (w, y))
            
        pygame.draw.line(self.display, (0, 0, 0), (w, 0), (w, h), 2)

    def _draw_world(self, state: Any) -> None:
        bs: int = self.config.block_size
        
        for food in state.foods:
            pygame.draw.rect(self.display, self.config.colors.FOOD, (food.x, food.y, bs, bs))
            
        for snake in state.snakes:
            if not snake.is_alive: continue
            
            for i, pt in enumerate(snake.body):
                rect: tuple[int, int, int, int] = (pt.x, pt.y, bs, bs)
                color: tuple[int, int, int] = snake.color
                
                if i == 0: 
                    color = tuple(max(0, c - 50) for c in color)
                    
                pygame.draw.rect(self.display, color, rect)
                pygame.draw.rect(self.display, (50, 50, 50), rect, 1)

            center_x: int = snake.head.x + bs // 2
            center_y: int = snake.head.y + bs // 2
            
            if snake.role == "Hunter":
                pygame.draw.polygon(self.display, (255, 50, 50), [
                    (center_x, snake.head.y + 4),
                    (snake.head.x + bs - 4, center_y),
                    (center_x, snake.head.y + bs - 4),
                    (snake.head.x + 4, center_y)
                ])
            elif snake.role == "Defender":
                pygame.draw.rect(self.display, (50, 200, 50), 
                                 (snake.head.x + 5, snake.head.y + 5, bs - 10, bs - 10))
            else:
                pygame.draw.circle(self.display, (50, 50, 255), (center_x, center_y), 3)

    def _draw_sidebar(self, state: Any) -> None:
        x_offset: int = self.config.map_width_px + 10
        y: int = 10
        text_color: tuple[int, int, int] = self.config.colors.TEXT
        
        title: pygame.Surface = self.font_bold.render(f"Iter: {state.global_stats.total_iterations}", True, text_color)
        self.display.blit(title, (x_offset, y))
        y += 25
        
        deaths_txt: pygame.Surface = self.font.render(f"Deaths: {state.global_stats.total_deaths}", True, text_color)
        self.display.blit(deaths_txt, (x_offset, y))
        y += 25
        
        time_txt: pygame.Surface = self.font.render(f"Time: {int(state.global_stats.total_time)}s", True, text_color)
        self.display.blit(time_txt, (x_offset, y))
        y += 35
        
        pygame.draw.line(self.display, (200, 200, 200), (x_offset, y), (self.config.window_width - 10, y))
        y += 10
        
        alive_by_team: dict[str, list[Any]] = {t.name: [] for t in self.config.teams}
        for s in state.snakes:
            if not (s.is_alive and s.team_name in alive_by_team): continue
            alive_by_team[s.team_name].append(s)
        
        for name, stats in state.team_stats.items():
            team_conf: Any = next((t for t in self.config.teams if t.name == name), None)
            color: tuple[int, int, int] = team_conf.color if team_conf else (0, 0, 0)
            
            header: pygame.Surface = self.font_bold.render(name, True, color)
            self.display.blit(header, (x_offset, y))
            y += 20
            
            metrics: list[str] = [
                f"  Record: {stats.record}",
                f"  Deaths: {stats.deaths}",
                f"  Score: {stats.current_score}"
            ]
            
            if team_conf and team_conf.brain_type == "GA":
                metrics.append(f"  Gen: {getattr(stats, 'generation', 0)}")
                
            for metric_text in metrics:
                self.display.blit(self.font.render(metric_text, True, text_color), (x_offset, y))
                y += 18

            for i, s in enumerate(alive_by_team.get(name, [])):
                hp_ratio: float = min(max(s.hp / getattr(s, 'max_hp', 100.0), 0.0), 1.0)
                fill_width: int = int(100 * hp_ratio)
                bar_color: tuple[int, int, int] = (int(255 * (1 - hp_ratio)), int(255 * hp_ratio), 0)
                
                role_txt: str = s.role[:3].upper()
                self.display.blit(self.font.render(f"  S{i+1} [{role_txt}]:", True, text_color), (x_offset, y))
                
                pygame.draw.rect(self.display, (200, 200, 200), (x_offset + 80, y + 5, 100, 10))
                pygame.draw.rect(self.display, bar_color, (x_offset + 80, y + 5, fill_width, 10))
                
                y += 20
            
            y += 10

    def get_input(self) -> dict[str, bool]:
        res: dict[str, bool] = {
            'quit': False, 
            'toggle_speed': False, 
            'save': False, 
            'load': False,
            'toggle_graph': False,
            'toggle_visuals': False
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                res['quit'] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    res['toggle_speed'] = True
                elif event.key == pygame.K_s:
                    res['save'] = True
                elif event.key == pygame.K_l:
                    res['load'] = True
                elif event.key == pygame.K_g:
                    res['toggle_graph'] = True
                elif event.key == pygame.K_v:
                    res['toggle_visuals'] = True
                    
        return res
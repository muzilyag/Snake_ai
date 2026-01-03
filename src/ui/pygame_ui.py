import pygame
from src.config import GameConfig

class PygameRenderer:
    def __init__(self, config: GameConfig):
        self.config = config
        pygame.init()
        self.font = pygame.font.SysFont('arial', 16)
        self.font_bold = pygame.font.SysFont('arial', 16, bold=True)
        self.display = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption('Multi-Brain Snake AI')
        self.clock = pygame.time.Clock()

    def render(self, state):
        self.display.fill(self.config.colors.BACKGROUND)
        self._draw_grid()
        self._draw_world(state)
        self._draw_sidebar(state)
        pygame.display.flip()

    def _draw_grid(self):
        for x in range(0, self.config.map_width_px + 1, self.config.block_size):
            pygame.draw.line(self.display, self.config.colors.GRID, (x, 0), (x, self.config.map_height_px))
        for y in range(0, self.config.map_height_px + 1, self.config.block_size):
            pygame.draw.line(self.display, self.config.colors.GRID, (0, y), (self.config.map_width_px, y))
        pygame.draw.line(self.display, (0, 0, 0), (self.config.map_width_px, 0), (self.config.map_width_px, self.config.window_height), 2)

    def _draw_world(self, state):
        for f in state.foods:
            pygame.draw.rect(self.display, self.config.colors.FOOD, (f.x, f.y, self.config.block_size, self.config.block_size))
        for snake in state.snakes:
            for pt in snake.body:
                pygame.draw.rect(self.display, snake.color, (pt.x, pt.y, self.config.block_size, self.config.block_size))

    def _draw_sidebar(self, state):
        menu_x = self.config.map_width_px
        pygame.draw.rect(self.display, self.config.colors.SIDEBAR_BG, (menu_x, 0, self.config.sidebar_width, self.config.window_height))
        x, y = menu_x + 10, 10
        
        lines = [
            ("=== GLOBAL STATS ===", True),
            (f"Iterations: {state.global_stats.total_iterations}", False),
            (f"Deaths: {state.global_stats.total_deaths}", False),
            ("", False),
            ("=== TEAMS ===", True)
        ]
        
        for name, stats in state.team_stats.items():
            lines.append((f"[{name}]", True))
            lines.append((f"  Current Score: {stats.current_score}", False))
            lines.append((f"  Team Record: {stats.record}", False))
            lines.append((f"  Team Deaths: {stats.deaths}", False))
            if "GA" in name or any(t.name == name and t.brain_type == "GA" for t in self.config.teams):
                lines.append((f"  Generation: {stats.generation}", False))
            lines.append(("", False))

        for text, is_bold in lines:
            f = self.font_bold if is_bold else self.font
            surf = f.render(text, True, self.config.colors.TEXT)
            self.display.blit(surf, (x, y))
            y += 22

    def get_input(self):
        res = {'quit': False, 'toggle_speed': False, 'save': False, 'load': False}
        for event in pygame.event.get():
            if event.type == pygame.QUIT: res['quit'] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: res['toggle_speed'] = True
                if event.key == pygame.K_s: res['save'] = True
                if event.key == pygame.K_l: res['load'] = True
        return res
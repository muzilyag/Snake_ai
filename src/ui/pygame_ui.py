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
            if not snake.is_alive: continue
            for pt in snake.body:
                pygame.draw.rect(self.display, snake.color, (pt.x, pt.y, self.config.block_size, self.config.block_size))

    def _draw_sidebar(self, state):
        menu_x = self.config.map_width_px
        pygame.draw.rect(self.display, self.config.colors.SIDEBAR_BG, (menu_x, 0, self.config.sidebar_width, self.config.window_height))
        x, y = menu_x + 10, 10
        
        lines = [
            ("GLOBAL STATS", True),
            (f"Iterations: {state.global_stats.total_iterations}", False),
            (f"Deaths: {state.global_stats.total_deaths}", False),
            ("", False),
            ("TEAMS", True)
        ]
        
        for text, is_bold in lines:
            f = self.font_bold if is_bold else self.font
            self.display.blit(f.render(text, True, self.config.colors.TEXT), (x, y))
            y += 22

        for name, stats in state.team_stats.items():
            # Находим конфиг команды, чтобы проверить brain_type
            team_conf = next((t for t in self.config.teams if t.name == name), None)
            
            self.display.blit(self.font_bold.render(f"[{name}]", True, self.config.colors.TEXT), (x, y))
            y += 20
            self.display.blit(self.font.render(f"  Score: {stats.current_score}", True, self.config.colors.TEXT), (x, y))
            y += 18
            self.display.blit(self.font.render(f"  Rec: {stats.record}", True, self.config.colors.TEXT), (x, y))
            y += 18
            self.display.blit(self.font.render(f"  Deaths: {stats.deaths}", True, self.config.colors.TEXT), (x, y))
            y += 18
            
            # Показываем поколение ТОЛЬКО для GA
            if team_conf and team_conf.brain_type == "GA":
                self.display.blit(self.font.render(f"  Generation: {stats.generation}", True, self.config.colors.TEXT), (x, y))
                y += 18

            # --- Индикаторы голода ---
            team_snakes = [s for s in state.snakes if s.team_name == name and s.is_alive]
            for i, s in enumerate(team_snakes):
                h = s.steps_since_last_food
                limit = self.config.max_steps_without_food
                
                # Цвет: Зеленый (сыт) -> Красный (голоден)
                ratio = min(h / limit, 1.0)
                color = (int(255 * ratio), int(255 * (1 - ratio)), 0)
                
                self.display.blit(self.font.render(f"  S{i+1} Hunger: {h}/{limit}", True, color), (x, y))
                y += 16
            
            y += 14 # Отступ между командами

    def get_input(self):
        res = {'quit': False, 'toggle_speed': False, 'save': False, 'load': False}
        for event in pygame.event.get():
            if event.type == pygame.QUIT: res['quit'] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: res['toggle_speed'] = True
                if event.key == pygame.K_s: res['save'] = True
                if event.key == pygame.K_l: res['load'] = True
        return res
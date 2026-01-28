import pygame
from src.core.types import Direction

class PygameRenderer:
    def __init__(self, config):
        self.config = config
        pygame.init()
        self.font = pygame.font.SysFont('arial', 16)
        self.font_bold = pygame.font.SysFont('arial', 16, bold=True)
        self.display = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption('Multi-Brain Snake AI (HP & Roles)')
        self.clock = pygame.time.Clock()

    def render(self, state):
        self._handle_events()
        self.display.fill(self.config.colors.BACKGROUND)
        self._draw_grid()
        self._draw_world(state)
        self._draw_sidebar(state)
        pygame.display.flip()
        
    def _handle_events(self):
        pygame.event.pump()

    def _draw_grid(self):
        for x in range(0, self.config.map_width_px + 1, self.config.block_size):
            pygame.draw.line(self.display, self.config.colors.GRID, (x, 0), (x, self.config.map_height_px))
        for y in range(0, self.config.map_height_px + 1, self.config.block_size):
            pygame.draw.line(self.display, self.config.colors.GRID, (0, y), (self.config.map_width_px, y))
        pygame.draw.line(self.display, (0,0,0), (self.config.map_width_px, 0), (self.config.map_width_px, self.config.map_height_px), 2)

    def _draw_world(self, state):
        bs = self.config.block_size
        
        for food in state.foods:
            rect = (food.x, food.y, bs, bs)
            pygame.draw.rect(self.display, self.config.colors.FOOD, rect)
            
        for snake in state.snakes:
            if not snake.is_alive: continue
            
            for i, pt in enumerate(snake.body):
                rect = pygame.Rect(pt.x, pt.y, bs, bs)
                color = snake.color
                if i == 0: 
                    color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
                pygame.draw.rect(self.display, color, rect)
                pygame.draw.rect(self.display, (50,50,50), rect, 1)

            center_x = snake.head.x + bs // 2
            center_y = snake.head.y + bs // 2
            
            if snake.role == "Hunter":
                pygame.draw.polygon(self.display, (255, 50, 50), [
                    (center_x, snake.head.y + 4),
                    (snake.head.x + bs - 4, center_y),
                    (center_x, snake.head.y + bs - 4),
                    (snake.head.x + 4, center_y)
                ])
            else:
                pygame.draw.circle(self.display, (50, 50, 255), (center_x, center_y), 3)

    def _draw_sidebar(self, state):
        x_offset = self.config.map_width_px + 10
        y = 10
        
        title = self.font_bold.render(f"Iter: {state.global_stats.total_iterations}", True, self.config.colors.TEXT)
        self.display.blit(title, (x_offset, y))
        y += 25
        
        deaths_txt = self.font.render(f"Deaths: {state.global_stats.total_deaths}", True, self.config.colors.TEXT)
        self.display.blit(deaths_txt, (x_offset, y))
        y += 25
        
        time_txt = self.font.render(f"Time: {int(state.global_stats.total_time)}s", True, self.config.colors.TEXT)
        self.display.blit(time_txt, (x_offset, y))
        y += 35
        
        pygame.draw.line(self.display, (200,200,200), (x_offset, y), (self.config.window_width-10, y))
        y += 10
        
        for name, stats in state.team_stats.items():
            team_conf = next((t for t in self.config.teams if t.name == name), None)
            color = team_conf.color if team_conf else (0,0,0)
            
            header = self.font_bold.render(f"{name}", True, color)
            self.display.blit(header, (x_offset, y))
            y += 20
            
            self.display.blit(self.font.render(f"  Record: {stats.record}", True, self.config.colors.TEXT), (x_offset, y))
            y += 18
            self.display.blit(self.font.render(f"  Deaths: {stats.deaths}", True, self.config.colors.TEXT), (x_offset, y))
            y += 18
            self.display.blit(self.font.render(f"  Current Score: {stats.current_score}", True, self.config.colors.TEXT), (x_offset, y))
            y += 18
            
            if team_conf and team_conf.brain_type == "GA":
                self.display.blit(self.font.render(f"  Gen: {stats.generation}", True, self.config.colors.TEXT), (x_offset, y))
                y += 18

            team_snakes = [s for s in state.snakes if s.team_name == name and s.is_alive]
            for i, s in enumerate(team_snakes):
                hp_ratio = min(s.hp / s.max_hp, 1.0)
                bar_width = 100
                fill_width = int(bar_width * hp_ratio)
                bar_color = (int(255 * (1 - hp_ratio)), int(255 * hp_ratio), 0)
                role_txt = "HNT" if s.role == "Hunter" else "HRV"
                self.display.blit(self.font.render(f"  S{i+1} [{role_txt}]:", True, self.config.colors.TEXT), (x_offset, y))                
                pygame.draw.rect(self.display, (200,200,200), (x_offset + 80, y + 5, bar_width, 10))
                pygame.draw.rect(self.display, bar_color, (x_offset + 80, y + 5, fill_width, 10))
                y += 20
            
            y += 10

    def get_input(self):
        res = {
            'quit': False, 
            'toggle_speed': False, 
            'save': False, 
            'load': False,
            'toggle_graph': False 
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                res['quit'] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    res['toggle_speed'] = True
                if event.key == pygame.K_s:
                    res['save'] = True
                if event.key == pygame.K_l:
                    res['load'] = True
                if event.key == pygame.K_g:
                    res['toggle_graph'] = True
        return res
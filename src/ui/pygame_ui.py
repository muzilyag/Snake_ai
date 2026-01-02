import pygame
from src.config import GameConfig
from src.core.types import GameStateDTO

class PygameRenderer:
    def __init__(self, config: GameConfig):
        self.config = config
        pygame.init()
        self.font = pygame.font.SysFont('arial', 18, bold=True)
        self.display = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption('AI Snake Training')
        self.clock = pygame.time.Clock()

    def render(self, state: GameStateDTO, gen: int = 0, fitness: float = 0):
        self.display.fill(self.config.colors.BACKGROUND)
        self._draw_grid()

        # Еда
        pygame.draw.rect(self.display, self.config.colors.FOOD, 
                         (state.food.x, state.food.y, self.config.block_size, self.config.block_size))

        # Змейка
        for point in state.snake_body:
            pygame.draw.rect(self.display, self.config.colors.SNAKE, 
                             (point.x, point.y, self.config.block_size, self.config.block_size))

        # Статистика
        info = f"Gen: {gen} | Score: {state.score} | Fitness: {int(fitness)} | Iter: {state.iteration}"
        text = self.font.render(info, True, self.config.colors.TEXT)
        self.display.blit(text, [10, 10])

        pygame.display.flip()

    def _draw_grid(self):
        bs = self.config.block_size
        for x in range(0, self.config.width, bs):
            pygame.draw.line(self.display, self.config.colors.GRID, (x, 0), (x, self.config.height))
        for y in range(0, self.config.height, bs):
            pygame.draw.line(self.display, self.config.colors.GRID, (0, y), (self.config.width, y))

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
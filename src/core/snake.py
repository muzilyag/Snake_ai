from src.core.types import Direction, Point

class Snake:
    def __init__(self, x, y, team_config, role="Harvester", initial_length=3, block_size=20):
        self.head = Point(x, y)
        self.body = []
        for i in range(initial_length):
            self.body.append(Point(x - (i * block_size), y))
            
        self.direction = Direction.RIGHT
        self.team_name = team_config.name
        self.color = team_config.color
        self.is_alive = True
        self.score = 0
        self.steps_alive = 0
        self.steps_since_last_food = 0
        self.deaths = 0
        self.brain_type = team_config.brain_type
        # Сохраняем режим расчета наград
        self.reward_mode = team_config.reward_mode
        self.role = role

    def set_direction(self, direction):
        if not self.is_alive: return
        # Запрет разворота на 180 градусов
        if (direction == Direction.UP and self.direction == Direction.DOWN) or \
           (direction == Direction.DOWN and self.direction == Direction.UP) or \
           (direction == Direction.LEFT and self.direction == Direction.RIGHT) or \
           (direction == Direction.RIGHT and self.direction == Direction.LEFT):
            return
        self.direction = direction

    def move(self, block_size):
        if not self.is_alive: return
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += block_size
        elif self.direction == Direction.LEFT: x -= block_size
        elif self.direction == Direction.DOWN: y += block_size
        elif self.direction == Direction.UP: y -= block_size
        self.head = Point(x, y)
        self.steps_alive += 1
        self.steps_since_last_food += 1
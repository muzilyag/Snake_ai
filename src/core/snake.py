from src.core.types import Point, Direction

class Snake:
    def __init__(self, start_x: int, start_y: int):
        self.head = Point(start_x, start_y)
        self.body = [self.head]
        self.direction = Direction.RIGHT

    def set_direction(self, new_direction: Direction):
        if len(self.body) > 1:
            if new_direction == Direction.RIGHT and self.direction == Direction.LEFT: return
            if new_direction == Direction.LEFT and self.direction == Direction.RIGHT: return
            if new_direction == Direction.UP and self.direction == Direction.DOWN: return
            if new_direction == Direction.DOWN and self.direction == Direction.UP: return
        
        self.direction = new_direction

    def move(self, block_size: int):
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += block_size
        elif self.direction == Direction.LEFT:
            x -= block_size
        elif self.direction == Direction.DOWN:
            y += block_size
        elif self.direction == Direction.UP:
            y -= block_size

        self.head = Point(x, y)
import random
import math
from src.core.types import Point, GameStateDTO
from src.core.snake import Snake

class GameEngine:
    def __init__(self, config, strategy):
        self.config = config
        self.strategy = strategy
        self.iteration = 0
        self.food = None
        self.score = 0
        self.steps_since_last_meal = 0
        self.reset_game()

    def reset_game(self):
        self.score = 0
        self.steps_since_last_meal = 0
        self._respawn_snake()
        if self.food is None:
            self._place_food()

    def _respawn_snake(self):
        x = random.randint(0, (self.config.width - 20) // 20) * 20
        y = random.randint(0, (self.config.height - 20) // 20) * 20
        self.snake = Snake(x, y)

    def _place_food(self):
        while True:
            x = random.randint(0, (self.config.width - 20) // 20) * 20
            y = random.randint(0, (self.config.height - 20) // 20) * 20
            p = Point(x, y)
            if p not in self.snake.body:
                self.food = p
                break

    def _get_dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def get_state(self):
        return GameStateDTO(list(self.snake.body), self.food, self.score, False, self.iteration)

    def step(self):
        state_dto = self.get_state()
        next_move, action_idx = self.strategy.get_next_move(state_dto)
        self.snake.set_direction(next_move)
        return self._process_movement(action_idx)

    def step_manual(self, action_idx):
        return self._process_movement(action_idx)

    def _process_movement(self, action_idx):
        self.iteration += 1
        self.steps_since_last_meal += 1
        dist_before = self._get_dist(self.snake.head, self.food)
        self.snake.move(20)
        self.snake.body.insert(0, self.snake.head)

        reward = 0.0
        done = False

        if self._is_collision() or self.steps_since_last_meal > 100:
            reward = -10.0
            done = True
        elif self.snake.head == self.food:
            reward = 50.0
            self.score += 1
            self.steps_since_last_meal = 0
            self._place_food()
        else:
            dist_after = self._get_dist(self.snake.head, self.food)
            reward = 1.0 if dist_after < dist_before else -1.5
            self.snake.body.pop()

        return reward, done, action_idx

    def _is_collision(self):
        h = self.snake.head
        if h.x < 0 or h.x >= self.config.width or h.y < 0 or h.y >= self.config.height:
            return True
        if h in self.snake.body[1:]:
            return True
        return False
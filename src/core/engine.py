import random
import time
import math
from src.core.types import Point, GameStateDTO, GlobalStats, TeamStats
from src.core.snake import Snake

class GameEngine:
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.iteration = 0
        self.total_deaths = 0
        self.snakes = []
        self.foods = []
        # Инициализируем TeamStats, используя dataclass
        self.team_stats = {t.name: TeamStats() for t in config.teams}
        
        for team in self.config.teams:
            for _ in range(team.count):
                self.snakes.append(self._create_initial_snake(team))
        
        while len(self.foods) < self.config.food_count:
            self._place_food()

    def _create_initial_snake(self, team_config):
        x = random.randint(5, self.config.grid_width - 5) * self.config.block_size
        y = random.randint(5, self.config.grid_height - 5) * self.config.block_size
        return Snake(x, y, team_config, self.config.initial_snake_length, self.config.block_size)

    def _respawn_snake_at_random(self, snake):
        while True:
            x = random.randint(2, self.config.grid_width - 3) * self.config.block_size
            y = random.randint(2, self.config.grid_height - 3) * self.config.block_size
            p = Point(x, y)
            occupied = any(p in s.body for s in self.snakes if s != snake) or any(p == f for f in self.foods)
            if not occupied:
                snake.head = p
                snake.body = []
                for i in range(self.config.initial_snake_length):
                    snake.body.append(Point(p.x - (i * self.config.block_size), p.y))
                snake.direction = random.choice([1, 2, 3, 4])
                snake.is_alive = True
                snake.score = 0
                snake.steps_alive = 0
                snake.steps_since_last_food = 0
                break

    def _place_food(self):
        attempts = 0
        while attempts < 100:
            x = random.randint(0, self.config.grid_width - 1) * self.config.block_size
            y = random.randint(0, self.config.grid_height - 1) * self.config.block_size
            p = Point(x, y)
            occupied = any(p in s.body for s in self.snakes) or (p in self.foods)
            if not occupied:
                self.foods.append(p)
                break
            attempts += 1

    def _calculate_reward(self, snake, dist_before, dist_after, event_type):
        r = self.config.rewards
        
        if snake.reward_mode == "linear":
            if event_type == 'starve': return r.starvation_penalty
            if event_type == 'death': return r.death_penalty_base
            if event_type == 'food':  return r.food_reward_base
            return r.step_closer if dist_after < dist_before else r.step_farther

        elif snake.reward_mode == "dynamic":
            if event_type == 'starve': return r.starvation_penalty
            if event_type == 'death': return -10.0 - (len(snake.body) * 0.5)
            if event_type == 'food': return 25.0
            
            if event_type == 'move':
                change = dist_before - dist_after
                vel_reward = change * r.dynamic_velocity_multiplier
                
                danger_penalty = 0.0
                bs = self.config.block_size
                # Проверка 4 соседних клеток на опасность
                neighbors = [
                    Point(snake.head.x, snake.head.y - bs),
                    Point(snake.head.x, snake.head.y + bs),
                    Point(snake.head.x - bs, snake.head.y),
                    Point(snake.head.x + bs, snake.head.y)
                ]
                
                for n in neighbors:
                    is_wall = (n.x < 0 or n.x >= self.config.map_width_px or 
                               n.y < 0 or n.y >= self.config.map_height_px)
                    is_body = False
                    if not is_wall:
                        for s in self.snakes:
                            if s.is_alive and n in s.body:
                                is_body = True
                                break
                    
                    if is_wall or is_body:
                        danger_penalty += r.danger_sensing_penalty

                return vel_reward + danger_penalty + r.survival_bonus
                
        return 0.0

    def step(self, actions):
        self.iteration += 1
        results = []
        
        # Сбрасываем текущий счет (current_score) для отрисовки, но не рекорд
        for t_name in self.team_stats:
            self.team_stats[t_name].current_score = 0
            
        for i, snake in enumerate(self.snakes):
            dist_before = self._get_closest_food_dist(snake)
            snake.move(self.config.block_size)
            snake.body.insert(0, snake.head)
            
            reward = 0
            done = False
            
            # --- Логика коллизий и наград ---
            if self._check_collision(snake):
                reward = self._calculate_reward(snake, 0, 0, 'death')
                done = True
                self.total_deaths += 1
                self.team_stats[snake.team_name].deaths += 1
                self._respawn_snake_at_random(snake)
                
            elif snake.steps_since_last_food >= self.config.max_steps_without_food:
                reward = self._calculate_reward(snake, 0, 0, 'starve')
                done = True
                self.total_deaths += 1
                self.team_stats[snake.team_name].deaths += 1
                self._respawn_snake_at_random(snake)
                
            elif snake.head in self.foods:
                reward = self._calculate_reward(snake, 0, 0, 'food')
                snake.score += 1
                snake.steps_since_last_food = 0
                
                # Обновляем рекорд команды
                if snake.score > self.team_stats[snake.team_name].record:
                    self.team_stats[snake.team_name].record = snake.score
                    
                self.foods.remove(snake.head)
                self._place_food()
                
            else:
                dist_after = self._get_closest_food_dist(snake)
                reward = self._calculate_reward(snake, dist_before, dist_after, 'move')
                snake.body.pop()
            
            # Обновляем текущий счет команды (суммарный по живым змейкам)
            self.team_stats[snake.team_name].current_score += snake.score
            
            results.append((reward, done, snake.score))
            
        return results, False

    def _get_closest_food_dist(self, snake):
        if not self.foods: return 0
        dists = [math.sqrt((snake.head.x - f.x)**2 + (snake.head.y - f.y)**2) for f in self.foods]
        return min(dists)

    def _check_collision(self, snake):
        h = snake.head
        if h.x < 0 or h.x >= self.config.map_width_px or h.y < 0 or h.y >= self.config.map_height_px: return True
        if h in snake.body[1:]: return True
        for other in self.snakes:
            if other is not snake and h in other.body: return True
        return False

    def get_state(self):
        # Создаем GlobalStats, используя поля из types.py (total_iterations, total_time, total_deaths)
        g_stats = GlobalStats(
            total_iterations=self.iteration, 
            total_time=time.time() - self.start_time, 
            total_deaths=self.total_deaths
        )
        return GameStateDTO(self.snakes, self.foods, g_stats, self.team_stats, False)
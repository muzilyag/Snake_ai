import random
import time
import math
from src.core.types import Point, GameStateDTO, GlobalStats, TeamStats, DeathReason
from src.core.snake import Snake
from src.core.analytics import AnalyticsEngine

class GameEngine:
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.iteration = 0
        self.total_deaths = 0
        self.snakes = []
        self.foods = []
        
        self.team_stats = {t.name: TeamStats() for t in config.teams}
        self.analytics = AnalyticsEngine(config)
        
        for team in self.config.teams:
            for i in range(team.count):
                role = team.agent_roles[i]
                self.snakes.append(self._create_initial_snake(team, role))
        
        while len(self.foods) < self.config.food_count:
            self._place_food()

    def _create_initial_snake(self, team_config, role="Harvester"):
        x = random.randint(5, self.config.grid_width - 5) * self.config.block_size
        y = random.randint(5, self.config.grid_height - 5) * self.config.block_size
        safe_role = role if role in self.config.reward_presets else "Harvester"
        return Snake(x, y, team_config, safe_role, self.config.initial_snake_length, self.config.block_size)

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

    def _get_closest_food_dist(self, snake):
        if not self.foods: return 0
        dists = [math.sqrt((snake.head.x - f.x)**2 + (snake.head.y - f.y)**2) for f in self.foods]
        return min(dists)

    def _get_closest_enemy_dist(self, snake):
        min_dist = float('inf')
        for other in self.snakes:
            if other is not snake and other.is_alive and other.team_name != snake.team_name:
                dist = math.sqrt((snake.head.x - other.head.x)**2 + (snake.head.y - other.head.y)**2)
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist != float('inf') else 0

    def _calculate_reward(self, snake, 
                          dist_food_before, dist_food_after, 
                          dist_enemy_before, dist_enemy_after, 
                          event_type):
        
        presets = self.config.reward_presets.get(snake.role, self.config.reward_presets["Harvester"])
        
        # 1. События (смерть, еда) - фиксированы всегда
        if event_type == 'starve': return presets.get('starve', -100.0)
        if event_type == 'death': return presets.get('death', -50.0)
        if event_type == 'food': return presets.get('food', 10.0)
        
        reward = presets.get('idle_penalty', 0.0)
        bs = self.config.block_size

        # --- LINEAR MODE (Старый добрый if-else) ---
        if snake.reward_mode == "linear":
            # Еда
            if dist_food_after < dist_food_before:
                reward += presets.get('step_closer_food', 0.0)
            else:
                reward += presets.get('step_farther_food', 0.0)
            
            # Враг (только если есть)
            if dist_enemy_before > 0:
                if dist_enemy_after < dist_enemy_before:
                    reward += presets.get('step_closer_enemy', 0.0)
                else:
                    reward += presets.get('step_farther_enemy', 0.0)

        # --- DYNAMIC MODE (Учитывает магнитуду движения) ---
        elif snake.reward_mode == "dynamic":
            # Считаем изменение (delta). Делим на bs, чтобы нормализовать к 1 шагу.
            # Пример: приблизился на 20px (1 блок) -> delta = 20. 20/20 = 1.0. 
            # Награда = 1.0 * coefficient.
            
            # Еда
            delta_food = dist_food_before - dist_food_after
            normalized_food = delta_food / bs
            # Если приблизился (normalized > 0), умножаем на 'step_closer', иначе на 'step_farther'
            # Но для упрощения динамики часто используют один коэффициент на сближение. 
            # Мы используем логику пресетов:
            if normalized_food > 0:
                reward += normalized_food * presets.get('step_closer_food', 0.0)
            else:
                # normalized_food отрицательный, берем модуль для умножения на штраф
                reward += abs(normalized_food) * presets.get('step_farther_food', 0.0)
            
            # Враг
            if dist_enemy_before > 0:
                delta_enemy = dist_enemy_before - dist_enemy_after
                normalized_enemy = delta_enemy / bs
                
                if normalized_enemy > 0:
                    reward += normalized_enemy * presets.get('step_closer_enemy', 0.0)
                else:
                    reward += abs(normalized_enemy) * presets.get('step_farther_enemy', 0.0)

        # Штраф за стены (одинаков для всех)
        h = snake.head
        if h.x < bs or h.x > self.config.map_width_px - bs or \
           h.y < bs or h.y > self.config.map_height_px - bs:
            reward += presets.get('wall_penalty', 0.0)
            
        return reward

    def step(self, actions):
        self.iteration += 1
        results = []
        
        for t_name in self.team_stats:
            self.team_stats[t_name].current_score = 0
            
        for i, snake in enumerate(self.snakes):
            dist_food_before = self._get_closest_food_dist(snake)
            dist_enemy_before = self._get_closest_enemy_dist(snake)

            snake.move(self.config.block_size)
            snake.body.insert(0, snake.head)
            
            reward = 0
            done = False
            event = 'move'
            
            death_reason = self._get_death_reason(snake)
            
            if death_reason != DeathReason.ALIVE:
                event = 'death'
                done = True
                self.total_deaths += 1
                self.team_stats[snake.team_name].deaths += 1
                self.analytics.log_death(snake.team_name, death_reason)
                self._respawn_snake_at_random(snake)
                
            elif snake.steps_since_last_food >= self.config.max_steps_without_food:
                event = 'starve'
                done = True
                self.total_deaths += 1
                self.team_stats[snake.team_name].deaths += 1
                self.analytics.log_death(snake.team_name, DeathReason.STARVATION)
                self._respawn_snake_at_random(snake)
                
            elif snake.head in self.foods:
                event = 'food'
                snake.score += 1
                snake.steps_since_last_food = 0
                self.analytics.log_food(snake.team_name)
                
                if snake.score > self.team_stats[snake.team_name].record:
                    self.team_stats[snake.team_name].record = snake.score
                    
                self.foods.remove(snake.head)
                self._place_food()
            
            else:
                snake.body.pop()

            dist_food_after = self._get_closest_food_dist(snake)
            dist_enemy_after = self._get_closest_enemy_dist(snake)

            reward = self._calculate_reward(
                snake, 
                dist_food_before, dist_food_after, 
                dist_enemy_before, dist_enemy_after, 
                event
            )
            
            self.team_stats[snake.team_name].current_score += snake.score
            results.append((reward, done, snake.score))
        
        self.analytics.update(self.iteration)
        return results, False

    def _get_closest_food_dist(self, snake):
        if not self.foods: return 0
        dists = [math.sqrt((snake.head.x - f.x)**2 + (snake.head.y - f.y)**2) for f in self.foods]
        return min(dists)
    
    def _get_closest_enemy_dist(self, snake):
        min_dist = float('inf')
        for other in self.snakes:
            if other is not snake and other.is_alive and other.team_name != snake.team_name:
                dist = math.sqrt((snake.head.x - other.head.x)**2 + (snake.head.y - other.head.y)**2)
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist != float('inf') else 0

    def _get_death_reason(self, snake) -> int:
        h = snake.head
        if h.x < 0 or h.x >= self.config.map_width_px or h.y < 0 or h.y >= self.config.map_height_px: 
            return DeathReason.WALL
        if h in snake.body[1:]: 
            return DeathReason.SELF_COLLISION
        for other in self.snakes:
            if other is not snake and h in other.body: 
                return DeathReason.ENEMY_COLLISION
        return DeathReason.ALIVE

    def get_state(self):
        g_stats = GlobalStats(
            total_iterations=self.iteration, 
            total_time=time.time() - self.start_time, 
            total_deaths=self.total_deaths
        )
        return GameStateDTO(self.snakes, self.foods, g_stats, self.team_stats, False)
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
        return Snake(x, y, team_config, role, self.config)

    def _respawn_snake_at_random(self, snake):
        while True:
            x = random.randint(2, self.config.grid_width - 3) * self.config.block_size
            y = random.randint(2, self.config.grid_height - 3) * self.config.block_size
            p = Point(x, y)
            occupied = any(p in s.body for s in self.snakes if s != snake) or any(p == f for f in self.foods)
            if not occupied:
                role_cfg = self.config.role_settings.get(snake.role, self.config.role_settings["Harvester"])
                snake.head = p
                snake.body = []
                for i in range(self.config.initial_snake_length):
                    snake.body.append(Point(p.x - (i * self.config.block_size), p.y))
                snake.direction = random.choice([1, 2, 3, 4])
                snake.is_alive = True
                snake.score = 0
                snake.hp = role_cfg.start_hp
                snake.steps_alive = 0
                snake.pending_reward = 0.0
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

    def _get_snake_at_body_pos(self, point):
        for s in self.snakes:
            if s.is_alive and point in s.body:
                return s
        return None

    def _calculate_reward(self, snake, 
                          dist_food_before, dist_food_after, 
                          dist_enemy_before, dist_enemy_after,
                          dist_ally_before, dist_ally_after,
                          event_type):
        
        presets = self.config.reward_presets.get(snake.role, self.config.reward_presets["Harvester"])
        
        reward = snake.pending_reward
        snake.pending_reward = 0.0
        
        if event_type == 'starve': return reward + presets.get('starve', -50.0)
        if event_type == 'death': return reward + presets.get('death', -50.0)
        if event_type == 'food': return reward + presets.get('food', 10.0)
        
        reward += presets.get('idle_penalty', 0.0)
        bs = self.config.block_size

        if snake.reward_mode == "linear":
            if dist_food_after < dist_food_before:
                reward += presets.get('step_closer_food', 0.0)
            else:
                reward += presets.get('step_farther_food', 0.0)
            
            if dist_enemy_before > 0:
                if dist_enemy_after < dist_enemy_before:
                    reward += presets.get('step_closer_enemy', 0.0)
                else:
                    reward += presets.get('step_farther_enemy', 0.0)
            
            if dist_ally_before > 0:
                if dist_ally_after < dist_ally_before:
                    reward += presets.get('step_closer_team', 0.0)
                else:
                    reward += presets.get('step_farther_team', 0.0)

        elif snake.reward_mode == "dynamic":
            delta_food = dist_food_before - dist_food_after
            normalized_food = delta_food / bs
            if normalized_food > 0:
                reward += normalized_food * presets.get('step_closer_food', 0.0)
            else:
                reward += abs(normalized_food) * presets.get('step_farther_food', 0.0)
            
            if dist_enemy_before > 0:
                delta_enemy = dist_enemy_before - dist_enemy_after
                normalized_enemy = delta_enemy / bs
                if normalized_enemy > 0:
                    reward += normalized_enemy * presets.get('step_closer_enemy', 0.0)
                else:
                    reward += abs(normalized_enemy) * presets.get('step_farther_enemy', 0.0)
            
            if dist_ally_before > 0:
                delta_ally = dist_ally_before - dist_ally_after
                normalized_ally = delta_ally / bs
                if normalized_ally > 0:
                    reward += normalized_ally * presets.get('step_closer_team', 0.0)
                else:
                    reward += abs(normalized_ally) * presets.get('step_farther_team', 0.0)

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
            if not snake.is_alive:
                self._respawn_snake_at_random(snake)
                results.append((0, True, 0))
                continue

            dist_food_before = self._get_closest_food_dist(snake)
            dist_enemy_before = self._get_closest_enemy_dist(snake)
            dist_ally_before = self._get_closest_ally_dist(snake)

            next_head = snake.move_head_prediction(self.config.block_size)
            
            reward = 0
            done = False
            event = 'move'
            death_reason = DeathReason.ALIVE

            snake.hp -= self.config.hp_decay_per_step

            collision_wall = (next_head.x < 0 or next_head.x >= self.config.map_width_px or 
                              next_head.y < 0 or next_head.y >= self.config.map_height_px)
            
            victim_snake = self._get_snake_at_body_pos(next_head)
            
            if collision_wall:
                snake.is_alive = False
                death_reason = DeathReason.WALL
                event = 'death'
                done = True

            elif next_head in self.foods:
                snake.commit_move(next_head)
                snake.body.insert(0, snake.head)
                snake.score += 1
                snake.heal(self.config.food_heal_amount)
                
                self.foods.remove(next_head)
                self._place_food()
                event = 'food'
                self.analytics.log_food(snake.team_name)

            elif victim_snake:
                victim_snake.take_damage(snake.damage_dealt)
                snake.take_damage(victim_snake.victim_return_damage)

                if victim_snake.team_name == snake.team_name and victim_snake is not snake:
                     presets = self.config.reward_presets.get(snake.role, {})
                     snake.pending_reward += presets.get('friendly_fire', -10.0)

                if victim_snake.team_name != snake.team_name:
                    presets = self.config.reward_presets.get(snake.role, {})
                    snake.pending_reward += presets.get('damage_dealt_reward', 0.0)
                    
                    if victim_snake.victim_return_damage > 0:
                        v_presets = self.config.reward_presets.get(victim_snake.role, {})
                        victim_snake.pending_reward += v_presets.get('damage_dealt_reward', 0.0)

                    if not snake.is_alive:
                         v_presets = self.config.reward_presets.get(victim_snake.role, {})
                         victim_snake.pending_reward += v_presets.get('kill_reward', 0.0)

                if snake.collision_survivable:
                    snake.take_damage(snake.self_damage)
                    if snake.is_alive:
                        try:
                            if victim_snake is snake:
                                snake.is_alive = False
                                death_reason = DeathReason.SELF_COLLISION
                                done = True
                                event = 'death'
                            else:
                                cut_idx = victim_snake.body.index(next_head)
                                victim_snake.body = victim_snake.body[:cut_idx]
                                
                                if len(victim_snake.body) < 1:
                                    victim_snake.is_alive = False
                                    victim_snake.body = []
                                    self.analytics.log_death(victim_snake.team_name, DeathReason.ENEMY_COLLISION)
                                    self.total_deaths += 1
                                    self.team_stats[victim_snake.team_name].deaths += 1
                                    
                                    if victim_snake.team_name != snake.team_name:
                                        presets = self.config.reward_presets.get(snake.role, {})
                                        snake.pending_reward += presets.get('kill_reward', 0.0)

                                victim_snake.score = max(0, len(victim_snake.body) - self.config.initial_snake_length)
                                
                                snake.commit_move(next_head)
                                snake.body.insert(0, snake.head)
                                snake.body.pop()
                                
                        except ValueError:
                            snake.is_alive = False
                            done = True
                            event = 'death'
                    else:
                        death_reason = DeathReason.ENEMY_COLLISION
                        done = True
                        event = 'death'
                else:
                    snake.is_alive = False
                    death_reason = DeathReason.ENEMY_COLLISION
                    done = True
                    event = 'death'

            else:
                snake.commit_move(next_head)
                snake.body.insert(0, snake.head)
                snake.body.pop()

            if snake.is_alive and snake.hp <= 0:
                snake.is_alive = False
                death_reason = DeathReason.STARVATION
                event = 'starve'
                done = True

            if not snake.is_alive and done:
                self.total_deaths += 1
                self.team_stats[snake.team_name].deaths += 1
                if death_reason != DeathReason.ALIVE:
                    self.analytics.log_death(snake.team_name, death_reason)
                
                if death_reason == DeathReason.ENEMY_COLLISION and victim_snake and victim_snake.is_alive:
                    if victim_snake.team_name != snake.team_name:
                         v_presets = self.config.reward_presets.get(victim_snake.role, {})
                         victim_snake.pending_reward += v_presets.get('kill_reward', 0.0)

            dist_food_after = self._get_closest_food_dist(snake)
            dist_enemy_after = self._get_closest_enemy_dist(snake)
            dist_ally_after = self._get_closest_ally_dist(snake)

            reward += self._calculate_reward(
                snake, 
                dist_food_before, dist_food_after, 
                dist_enemy_before, dist_enemy_after,
                dist_ally_before, dist_ally_after,
                event
            )
            
            current_len = max(0, len(snake.body) - 1) if snake.is_alive else 0
            self.team_stats[snake.team_name].current_score += current_len
            
            results.append((reward, done, snake.score))

        for t_name in self.team_stats:
            if self.team_stats[t_name].current_score > self.team_stats[t_name].record:
                self.team_stats[t_name].record = self.team_stats[t_name].current_score
        
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

    def _get_closest_ally_dist(self, snake):
        min_dist = float('inf')
        for other in self.snakes:
            if other is not snake and other.is_alive and other.team_name == snake.team_name:
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
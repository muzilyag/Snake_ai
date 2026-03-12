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
        
        self.grid = [[None for _ in range(config.grid_height)] for _ in range(config.grid_width)]
        
        self.team_stats = {t.name: TeamStats() for t in config.teams}
        self.analytics = AnalyticsEngine(config)
        
        for team in self.config.teams:
            for i in range(team.count):
                role = team.agent_roles[i]
                self.snakes.append(self._create_initial_snake(team, role))
        
        self._rebuild_grid()
        
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
            
            gx = x // self.config.block_size
            gy = y // self.config.block_size
            
            if self.grid[gx][gy] is None:
                role_cfg = self.config.role_settings.get(snake.role, self.config.role_settings["Harvester"])
                snake.head = Point(x, y)
                snake.body = []
                for i in range(self.config.initial_snake_length):
                    snake.body.append(Point(x - (i * self.config.block_size), y))
                snake.direction = random.choice([1, 2, 3, 4])
                snake.is_alive = True
                snake.score = 0
                snake.hp = role_cfg.start_hp
                snake.steps_alive = 0
                snake.pending_reward = 0.0
                break

    def _rebuild_grid(self):
        w = self.config.grid_width
        h = self.config.grid_height
        bs = self.config.block_size
        
        self.grid = [[None for _ in range(h)] for _ in range(w)]
        
        for s in self.snakes:
            if not s.is_alive: continue
            for pt in s.body:
                gx = pt.x // bs
                gy = pt.y // bs
                if 0 <= gx < w and 0 <= gy < h:
                    self.grid[gx][gy] = s
        
        for f in self.foods:
            gx = f.x // bs
            gy = f.y // bs
            if 0 <= gx < w and 0 <= gy < h:
                self.grid[gx][gy] = "FOOD"

    def _place_food(self):
        attempts = 0
        w = self.config.grid_width
        h = self.config.grid_height
        bs = self.config.block_size
        
        while attempts < 100:
            gx = random.randint(0, w - 1)
            gy = random.randint(0, h - 1)
            
            if self.grid[gx][gy] is None:
                x = gx * bs
                y = gy * bs
                self.foods.append(Point(x, y))
                self.grid[gx][gy] = "FOOD"
                break
            attempts += 1

    def _get_snake_at_body_pos(self, point):
        gx = point.x // self.config.block_size
        gy = point.y // self.config.block_size
        
        if 0 <= gx < self.config.grid_width and 0 <= gy < self.config.grid_height:
            obj = self.grid[gx][gy]
            if isinstance(obj, Snake) and obj.is_alive:
                return obj
        return None

    def _calculate_reward(self, snake, 
                          dist_food_before, dist_food_after, 
                          closest_enemy_before, closest_enemy_after,
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

        if dist_food_after < dist_food_before:
            reward += presets.get('step_closer_food', 0.0)
        else:
            reward += presets.get('step_farther_food', 0.0)
            
        enemy_snake, dist_before = closest_enemy_before
        _, dist_after = closest_enemy_after
        
        if enemy_snake and dist_before > 0:
            role_key_closer = f"step_closer_enemy_{enemy_snake.role.lower()}" if hasattr(enemy_snake, 'role') else "step_closer_enemy"
            val_closer = presets.get(role_key_closer, presets.get('step_closer_enemy', 0.0))
            val_farther = -val_closer * 0.5
            
            if dist_after < dist_before:
                reward += val_closer
            else:
                reward += val_farther
            
        if dist_ally_before > 0:
            if dist_ally_after < dist_ally_before:
                reward += presets.get('step_closer_team', 0.0)
            else:
                reward += presets.get('step_farther_team', 0.0)

        h = snake.head
        if h.x < bs or h.x > self.config.map_width_px - bs or \
           h.y < bs or h.y > self.config.map_height_px - bs:
            reward += presets.get('wall_penalty', 0.0)
            
        return reward

    def step(self, actions):
        self.iteration += 1
        
        self._rebuild_grid()
        
        results = []
        
        for t_name in self.team_stats:
            self.team_stats[t_name].current_score = 0
            
        for i, snake in enumerate(self.snakes):
            if not snake.is_alive:
                self._respawn_snake_at_random(snake)
                results.append((0, True, 0))
                continue

            dist_food_before = self._get_closest_food_dist(snake)
            closest_enemy_before = self._get_closest_enemy_info(snake)
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

                presets = self.config.reward_presets.get(snake.role, {})

                if victim_snake.team_name == snake.team_name and victim_snake is not snake:
                     snake.pending_reward += presets.get('friendly_fire', -10.0)

                if victim_snake.team_name != snake.team_name:
                    snake.pending_reward += presets.get('damage_dealt_reward', 0.0)
                    if victim_snake.victim_return_damage > 0:
                        v_presets = self.config.reward_presets.get(victim_snake.role, {})
                        victim_snake.pending_reward += v_presets.get('damage_dealt_reward', 0.0)

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
                                        kill_key = f"kill_{victim_snake.role.lower()}"
                                        kill_reward = presets.get(kill_key, presets.get('kill_generic', 5.0))
                                        snake.pending_reward += kill_reward

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
                         kill_key = f"kill_{snake.role.lower()}"
                         kill_reward = v_presets.get(kill_key, v_presets.get('kill_generic', 5.0))
                         victim_snake.pending_reward += kill_reward

            dist_food_after = self._get_closest_food_dist(snake)
            closest_enemy_after = self._get_closest_enemy_info(snake)
            dist_ally_after = self._get_closest_ally_dist(snake)

            reward += self._calculate_reward(
                snake, 
                dist_food_before, dist_food_after, 
                closest_enemy_before, closest_enemy_after,
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
    
    def _get_closest_enemy_info(self, snake):
        min_dist = float('inf')
        closest_s = None
        for other in self.snakes:
            if other is not snake and other.is_alive and other.team_name != snake.team_name:
                dist = math.sqrt((snake.head.x - other.head.x)**2 + (snake.head.y - other.head.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_s = other
        
        return (closest_s, min_dist if min_dist != float('inf') else 0)

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
        dto = GameStateDTO(self.snakes, self.foods, g_stats, self.team_stats, False)
        dto.grid = self.grid 
        return dto
import random
import time
import math
from typing import Any
from .types import Point, GameStateDTO, GlobalStats, TeamStats, DeathReason
from .snake import Snake
from .analytics import AnalyticsEngine

class GameEngine:
    def __init__(self, config: Any) -> None:
        self.config: Any = config
        self.start_time: float = time.time()
        self.iteration: int = 0
        self.total_deaths: int = 0
        self.snakes: list[Snake] = []
        self.foods: list[Point] = []
        
        w: int = config.grid_width
        h: int = config.grid_height
        self.grid: list[list[Snake | str | None]] = [[None for _ in range(h)] for _ in range(w)]
        self.team_stats: dict[str, TeamStats] = {t.name: TeamStats() for t in config.teams}
        self.analytics: AnalyticsEngine = AnalyticsEngine(config)
        
        for team in self.config.teams:
            for i in range(team.count):
                self.snakes.append(self._create_initial_snake(team, team.agent_roles[i]))
        
        self._rebuild_grid()
        
        while len(self.foods) < self.config.food_count: self._place_food()

    def _create_initial_snake(self, team_config: Any, role: str = "Harvester") -> Snake:
        x: int = random.randint(5, self.config.grid_width - 5) * self.config.block_size
        y: int = random.randint(5, self.config.grid_height - 5) * self.config.block_size
        return Snake(x, y, team_config, role, self.config)

    def _respawn_snake_at_random(self, snake: Snake) -> None:
        while True:
            x: int = random.randint(2, self.config.grid_width - 3) * self.config.block_size
            y: int = random.randint(2, self.config.grid_height - 3) * self.config.block_size
            gx: int = x // self.config.block_size
            gy: int = y // self.config.block_size
            
            if self.grid[gx][gy] is not None: continue

            role_cfg: Any = self.config.role_settings.get(snake.role, self.config.role_settings["Harvester"])
            init_len: int = getattr(role_cfg, 'initial_length', self.config.initial_snake_length)
            
            snake.head = Point(x, y)
            snake.body = [Point(x - (i * self.config.block_size), y) for i in range(init_len)]
            snake.direction = random.choice([1, 2, 3, 4])
            snake.is_alive = True
            snake.score = 0
            snake.hp = role_cfg.start_hp
            snake.steps_alive = 0
            snake.pending_reward = 0.0
            break

    def _rebuild_grid(self) -> None:
        w: int = self.config.grid_width
        h: int = self.config.grid_height
        bs: int = self.config.block_size
        
        self.grid = [[None for _ in range(h)] for _ in range(w)]
        
        for s in self.snakes:
            if not s.is_alive: continue
            for pt in s.body:
                gx: int = pt.x // bs
                gy: int = pt.y // bs
                if not (0 <= gx < w and 0 <= gy < h): continue
                self.grid[gx][gy] = s
        
        for f in self.foods:
            gx: int = f.x // bs
            gy: int = f.y // bs
            if not (0 <= gx < w and 0 <= gy < h): continue
            self.grid[gx][gy] = "FOOD"

    def _place_food(self) -> None:
        w: int = self.config.grid_width
        h: int = self.config.grid_height
        bs: int = self.config.block_size
        
        for _ in range(100):
            gx: int = random.randint(0, w - 1)
            gy: int = random.randint(0, h - 1)
            if self.grid[gx][gy] is not None: continue
            self.foods.append(Point(gx * bs, gy * bs))
            self.grid[gx][gy] = "FOOD"
            break

    def _get_snake_at_body_pos(self, point: Point) -> Snake | None:
        gx, gy = point.x // self.config.block_size, point.y // self.config.block_size
        
        if (0 <= gx < self.config.grid_width and 0 <= gy < self.config.grid_height and 
            isinstance(obj := self.grid[gx][gy], Snake) and obj.is_alive):
            return obj
        return None

    def _calculate_reward(self, snake: Snake, dist_food_before: float, dist_food_after: float, 
                          closest_enemy_before: tuple[Snake | None, float], closest_enemy_after: tuple[Snake | None, float],
                          dist_ally_before: float, dist_ally_after: float, event_type: str) -> float:
        
        presets: dict[str, float] = self.config.reward_presets.get(snake.role, self.config.reward_presets["Harvester"])
        reward: float = snake.pending_reward
        snake.pending_reward = 0.0
        
        if event_type == 'starve': 
            return reward + presets.get('starve', -50.0)
        if event_type == 'death': 
            return reward + presets.get('death', -50.0)
        if event_type == 'food': 
            return reward + presets.get('food', 10.0)
        
        reward += presets.get('idle_penalty', 0.0)
        
        if dist_food_after < dist_food_before:
            reward += presets.get('step_closer_food', 0.0)
        else:
            reward += presets.get('step_farther_food', 0.0)
            
        enemy_snake: Snake | None = closest_enemy_before[0]
        dist_before: float = closest_enemy_before[1]
        dist_after: float = closest_enemy_after[1]
        
        if enemy_snake and dist_before > 0:
            enemy_role: str = getattr(enemy_snake, 'role', '').lower()
            role_key: str = f"step_closer_enemy_{enemy_role}"
            val_closer: float = presets.get(role_key, presets.get('step_closer_enemy', 0.0))
            
            reward += val_closer if dist_after < dist_before else -val_closer * 0.5
            
        if dist_ally_before > 0:
            reward += presets.get('step_closer_team', 0.0) if dist_ally_after < dist_ally_before else presets.get('step_farther_team', 0.0)

        bs: int = self.config.block_size
        if not (bs <= snake.head.x <= self.config.map_width_px - bs and 
                bs <= snake.head.y <= self.config.map_height_px - bs):
            reward += presets.get('wall_penalty', 0.0)
            
        return reward

    def _handle_combat(self, snake: Snake, victim: Snake, next_head: Point) -> tuple[DeathReason, bool]:
        is_self: bool = (victim is snake)
        v_role: str = getattr(victim, 'role', '')
        presets: dict[str, float] = self.config.reward_presets.get(snake.role, {})
        
        if is_self:
            snake.take_damage(snake.self_damage)
        else:
            victim.take_damage(snake.damage_dealt)
            snake.take_damage(victim.victim_return_damage)
            
            if victim.team_name == snake.team_name:
                snake.pending_reward += presets.get('friendly_fire', -10.0)
            else:
                snake.pending_reward += presets.get('damage_dealt_reward', 0.0)
                if victim.victim_return_damage > 0:
                    v_presets: dict[str, float] = self.config.reward_presets.get(v_role, {})
                    victim.pending_reward += v_presets.get('damage_dealt_reward', 0.0)

        if not snake.collision_survivable or not snake.is_alive:
            snake.is_alive = False
            return DeathReason.SELF_COLLISION if is_self else DeathReason.ENEMY_COLLISION, True

        if next_head in victim.body:
            cut_idx: int = victim.body.index(next_head)
            victim.body = victim.body[:cut_idx]
            
            if not victim.body:
                victim.is_alive = False
                reason: DeathReason = DeathReason.SELF_COLLISION if is_self else DeathReason.ENEMY_COLLISION
                self.analytics.log_death(victim.team_name, reason)
                self.total_deaths += 1
                self.team_stats[victim.team_name].deaths += 1
                
                if not is_self and victim.team_name != snake.team_name:
                    snake.pending_reward += presets.get(f"kill_{v_role.lower()}", presets.get('kill_generic', 5.0))

            v_role_cfg: Any = self.config.role_settings.get(v_role, {})
            init_len: int = getattr(v_role_cfg, 'initial_length', self.config.initial_snake_length)
            victim.score = max(0, len(victim.body) - init_len)
            
            snake.commit_move(next_head)
            snake.body.insert(0, snake.head)
            snake.body.pop()
            return DeathReason.ALIVE, False

        snake.is_alive = False
        return DeathReason.SELF_COLLISION if is_self else DeathReason.ENEMY_COLLISION, True

    def step(self, actions: list[int]) -> tuple[list[tuple[float, bool, int]], bool]:
        self.iteration += 1
        self._rebuild_grid()
        results: list[tuple[float, bool, int]] = []
        
        for t_name in self.team_stats:
            self.team_stats[t_name].current_score = 0
            
        for _, snake in enumerate(self.snakes):
            if not snake.is_alive:
                self._respawn_snake_at_random(snake)
                results.append((0.0, True, 0))
                continue

            dist_food_b: float = self._get_closest_food_dist(snake)
            enemy_b: tuple[Snake | None, float] = self._get_closest_enemy_info(snake)
            dist_ally_b: float = self._get_closest_ally_dist(snake)

            next_head: Point = snake.move_head_prediction(self.config.block_size)
            snake.hp -= self.config.hp_decay_per_step
            
            done: bool = False
            event: str = 'move'
            death_reason: DeathReason = DeathReason.ALIVE
            victim: Snake | None = self._get_snake_at_body_pos(next_head)

            if not (0 <= next_head.x < self.config.map_width_px and 0 <= next_head.y < self.config.map_height_px):
                snake.is_alive = False
                done = True
                event = 'death'
                death_reason = DeathReason.WALL
            elif next_head in self.foods:
                snake.commit_move(next_head)
                snake.body.insert(0, snake.head)
                snake.score += 1
                snake.heal(self.config.food_heal_amount)
                self.foods.remove(next_head)
                self._place_food()
                event = 'food'
                self.analytics.log_food(snake.team_name)
            elif victim:
                is_friendly: bool = victim.team_name == snake.team_name and victim is not snake
                v_role: str = getattr(victim, 'role', '')
                if is_friendly and (snake.role == "Defender" or v_role == "Defender"):
                    snake.commit_move(next_head)
                    snake.body.insert(0, snake.head)
                    snake.body.pop()
                else:
                    death_reason, done = self._handle_combat(snake, victim, next_head)
                    if done: 
                        event = 'death'
            else:
                snake.commit_move(next_head)
                snake.body.insert(0, snake.head)
                snake.body.pop()

            if snake.is_alive and snake.hp <= 0:
                snake.is_alive = False
                done = True
                event = 'starve'
                death_reason = DeathReason.STARVATION

            if done and not snake.is_alive:
                self.total_deaths += 1
                self.team_stats[snake.team_name].deaths += 1
                if death_reason != DeathReason.ALIVE:
                    self.analytics.log_death(snake.team_name, death_reason)
                
                if death_reason == DeathReason.ENEMY_COLLISION and victim and victim.is_alive and victim.team_name != snake.team_name:
                    v_role = getattr(victim, 'role', '')
                    v_presets: dict[str, float] = self.config.reward_presets.get(v_role, {})
                    victim.pending_reward += v_presets.get(f"kill_{snake.role.lower()}", v_presets.get('kill_generic', 5.0))

            reward: float = self._calculate_reward(
                snake, dist_food_b, self._get_closest_food_dist(snake), 
                enemy_b, self._get_closest_enemy_info(snake),
                dist_ally_b, self._get_closest_ally_dist(snake), event
            )
            
            self.team_stats[snake.team_name].current_score += max(0, len(snake.body) - 1) if snake.is_alive else 0
            results.append((reward, done, snake.score))
        
        for t_name, stats in self.team_stats.items():
            if stats.current_score > stats.record:
                stats.record = stats.current_score
        
        self.analytics.update(self.iteration)
        return results, False

    def _get_closest_food_dist(self, snake: Snake) -> float:
        if not self.foods: 
            return 0.0
        
        min_dist: float = float('inf')
        for f in self.foods:
            d: float = math.hypot(snake.head.x - f.x, snake.head.y - f.y)
            if d < min_dist:
                min_dist = d
        return min_dist if min_dist != float('inf') else 0.0
    
    def _get_closest_enemy_info(self, snake: Snake) -> tuple[Snake | None, float]:
        closest: Snake | None = None
        min_dist: float = float('inf')
        
        for s in self.snakes:
            if s is not snake and s.is_alive and s.team_name != snake.team_name:
                d: float = math.hypot(snake.head.x - s.head.x, snake.head.y - s.head.y)
                if d >= min_dist: continue
                min_dist = d
                closest = s
                    
        return closest, min_dist if min_dist != float('inf') else 0.0

    def _get_closest_ally_dist(self, snake: Snake) -> float:
        min_dist: float = float('inf')
        for s in self.snakes:
            if s is not snake and s.is_alive and s.team_name == snake.team_name:
                d: float = math.hypot(snake.head.x - s.head.x, snake.head.y - s.head.y)
                if d >= min_dist: continue
                min_dist = d
                    
        return min_dist if min_dist != float('inf') else 0.0

    def get_state(self) -> GameStateDTO:
        total_time: float = time.time() - self.start_time
        g_stats: GlobalStats = GlobalStats(self.iteration, total_time, self.total_deaths)
        dto: GameStateDTO = GameStateDTO(self.snakes, self.foods, g_stats, self.team_stats, False)
        dto.grid = self.grid 
        return dto
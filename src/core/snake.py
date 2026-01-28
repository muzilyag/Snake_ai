from src.core.types import Direction, Point

class Snake:
    def __init__(self, x, y, team_config, role, config):
        self.head = Point(x, y)
        self.body = []
        for i in range(config.initial_snake_length):
            self.body.append(Point(x - (i * config.block_size), y))
            
        self.direction = Direction.RIGHT
        self.team_name = team_config.name
        self.color = team_config.color
        self.role = role
        
        role_cfg = config.role_settings.get(role, config.role_settings["Harvester"])
        self.max_hp = role_cfg.max_hp
        self.hp = role_cfg.start_hp
        self.damage_dealt = role_cfg.damage_dealt
        self.self_damage = role_cfg.self_damage
        self.collision_survivable = role_cfg.collision_survivable
        
        self.is_alive = True
        self.score = 0
        self.steps_alive = 0
        self.deaths = 0
        
        self.brain_type = team_config.brain_type
        self.reward_mode = team_config.reward_mode
        self.pending_reward = 0.0

    def set_direction(self, direction):
        if not self.is_alive: return
        if (direction == Direction.UP and self.direction == Direction.DOWN) or \
           (direction == Direction.DOWN and self.direction == Direction.UP) or \
           (direction == Direction.LEFT and self.direction == Direction.RIGHT) or \
           (direction == Direction.RIGHT and self.direction == Direction.LEFT):
            return
        self.direction = direction

    def move_head_prediction(self, block_size):
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += block_size
        elif self.direction == Direction.LEFT: x -= block_size
        elif self.direction == Direction.DOWN: y += block_size
        elif self.direction == Direction.UP: y -= block_size
        return Point(x, y)

    def commit_move(self, new_head):
        self.head = new_head
        self.steps_alive += 1

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.is_alive = False
            self.hp = 0

    def heal(self, amount):
        if not self.is_alive: return
        self.hp += amount
        if self.hp > self.max_hp:
            self.hp = self.max_hp
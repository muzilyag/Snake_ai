import random
import os
from src import *

def main():
    ui = PygameRenderer(SETTINGS)
    engine = GameEngine(SETTINGS)
    strategy = MultiAgentStrategy(SETTINGS)
    
    models_pool = {t.name: [] for t in SETTINGS.teams}
    rl_trainers = {}
    ga_trainers = {}
    last_known_records = {t.name: 0 for t in SETTINGS.teams}
    
    for team in SETTINGS.teams:
        if team.brain_type == "RL":
            m = SnakeNet()
            models_pool[team.name] = [m for _ in range(team.count)]
            rl_trainers[team.name] = RLTrainer(m)
        else:
            ga_trainers[team.name] = GATrainer(SnakeNet)
            models_pool[team.name] = [SnakeNet() for _ in range(team.count)]

    epsilon = 80
    current_fps = SETTINGS.fps_train
    
    print("--- Proccessing ---")
    print("Controls: SPACE (speed), G (graphs), S/L (save/load)")

    while True:
        inputs = ui.get_input()
        if inputs['quit']: break
        if inputs['toggle_speed']:
            current_fps = SETTINGS.fps_watch if current_fps == SETTINGS.fps_train else SETTINGS.fps_train
        
        if inputs.get('toggle_graph', False):
            csv_path = engine.analytics.get_current_filename()
            if os.path.exists(csv_path):
                print(f"Opening stats: {csv_path}")
                SnakePlotter(csv_path)
            else:
                print("Stats file not created yet (wait for first interval).")

        state_dto = engine.get_state()
        indices = []
        old_states = []
        
        for i, snake in enumerate(engine.snakes):
            model = models_pool[snake.team_name][i % len(models_pool[snake.team_name])]
            sensors = strategy._get_sensors(snake, state_dto)
            old_states.append(sensors)
            
            if snake.brain_type == "RL" and random.randint(0, 100) < epsilon:
                action_idx = random.randint(0, 2)
            else:
                _, action_idx, _ = strategy.get_action(model, snake, state_dto)
            
            indices.append(action_idx)
            snake.set_direction(strategy._transform_action(snake, action_idx))

        results, _ = engine.step(indices) 
        new_state_dto = engine.get_state()
        
        for i, snake in enumerate(engine.snakes):
            reward, done, score = results[i]
            
            if engine.team_stats[snake.team_name].record > last_known_records[snake.team_name]:
                last_known_records[snake.team_name] = engine.team_stats[snake.team_name].record
                print(f"RECORD! [{snake.team_name}] Score: {last_known_records[snake.team_name]} (Brain: {snake.brain_type})")

            if snake.brain_type == "RL":
                trainer = rl_trainers[snake.team_name]
                new_sensors = strategy._get_sensors(snake, new_state_dto)
                trainer.train_step(old_states[i], indices[i], reward, new_sensors, done)
                if done and epsilon > 5: epsilon -= 0.05
            else:
                if done:
                    fitness = (snake.score * 500) + snake.steps_alive                    
                    ga_manager = ga_trainers[snake.team_name]
                    ga_manager.save_candidate(models_pool[snake.team_name][i % len(SETTINGS.teams)], fitness)
                    models_pool[snake.team_name][i % len(SETTINGS.teams)] = ga_manager.get_offspring()
                    engine.team_stats[snake.team_name].generation += 1

        ui.render(new_state_dto)
        if current_fps > 0: ui.clock.tick(current_fps)

if __name__ == "__main__":
    main()
import random
import os
from src import *
import torch

torch.set_num_threads(1)

def main():
    ui = PygameRenderer(SETTINGS)
    engine = GameEngine(SETTINGS)
    strategy = MultiAgentStrategy(SETTINGS)
    
    models = []
    rl_trainers = []
    
    for team in SETTINGS.teams:
        for _ in range(team.count):
            m = SnakeNet(input_size=28)
            m = torch.jit.script(m)
            models.append(m)
            rl_trainers.append(RLTrainer(m))

    epsilon = 80
    current_fps = SETTINGS.fps_train
    visuals_on = True
    
    print("--- Proccessing ---")
    print("Controls: SPACE (Speed), V (Visuals ON/OFF), G (Graphs), S/L (Save/Load)")

    while True:
        inputs = ui.get_input()
        if inputs['quit']: 
            break
        
        if inputs['toggle_speed']:
            current_fps = SETTINGS.fps_watch if current_fps == SETTINGS.fps_train else SETTINGS.fps_train
            print(f"Speed switched. FPS limit: {current_fps}")

        if inputs['toggle_visuals']:
            visuals_on = not visuals_on
            print(f"Visuals: {'ON' if visuals_on else 'OFF'}")
        
        if inputs.get('toggle_graph', False):
            csv_path = engine.analytics.get_current_filename()
            if os.path.exists(csv_path):
                print(f"Opening stats: {csv_path}")
                try:
                    SnakePlotter(csv_path)
                except Exception as e:
                    print(f"Graph error: {e}")
            else:
                print("Stats file not created yet.")

        if inputs['save']:
            for i, trainer in enumerate(rl_trainers):
                snake = engine.snakes[i]
                trainer.model.save(f"{snake.team_name}_{snake.role}_{i}_model.pth")
            print("All models saved.")
            
        if inputs['load']:
            for i, trainer in enumerate(rl_trainers):
                snake = engine.snakes[i]
                trainer.model.load(f"{snake.team_name}_{snake.role}_{i}_model.pth")
            print("All models loaded.")

        state_dto = engine.get_state()
        indices = []
        old_states = []
        
        for i, snake in enumerate(engine.snakes):
            model = models[i]
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
            
            if snake.brain_type == "RL":
                trainer = rl_trainers[i]
                new_sensors = strategy._get_sensors(snake, new_state_dto)
                trainer.train_step(old_states[i], indices[i], reward, new_sensors, done)
                if done and epsilon > 5: 
                    epsilon -= 0.05

        if visuals_on:
            ui.render(new_state_dto)
            if current_fps > 0: 
                ui.clock.tick(current_fps)
        else:
            if engine.iteration % 1000 == 0:
                ui.render(new_state_dto)

if __name__ == "__main__":
    main()
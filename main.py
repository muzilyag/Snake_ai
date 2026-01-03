import torch
import random
from src.config import SETTINGS
from src.core.engine import GameEngine
from src.ui.pygame_ui import PygameRenderer
from src.ai.model import SnakeNet
from src.ai.rl_trainer import RLTrainer
from src.ai.ga_trainer import GATrainer
from src.input.strategies import MultiAgentStrategy

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
    
    print("--- Процесс пошел ---")

    while True:
        inputs = ui.get_input()
        if inputs['quit']: break
        if inputs['toggle_speed']:
            current_fps = SETTINGS.fps_watch if current_fps == SETTINGS.fps_train else SETTINGS.fps_train

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
            
            # Вывод в консоль только при новом рекорде команды
            if engine.team_stats[snake.team_name].record > last_known_records[snake.team_name]:
                last_known_records[snake.team_name] = engine.team_stats[snake.team_name].record
                print(f"РЕКОРД! [{snake.team_name}] Счет: {last_known_records[snake.team_name]} (Мозг: {snake.brain_type})")

            if snake.brain_type == "RL":
                trainer = rl_trainers[snake.team_name]
                new_sensors = strategy._get_sensors(snake, new_state_dto)
                trainer.train_step(old_states[i], indices[i], reward, new_sensors, done)
                if done and epsilon > 5: epsilon -= 0.05
            else:
                if done:
                    # Упрощенный Fitness: яблоки + время жизни
                    fitness = (snake.score * 500) + snake.steps_alive
                    
                    ga_manager = ga_trainers[snake.team_name]
                    # Сохраняем результат. Если это новый чемпион команды - ga_manager это запомнит
                    ga_manager.save_candidate(models_pool[snake.team_name][i % len(SETTINGS.teams)], fitness)
                    
                    # Мгновенно заменяем умершую змейку мутировавшим потомком текущего чемпиона
                    models_pool[snake.team_name][i % len(SETTINGS.teams)] = ga_manager.get_offspring()
                    engine.team_stats[snake.team_name].generation += 1

        ui.render(new_state_dto)
        if current_fps > 0: ui.clock.tick(current_fps)

if __name__ == "__main__":
    main()
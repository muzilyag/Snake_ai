import random
import torch
from src.config import SETTINGS
from src.core.engine import GameEngine
from src.ui.pygame_ui import PygameRenderer
from src.ai.model import SnakeNet
from src.ai.rl_trainer import RLTrainer
from src.ai.ga_trainer import GATrainer
from src.input.strategies import AIStrategy

MODE = "RL"

def main():
    ui = PygameRenderer(SETTINGS)
    model = SnakeNet()

    if MODE == "RL":
        trainer = RLTrainer(model)
        strategy = AIStrategy(model)
        engine = GameEngine(SETTINGS, strategy)
        epsilon = 80
        
        while True:
            ui.process_events()
            state_old = strategy._get_sensors(engine.get_state())
            
            if random.randint(0, 100) < epsilon:
                action_idx = random.randint(0, 2)
                move = strategy._transform_action(engine.get_state(), action_idx)
                engine.snake.set_direction(move)
                reward, done, _ = engine.step_manual(action_idx)
            else:
                reward, done, action_idx = engine.step()

            state_new = strategy._get_sensors(engine.get_state())
            trainer.train_step(state_old, action_idx, reward, state_new, done)

            if done:
                engine.reset_game()
                if epsilon > 5: epsilon -= 0.2
            
            ui.render(engine.get_state())

    elif MODE == "GA":
        POP_SIZE = 40
        ga_trainer = GATrainer(SnakeNet)
        population = [SnakeNet() for _ in range(POP_SIZE)]
        
        for gen in range(1000):
            fitness_results = []
            for i, net in enumerate(population):
                strat = AIStrategy(net)
                eng = GameEngine(SETTINGS, strat)
                
                for _ in range(500):
                    ui.process_events()
                    reward, done, _ = eng.step()
                    if i % 10 == 0:
                        ui.render(eng.get_state())
                    if done: break
                
                fitness = (eng.score * 1000) + eng.iteration
                fitness_results.append((net, fitness))

            fitness_results.sort(key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in fitness_results[:8]]
            new_pop = elites[:]
            
            while len(new_pop) < POP_SIZE:
                p1, p2 = random.sample(elites, 2)
                child = ga_trainer.crossover(p1, p2)
                new_pop.append(ga_trainer.mutate(child, mutation_rate=0.15))
            
            population = new_pop

if __name__ == "__main__":
    main()
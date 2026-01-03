import torch
import copy
import random

class GATrainer:
    def __init__(self, model_class):
        self.model_class = model_class
        self.best_model_state = None
        self.best_fitness = -1e9

    def save_candidate(self, model, fitness):
        # Если эта змейка лучше всех, что мы видели - запоминаем её "гены"
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True # Сообщаем, что рекорд побит
        return False

    def get_offspring(self, mutation_rate=0.2):
        new_model = self.model_class()
        
        # Если у нас еще нет "чемпиона", даем случайные веса
        if self.best_model_state is None:
            return new_model
            
        # Загружаем гены лучшего
        new_model.load_state_dict(self.best_model_state)
        
        # Жестко мутируем веса
        with torch.no_grad():
            for param in new_model.parameters():
                # Мутируем только часть весов, чтобы не разрушить мозг полностью
                mask = torch.rand_like(param) < mutation_rate
                # Добавляем сильный шум
                param.add_(mask * torch.randn_like(param) * 0.3)
                
        return new_model
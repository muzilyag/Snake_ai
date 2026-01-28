import torch
import copy

class GATrainer:
    def __init__(self, model_class):
        self.model_class = model_class
        self.best_model_state = None
        self.best_fitness = -1e9

    def save_candidate(self, model, fitness):
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True
        return False

    def get_offspring(self, mutation_rate=0.2):
        new_model = self.model_class()
        
        if self.best_model_state is None:
            return new_model
            
        new_model.load_state_dict(self.best_model_state)
        
        with torch.no_grad():
            for param in new_model.parameters():
                mask = torch.rand_like(param) < mutation_rate
                param.add_(mask * torch.randn_like(param) * 0.3)
                
        return new_model
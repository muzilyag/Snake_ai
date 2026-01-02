import torch
import copy
import random

class GATrainer:
    def __init__(self, model_class):
        self.model_class = model_class

    def mutate(self, model, mutation_rate=0.1):
        child = copy.deepcopy(model)
        with torch.no_grad():
            for param in child.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(param) * 0.2
                    param.add_(noise)
        return child

    def crossover(self, parent1, parent2):
        child = self.model_class()
        with torch.no_grad():
            for cp, p1p, p2p in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                mask = torch.rand_like(cp) > 0.5
                cp.data.copy_(torch.where(mask, p1p.data, p2p.data))
        return child
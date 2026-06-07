import os
import json

class CheckpointManager:
    def __init__(self, agents: dict, monitor: any, logger: any, base_dir: str = "checkpoints"):
        self.agents = agents
        self.monitor = monitor
        self.logger = logger
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, current_csv_path: str) -> None:
        iteration = self.monitor.iteration
        
        weights_dir = os.path.join(self.base_dir, f"weights_{iteration}")
        os.makedirs(weights_dir, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent.save(weights_dir)
            
        state = {
            "iteration": iteration,
            "csv_path": current_csv_path,
            "weights_dir": weights_dir,
            "monitor_state": self.monitor.get_state()
        }
        
        chk_path = os.path.join(self.base_dir, f"checkpoint_{iteration}.json")
        with open(chk_path, 'w') as f:
            json.dump(state, f, indent=4)
            
        print(f"\n[{iteration}] УСПЕХ: Чекпоинт сохранен -> {chk_path}")

    def load(self, checkpoint_path: str) -> str:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
            
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
            
        iteration = state["iteration"]
        weights_dir = state["weights_dir"]
        csv_path = state["csv_path"]
        
        for agent_id, agent in self.agents.items():
            agent.load(weights_dir)
            
        self.monitor.load_state(state["monitor_state"])
        self.logger.set_csv_file(csv_path, iteration)
        
        print(f"\n[{iteration}] УСПЕХ: Чекпоинт загружен <- {checkpoint_path}")
        return csv_path
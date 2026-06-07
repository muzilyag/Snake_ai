import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import glob
import torch
import numpy as np
from typing import Dict, Any, List

from src.config import SETTINGS
from src.env.env_wrapper import CppVecEnv
from src.agents.actor import MAPPOAgent
from src.agents.mappo_trainer import MAPPOTrainer
from src.ui.renderer import PygameRenderer
from src.utils.metrics import MetricsLogger
from src.utils.monitor import GameMonitor
from src.plotter import SnakePlotter
from src.utils.checkpointer import CheckpointManager

torch.set_num_threads(8)

def main() -> None:
    total_envs = SETTINGS.total_envs
    
    vec_env = CppVecEnv(total_envs, SETTINGS)
    
    monitor = GameMonitor(SETTINGS)
    logger = MetricsLogger(SETTINGS)
    ui = PygameRenderer(SETTINGS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    observations = vec_env.reset()
    global_states = vec_env.get_global_states()
    global_state_size = len(global_states[0])
    
    agents: Dict[str, MAPPOAgent] = {}
    for team in SETTINGS.teams:
        for i, role in enumerate(team.agent_roles):
            agent_id = f"{team.name}_{role}_{i}"
            agents[agent_id] = MAPPOAgent(agent_id, role, global_state_size, device)

    trainer = MAPPOTrainer(agents, device)
    
    # Инициализация менеджера чекпоинтов
    checkpointer = CheckpointManager(agents, monitor, logger)
    
    current_fps = SETTINGS.fps_train
    visuals_on = True
    
    ROLLOUT_STEPS = 8192
    steps_collected = 0
    
    print("--- Processing MAPPO Environment ---")
    print(f"Hardware: Running {total_envs} parallel environments on C++ ENGINE")
    print("Controls: SPACE (Speed), V (Visuals ON/OFF), G (Graphs), S/L (Save/Load)")

    try:
        while True:
            inputs = ui.get_input()
            if inputs['quit']: break
            if inputs['toggle_speed']:
                current_fps = SETTINGS.fps_watch if current_fps == SETTINGS.fps_train else SETTINGS.fps_train
            if inputs['toggle_visuals']:
                visuals_on = not visuals_on
                print(f"Visuals: {'ON' if visuals_on else 'OFF'}")
                
            if inputs['toggle_graph']:
                SnakePlotter(logger.current_csv_path)

            if inputs.get('save'):
                checkpointer.save(logger.current_csv_path)

            if inputs.get('load'):
                files = glob.glob(os.path.join("checkpoints", "*.json"))
                if files:
                    latest_chk = max(files, key=os.path.getmtime)
                    try:
                        vec_env.main_env.iteration = checkpointer.monitor.iteration
                        logger.current_csv_path = checkpointer.load(latest_chk)
                        vec_env.main_env.iteration = checkpointer.monitor.iteration
                    except Exception as e:
                        print(f"Ошибка загрузки: {e}")
                else:
                    print("Папка 'checkpoints' пуста или не найдена!")

            global_states = vec_env.get_global_states()
            
            agent_obs_batch = {agent_id: [] for agent_id in agents.keys()}
            agent_global_batch = {agent_id: [] for agent_id in agents.keys()}
            
            for env_idx in range(total_envs):
                for agent_id in agents.keys():
                    agent_obs_batch[agent_id].append(observations[env_idx][agent_id])
                    agent_global_batch[agent_id].append(global_states[env_idx])

            actions_list = [{} for _ in range(total_envs)]
            log_probs_list = [{} for _ in range(total_envs)]
            values_list = [{} for _ in range(total_envs)]

            for agent_id, agent in agents.items():
                obs_stack = np.array(agent_obs_batch[agent_id], dtype=np.float32)
                global_stack = np.array(agent_global_batch[agent_id], dtype=np.float32)
                
                acts, logs, vals = agent.act_batched(obs_stack, global_stack)
                
                for env_idx in range(total_envs):
                    actions_list[env_idx][agent_id] = int(acts[env_idx])
                    log_probs_list[env_idx][agent_id] = float(logs[env_idx])
                    values_list[env_idx][agent_id] = float(vals[env_idx])

            next_obs_list, rewards_list, dones_list, infos_list = vec_env.step(actions_list, render=visuals_on)
            
            combined_infos = {}
            for e_idx, inf_dict in enumerate(infos_list):
                for agent_id, info in inf_dict.items():
                    combined_infos[f"{agent_id}_env{e_idx}"] = info
            
            monitor.update(vec_env.main_env.iteration, infos_list[0])
            logger.log_step(combined_infos, monitor) 
            
            for env_idx in range(total_envs):
                for agent_id in agents.keys():
                    agents[agent_id].buffer.push(
                        observations[env_idx][agent_id], global_states[env_idx], 
                        actions_list[env_idx][agent_id], log_probs_list[env_idx][agent_id],
                        rewards_list[env_idx][agent_id], values_list[env_idx][agent_id],
                        dones_list[env_idx].get(agent_id, False)
                    )

            observations = next_obs_list
            steps_collected += total_envs
            
            if steps_collected >= ROLLOUT_STEPS:
                next_global_states = vec_env.get_global_states()
                for agent_id in agents.keys():
                    trainer.train_agent(agent_id, next_global_states)
                steps_collected = 0
                
                print(f"[{vec_env.main_env.iteration}] PyTorch Training Completed. Buffer cleared.")

            if visuals_on:
                ui.render(vec_env.main_env, monitor)
                if current_fps > 0: ui.clock.tick(current_fps)

    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")

if __name__ == "__main__":
    main()
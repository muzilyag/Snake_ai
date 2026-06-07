#include "rewards.h"

namespace RewardSystem {
    void calculate(int num_envs, int snakes_per_env,
                   const std::vector<std::vector<SnakeData>>& snakes,
                   const int* events, const int* killers,
                   const RewardConfig& config, float* rewards) 
    {
        for (int e = 0; e < num_envs; ++e) 
        {
            for (int s = 0; s < snakes_per_env; ++s) 
            {
                int idx = e * snakes_per_env + s;
                int event = events[idx];
                int role = snakes[e][s].role_idx;
                int team = snakes[e][s].team_idx;
                
                float base_reward = 0.0f;
                if (event == 1)
                { 
                    base_reward += config.params[role][0];
                }
                else if (event == 2) 
                {
                    base_reward += config.params[role][1];
                    int killer = killers[idx];
                    if (killer >= 0 && killer != s) 
                    {
                        int k_role = snakes[e][killer].role_idx;
                        int k_team = snakes[e][killer].team_idx;
                        float kill_rew = config.params[k_role][5 + role];
                        rewards[e * snakes_per_env + killer] += kill_rew;
                        for (int os = 0; os < snakes_per_env; ++os) 
                        {
                            if (os != killer && snakes[e][os].team_idx == k_team) 
                            {
                                rewards[e * snakes_per_env + os] += kill_rew * 0.5f;
                            }
                        }
                    }
                }
                else if (event == 3)
                { 
                    base_reward += config.params[role][2];
                }
                else if (event == 4)
                { 
                    base_reward += config.params[role][1] + config.params[role][3];
                }
                else if (event == 5)
                { 
                    base_reward += config.params[role][1];
                }
                else
                { 
                    base_reward += config.params[role][4];
                }

                rewards[idx] += base_reward;

                if (event == 1) 
                {
                    for (int os = 0; os < snakes_per_env; ++os) 
                    {
                        if (os != s && snakes[e][os].team_idx == team) 
                        {
                            rewards[e * snakes_per_env + os] += config.params[role][0] * 0.5f;
                        }
                    }
                }
            }
        }
    }
}
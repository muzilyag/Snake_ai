#include "rewards.h"
#include <unordered_map>

namespace RewardSystem 
{
    void calculate(int num_envs, int snakes_per_env,
                   const int* roles, const int* teams,
                   const int* events, const int* killers,
                   const RewardConfig& config, float* rewards) 
    {
        for (int e = 0; e < num_envs; ++e) 
        {
            std::unordered_map<int, float> team_bonuses;

            for (int s = 0; s < snakes_per_env; ++s) 
            {
                const int idx = e * snakes_per_env + s;
                const int event = events[idx];
                const int role = roles[idx];
                const int team = teams[idx];
                
                float base_reward = 0.0f;
                
                switch (event) 
                {
                    case 1:
                    {
                        base_reward += config.params[role][0];
                        const float bonus = config.params[role][0] * 0.5f;
                        team_bonuses[team] += bonus;
                        base_reward -= bonus;
                        break;
                    }
                    case 2:
                    {
                        base_reward += config.params[role][1];
                        const int killer = killers[idx];
                        if (killer >= 0 && killer != s) 
                        {
                            const int killer_idx = e * snakes_per_env + killer;
                            const int k_role = roles[killer_idx];
                            const int k_team = teams[killer_idx];
                            const float kill_rew = config.params[k_role][5 + role];
                            const float k_bonus = kill_rew * 0.5f;
                            
                            rewards[killer_idx] += kill_rew - k_bonus;
                            team_bonuses[k_team] += k_bonus;
                        }
                        break;
                    }
                    case 3:
                        base_reward += config.params[role][2];
                        break;
                    case 4:
                        base_reward += config.params[role][1] + config.params[role][3];
                        break;
                    case 5:
                        base_reward += config.params[role][1];
                        break;
                    default:
                        base_reward += config.params[role][4];
                        break;
                }

                rewards[idx] += base_reward;
            }

            for (int s = 0; s < snakes_per_env; ++s) 
            {
                const int idx = e * snakes_per_env + s;
                const int team = teams[idx];
                auto it = team_bonuses.find(team);
                if (it != team_bonuses.end()) 
                {
                    rewards[idx] += it->second;
                }
            }
        }
    }
}
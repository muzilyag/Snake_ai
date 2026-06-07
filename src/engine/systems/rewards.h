#ifndef REWARDS_H
#define REWARDS_H
#include "../core/types.h"
#include <vector>

namespace RewardSystem 
{
    void calculate(int num_envs, int snakes_per_env,
                   const std::vector<std::vector<SnakeData>>& snakes,
                   const int* events, const int* killers,
                   const RewardConfig& config, float* rewards);
}

#endif
#ifndef REWARDS_H
#define REWARDS_H
#include "../core/types.h"

namespace RewardSystem 
{
    void calculate(int num_envs, int snakes_per_env,
                   const int* roles, const int* teams,
                   const int* events, const int* killers,
                   const RewardConfig& config, float* rewards);
}

#endif
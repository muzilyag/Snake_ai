#pragma once
#include "../core/types.h"
#include <vector>
#include <random>

namespace PhysicsSystem {
    void update(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods,
                const int* actions,
                std::vector<std::vector<SnakeData>>& snakes,
                std::vector<std::vector<Point>>& foods,
                std::vector<std::mt19937>& rngs,
                int* dones, int* events, int* killers, int* spatial_grid);
}
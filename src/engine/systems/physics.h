#ifndef PHYSICS_H
#define PHYSICS_H
#include "../core/types.h"
#include <vector>
#include <random>

namespace PhysicsSystem 
{
    void update(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods, int max_body_len,
                const int* actions,
                int* alive, float* hp, int* directions, Point* heads, Point* bodies, int* body_lengths, int* scores, const int* roles,
                Point* foods,
                std::vector<std::mt19937>& rngs,
                int* dones, int* events, int* killers, int* spatial_grid);
}

#endif
#ifndef RADAR_H
#define RADAR_H
#include "../core/types.h"
#include <vector>

namespace RadarSystem 
{
    void generate(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods, int max_body_len,
                  const int* alive, const int* teams, const int* roles, const int* directions, const float* hps, const Point* heads, const Point* bodies, const int* body_lengths,
                  const Point* foods,
                  float* obs, float* global, int* spatial_grid);
}

#endif
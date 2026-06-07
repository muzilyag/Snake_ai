#ifndef RADAR_H
#define RADAR_H
#include "../core/types.h"
#include <vector>

namespace RadarSystem 
{
    void generate(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods,
                  const std::vector<std::vector<SnakeData>>& snakes,
                  const std::vector<std::vector<Point>>& foods,
                  float* obs, float* global, int* spatial_grid);
}

#endif
#ifndef VEC_ENV_H
#define VEC_ENV_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include "core/types.h"

namespace py = pybind11;

class VecSnakeEngine 
{
    int num_envs;
    int num_snakes_per_env;
    int obs_size;
    int global_state_size;
    int grid_width;
    int grid_height;
    int block_size;
    int max_body_length;
    int num_foods;

    RewardConfig reward_config;

    std::vector<float> obs_buffer;
    std::vector<float> global_state_buffer;
    std::vector<float> rewards_buffer;
    std::vector<int> dones_buffer;
    std::vector<int> render_buffer;
    std::vector<int> scores_buffer;
    std::vector<int> events_buffer;
    std::vector<int> killers_buffer;
    std::vector<int> spatial_grid;
    std::vector<int> roles_buffer;
    std::vector<int> teams_buffer;

    std::vector<std::vector<SnakeData>> env_snakes;
    std::vector<std::vector<Point>> env_foods;
    std::vector<std::mt19937> env_rngs;

public:
    VecSnakeEngine(int num_envs, int num_snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods, std::vector<float> flat_config);
    void reset_all();
    py::tuple step(py::array_t<int> actions_array);
};

#endif
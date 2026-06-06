#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include "snake.h"

namespace py = pybind11;

class VecSnakeEngine
{
private:
    int num_envs;
    int num_snakes_per_env;
    int obs_size;
    int global_state_size;
    int grid_width;
    int grid_height;
    int block_size;
    int max_body_length;
    int num_foods;

    std::vector<float> obs_buffer;
    std::vector<float> global_state_buffer;
    std::vector<float> rewards_buffer;
    std::vector<int> dones_buffer;
    std::vector<int> render_buffer;
    std::vector<int> scores_buffer;
    std::vector<int> events_buffer;
    std::vector<int> killers_buffer;

    std::vector<std::vector<Snake>> env_snakes;
    std::vector<std::vector<Point>> env_foods;
    std::mt19937 rng;

    void update_observations();
    void generate_obs(int env_idx, int snake_idx, float* obs_ptr);
    void check_collisions(int env_idx);
    void place_food(int env_idx, int food_idx);
    void respawn_snake(int env_idx, int snake_idx);

public:
    VecSnakeEngine(int num_envs, int num_snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods);
    void reset_all();
    py::tuple step(py::array_t<int> actions_array);
};
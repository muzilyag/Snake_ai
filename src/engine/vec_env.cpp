#include "vec_env.h"
#include "systems/physics.h"
#include "systems/rewards.h"
#include "systems/radar.h"
#include <algorithm>

VecSnakeEngine::VecSnakeEngine(int num_envs, int num_snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods, std::vector<float> flat_config)
    : num_envs(num_envs), num_snakes_per_env(num_snakes_per_env),
      obs_size(28), grid_width(grid_w), grid_height(grid_h), block_size(block_s), num_foods(num_foods)
{
    std::random_device rd;
    env_rngs.resize(num_envs);
    for (int i = 0; i < num_envs; ++i) 
    {
        env_rngs[i].seed(rd() + i);
    }

    for (int r = 0; r < 3; ++r) 
    {
        for (int p = 0; p < 9; ++p) 
        {
            reward_config.params[r][p] = flat_config[r * 9 + p];
        }
    }

    global_state_size = (num_snakes_per_env * 4) + (num_foods * 2);
    max_body_length = 100;

    obs_buffer.resize(num_envs * num_snakes_per_env * obs_size, 0.0f);
    global_state_buffer.resize(num_envs * global_state_size, 0.0f);
    rewards_buffer.resize(num_envs * num_snakes_per_env, 0.0f);
    dones_buffer.resize(num_envs * num_snakes_per_env, 0);
    render_buffer.resize(num_envs * num_snakes_per_env * max_body_length * 2, -1000);
    scores_buffer.resize(num_envs * num_snakes_per_env, 0);
    events_buffer.resize(num_envs * num_snakes_per_env, 0);
    killers_buffer.resize(num_envs * num_snakes_per_env, -1);
    
    spatial_grid.resize(num_envs * grid_width * grid_height, -1);
    roles_buffer.resize(num_envs * num_snakes_per_env, 0);
    teams_buffer.resize(num_envs * num_snakes_per_env, 0);

    env_snakes.resize(num_envs);
    env_foods.resize(num_envs, std::vector<Point>(num_foods));

    reset_all();
}

void VecSnakeEngine::reset_all() 
{
    constexpr float max_hps[3] = {100.0f, 150.0f, 200.0f};

    for (int i = 0; i < num_envs; ++i) 
    {
        env_snakes[i].clear();
        for (int s = 0; s < num_snakes_per_env; ++s) 
        {
            SnakeData snake;
            snake.is_alive = true;
            snake.role_idx = s % 3;
            snake.team_idx = s / 3;
            snake.hp = max_hps[snake.role_idx];
            snake.score = 0;
            snake.head = {((s * 2) % grid_width) * block_size, ((s * 2) / grid_width + 5) * block_size};
            snake.direction = 2;
            for (int b = 1; b <= 3; ++b) 
            {
                snake.body.push_back({snake.head.x - b * block_size, snake.head.y});
            }
            env_snakes[i].push_back(snake);
            
            roles_buffer[i * num_snakes_per_env + s] = snake.role_idx;
            teams_buffer[i * num_snakes_per_env + s] = snake.team_idx;
        }
        for (int f = 0; f < num_foods; ++f) 
        {
            std::uniform_int_distribution<int> dist_x(0, grid_width - 1);
            std::uniform_int_distribution<int> dist_y(0, grid_height - 1);
            env_foods[i][f] = {dist_x(env_rngs[i]) * block_size, dist_y(env_rngs[i]) * block_size};
        }
    }
    RadarSystem::generate(num_envs, num_snakes_per_env, grid_width, grid_height, block_size, num_foods, env_snakes, env_foods, obs_buffer.data(), global_state_buffer.data(), spatial_grid.data());
}

py::tuple VecSnakeEngine::step(py::array_t<int> actions_array) 
{
    const int* actions_ptr = actions_array.data();
    
    std::fill(rewards_buffer.begin(), rewards_buffer.end(), 0.0f);
    std::fill(dones_buffer.begin(), dones_buffer.end(), 0);
    std::fill(events_buffer.begin(), events_buffer.end(), 0);
    std::fill(scores_buffer.begin(), scores_buffer.end(), 0);
    std::fill(killers_buffer.begin(), killers_buffer.end(), -1);

    PhysicsSystem::update(num_envs, num_snakes_per_env, grid_width, grid_height, block_size, num_foods, 
                          actions_ptr, env_snakes, env_foods, env_rngs, 
                          dones_buffer.data(), events_buffer.data(), killers_buffer.data(), spatial_grid.data());

    RewardSystem::calculate(num_envs, num_snakes_per_env, roles_buffer.data(), teams_buffer.data(), events_buffer.data(), killers_buffer.data(), reward_config, rewards_buffer.data());

    RadarSystem::generate(num_envs, num_snakes_per_env, grid_width, grid_height, block_size, num_foods, env_snakes, env_foods, obs_buffer.data(), global_state_buffer.data(), spatial_grid.data());

    for (int e = 0; e < num_envs; ++e) 
    {
        for (int s = 0; s < num_snakes_per_env; ++s) 
        {
            scores_buffer[e * num_snakes_per_env + s] = env_snakes[e][s].score;
        }
    }

    std::fill(render_buffer.begin(), render_buffer.end(), -1000);
    for (int e = 0; e < num_envs; ++e) 
    {
        for (int s = 0; s < num_snakes_per_env; ++s) 
        {
            SnakeData& snake = env_snakes[e][s];
            if (snake.is_alive) 
            {
                int base_idx = (e * num_snakes_per_env + s) * max_body_length * 2;
                for (size_t i = 0; i < snake.body.size() && i < max_body_length; ++i) 
                {
                    render_buffer[base_idx + i * 2] = snake.body[i].x;
                    render_buffer[base_idx + i * 2 + 1] = snake.body[i].y;
                }
            } 
            else 
            {
                std::uniform_int_distribution<int> dist_x(2, grid_width - 3);
                std::uniform_int_distribution<int> dist_y(2, grid_height - 3);
                snake.head = {dist_x(env_rngs[e]) * block_size, dist_y(env_rngs[e]) * block_size};
                snake.body.clear();
                for (int b = 1; b <= 3; ++b) 
                {
                    snake.body.push_back({snake.head.x - b * block_size, snake.head.y});
                }
                snake.direction = 2;
                constexpr float max_hps[3] = {100.0f, 150.0f, 200.0f};
                snake.hp = max_hps[snake.role_idx];
                snake.score = 0;
                snake.is_alive = true;
            }
        }
    }

    auto py_obs = py::array_t<float>({num_envs * num_snakes_per_env, obs_size}, obs_buffer.data());
    auto py_global = py::array_t<float>({num_envs, global_state_size}, global_state_buffer.data());
    auto py_rewards = py::array_t<float>(rewards_buffer.size(), rewards_buffer.data());
    auto py_dones = py::array_t<int>(dones_buffer.size(), dones_buffer.data());
    auto py_render = py::array_t<int>({num_envs, num_snakes_per_env, max_body_length, 2}, render_buffer.data());
    auto py_scores = py::array_t<int>(scores_buffer.size(), scores_buffer.data());
    auto py_events = py::array_t<int>(events_buffer.size(), events_buffer.data());
    auto py_killers = py::array_t<int>(killers_buffer.size(), killers_buffer.data());

    return py::make_tuple(py_obs, py_global, py_rewards, py_dones, py_render, py_scores, py_events, py_killers);
}
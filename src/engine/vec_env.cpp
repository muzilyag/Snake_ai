#include "vec_env.h"
#include "systems/physics.h"
#include "systems/rewards.h"
#include "systems/radar.h"
#include <algorithm>

VecSnakeEngine::VecSnakeEngine(int num_envs, int num_snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods, std::vector<float> flat_config,
                               py::array_t<float> obs_arr, py::array_t<float> global_arr, py::array_t<float> rewards_arr,
                               py::array_t<int> dones_arr, py::array_t<int> render_arr, py::array_t<int> scores_arr,
                               py::array_t<int> events_arr, py::array_t<int> killers_arr)
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

    int total_snakes = num_envs * num_snakes_per_env;

    obs_ptr = obs_arr.mutable_data();
    global_state_ptr = global_arr.mutable_data();
    rewards_ptr = rewards_arr.mutable_data();
    dones_ptr = dones_arr.mutable_data();
    render_ptr = render_arr.mutable_data();
    scores_ptr = scores_arr.mutable_data();
    events_ptr = events_arr.mutable_data();
    killers_ptr = killers_arr.mutable_data();

    spatial_grid.resize(num_envs * grid_width * grid_height, -1);

    alive_buffer.resize(total_snakes, 0);
    hp_buffer.resize(total_snakes, 0.0f);
    roles_buffer.resize(total_snakes, 0);
    teams_buffer.resize(total_snakes, 0);
    directions_buffer.resize(total_snakes, 2);
    heads_buffer.resize(total_snakes, {0, 0});
    bodies_buffer.resize(total_snakes * max_body_length, {0, 0});
    body_lengths_buffer.resize(total_snakes, 0);

    foods_buffer.resize(num_envs * num_foods, {0, 0});

    reset_all();
}

void VecSnakeEngine::reset_all() 
{
    constexpr float max_hps[3] = {100.0f, 150.0f, 200.0f};

    for (int e = 0; e < num_envs; ++e) 
    {
        for (int s = 0; s < num_snakes_per_env; ++s) 
        {
            int g_idx = e * num_snakes_per_env + s;
            
            alive_buffer[g_idx] = 1;
            roles_buffer[g_idx] = s % 3;
            teams_buffer[g_idx] = s / 3;
            hp_buffer[g_idx] = max_hps[roles_buffer[g_idx]];
            scores_ptr[g_idx] = 0;
            heads_buffer[g_idx] = {((s * 2) % grid_width) * block_size, ((s * 2) / grid_width + 5) * block_size};
            directions_buffer[g_idx] = 2;
            
            body_lengths_buffer[g_idx] = 3;
            for (int b = 1; b <= 3; ++b) 
            {
                bodies_buffer[g_idx * max_body_length + (b - 1)] = {heads_buffer[g_idx].x - b * block_size, heads_buffer[g_idx].y};
            }
        }
        for (int f = 0; f < num_foods; ++f) 
        {
            std::uniform_int_distribution<int> dist_x(0, grid_width - 1);
            std::uniform_int_distribution<int> dist_y(0, grid_height - 1);
            foods_buffer[e * num_foods + f] = {dist_x(env_rngs[e]) * block_size, dist_y(env_rngs[e]) * block_size};
        }
    }
    
    RadarSystem::generate(num_envs, num_snakes_per_env, grid_width, grid_height, block_size, num_foods, max_body_length,
                          alive_buffer.data(), teams_buffer.data(), roles_buffer.data(), directions_buffer.data(), 
                          hp_buffer.data(), heads_buffer.data(), bodies_buffer.data(), body_lengths_buffer.data(),
                          foods_buffer.data(), obs_ptr, global_state_ptr, spatial_grid.data());
}

void VecSnakeEngine::step(py::array_t<int> actions_array) 
{
    const int* actions_ptr = actions_array.data();
    const int total_snakes = num_envs * num_snakes_per_env;
    
    std::fill_n(rewards_ptr, total_snakes, 0.0f);
    std::fill_n(dones_ptr, total_snakes, 0);
    std::fill_n(events_ptr, total_snakes, 0);
    std::fill_n(killers_ptr, total_snakes, -1);

    PhysicsSystem::update(num_envs, num_snakes_per_env, grid_width, grid_height, block_size, num_foods, max_body_length,
                          actions_ptr, alive_buffer.data(), hp_buffer.data(), directions_buffer.data(), 
                          heads_buffer.data(), bodies_buffer.data(), body_lengths_buffer.data(), scores_ptr, 
                          roles_buffer.data(), foods_buffer.data(), env_rngs, dones_ptr, events_ptr, 
                          killers_ptr, spatial_grid.data());

    RewardSystem::calculate(num_envs, num_snakes_per_env, roles_buffer.data(), teams_buffer.data(), events_ptr, 
                            killers_ptr, reward_config, rewards_ptr);

    RadarSystem::generate(num_envs, num_snakes_per_env, grid_width, grid_height, block_size, num_foods, max_body_length,
                          alive_buffer.data(), teams_buffer.data(), roles_buffer.data(), directions_buffer.data(), 
                          hp_buffer.data(), heads_buffer.data(), bodies_buffer.data(), body_lengths_buffer.data(),
                          foods_buffer.data(), obs_ptr, global_state_ptr, spatial_grid.data());

    std::fill_n(render_ptr, total_snakes * max_body_length * 2, -1000);
    for (int e = 0; e < num_envs; ++e) 
    {
        for (int s = 0; s < num_snakes_per_env; ++s) 
        {
            const int g_idx = e * num_snakes_per_env + s;
            if (alive_buffer[g_idx]) 
            {
                int base_idx = g_idx * max_body_length * 2;
                int len = body_lengths_buffer[g_idx];
                for (int i = 0; i < len; ++i) 
                {
                    render_ptr[base_idx + i * 2] = bodies_buffer[g_idx * max_body_length + i].x;
                    render_ptr[base_idx + i * 2 + 1] = bodies_buffer[g_idx * max_body_length + i].y;
                }
            } 
            else 
            {
                std::uniform_int_distribution<int> dist_x(2, grid_width - 3);
                std::uniform_int_distribution<int> dist_y(2, grid_height - 3);
                heads_buffer[g_idx] = {dist_x(env_rngs[e]) * block_size, dist_y(env_rngs[e]) * block_size};
                
                body_lengths_buffer[g_idx] = 3;
                for (int b = 1; b <= 3; ++b) 
                {
                    bodies_buffer[g_idx * max_body_length + (b - 1)] = {heads_buffer[g_idx].x - b * block_size, heads_buffer[g_idx].y};
                }
                directions_buffer[g_idx] = 2;
                constexpr float max_hps[3] = {100.0f, 150.0f, 200.0f};
                hp_buffer[g_idx] = max_hps[roles_buffer[g_idx]];
                scores_ptr[g_idx] = 0;
                alive_buffer[g_idx] = 1;
            }
        }
    }
}
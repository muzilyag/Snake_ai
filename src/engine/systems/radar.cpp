#include "radar.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

namespace 
{
    inline float get_distance(Point a, Point b) 
    {
        return std::sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
    }
}

namespace RadarSystem 
{
    void generate(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods, int max_body_len,
                  const int* alive, const int* teams, const int* roles, const int* directions, const float* hps, const Point* heads, const Point* bodies, const int* body_lengths,
                  const Point* foods,
                  float* obs, float* global, int* spatial_grid) 
    {
        int obs_size = 28;
        int global_size = (snakes_per_env * 4) + (num_foods * 2);
        int grid_size = grid_w * grid_h;

        for (int e = 0; e < num_envs; ++e) 
        {
            int* grid = spatial_grid + (e * grid_size);
            std::fill(grid, grid + grid_size, -1);

            for (int os = 0; os < snakes_per_env; ++os) 
            {
                const int os_idx = e * snakes_per_env + os;
                if (!alive[os_idx]) 
                {
                    continue;
                }
                
                auto add_point = [&](Point p) 
                {
                    int cx = p.x / block_s; int cy = p.y / block_s;
                    if (cx >= 0 && cx < grid_w && cy >= 0 && cy < grid_h) 
                    {
                        grid[cy * grid_w + cx] = os;
                    }
                };
                
                add_point(heads[os_idx]);
                const int len = body_lengths[os_idx];
                const Point* body = &bodies[os_idx * max_body_len];
                for (int i = 0; i < len; ++i) 
                {
                    add_point(body[i]);
                }
            }

            for (int s = 0; s < snakes_per_env; ++s) 
            {
                const int g_idx = e * snakes_per_env + s;
                int o_idx = g_idx * obs_size;

                if (!alive[g_idx]) 
                {
                    for (int i = 0; i < obs_size; ++i) 
                    {
                        obs[o_idx + i] = 0.0f;
                    }
                    continue;
                }

                std::vector<Point> check_dirs;
                if (directions[g_idx] == 1) check_dirs = {{0, -block_s}, {block_s, 0}, {-block_s, 0}};
                else if (directions[g_idx] == 2) check_dirs = {{block_s, 0}, {0, block_s}, {0, -block_s}};
                else if (directions[g_idx] == 3) check_dirs = {{0, block_s}, {-block_s, 0}, {block_s, 0}};
                else check_dirs = {{-block_s, 0}, {0, -block_s}, {0, block_s}};

                for (const auto& d : check_dirs) 
                {
                    int px = (heads[g_idx].x + d.x) / block_s;
                    int py = (heads[g_idx].y + d.y) / block_s;
                    bool is_wall = (px < 0 || px >= grid_w || py < 0 || py >= grid_h);
                    bool is_friend = false;
                    int enemy_role = -1;

                    if (!is_wall) 
                    {
                        int occ = grid[py * grid_w + px]; 
                        if (occ != -1) 
                        {
                            const int occ_idx = e * snakes_per_env + occ;
                            if (teams[occ_idx] == teams[g_idx]) 
                            {
                                is_friend = true;
                            }
                            else
                            {
                                enemy_role = roles[occ_idx];
                            }
                        }
                    }

                    obs[o_idx++] = (is_wall || is_friend) ? 1.0f : 0.0f;
                    obs[o_idx++] = (enemy_role == 0) ? 1.0f : 0.0f;
                    obs[o_idx++] = (enemy_role == 1) ? 1.0f : 0.0f;
                    obs[o_idx++] = (enemy_role == 2) ? 1.0f : 0.0f;
                }

                obs[o_idx++] = (directions[g_idx] == 4) ? 1.0f : 0.0f;
                obs[o_idx++] = (directions[g_idx] == 2) ? 1.0f : 0.0f;
                obs[o_idx++] = (directions[g_idx] == 1) ? 1.0f : 0.0f;
                obs[o_idx++] = (directions[g_idx] == 3) ? 1.0f : 0.0f;

                float min_dist = std::numeric_limits<float>::max();
                Point closest_food = heads[g_idx];
                for (int f = 0; f < num_foods; ++f) 
                {
                    const int f_idx = e * num_foods + f;
                    float d = get_distance(heads[g_idx], foods[f_idx]);
                    if (d < min_dist) 
                    { 
                        min_dist = d; closest_food = foods[f_idx]; 
                    }
                }

                obs[o_idx++] = (closest_food.x < heads[g_idx].x) ? 1.0f : 0.0f;
                obs[o_idx++] = (closest_food.x > heads[g_idx].x) ? 1.0f : 0.0f;
                obs[o_idx++] = (closest_food.y < heads[g_idx].y) ? 1.0f : 0.0f;
                obs[o_idx++] = (closest_food.y > heads[g_idx].y) ? 1.0f : 0.0f;

                Point closest_ally = heads[g_idx]; Point closest_enemy = heads[g_idx];
                float min_ally_dist = std::numeric_limits<float>::max();
                float min_enemy_dist = std::numeric_limits<float>::max();

                for (int other = 0; other < snakes_per_env; ++other) 
                {
                    const int other_idx = e * snakes_per_env + other;
                    if (!alive[other_idx] || other == s) 
                    {
                        continue;
                    }
                    float d = get_distance(heads[g_idx], heads[other_idx]);
                    if (teams[other_idx] == teams[g_idx] && d < min_ally_dist) 
                    {
                        min_ally_dist = d; closest_ally = heads[other_idx];
                    } 
                    else if (teams[other_idx] != teams[g_idx] && d < min_enemy_dist) 
                    {
                        min_enemy_dist = d; closest_enemy = heads[other_idx];
                    }
                }

                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.x < heads[g_idx].x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.x > heads[g_idx].x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.y < heads[g_idx].y) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.y > heads[g_idx].y) ? 1.0f : 0.0f;

                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.x < heads[g_idx].x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.x > heads[g_idx].x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.y < heads[g_idx].y) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.y > heads[g_idx].y) ? 1.0f : 0.0f;
            }

            int g_idx = e * global_size;
            for (int s = 0; s < snakes_per_env; ++s) 
            {
                const int s_idx = e * snakes_per_env + s;
                global[g_idx++] = alive[s_idx] ? 1.0f : 0.0f;
                global[g_idx++] = static_cast<float>(heads[s_idx].x);
                global[g_idx++] = static_cast<float>(heads[s_idx].y);
                global[g_idx++] = hps[s_idx];
            }
            for (int f = 0; f < num_foods; ++f) 
            {
                const int f_idx = e * num_foods + f;
                global[g_idx++] = static_cast<float>(foods[f_idx].x);
                global[g_idx++] = static_cast<float>(foods[f_idx].y);
            }
        }
    }
}
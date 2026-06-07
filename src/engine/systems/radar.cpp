#include "radar.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace 
{
    inline float get_distance(Point a, Point b) 
    {
        return std::sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
    }
}

namespace RadarSystem 
{
    void generate(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods,
                  const std::vector<std::vector<SnakeData>>& snakes,
                  const std::vector<std::vector<Point>>& foods,
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
                if (!snakes[e][os].is_alive) 
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
                add_point(snakes[e][os].head);
                for (const auto& part : snakes[e][os].body) 
                {
                    add_point(part);
                }
            }

            for (int s = 0; s < snakes_per_env; ++s) 
            {
                const SnakeData& snake = snakes[e][s];
                int o_idx = (e * snakes_per_env + s) * obs_size;

                if (!snake.is_alive) 
                {
                    for (int i = 0; i < obs_size; ++i) 
                    {
                        obs[o_idx + i] = 0.0f;
                    }
                    continue;
                }

                std::vector<Point> check_dirs;
                if (snake.direction == 1) check_dirs = {{0, -block_s}, {block_s, 0}, {-block_s, 0}};
                else if (snake.direction == 2) check_dirs = {{block_s, 0}, {0, block_s}, {0, -block_s}};
                else if (snake.direction == 3) check_dirs = {{0, block_s}, {-block_s, 0}, {block_s, 0}};
                else check_dirs = {{-block_s, 0}, {0, -block_s}, {0, block_s}};

                for (const auto& d : check_dirs) 
                {
                    int px = (snake.head.x + d.x) / block_s;
                    int py = (snake.head.y + d.y) / block_s;
                    bool is_wall = (px < 0 || px >= grid_w || py < 0 || py >= grid_h);
                    bool is_friend = false;
                    int enemy_role = -1;

                    if (!is_wall) 
                    {
                        int occ = grid[py * grid_w + px]; 
                        if (occ != -1) 
                        {
                            const SnakeData& other = snakes[e][occ];
                            if (other.team_idx == snake.team_idx) 
                            {
                                is_friend = true;
                            }
                            else
                            {
                                enemy_role = other.role_idx;
                            }
                        }
                    }

                    obs[o_idx++] = (is_wall || is_friend) ? 1.0f : 0.0f;
                    obs[o_idx++] = (enemy_role == 0) ? 1.0f : 0.0f;
                    obs[o_idx++] = (enemy_role == 1) ? 1.0f : 0.0f;
                    obs[o_idx++] = (enemy_role == 2) ? 1.0f : 0.0f;
                }

                obs[o_idx++] = (snake.direction == 4) ? 1.0f : 0.0f;
                obs[o_idx++] = (snake.direction == 2) ? 1.0f : 0.0f;
                obs[o_idx++] = (snake.direction == 1) ? 1.0f : 0.0f;
                obs[o_idx++] = (snake.direction == 3) ? 1.0f : 0.0f;

                float min_dist = std::numeric_limits<float>::max();
                Point closest_food = snake.head;
                for (const auto& f : foods[e]) 
                {
                    float d = get_distance(snake.head, f);
                    if (d < min_dist) 
                    { 
                        min_dist = d; closest_food = f; 
                    }
                }

                obs[o_idx++] = (closest_food.x < snake.head.x) ? 1.0f : 0.0f;
                obs[o_idx++] = (closest_food.x > snake.head.x) ? 1.0f : 0.0f;
                obs[o_idx++] = (closest_food.y < snake.head.y) ? 1.0f : 0.0f;
                obs[o_idx++] = (closest_food.y > snake.head.y) ? 1.0f : 0.0f;

                Point closest_ally = snake.head; Point closest_enemy = snake.head;
                float min_ally_dist = std::numeric_limits<float>::max();
                float min_enemy_dist = std::numeric_limits<float>::max();

                for (const auto& other : snakes[e]) 
                {
                    if (!other.is_alive || &other == &snake) 
                    {
                        continue;
                    }
                    float d = get_distance(snake.head, other.head);
                    if (other.team_idx == snake.team_idx && d < min_ally_dist) 
                    {
                        min_ally_dist = d; closest_ally = other.head;
                    } 
                    else if (other.team_idx != snake.team_idx && d < min_enemy_dist) 
                    {
                        min_enemy_dist = d; closest_enemy = other.head;
                    }
                }

                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.x < snake.head.x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.x > snake.head.x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.y < snake.head.y) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.y > snake.head.y) ? 1.0f : 0.0f;

                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.x < snake.head.x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.x > snake.head.x) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.y < snake.head.y) ? 1.0f : 0.0f;
                obs[o_idx++] = (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.y > snake.head.y) ? 1.0f : 0.0f;
            }

            int g_idx = e * global_size;
            for (int s = 0; s < snakes_per_env; ++s) 
            {
                const SnakeData& snake = snakes[e][s];
                global[g_idx++] = snake.is_alive ? 1.0f : 0.0f;
                global[g_idx++] = static_cast<float>(snake.head.x);
                global[g_idx++] = static_cast<float>(snake.head.y);
                global[g_idx++] = snake.hp;
            }
            for (int f = 0; f < num_foods; ++f) 
            {
                global[g_idx++] = static_cast<float>(foods[e][f].x);
                global[g_idx++] = static_cast<float>(foods[e][f].y);
            }
        }
    }
}
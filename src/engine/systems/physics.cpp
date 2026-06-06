#include "physics.h"
#include <algorithm>

namespace PhysicsSystem {
    void move_snake(SnakeData& snake, int block_size, bool ate_food) {
        if (!ate_food && !snake.body.empty()) snake.body.pop_back();
        snake.body.insert(snake.body.begin(), snake.head);
        if (snake.direction == 1) snake.head.y -= block_size;
        else if (snake.direction == 2) snake.head.x += block_size;
        else if (snake.direction == 3) snake.head.y += block_size;
        else if (snake.direction == 4) snake.head.x -= block_size;
    }

    void update(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods,
                const int* actions,
                std::vector<std::vector<SnakeData>>& snakes,
                std::vector<std::vector<Point>>& foods,
                std::vector<std::mt19937>& rngs,
                int* dones, int* events, int* killers, int* spatial_grid) 
    {
        float max_hps[3] = {100.0f, 150.0f, 200.0f};
        float damage_dealt[3] = {10.0f, 50.0f, 20.0f};
        float victim_return[3] = {0.0f, 10.0f, 60.0f};
        float self_dmg[3] = {100.0f, 15.0f, 100.0f};

        int grid_size = grid_w * grid_h;

        for (int e = 0; e < num_envs; ++e) {
            for (int s = 0; s < snakes_per_env; ++s) {
                int g_idx = e * snakes_per_env + s;
                SnakeData& snake = snakes[e][s];
                if (!snake.is_alive) { dones[g_idx] = 1; continue; }

                int action = actions[g_idx];
                if (action == 0 && snake.direction != 3) snake.direction = 1;
                else if (action == 1 && snake.direction != 4) snake.direction = 2;
                else if (action == 2 && snake.direction != 1) snake.direction = 3;
                else if (action == 3 && snake.direction != 2) snake.direction = 4;

                bool ate_food = false;
                for (int f = 0; f < num_foods; ++f) {
                    if (snake.head == foods[e][f]) {
                        ate_food = true;
                        snake.score += 1;
                        snake.hp = std::min(snake.hp + 30.0f, max_hps[snake.role_idx]);
                        events[g_idx] = 1;
                        std::uniform_int_distribution<int> dx(0, grid_w - 1);
                        std::uniform_int_distribution<int> dy(0, grid_h - 1);
                        foods[e][f] = {dx(rngs[e]) * block_s, dy(rngs[e]) * block_s};
                        break;
                    }
                }

                move_snake(snake, block_s, ate_food);
                snake.hp -= 0.5f;

                if (snake.hp <= 0.0f) {
                    snake.is_alive = false;
                    dones[g_idx] = 1;
                    events[g_idx] = 3;
                }
            }

            int* grid = spatial_grid + (e * grid_size);
            std::fill(grid, grid + grid_size, -1);

            for (int os = 0; os < snakes_per_env; ++os) {
                if (!snakes[e][os].is_alive) continue;
                for (const auto& part : snakes[e][os].body) {
                    int cx = part.x / block_s; int cy = part.y / block_s;
                    if (cx >= 0 && cx < grid_w && cy >= 0 && cy < grid_h) {
                        grid[cy * grid_w + cx] = os; 
                    }
                }
            }

            for (int s = 0; s < snakes_per_env; ++s) {
                SnakeData& snake = snakes[e][s];
                if (!snake.is_alive) continue;
                int g_idx = e * snakes_per_env + s;

                int hx = snake.head.x / block_s;
                int hy = snake.head.y / block_s;

                if (hx < 0 || hx >= grid_w || hy < 0 || hy >= grid_h) {
                    snake.hp = 0.0f; snake.is_alive = false;
                    dones[g_idx] = 1; events[g_idx] = 4;
                    continue;
                }

                int body_occupant = grid[hy * grid_w + hx];
                
                if (body_occupant == s) {
                    snake.hp = 0.0f; snake.is_alive = false;
                    dones[g_idx] = 1; events[g_idx] = 5;
                    continue;
                }

                for (int os = 0; os < snakes_per_env; ++os) {
                    if (s == os) continue;
                    SnakeData& other = snakes[e][os];
                    if (!other.is_alive) continue;

                    bool head_to_head = (snake.head == other.head);
                    bool head_to_body = (body_occupant == os);

                    if (head_to_head || head_to_body) {
                        if (head_to_head && s > os) continue;
                        
                        other.hp -= damage_dealt[snake.role_idx];
                        snake.hp -= (self_dmg[snake.role_idx] + victim_return[other.role_idx]);
                        
                        int os_idx = e * snakes_per_env + os;
                        if (other.hp <= 0.0f && other.is_alive) {
                            other.is_alive = false; dones[os_idx] = 1;
                            events[os_idx] = 2; killers[os_idx] = s;
                        }
                        if (snake.hp <= 0.0f && snake.is_alive) {
                            snake.is_alive = false; dones[g_idx] = 1;
                            events[g_idx] = 2; killers[g_idx] = os;
                        }
                        if (!snake.is_alive) break;
                    }
                }
            }
        }
    }
}
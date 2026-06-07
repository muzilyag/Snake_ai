#include "physics.h"
#include <algorithm>
#include <array>
#include <vector>

namespace PhysicsSystem 
{
    void move_snake(SnakeData& snake, int block_size, bool ate_food) 
    {
        if (!ate_food && !snake.body.empty()) 
        {
            snake.body.pop_back();
        }
        
        snake.body.push_front(snake.head);
        
        switch (snake.direction) 
        {
            case 1:
                snake.head.y -= block_size;
                break;
            case 2:
                snake.head.x += block_size;
                break;
            case 3:
                snake.head.y += block_size;
                break;
            case 4:
                snake.head.x -= block_size;
                break;
        }
    }

    void update(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods,
                const int* actions,
                std::vector<std::vector<SnakeData>>& snakes,
                std::vector<std::vector<Point>>& foods,
                std::vector<std::mt19937>& rngs,
                int* dones, int* events, int* killers, int* spatial_grid) 
    {
        constexpr std::array<float, 3> max_hps = {100.0f, 150.0f, 200.0f};
        constexpr std::array<float, 3> damage_dealt = {10.0f, 50.0f, 20.0f};
        constexpr std::array<float, 3> victim_return = {0.0f, 10.0f, 60.0f};
        constexpr std::array<float, 3> self_dmg = {100.0f, 15.0f, 100.0f};

        const int grid_size = grid_w * grid_h;
        
        std::vector<int> head_first(grid_size, -1);
        std::vector<int> head_next(snakes_per_env, -1);
        std::vector<int> touched_cells;
        touched_cells.reserve(snakes_per_env);
        std::vector<int> candidates;
        candidates.reserve(snakes_per_env);

        for (int e = 0; e < num_envs; ++e) 
        {
            for (int s = 0; s < snakes_per_env; ++s) 
            {
                const int g_idx = e * snakes_per_env + s;
                SnakeData& snake = snakes[e][s];
                
                if (!snake.is_alive) 
                {
                    dones[g_idx] = 1; 
                    continue; 
                }

                const int action = actions[g_idx];
                switch (action) 
                {
                    case 0:
                        if (snake.direction != 3) snake.direction = 1;
                        break;
                    case 1:
                        if (snake.direction != 4) snake.direction = 2;
                        break;
                    case 2:
                        if (snake.direction != 1) snake.direction = 3;
                        break;
                    case 3:
                        if (snake.direction != 2) snake.direction = 4;
                        break;
                }

                bool ate_food = false;
                for (int f = 0; f < num_foods; ++f) 
                {
                    if (snake.head == foods[e][f]) 
                    {
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

                if (snake.hp <= 0.0f) 
                {
                    snake.is_alive = false;
                    dones[g_idx] = 1;
                    events[g_idx] = 3;
                }
            }

            int* const grid = spatial_grid + (e * grid_size);
            std::fill_n(grid, grid_size, -1);

            for (int os = 0; os < snakes_per_env; ++os) 
            {
                if (!snakes[e][os].is_alive) 
                {
                    continue;
                }
                for (const auto& part : snakes[e][os].body) 
                {
                    const int cx = part.x / block_s; 
                    const int cy = part.y / block_s;
                    if (cx >= 0 && cx < grid_w && cy >= 0 && cy < grid_h) 
                    {
                        grid[cy * grid_w + cx] = os; 
                    }
                }
            }

            for (int os = snakes_per_env - 1; os >= 0; --os) 
            {
                if (!snakes[e][os].is_alive) 
                {
                    continue;
                }
                const int hx = snakes[e][os].head.x / block_s;
                const int hy = snakes[e][os].head.y / block_s;
                if (hx >= 0 && hx < grid_w && hy >= 0 && hy < grid_h) 
                {
                    const int pos = hy * grid_w + hx;
                    if (head_first[pos] == -1) 
                    {
                        touched_cells.push_back(pos);
                    }
                    head_next[os] = head_first[pos];
                    head_first[pos] = os;
                }
            }

            for (int s = 0; s < snakes_per_env; ++s) 
            {
                SnakeData& snake = snakes[e][s];
                if (!snake.is_alive) 
                {
                    continue;
                }
                const int g_idx = e * snakes_per_env + s;

                const int hx = snake.head.x / block_s;
                const int hy = snake.head.y / block_s;

                if (hx < 0 || hx >= grid_w || hy < 0 || hy >= grid_h) 
                {
                    snake.hp = 0.0f; 
                    snake.is_alive = false;
                    dones[g_idx] = 1; 
                    events[g_idx] = 4;
                    continue;
                }

                const int pos = hy * grid_w + hx;
                const int body_occupant = grid[pos];
                
                if (body_occupant == s) 
                {
                    snake.hp = 0.0f; 
                    snake.is_alive = false;
                    dones[g_idx] = 1; 
                    events[g_idx] = 5;
                    continue;
                }

                candidates.clear();
                int curr_head_os = head_first[pos];
                
                while (curr_head_os != -1) 
                {
                    if (curr_head_os != s) 
                    {
                        candidates.push_back(curr_head_os);
                    }
                    curr_head_os = head_next[curr_head_os];
                }
                
                if (body_occupant != -1 && body_occupant != s) 
                {
                    if (std::find(candidates.begin(), candidates.end(), body_occupant) == candidates.end()) 
                    {
                        candidates.push_back(body_occupant);
                    }
                }
                
                std::sort(candidates.begin(), candidates.end());

                for (int os : candidates) 
                {
                    SnakeData& other = snakes[e][os];
                    if (!other.is_alive) 
                    {
                        continue;
                    }

                    const bool head_to_head = (snake.head == other.head);
                    const bool head_to_body = (body_occupant == os);

                    if (head_to_head || head_to_body) 
                    {
                        if (head_to_head && s > os) 
                        {
                            continue;
                        }
                        
                        other.hp -= damage_dealt[snake.role_idx];
                        snake.hp -= (self_dmg[snake.role_idx] + victim_return[other.role_idx]);
                        
                        const int os_idx = e * snakes_per_env + os;
                        if (other.hp <= 0.0f && other.is_alive) 
                        {
                            other.is_alive = false; 
                            dones[os_idx] = 1;
                            events[os_idx] = 2; 
                            killers[os_idx] = s;
                        }
                        if (snake.hp <= 0.0f && snake.is_alive) 
                        {
                            snake.is_alive = false; 
                            dones[g_idx] = 1;
                            events[g_idx] = 2; 
                            killers[g_idx] = os;
                        }
                        if (!snake.is_alive) 
                        {
                            break;
                        }
                    }
                }
            }

            for (int pos : touched_cells) 
            {
                head_first[pos] = -1;
            }
            touched_cells.clear();
        }
    }
}
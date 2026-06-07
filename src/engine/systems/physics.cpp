#include "physics.h"
#include <algorithm>
#include <array>
#include <vector>

namespace PhysicsSystem 
{
    void move_snake(Point* body, int& body_len, Point head, int max_body_len, int block_size, int direction, bool ate_food) 
    {
        if (ate_food) 
        {
            if (body_len < max_body_len) 
            {
                body_len++;
            }
        } 
        
        if (body_len > 0) 
        {
            for (int i = body_len - 1; i > 0; --i) 
            {
                body[i] = body[i - 1];
            }
            body[0] = head;
        }
    }

    void update(int num_envs, int snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods, int max_body_len,
                const int* actions,
                int* alive, float* hp, int* directions, Point* heads, Point* bodies, int* body_lengths, int* scores, const int* roles,
                Point* foods,
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
                
                if (!alive[g_idx]) 
                {
                    dones[g_idx] = 1; 
                    continue; 
                }

                const int action = actions[g_idx];
                switch (action) 
                {
                    case 0:
                        if (directions[g_idx] != 3) directions[g_idx] = 1;
                        break;
                    case 1:
                        if (directions[g_idx] != 4) directions[g_idx] = 2;
                        break;
                    case 2:
                        if (directions[g_idx] != 1) directions[g_idx] = 3;
                        break;
                    case 3:
                        if (directions[g_idx] != 2) directions[g_idx] = 4;
                        break;
                }

                bool ate_food = false;
                for (int f = 0; f < num_foods; ++f) 
                {
                    const int f_idx = e * num_foods + f;
                    if (heads[g_idx] == foods[f_idx]) 
                    {
                        ate_food = true;
                        scores[g_idx] += 1;
                        hp[g_idx] = std::min(hp[g_idx] + 30.0f, max_hps[roles[g_idx]]);
                        events[g_idx] = 1;
                        
                        std::uniform_int_distribution<int> dx(0, grid_w - 1);
                        std::uniform_int_distribution<int> dy(0, grid_h - 1);
                        foods[f_idx] = {dx(rngs[e]) * block_s, dy(rngs[e]) * block_s};
                        break;
                    }
                }

                move_snake(&bodies[g_idx * max_body_len], body_lengths[g_idx], heads[g_idx], max_body_len, block_s, directions[g_idx], ate_food);
                
                switch (directions[g_idx]) 
                {
                    case 1: heads[g_idx].y -= block_s; break;
                    case 2: heads[g_idx].x += block_s; break;
                    case 3: heads[g_idx].y += block_s; break;
                    case 4: heads[g_idx].x -= block_s; break;
                }

                hp[g_idx] -= 0.5f;

                if (hp[g_idx] <= 0.0f) 
                {
                    alive[g_idx] = 0;
                    dones[g_idx] = 1;
                    events[g_idx] = 3;
                }
            }

            int* const grid = spatial_grid + (e * grid_size);
            std::fill_n(grid, grid_size, -1);

            for (int os = 0; os < snakes_per_env; ++os) 
            {
                const int os_idx = e * snakes_per_env + os;
                if (!alive[os_idx]) 
                {
                    continue;
                }
                const int len = body_lengths[os_idx];
                Point* body = &bodies[os_idx * max_body_len];
                for (int i = 0; i < len; ++i) 
                {
                    const int cx = body[i].x / block_s; 
                    const int cy = body[i].y / block_s;
                    if (cx >= 0 && cx < grid_w && cy >= 0 && cy < grid_h) 
                    {
                        grid[cy * grid_w + cx] = os; 
                    }
                }
            }

            for (int os = snakes_per_env - 1; os >= 0; --os) 
            {
                const int os_idx = e * snakes_per_env + os;
                if (!alive[os_idx]) 
                {
                    continue;
                }
                const int hx = heads[os_idx].x / block_s;
                const int hy = heads[os_idx].y / block_s;
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
                const int g_idx = e * snakes_per_env + s;
                if (!alive[g_idx]) 
                {
                    continue;
                }

                const int hx = heads[g_idx].x / block_s;
                const int hy = heads[g_idx].y / block_s;

                if (hx < 0 || hx >= grid_w || hy < 0 || hy >= grid_h) 
                {
                    hp[g_idx] = 0.0f; 
                    alive[g_idx] = 0;
                    dones[g_idx] = 1; 
                    events[g_idx] = 4;
                    continue;
                }

                const int pos = hy * grid_w + hx;
                const int body_occupant = grid[pos];
                
                if (body_occupant == s) 
                {
                    hp[g_idx] = 0.0f; 
                    alive[g_idx] = 0;
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
                    const int os_idx = e * snakes_per_env + os;
                    if (!alive[os_idx]) 
                    {
                        continue;
                    }

                    const bool head_to_head = (heads[g_idx] == heads[os_idx]);
                    const bool head_to_body = (body_occupant == os);

                    if (head_to_head || head_to_body) 
                    {
                        if (head_to_head && s > os) 
                        {
                            continue;
                        }
                        
                        hp[os_idx] -= damage_dealt[roles[g_idx]];
                        hp[g_idx] -= (self_dmg[roles[g_idx]] + victim_return[roles[os_idx]]);
                        
                        if (hp[os_idx] <= 0.0f && alive[os_idx]) 
                        {
                            alive[os_idx] = 0; 
                            dones[os_idx] = 1;
                            events[os_idx] = 2; 
                            killers[os_idx] = s;
                        }
                        if (hp[g_idx] <= 0.0f && alive[g_idx]) 
                        {
                            alive[g_idx] = 0; 
                            dones[g_idx] = 1;
                            events[g_idx] = 2; 
                            killers[g_idx] = os;
                        }
                        if (!alive[g_idx]) 
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
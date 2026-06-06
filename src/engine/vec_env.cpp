#include "vec_env.h"
#include <stdexcept>
#include <algorithm>
#include <limits>

VecSnakeEngine::VecSnakeEngine(int num_envs, int num_snakes_per_env, int grid_w, int grid_h, int block_s, int num_foods)
    : num_envs(num_envs), num_snakes_per_env(num_snakes_per_env),
      obs_size(28), grid_width(grid_w), grid_height(grid_h), block_size(block_s), num_foods(num_foods)
{
    std::random_device rd;
    rng.seed(rd());

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

    env_snakes.resize(num_envs);
    env_foods.resize(num_envs, std::vector<Point>(num_foods));

    reset_all();
}

void VecSnakeEngine::place_food(int env_idx, int food_idx)
{
    std::uniform_int_distribution<int> dist_x(0, grid_width - 1);
    std::uniform_int_distribution<int> dist_y(0, grid_height - 1);
    env_foods[env_idx][food_idx] = {dist_x(rng) * block_size, dist_y(rng) * block_size};
}

void VecSnakeEngine::respawn_snake(int env_idx, int snake_idx)
{
    Snake& snake = env_snakes[env_idx][snake_idx];
    std::uniform_int_distribution<int> dist_x(2, grid_width - 3);
    std::uniform_int_distribution<int> dist_y(2, grid_height - 3);

    snake.head = {dist_x(rng) * block_size, dist_y(rng) * block_size};
    snake.body.clear();

    for (int b = 1; b <= 3; ++b)
    {
        snake.body.push_back({snake.head.x - b * block_size, snake.head.y});
    }

    snake.direction = 2;
    float max_hps[3] = {100.0f, 150.0f, 200.0f};
    snake.hp = max_hps[snake.role_idx];
    snake.score = 0;
    snake.is_alive = true;
}

void VecSnakeEngine::reset_all()
{
    float max_hps[3] = {100.0f, 150.0f, 200.0f};

    for (int i = 0; i < num_envs; ++i)
    {
        env_snakes[i].clear();

        for (int s = 0; s < num_snakes_per_env; ++s)
        {
            Snake snake;
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
        }

        for (int f = 0; f < num_foods; ++f)
        {
            place_food(i, f);
        }
    }

    update_observations();
}

void VecSnakeEngine::generate_obs(int env_idx, int snake_idx, float* obs_ptr)
{
    Snake& snake = env_snakes[env_idx][snake_idx];

    if (!snake.is_alive)
    {
        for (int i = 0; i < obs_size; ++i)
        {
            obs_ptr[i] = 0.0f;
        }
        return;
    }

    int idx = 0;
    std::vector<Point> check_dirs;

    if (snake.direction == 1)
    {
        check_dirs = {{0, -block_size}, {block_size, 0}, {-block_size, 0}};
    }
    else if (snake.direction == 2)
    {
        check_dirs = {{block_size, 0}, {0, block_size}, {0, -block_size}};
    }
    else if (snake.direction == 3)
    {
        check_dirs = {{0, block_size}, {-block_size, 0}, {block_size, 0}};
    }
    else
    {
        check_dirs = {{-block_size, 0}, {0, -block_size}, {0, block_size}};
    }

    for (const auto& d : check_dirs)
    {
        Point p = {snake.head.x + d.x, snake.head.y + d.y};
        bool is_wall = (p.x < 0 || p.x >= grid_width * block_size || p.y < 0 || p.y >= grid_height * block_size);
        bool is_friend = false;
        int enemy_role = -1;

        if (!is_wall)
        {
            for (const auto& other : env_snakes[env_idx])
            {
                if (!other.is_alive)
                {
                    continue;
                }

                for (const auto& part : other.body)
                {
                    if (part == p)
                    {
                        if (other.team_idx == snake.team_idx)
                        {
                            is_friend = true;
                        }
                        else
                        {
                            enemy_role = other.role_idx;
                        }
                        break;
                    }
                }

                if (other.head == p)
                {
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
        }

        if (is_wall || is_friend)
        {
            obs_ptr[idx++] = 1.0f;
        }
        else
        {
            obs_ptr[idx++] = 0.0f;
        }

        if (enemy_role == 0)
        {
            obs_ptr[idx++] = 1.0f;
        }
        else
        {
            obs_ptr[idx++] = 0.0f;
        }

        if (enemy_role == 1)
        {
            obs_ptr[idx++] = 1.0f;
        }
        else
        {
            obs_ptr[idx++] = 0.0f;
        }

        if (enemy_role == 2)
        {
            obs_ptr[idx++] = 1.0f;
        }
        else
        {
            obs_ptr[idx++] = 0.0f;
        }
    }

    if (snake.direction == 4)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (snake.direction == 2)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (snake.direction == 1)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (snake.direction == 3)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    float min_dist = std::numeric_limits<float>::max();
    Point closest_food = snake.head;

    for (const auto& f : env_foods[env_idx])
    {
        float d = get_distance(snake.head, f);

        if (d < min_dist)
        {
            min_dist = d;
            closest_food = f;
        }
    }

    if (closest_food.x < snake.head.x)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (closest_food.x > snake.head.x)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (closest_food.y < snake.head.y)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (closest_food.y > snake.head.y)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    Point closest_ally = snake.head;
    Point closest_enemy = snake.head;
    float min_ally_dist = std::numeric_limits<float>::max();
    float min_enemy_dist = std::numeric_limits<float>::max();

    for (const auto& other : env_snakes[env_idx])
    {
        if (!other.is_alive || &other == &snake)
        {
            continue;
        }

        float d = get_distance(snake.head, other.head);

        if (other.team_idx == snake.team_idx && d < min_ally_dist)
        {
            min_ally_dist = d;
            closest_ally = other.head;
        }
        else if (other.team_idx != snake.team_idx && d < min_enemy_dist)
        {
            min_enemy_dist = d;
            closest_enemy = other.head;
        }
    }

    if (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.x < snake.head.x)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.x > snake.head.x)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.y < snake.head.y)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (min_ally_dist != std::numeric_limits<float>::max() && closest_ally.y > snake.head.y)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.x < snake.head.x)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.x > snake.head.x)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.y < snake.head.y)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }

    if (min_enemy_dist != std::numeric_limits<float>::max() && closest_enemy.y > snake.head.y)
    {
        obs_ptr[idx++] = 1.0f;
    }
    else
    {
        obs_ptr[idx++] = 0.0f;
    }
}

void VecSnakeEngine::update_observations()
{
    for (int e = 0; e < num_envs; ++e)
    {
        for (int s = 0; s < num_snakes_per_env; ++s)
        {
            generate_obs(e, s, &obs_buffer[(e * num_snakes_per_env + s) * obs_size]);
        }

        int g_idx = e * global_state_size;

        for (int s = 0; s < num_snakes_per_env; ++s)
        {
            Snake& snake = env_snakes[e][s];

            if (snake.is_alive)
            {
                global_state_buffer[g_idx++] = 1.0f;
            }
            else
            {
                global_state_buffer[g_idx++] = 0.0f;
            }

            global_state_buffer[g_idx++] = static_cast<float>(snake.head.x);
            global_state_buffer[g_idx++] = static_cast<float>(snake.head.y);
            global_state_buffer[g_idx++] = snake.hp;
        }

        for (int f = 0; f < num_foods; ++f)
        {
            global_state_buffer[g_idx++] = static_cast<float>(env_foods[e][f].x);
            global_state_buffer[g_idx++] = static_cast<float>(env_foods[e][f].y);
        }
    }
}

void VecSnakeEngine::check_collisions(int env_idx)
{
    float max_hps[3] = {100.0f, 150.0f, 200.0f};
    float damage_dealt[3] = {10.0f, 50.0f, 20.0f};
    float victim_return[3] = {0.0f, 10.0f, 60.0f};
    float self_dmg[3] = {100.0f, 15.0f, 100.0f};

    for (int s = 0; s < num_snakes_per_env; ++s)
    {
        Snake& snake = env_snakes[env_idx][s];

        if (!snake.is_alive)
        {
            continue;
        }

        int g_idx = env_idx * num_snakes_per_env + s;

        if (snake.head.x < 0 || snake.head.x >= grid_width * block_size ||
            snake.head.y < 0 || snake.head.y >= grid_height * block_size)
        {
            snake.hp = 0.0f;
            snake.is_alive = false;
            dones_buffer[g_idx] = 1;
            events_buffer[g_idx] = 4;
            continue;
        }

        bool self_collided = false;
        for (const auto& part : snake.body)
        {
            if (snake.head == part)
            {
                self_collided = true;
                break;
            }
        }

        if (self_collided)
        {
            snake.hp = 0.0f;
            snake.is_alive = false;
            dones_buffer[g_idx] = 1;
            events_buffer[g_idx] = 5;
            continue;
        }

        for (int os = 0; os < num_snakes_per_env; ++os)
        {
            if (s == os)
            {
                continue;
            }

            Snake& other = env_snakes[env_idx][os];

            if (!other.is_alive)
            {
                continue;
            }

            bool head_to_head = (snake.head == other.head);
            bool head_to_body = false;
            
            for (const auto& part : other.body)
            {
                if (snake.head == part)
                {
                    head_to_body = true;
                    break;
                }
            }

            if (head_to_head || head_to_body)
            {
                if (head_to_head && s > os)
                {
                    continue;
                }

                other.hp -= damage_dealt[snake.role_idx];
                snake.hp -= (self_dmg[snake.role_idx] + victim_return[other.role_idx]);

                int os_g_idx = env_idx * num_snakes_per_env + os;

                if (other.hp <= 0.0f && other.is_alive)
                {
                    other.is_alive = false;
                    dones_buffer[os_g_idx] = 1;
                    events_buffer[os_g_idx] = 2;
                    killers_buffer[os_g_idx] = s;
                }

                if (snake.hp <= 0.0f && snake.is_alive)
                {
                    snake.is_alive = false;
                    dones_buffer[g_idx] = 1;
                    events_buffer[g_idx] = 2;
                    killers_buffer[g_idx] = os;
                }

                if (!snake.is_alive)
                {
                    break;
                }
            }
        }
    }
}

py::tuple VecSnakeEngine::step(py::array_t<int> actions_array)
{
    auto actions = actions_array.unchecked<1>();

    if (actions.size() != num_envs * num_snakes_per_env)
    {
        throw std::runtime_error("Actions size mismatch");
    }

    std::fill(rewards_buffer.begin(), rewards_buffer.end(), 0.0f);
    std::fill(dones_buffer.begin(), dones_buffer.end(), 0);
    std::fill(events_buffer.begin(), events_buffer.end(), 0);
    std::fill(scores_buffer.begin(), scores_buffer.end(), 0);
    std::fill(killers_buffer.begin(), killers_buffer.end(), -1);

    float max_hps[3] = {100.0f, 150.0f, 200.0f};

    for (int e = 0; e < num_envs; ++e)
    {
        for (int s = 0; s < num_snakes_per_env; ++s)
        {
            int g_idx = e * num_snakes_per_env + s;
            Snake& snake = env_snakes[e][s];

            if (!snake.is_alive)
            {
                dones_buffer[g_idx] = 1;
                continue;
            }

            int action = actions(g_idx);

            if (action == 0 && snake.direction != 3)
            {
                snake.direction = 1;
            }
            else if (action == 1 && snake.direction != 4)
            {
                snake.direction = 2;
            }
            else if (action == 2 && snake.direction != 1)
            {
                snake.direction = 3;
            }
            else if (action == 3 && snake.direction != 2)
            {
                snake.direction = 4;
            }

            bool ate_food = false;

            for (int f = 0; f < num_foods; ++f)
            {
                if (snake.head == env_foods[e][f])
                {
                    ate_food = true;
                    snake.score += 1;

                    if (snake.hp + 30.0f < max_hps[snake.role_idx])
                    {
                        snake.hp += 30.0f;
                    }
                    else
                    {
                        snake.hp = max_hps[snake.role_idx];
                    }

                    events_buffer[g_idx] = 1;
                    place_food(e, f);
                    break;
                }
            }

            move_snake(snake, block_size, ate_food);
            snake.hp -= 0.5f;

            if (snake.hp <= 0.0f)
            {
                snake.is_alive = false;
                dones_buffer[g_idx] = 1;
                events_buffer[g_idx] = 3;
            }
        }

        check_collisions(e);
    }

    for (int e = 0; e < num_envs; ++e)
    {
        for (int s = 0; s < num_snakes_per_env; ++s)
        {
            int g_idx = e * num_snakes_per_env + s;
            scores_buffer[g_idx] = env_snakes[e][s].score;
        }
    }

    update_observations();
    std::fill(render_buffer.begin(), render_buffer.end(), -1000);

    for (int e = 0; e < num_envs; ++e)
    {
        for (int s = 0; s < num_snakes_per_env; ++s)
        {
            Snake& snake = env_snakes[e][s];

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
                respawn_snake(e, s);
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
#include "snake.h"

float get_distance(const Point& a, const Point& b)
{
    return std::hypot(a.x - b.x, a.y - b.y);
}

void move_snake(Snake& snake, int block_size, bool grow)
{
    if (!snake.is_alive)
    {
        return;
    }

    snake.body.insert(snake.body.begin(), snake.head);

    if (snake.direction == 1)
    {
        snake.head.y -= block_size;
    }
    else if (snake.direction == 2)
    {
        snake.head.x += block_size;
    }
    else if (snake.direction == 3)
    {
        snake.head.y += block_size;
    }
    else if (snake.direction == 4)
    {
        snake.head.x -= block_size;
    }

    if (!grow && !snake.body.empty())
    {
        snake.body.pop_back();
    }
}
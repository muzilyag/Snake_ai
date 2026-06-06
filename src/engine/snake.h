#pragma once
#include <vector>
#include <cmath>

struct Point
{
    int x;
    int y;

    bool operator==(const Point& other) const
    {
        return x == other.x && y == other.y;
    }
};

struct Snake
{
    bool is_alive;
    int role_idx;
    int team_idx;
    float hp;
    int score;
    Point head;
    std::vector<Point> body;
    int direction;
};

float get_distance(const Point& a, const Point& b);
void move_snake(Snake& snake, int block_size, bool grow);
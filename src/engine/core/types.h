#ifndef TYPES_H
#define TYPES_H
#include <deque>

struct Point 
{
    int x; 
    int y;
    bool operator==(const Point& o) const 
    { 
        return x == o.x && y == o.y; 
    }
};

struct SnakeData 
{
    bool is_alive;
    int role_idx;
    int team_idx;
    float hp;
    int score;
    int direction;
    Point head;
    std::deque<Point> body;
};

struct RewardConfig 
{
    float params[3][9];
};

#endif
#ifndef TYPES_H
#define TYPES_H

struct Point 
{
    int x; 
    int y;
    bool operator==(const Point& o) const 
    { 
        return x == o.x && y == o.y; 
    }
};

struct RewardConfig 
{
    float params[3][9];
};

#endif
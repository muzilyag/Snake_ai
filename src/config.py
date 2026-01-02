from pydantic import BaseModel
from typing import Tuple

class Colors(BaseModel):
    # Светлая тема
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    GRAY_LIGHT: Tuple[int, int, int] = (230, 230, 230) # Для сетки
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    GREEN: Tuple[int, int, int] = (0, 255, 0)
    BLUE: Tuple[int, int, int] = (0, 0, 255)
    
    BACKGROUND: Tuple[int, int, int] = WHITE
    GRID: Tuple[int, int, int] = GRAY_LIGHT
    SNAKE: Tuple[int, int, int] = BLUE
    FOOD: Tuple[int, int, int] = RED
    TEXT: Tuple[int, int, int] = BLACK

class GameConfig(BaseModel):
    width: int = 640
    height: int = 480
    block_size: int = 20
    speed: int = 60
    colors: Colors = Colors()

SETTINGS = GameConfig()
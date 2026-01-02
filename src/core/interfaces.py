from typing import Protocol
from src.core.types import Direction, GameStateDTO

class MoveStrategy(Protocol):
    def get_next_move(self, state: GameStateDTO) -> Direction:
        ...

class GameObserver(Protocol):
    def render(self, state: GameStateDTO) -> None:
        ...
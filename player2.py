import random
from typing import List

class Player:
    def __init__(self):
        self.gold = 0
    
    def MoveDecision(self, grid: List[List[int]], gold: int, gold_2: int) -> List[int]:
        """简单的随机策略"""
        return [random.randint(0, 3) for _ in range(3)]

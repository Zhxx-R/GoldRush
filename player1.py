
import random
import numpy as np
from typing import List

class Player:
    def __init__(self):
        self.gold = 0
        # 这里可以加载训练好的模型
        # self.model = load_model("path/to/model")
    
    def MoveDecision(self, grid: List[List[int]], gold: int, gold_2: int) -> List[int]:
        """
        根据当前状态做出移动决策
        """
        self.gold = gold
        # print("---------------")
        # print(grid)

        # 找到自己的位置
        my_pos = self._find_player_position(grid, -9)
        if my_pos is None:
            return [random.randint(0, 3) for _ in range(3)]
        
        # 简单策略：优先收集金币，避开炸弹
        actions = []
        current_pos = my_pos
        
        for step in range(3):
            # 寻找最近的金币
            target = self._find_nearest_coin(grid, current_pos)
            
            if target:
                # 朝目标移动
                action = self._move_towards_target(current_pos, target)
            else:
                # 没有金币就随机移动
                action = random.randint(0, 3)
            
            # 检查移动是否安全
            new_pos = self._get_new_position(current_pos, action)
            if not self._is_safe_move(grid, new_pos):
                # 不安全就随机选择其他方向
                safe_actions = []
                for a in range(4):
                    test_pos = self._get_new_position(current_pos, a)
                    if self._is_safe_move(grid, test_pos):
                        safe_actions.append(a)
                action = random.choice(safe_actions) if safe_actions else random.randint(0, 3)
                new_pos = self._get_new_position(current_pos, action)
            
            actions.append(action)
            # 重要：更新当前位置，为下一步做准备
            if self._is_safe_move(grid, new_pos):
                current_pos = new_pos
                # 模拟在grid中更新位置（用于后续步骤的计算）
                if 0 <= new_pos[0] < 17 and 0 <= new_pos[1] < 17:
                    grid[my_pos[0]][my_pos[1]] = 0  # 清除旧位置
                    grid[new_pos[0]][new_pos[1]] = -9  # 设置新位置
                    my_pos = new_pos
        
        return actions
    
    def _find_player_position(self, grid, player_code):
        """找到玩家位置"""
        for i in range(17):
            for j in range(17):
                if grid[i][j] == player_code:
                    return (i, j)
        return None
    
    def _find_nearest_coin(self, grid, pos):
        """找到最近的金币"""
        min_dist = float('inf')
        target = None
        
        for i in range(17):
            for j in range(17):
                if grid[i][j] >= 1:  # 有金币  
                    dist = abs(i - pos[0]) + abs(j - pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        target = (i, j)
        
        return target
    
    def _move_towards_target(self, current_pos, target_pos):
        """朝目标移动"""
        dx = target_pos[0] - current_pos[0]  # 行差值
        dy = target_pos[1] - current_pos[1]  # 列差值
        
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 0  # 下(1)或上(0)
        else:
            return 3 if dy > 0 else 2  # 右(3)或左(2)
    
    def _get_new_position(self, pos, action):
        """根据动作获取新位置"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        dx, dy = directions[action]
        return (pos[0] + dx, pos[1] + dy)
    
    def _is_safe_move(self, grid, pos):
        """检查移动是否安全"""
        x, y = pos
        
        # 边界检查
        if x < 0 or x >= 17 or y < 0 or y >= 17:
            return False
        
        # 障碍物和炸弹检查
        if grid[x][y] == -1 or grid[x][y] == -3:
            return False
        
        # 对手位置检查
        if grid[x][y] == -2:
            return False
        
        return True


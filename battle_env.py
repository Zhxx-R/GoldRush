
import gym
import numpy as np
from gym import spaces
from config import *
import random

class BattleEnv(gym.Env):
    def __init__(self, bomb_damage_percent=10, bomb_refresh_rounds=50, npc_spawn_rounds=20):#参数
        super().__init__()
        
        # 地图配置
        self.map_size = 17
        self.center_zone = 9  # 9x9中央区域
        self.max_rounds = MAX_ROUNDS
        
        # 游戏参数
        self.bomb_damage_percent = bomb_damage_percent
        self.bomb_refresh_rounds = bomb_refresh_rounds  # 50回合刷新炸弹
        self.npc_spawn_rounds = npc_spawn_rounds
        
        # 动作空间：每个玩家一次选择3步移动
        self.action_space = spaces.MultiDiscrete([5, 5, 5])  # 3次移动，每次5个方向
        
        # 地图状态：0:空, 1:障碍物, 2:炸弹, 3:玩家1, 4:玩家2
        # 金币数量单独存储
        self.observation_space = spaces.Dict({
            'map': spaces.Box(low=0, high=4, shape=(17, 17), dtype=np.int32),
            'coin_map': spaces.Box(low=0, high=99, shape=(17, 17), dtype=np.int32),  # 每个位置的金币数量
            'player_pos': spaces.Box(low=0, high=16, shape=(2,), dtype=np.int32),
            'player_coins': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'enemy_pos': spaces.Box(low=0, high=16, shape=(2,), dtype=np.int32),
            'enemy_coins': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'round': spaces.Box(low=0, high=300, shape=(1,), dtype=np.int32),
            'grid': spaces.Box(low=-9, high=99, shape=(17, 17), dtype=np.int32)  # 新增：比赛接口格式的grid
        })
        
        self.reset()
    
    def reset(self):
        # 初始化地图
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        self.coin_map = np.zeros((self.map_size, self.map_size), dtype=np.int32)  # 金币数量地图
        
        pos_init = random.randint(0, 1)

        if(pos_init == 0):
            self.player1_pos = np.array([0, 0])
            self.player2_pos = np.array([16, 16])
        else:
            self.player1_pos = np.array([16, 16])
            self.player2_pos = np.array([0, 0])
            
        # 玩家金币
        self.player1_coins = 0
        self.player2_coins = 0
        
        # 回合数
        self.current_round = 0
        
        # 生成初始障碍物
        self._generate_obstacles()
        
        # 生成初始炸弹
        self._generate_initial_bombs()
        
        # 生成初始金币
        self._generate_coins()
        
        self._spawn_npc()

        # 更新地图
        self._update_map()
        
        return self._get_observation()

    def _generate_initial_bombs(self):
        """生成初始炸弹"""
        num_bombs = random.randint(4, 8)  # 初始炸弹数量
        placed_bombs = []
        
        for _ in range(num_bombs):
            attempts = 0
            while attempts < 20:  # 最多尝试20次
                x, y = random.randint(0, 16), random.randint(0, 16)
                if (self.map[x, y] == 0 and 
                    (x, y) not in [(0, 0), (16, 16)]):  # 避免在出生点生成
                    self.map[x, y] = 2
                    placed_bombs.append((x, y))
                    break
                attempts += 1
        
        # 为公平性，尝试对称放置一些炸弹
        for x, y in placed_bombs[:num_bombs//2]:
            if random.random() < 0.3:  # 30%概率对称放置
                sym_x, sym_y = 16 - x, 16 - y
                if (0 <= sym_x < 17 and 0 <= sym_y < 17 and 
                    self.map[sym_x, sym_y] == 0 and
                    (sym_x, sym_y) not in [(0, 0), (16, 16)]):
                    self.map[sym_x, sym_y] = 2
        
        initial_bomb_count = np.sum(self.map == 2)
        # print(f"初始化: 生成 {initial_bomb_count} 个炸弹")
    
    def step(self, actions, first_player=0):
        """
        actions: [player1_actions, player2_actions]
        每个actions是长度为3的数组，表示3次移动
        """
        prev_p1_coins = self.player1_coins
        prev_p2_coins = self.player2_coins
        
        
        collision_info = {
            'player1': {'obstacle_hits': 0, 'bomb_hits': 0, 'bomb_damage': 0},
            'player2': {'obstacle_hits': 0, 'bomb_hits': 0, 'bomb_damage': 0}
        }
        
        # 1) 回合计数
        self.current_round += 1
    
        # NPC生成
        if self.current_round % self.npc_spawn_rounds == 1 and self.current_round != 1:
            self._spawn_npc()

        # 2) 执行顺序
        order = [first_player, 1 - first_player]

        # 3) 先手决策后执行，执行后立即更新地图状态
        for pid in order:
            player_collision = self._execute_player_moves(pid, actions[pid])
            collision_info[f'player{pid+1}'] = player_collision
            self._update_map()

        # 4) 回合结束的全局更新
        self._generate_coins()
        
        # 每50回合刷新炸弹
        if self.current_round % self.bomb_refresh_rounds == 1:
            self._refresh_bombs()
        
        done = self.current_round >= self.max_rounds
        
        reward = self._calculate_reward(prev_p1_coins, prev_p2_coins, collision_info, done)
        
        return self._get_observation(), reward, done, {}
    
    
    def _execute_player_moves(self, player_id, actions):
        """执行单个玩家的3次移动"""
        current_pos = self.player1_pos if player_id == 0 else self.player2_pos
        collision_info = {'obstacle_hits': 0, 'bomb_hits': 0, 'bomb_damage': 0}

        for action in actions:  # 执行3次移动
            new_pos = self._get_new_position(current_pos, action)
            
            # 检查移动合法性
            if self._is_valid_move(new_pos, player_id):
                current_pos = new_pos
                
                # 检查金币
                if self.coin_map[new_pos[0], new_pos[1]] > 0:
                    self._collect_coins(player_id, new_pos)
                
                # 检查炸弹
                elif self.map[new_pos[0], new_pos[1]] == 2:
                    damage = self._trigger_bomb(player_id, new_pos)
                    collision_info['bomb_hits'] += 1
                    collision_info['bomb_damage'] += damage
            else:
                x, y = new_pos
                if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                    # 超出边界，撞墙
                    collision_info['obstacle_hits'] += 1
                elif self.map[x, y] == 1:
                    # 撞到障碍物
                    collision_info['obstacle_hits'] += 1

        # 更新玩家位置
        if player_id == 0:
            self.player1_pos = current_pos
        else:
            self.player2_pos = current_pos

        return collision_info
    
    
    def _get_new_position(self, pos, action):
        """根据动作获取新位置"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # 上下左右停留
        dx, dy = directions[action]
        return np.array([pos[0] + dx, pos[1] + dy])
    
    def _is_valid_move(self, pos, player_id):
        """检查移动是否合法"""
        x, y = pos
        
        # 边界检查
        if x < 0 or x >= self.map_size or y < 0 or y >= self.map_size:
            return False
        
        # 障碍物检查
        if self.map[x, y] == 1:
            return False
        
        # 其他玩家检查
        other_player_pos = self.player2_pos if player_id == 0 else self.player1_pos
        if np.array_equal(pos, other_player_pos):
            return False
        
        return True
    
    def _collect_coins(self, player_id, pos):
        """收集该位置的所有金币"""
        coins = self.coin_map[pos[0], pos[1]]
        
        if player_id == 0:
            self.player1_coins += coins
        else:
            self.player2_coins += coins
        
        # 清空该位置的金币
        self.coin_map[pos[0], pos[1]] = 0
    
    def _trigger_bomb(self, player_id, pos):
        """触发炸弹 - 损失X%金币向上取整"""
        damage = 0
        if player_id == 0:
            if self.player1_coins > 0:
                damage = int(np.ceil(self.player1_coins * self.bomb_damage_percent / 100))
                self.player1_coins = max(0, self.player1_coins - damage)
        else:
            if self.player2_coins > 0:
                damage = int(np.ceil(self.player2_coins * self.bomb_damage_percent / 100))
                self.player2_coins = max(0, self.player2_coins - damage)
        
        # 移除炸弹
        self.map[pos[0], pos[1]] = 0
        return damage
    
    def _generate_obstacles(self):
        """生成固定障碍物模式"""
        # 清空地图
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        
        # 固定障碍物模式1：十字形 + 边角障碍
        obstacles = [
            # 中央十字形
            (8, 5),(8, 6), (8, 11), (8, 10),  # 水平线
            (6, 8), (11, 8), (10, 8),  # 垂直线
            
            # 四个角落的小障碍群
            (2, 2), (2, 3), (3, 2),           # 左上角
            (2, 13), (2, 14), (3, 14),        # 右上角  
            (13, 2), (14, 2), (14, 3),        # 左下角
            (13, 14), (14, 13), (14, 14),     # 右下角
            
            # 中间区域的对称障碍
            (5, 5), (5, 11), (11, 5), (11, 11),  # 四个对称点
            (4, 8), (5, 8), (12, 8), (8, 4), (8, 12),    # 十字延伸
        ]
        
        # 放置障碍物
        for x, y in obstacles:
            if 0 <= x < 17 and 0 <= y < 17:
                # 确保不在玩家出生点
                if (x, y) not in [(0, 0), (16, 16)]:
                    self.map[x, y] = 1
        
        # print(f"固定障碍物: 生成 {np.sum(self.map == 1)} 个障碍物")
    
    def _is_map_connected(self, test_map):
        """检查地图是否连通（使用BFS）"""
        # 从(0,0)开始BFS，看能否到达(16,16)
        visited = np.zeros((17, 17), dtype=bool)
        queue = [(0, 0)]
        visited[0, 0] = True
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            x, y = queue.pop(0)
            
            # 如果到达了(16,16)，说明连通
            if x == 16 and y == 16:
                return True
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < 17 and 0 <= ny < 17 and 
                    not visited[nx, ny] and test_map[nx, ny] != 1):
                    visited[nx, ny] = True
                    queue.append((nx, ny))
        
        return False  # 无法到达(16,16)
    
    def _generate_coins(self):
        """在中央9x9区域生成金币 - 大幅增加金币数量和位置"""
        center_start = (self.map_size - self.center_zone) // 2
        center_end = center_start + self.center_zone
        
        if random.random() < 0.8:
            # 生成1-3个位置的金币（大幅增加位置数）
            num_positions = random.randint(1, 3)

            for _ in range(num_positions):
                attempts = 0
                while attempts < 5:  # 增加尝试次数确保生成
                    x = random.randint(center_start, center_end - 1)
                    y = random.randint(center_start, center_end - 1)
                    
                    # 只在空白处且没有金币的地方生成
                    if self.map[x, y] == 0:
                        # 每个位置生成1-15个金币--- fix no use 1 or 2 to  avoid player id
                        coin_count = random.randint(1, 15)
                        self.coin_map[x, y] += coin_count  
                        break
                    attempts += 1
    
    def _refresh_bombs(self):
        """每50回合刷新炸弹 - 不重置金币"""
        # 清除旧炸弹
        old_bomb_count = np.sum(self.map == 2)
        self.map[self.map == 2] = 0
        
        # 生成新炸弹 - 增加数量并确保分布相对公平
        num_bombs = random.randint(8, 15)
        placed_bombs = []
        
        for _ in range(num_bombs):
            attempts = 0
            while attempts < 20:
                x, y = random.randint(0, 16), random.randint(0, 16)
                if (self.map[x, y] == 0 and self.coin_map[x, y] == 0 and
                    (x, y) not in [(0, 0), (16, 16)]):
                    self.map[x, y] = 2
                    placed_bombs.append((x, y))
                    break
                attempts += 1
        
        # 尝试在对称位置也放置一些炸弹以保证公平
        for x, y in placed_bombs[:num_bombs//2]:
            if random.random() < 0.4:
                sym_x, sym_y = 16 - x, 16 - y
                if (0 <= sym_x < 17 and 0 <= sym_y < 17 and 
                    self.map[sym_x, sym_y] == 0 and self.coin_map[sym_x, sym_y] == 0 and
                    (sym_x, sym_y) not in [(0, 0), (16, 16)]):
                    self.map[sym_x, sym_y] = 2
                    placed_bombs.append((sym_x, sym_y))
        
        new_bomb_count = np.sum(self.map == 2)
        # print(f"炸弹刷新: 清除 {old_bomb_count} 个旧炸弹, 生成 {new_bomb_count} 个新炸弹")
        
    
    def _spawn_npc(self):
        """生成NPC - 固定对称路径掉落金币，保证公平性"""
        # print(f"第 {self.current_round} 回合: NPC出现!")
        
        # 定义固定的基础路径模式（上半部分）
        base_paths = [
            # 路径1
            [(0, 5), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (0, 11)],
            # 路径2
            [(5, 0), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (11, 0)],
           
        ]
        
        # 生成所有路径的四象限对称位置
        all_paths = set()
        
        # 对每条基础路径都生成对称位置
        for path in base_paths:
            for x, y in path:
                # 四个象限的对称位置
                positions = [
                    (x, y),                    # 原始位置（左上）
                    (x, 16 - y),              # 左下对称
                    (16 - x, y),              # 右上对称  
                    (16 - x, 16 - y)          # 右下对称
                ]
                
                # 添加所有有效位置
                for px, py in positions:
                    if (0 <= px < 17 and 0 <= py < 17):
                        all_paths.add((px, py))
        
        # 在所有路径位置掉落金币
        coins_dropped = 0
        for x, y in all_paths:
            if (0 <= x < 17 and 0 <= y < 17 and 
                self.map[x, y] == 0 and  # 只在空白处掉落
                not np.array_equal([x, y], self.player1_pos) and  # 避免在玩家1位置
                not np.array_equal([x, y], self.player2_pos)):    # 避免在玩家2位置
                # 每个位置掉落5-10个金币
                coin_count = 3
                self.coin_map[x, y] += coin_count#fix
                coins_dropped += coin_count
        
        # print(f"掉落金币总数: {coins_dropped}")

    def _get_non_center_position(self, center_start, center_end):
        """在非中央9x9区域获取随机位置"""
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(0, 16)
            y = random.randint(0, 16)
            
            # 确保不在中央9x9区域
            if not (center_start <= x < center_end and center_start <= y < center_end):
                # 确保不是障碍物
                if self.map[x, y] != 1:
                    return x, y
        
        # 如果找不到位置，返回一个默认位置
        return 1, 1
    
    def _update_map(self):
        """更新地图显示玩家位置"""
        # 清除之前的玩家位置
        self.map[self.map == 3] = 0
        self.map[self.map == 4] = 0
        
        # 设置当前玩家位置
        self.map[self.player1_pos[0], self.player1_pos[1]] = 3
        self.map[self.player2_pos[0], self.player2_pos[1]] = 4
    
    def _calculate_reward(self, prev_p1_coins, prev_p2_coins, collision_info, done):
        """计算奖励"""
        reward = 0.0
        my_coin_gain = self.player1_coins - prev_p1_coins
        reward += np.sign(my_coin_gain) * np.sqrt(abs(my_coin_gain))

        # # coin_diff = self.player1_coins - self.player2_coins
        # enemy_coin_gain = self.player2_coins - prev_p2_coins
        # coin_reward = my_coin_gain - enemy_coin_gain
        # reward += 0.3 * np.sign(coin_reward) * np.sqrt(abs(coin_reward)) if coin_reward != 0 else 0


        # enemy_coin_gain = self.player2_coins - prev_p2_coins
        # coin_reward = my_coin_gain - enemy_coin_gain
        # reward += (coin_reward/10)

        bomb_penalty = collision_info['player1']['bomb_damage']

        if collision_info['player1']['bomb_hits'] == 0:
            bomb_penalty = 0
        else:
            bomb_penalty = min(bomb_penalty / 5, 10)
        
        collision_penalty = collision_info['player1']['obstacle_hits'] * 2
        
        # 总惩罚：炸弹损伤 + 碰撞惩罚
        total_penalty = bomb_penalty + collision_penalty
        reward -= total_penalty


        # 游戏结束奖励/惩罚（第300回合）
        if done:
            coin_diff = self.player1_coins - self.player2_coins     
            # reward += (coin_diff/10)
            reward += np.sign(coin_diff) * min(np.sqrt(abs(coin_diff)),20)
        
        return reward
    
    def _get_observation(self):
        """获取观察状态"""
        # 创建比赛接口格式的grid (17x17)
        grid = []
        for i in range(17):
            row = []
            for j in range(17):
                if np.array_equal([i, j], self.player1_pos):
                    row.append(-9)  # 玩家1本人
                elif np.array_equal([i, j], self.player2_pos):
                    row.append(-2)  # 玩家2
                elif self.map[i, j] == 1:  # 障碍物
                    row.append(-1)
                elif self.map[i, j] == 2:  # 炸弹
                    row.append(-3)
                elif self.coin_map[i, j] > 0:  # 金币
                    row.append(int(self.coin_map[i, j]))
                else:  # 空白
                    row.append(0)
            grid.append(row)
        
        return {
            'map': self.map.copy(),
            'coin_map': self.coin_map.copy(),
            'player_pos': self.player1_pos.copy(),
            'player_coins': np.array([self.player1_coins]),
            'enemy_pos': self.player2_pos.copy(),
            'enemy_coins': np.array([self.player2_coins]),
            'round': np.array([self.current_round]),
            'grid': grid  # 新增：比赛接口格式的grid
        }
    
   
    
    def render(self, mode='human'):
        """渲染环境"""
        bomb_count = np.sum(self.map == 2)
        obstacle_count = np.sum(self.map == 1)
        total_coins = np.sum(self.coin_map)
        
        print(f"Round: {self.current_round}/300")
        print(f"Player1 coins: {self.player1_coins}, Player2 coins: {self.player2_coins}")
        print(f"Player1 pos: {self.player1_pos}, Player2 pos: {self.player2_pos}")
        print(f"地图状态: 障碍物={obstacle_count}, 炸弹={bomb_count}, 总金币={total_coins}")
        print("Map (# = obstacle, * = bomb, -1 = player1, -2 = player2):")
        
        symbols = {0: '.', 1: '#', 2: '*', 3: '-1', 4: '-2'}
        
        for i in range(self.map_size):
            row_str = ""
            for j in range(self.map_size):
                if self.coin_map[i, j] > 0:
                    row_str += f"{self.coin_map[i, j]:2d} "
                else:
                    row_str += f" {symbols[self.map[i, j]]} "
            print(row_str)
        print()



import numpy as np
import random
from battle_env import BattleEnv

class CompetitionSimulator:
    def __init__(self, player1_module, player2_module):
        """
        初始化比赛模拟器
        Args:
            player1_module: 玩家1的模块（包含Player类）
            player2_module: 玩家2的模块（包含Player类）
        """
        self.env = BattleEnv()
        self.player1 = player1_module.Player()
        self.player2 = player2_module.Player()
        
    def run_game(self, verbose=True):
        """运行一场完整的比赛"""
        obs = self.env.reset()
        
        if verbose:
            print("=== 比赛开始 ===")
            self.env.render()
        
        for round_num in range(300):
            # 转换环境状态为比赛接口格式
            grid1, gold1, gold2_1 = self._convert_obs_for_player(obs, player_id=1)
            grid2, gold2, gold1_2 = self._convert_obs_for_player(obs, player_id=2)
            # print("--------------------------")
            # print(grid1)
            # 获取玩家决策
            try:
                actions1 = self.player1.MoveDecision(grid1, gold1, gold2_1)
                actions2 = self.player2.MoveDecision(grid2, gold2, gold1_2)
            except Exception as e:
                print(f"玩家决策错误: {e}")
                actions1 = [random.randint(0, 3) for _ in range(3)]
                actions2 = [random.randint(0, 3) for _ in range(3)]
            
            # 验证动作格式
            actions1 = self._validate_actions(actions1)
            actions2 = self._validate_actions(actions2)
            
            # 执行动作
            # 强制 Player1 先手
            if round_num % 2 == 1:
                obs, reward, done, info = self.env.step([actions1, actions2], first_player=0)
            else:
                obs, reward, done, info = self.env.step([actions1, actions2], first_player=1)

            # 显示进度
            if verbose and (round_num % 50 == 0 or round_num < 5):
                print(f"\n=== 第 {round_num + 1} 回合 ===")
                print(f"Player1 actions: {actions1}")
                print(f"Player2 actions: {actions2}")
                self.env.render()
            
            if done:
                break
        
        # 比赛结果
        final_score1 = self.env.player1_coins
        final_score2 = self.env.player2_coins
        
        if verbose:
            print("=== 比赛结束 ===")
            print(f"最终得分 - Player1: {final_score1}, Player2: {final_score2}")
            if final_score1 > final_score2:
                print("Player1 获胜!")
            elif final_score2 > final_score1:
                print("Player2 获胜!")
            else:
                print("平局!")
        
        return {
            'player1_score': final_score1,
            'player2_score': final_score2,
            'winner': 1 if final_score1 > final_score2 else (2 if final_score2 > final_score1 else 0),
            'rounds': self.env.current_round
        }
    
    def _convert_obs_for_player(self, obs, player_id):
        """
        将环境观察转换为比赛接口格式
        返回: grid, my_gold, enemy_gold
        """
        map_data = obs['map']
        coin_map = obs['coin_map']
        
        if player_id == 1:
            my_pos = obs['player_pos']
            enemy_pos = obs['enemy_pos']
            my_gold = obs['player_coins'][0]
            enemy_gold = obs['enemy_coins'][0]
        else:
            my_pos = obs['enemy_pos']
            enemy_pos = obs['player_pos']
            my_gold = obs['enemy_coins'][0]
            enemy_gold = obs['player_coins'][0]
        
        # 创建17x17的grid
        grid = []
        for i in range(17):
            row = []
            for j in range(17):
                if np.array_equal([i, j], my_pos):
                    row.append(-9)  # 玩家本人
                elif np.array_equal([i, j], enemy_pos):
                    row.append(-2)  # 对手玩家
                elif map_data[i, j] == 1:  # 障碍物
                    row.append(-1)
                elif map_data[i, j] == 2:  # 炸弹
                    row.append(-3)
                elif coin_map[i, j] > 0:  # 金币
                    row.append(int(coin_map[i, j]))
                else:  # 空白
                    row.append(0)
            grid.append(row)
        
        return grid, int(my_gold), int(enemy_gold)
    
    def _validate_actions(self, actions):
        """验证并修正动作格式"""
        if not isinstance(actions, list) or len(actions) != 3:
            return [random.randint(0, 3) for _ in range(3)]
        
        validated = []
        for action in actions:
            if isinstance(action, int) and 0 <= action <= 3:
                validated.append(action)
            else:
                validated.append(random.randint(0, 3))
        
        return validated

def run_competition(player1_module_name, player2_module_name, num_games=1):
    """
    运行多场比赛
    Args:
        player1_module_name: 玩家1模块名（如 'player1'）
        player2_module_name: 玩家2模块名（如 'player2'）
        num_games: 比赛场数
    """
    # 动态导入玩家模块
    import importlib
    player1_module = importlib.import_module(player1_module_name)
    player2_module = importlib.import_module(player2_module_name)
    
    results = []
    wins = [0, 0, 0]  # [player1_wins, player2_wins, draws]
    
    for game_num in range(num_games):
        print(f"\n{'='*50}")
        print(f"第 {game_num + 1}/{num_games} 场比赛")
        print(f"{'='*50}")
    
        simulator = CompetitionSimulator(player1_module, player2_module)
        result = simulator.run_game(verbose=(num_games <= 3))
        
        results.append(result)
        wins[result['winner']] += 1
        
        print(f"比赛 {game_num + 1} 结果: Player1={result['player1_score']}, Player2={result['player2_score']}")
    
    # 总结
    print(f"\n{'='*50}")
    print("比赛总结")
    print(f"{'='*50}")
    print(f"总场数: {num_games}")
    print(f"Player1 胜利: {wins[1]} 场")
    print(f"Player2 胜利: {wins[2]} 场") 
    print(f"平局: {wins[0]} 场")
    
    if wins[1] > wins[2]:
        print("Player1 总体获胜!")
    elif wins[2] > wins[1]:
        print("Player2 总体获胜!")
    else:
        print("总体平局!")
    
    return results

if __name__ == "__main__":
    # 运行比赛示例
    results = run_competition('player1', 'player2', num_games=3)


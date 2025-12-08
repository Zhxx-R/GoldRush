
from competition_env import run_competition

# 测试比赛环境
if __name__ == "__main__":
    print("开始比赛测试...")
    
    # 运行3场比赛
    results = run_competition('player4', 'player3', num_games=1)
    
    # 分析结果
    total_score1 = sum(r['player1_score'] for r in results)
    total_score2 = sum(r['player2_score'] for r in results)
    
    print(f"\n详细统计:")
    print(f"Player1 总得分: {total_score1}")
    print(f"Player2 总得分: {total_score2}")
    print(f"平均得分 - Player1: {total_score1/len(results):.1f}, Player2: {total_score2/len(results):.1f}")


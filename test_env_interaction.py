from battle_env import BattleEnv
import numpy as np

# 创建环境
env = BattleEnv()

# 测试基本交互
print("=== 环境交互测试 ===")

# 重置环境
obs = env.reset()
print(f"观察空间键: {obs.keys()}")
print(f"地图形状: {obs['map'].shape}")
print(f"金币地图形状: {obs['coin_map'].shape}")
print(f"玩家1位置: {obs['player_pos']}")
print(f"玩家2位置: {obs['enemy_pos']}")

# 测试几步交互
for step in range(5):
    # 随机动作：每个玩家3次移动
    player1_actions = [np.random.randint(0, 5) for _ in range(3)]
    player2_actions = [np.random.randint(0, 5) for _ in range(3)]
    actions = [player1_actions, player2_actions]
    
    # 执行动作
    obs, reward, done, info = env.step(actions)
    
    print(f"\n步骤 {step + 1}:")
    print(f"动作: P1={player1_actions}, P2={player2_actions}")
    print(f"奖励: {reward}")
    print(f"玩家1金币: {obs['player_coins'][0]}")
    print(f"玩家2金币: {obs['enemy_coins'][0]}")
    print(f"回合: {obs['round'][0]}")
    env.render()
    if done:
        print("游戏结束!")
        break

print("\n=== 环境交互测试完成 ===")
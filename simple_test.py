#!/usr/bin/env python3
"""
简单测试日志功能
"""

import os
import ctypes
import numpy as np

# 设置环境变量
os.environ['CPP_LOG_LEVEL'] = '1'  # 关键函数
os.environ['CPP_LOG_CONSOLE'] = '1'  # 输出到控制台
os.environ['CPP_LOG_FILE'] = 'simple_test.log'

print("环境变量设置:")
print(f"CPP_LOG_LEVEL = {os.environ.get('CPP_LOG_LEVEL')}")
print(f"CPP_LOG_CONSOLE = {os.environ.get('CPP_LOG_CONSOLE')}")
print(f"CPP_LOG_FILE = {os.environ.get('CPP_LOG_FILE')}")

# 加载库
lib_path = "./submitv1fix.so"
try:
    lib = ctypes.CDLL(lib_path)
    print(f"成功加载库: {lib_path}")
except Exception as e:
    print(f"无法加载库: {e}")
    exit(1)

# 设置函数签名
lib.move_decision_c.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # grid_flat
    ctypes.c_int,                    # rows
    ctypes.c_int,                    # cols
    ctypes.c_int,                    # gold
    ctypes.c_int,                    # gold_2
    ctypes.POINTER(ctypes.c_int32)   # out_moves
]
lib.move_decision_c.restype = None

# 创建测试数据
print("\n创建测试网格...")
grid = np.zeros((17, 17), dtype=np.int32)
grid[8, 8] = -9  # 玩家
grid[8, 9] = -2  # 敌人
grid[7, 8] = 5   # 奖励
grid[9, 8] = 3   # 奖励

grid_flat = grid.flatten()
grid_ptr = grid_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

# 输出数组
out_moves = (ctypes.c_int32 * 3)()

print("调用 move_decision_c...")
print("应该看到日志输出:")

# 调用函数
lib.move_decision_c(grid_ptr, 17, 17, 50, 30, out_moves)

print(f"\n返回的移动: [{out_moves[0]}, {out_moves[1]}, {out_moves[2]}]")

# 检查日志文件
log_file = 'simple_test.log'
if os.path.exists(log_file):
    size = os.path.getsize(log_file)
    print(f"\n日志文件 {log_file} 大小: {size} 字节")
    if size > 0:
        print("日志内容:")
        with open(log_file, 'r') as f:
            print(f.read())
    else:
        print("日志文件为空")
else:
    print(f"\n日志文件 {log_file} 不存在")

print("\n测试完成")

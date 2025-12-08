#!/usr/bin/env python3
"""
测试日志系统的性能影响
"""

import os
import time
import ctypes
import numpy as np
from pathlib import Path

# 设置库路径
lib_path = "./submitv1fix.so"

def test_performance(log_level, num_calls=100):
    """测试不同日志级别下的性能"""
    
    # 设置环境变量
    os.environ['CPP_LOG_LEVEL'] = str(log_level)
    os.environ['CPP_LOG_CONSOLE'] = '0'  # 不输出到控制台
    os.environ['CPP_LOG_FILE'] = f'perf_test_level_{log_level}.log'
    
    # 重新加载库（如果已经加载过）
    try:
        lib = ctypes.CDLL(lib_path)
    except Exception as e:
        print(f"无法加载库: {e}")
        return None
    
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
    grid = np.zeros((17, 17), dtype=np.int32)
    grid[8, 8] = -9  # 玩家
    grid[8, 9] = -2  # 敌人
    grid[7, 8] = 5   # 奖励
    grid[9, 8] = 3   # 奖励
    
    grid_flat = grid.flatten()
    grid_ptr = grid_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    
    # 输出数组
    out_moves = (ctypes.c_int32 * 3)()
    
    print(f"测试日志级别 {log_level}，调用 {num_calls} 次...")
    
    # 预热
    for _ in range(5):
        lib.move_decision_c(grid_ptr, 17, 17, 50, 30, out_moves)
    
    # 性能测试
    start_time = time.time()
    for i in range(num_calls):
        lib.move_decision_c(grid_ptr, 17, 17, 50 + i, 30, out_moves)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_calls * 1000  # 毫秒
    
    # 检查日志文件大小
    log_file = f'perf_test_level_{log_level}.log'
    log_size = 0
    if os.path.exists(log_file):
        log_size = os.path.getsize(log_file)
    
    return {
        'level': log_level,
        'total_time': total_time,
        'avg_time_ms': avg_time,
        'calls_per_sec': num_calls / total_time,
        'log_size_bytes': log_size,
        'log_size_kb': log_size / 1024
    }

def main():
    print("=== C++ 日志系统性能测试 ===")
    print(f"库路径: {lib_path}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查库文件是否存在
    if not os.path.exists(lib_path):
        print(f"错误: 找不到库文件 {lib_path}")
        print("请先运行: make submitv1fix.so")
        return
    
    # 清理旧的日志文件
    for f in Path('.').glob('perf_test_level_*.log'):
        f.unlink()
    
    # 测试不同日志级别
    results = []
    num_calls = 50  # 减少调用次数以便快速测试
    
    for level in [0, 1, 2, 3]:
        try:
            result = test_performance(level, num_calls)
            if result:
                results.append(result)
                print(f"级别 {level}: {result['avg_time_ms']:.2f}ms/call, "
                      f"{result['calls_per_sec']:.1f} calls/sec, "
                      f"日志大小: {result['log_size_kb']:.1f}KB")
        except Exception as e:
            print(f"级别 {level} 测试失败: {e}")
    
    if results:
        print("\n=== 性能对比 ===")
        baseline = results[0]['avg_time_ms']  # 级别0作为基准
        
        print(f"{'级别':<4} {'平均时间(ms)':<12} {'相对开销':<10} {'调用/秒':<10} {'日志大小(KB)':<12}")
        print("-" * 60)
        
        for r in results:
            overhead = (r['avg_time_ms'] / baseline - 1) * 100 if baseline > 0 else 0
            print(f"{r['level']:<4} {r['avg_time_ms']:<12.2f} {overhead:+6.1f}%    "
                  f"{r['calls_per_sec']:<10.1f} {r['log_size_kb']:<12.1f}")
    
    print("\n=== 日志文件位置 ===")
    print(f"日志文件保存在当前目录: {os.getcwd()}")
    for f in Path('.').glob('perf_test_level_*.log'):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f}KB")
    
    print("\n=== 建议 ===")
    if results:
        level_0 = next((r for r in results if r['level'] == 0), None)
        level_1 = next((r for r in results if r['level'] == 1), None)
        
        if level_0 and level_1:
            overhead = (level_1['avg_time_ms'] / level_0['avg_time_ms'] - 1) * 100
            if overhead < 5:
                print("✓ 级别1的性能开销很小，适合生产环境使用")
            elif overhead < 15:
                print("⚠ 级别1有一定性能开销，建议仅在需要调试时使用")
            else:
                print("⚠ 级别1性能开销较大，生产环境建议使用级别0")
    
    print("\n使用方法:")
    print("export CPP_LOG_LEVEL=0  # 关闭日志，最佳性能")
    print("export CPP_LOG_LEVEL=1  # 仅关键信息，轻微影响")
    print("export CPP_LOG_LEVEL=2  # 重要函数，适度影响")
    print("export CPP_LOG_LEVEL=3  # 详细日志，明显影响")

if __name__ == "__main__":
    main()

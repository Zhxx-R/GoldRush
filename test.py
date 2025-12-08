import ray
import torch
import numpy as np
from runner import RLRunner
from model import NetworkModelActor
from parameter import INPUT_DIM, EMBEDDING_DIM

def test_parallel_runners():
    """测试并行runner和数据接收"""
    ray.init()
    
    print("=== 测试并行Runner ===")
    
    # 创建2个runner
    runner1 = RLRunner.remote(0)  # meta_agent_id=0
    runner2 = RLRunner.remote(1)  # meta_agent_id=1
    
    # 使用当前配置创建新的权重（不是从文件加载）
    dummy_net = NetworkModelActor(INPUT_DIM, EMBEDDING_DIM)
    fake_weights = dummy_net.state_dict()
    
    print(f"使用配置: INPUT_DIM={INPUT_DIM}, EMBEDDING_DIM={EMBEDDING_DIM}")
    print("权重结构:")
    for key, tensor in fake_weights.items():
        print(f"  {key}: {tensor.shape}")
    
    # 确保权重在CPU上
    cpu_weights = {}
    for key, value in fake_weights.items():
        if isinstance(value, torch.Tensor):
            cpu_weights[key] = value.cpu()
        else:
            cpu_weights[key] = value
    
    print("\n启动2个并行任务...")
    
    # 同时启动2个job
    job1 = runner1.job.remote([cpu_weights], 1)
    job2 = runner2.job.remote([cpu_weights], 2)
    
    job_list = [job1, job2]
    
    print("等待任务完成...")
    
    # 等待任务完成并收集结果
    results = []
    while job_list:
        done_id, job_list = ray.wait(job_list, num_returns=1)
        done_jobs = ray.get(done_id)
        
        for job_result in done_jobs:
            job_results, perf_metrics, info = job_result
            
            print(f"\n收到来自 meta_agent {info['id']} 的数据:")
            print(f"  episode_number: {info['episode_number']}")
            print(f"  buffer长度: {[len(buf) for buf in job_results]}")
            print(f"  性能指标: {perf_metrics}")
            
            if len(job_results[0]) > 0:
                sample_state = job_results[0][0]
                sample_reward = job_results[2][0] if len(job_results[2]) > 0 else 0
                print(f"  样本检查 - state shape: {sample_state.shape}, reward: {sample_reward}")
            
            results.append((info['id'], len(job_results[0])))
    
    print(f"\n=== 测试结果 ===")
    print(f"成功接收到 {len(results)} 个进程的数据:")
    for agent_id, buffer_size in results:
        print(f"  Agent {agent_id}: {buffer_size} 个样本")
    
    ray.shutdown()
    return len(results) == 2

if __name__ == "__main__":
    success = test_parallel_runners()
    if success:
        print("✅ 并行测试成功!")
    else:
        print("❌ 并行测试失败!")

import numpy as np
import copy
import os
import torch

from typing import List, Dict, Any
from battle_env import BattleEnv
from config import *


import logging
from datetime import datetime

def set_logger(meta_agent_id):
    logger = logging.getLogger(f"workflow_{meta_agent_id}")  # 使用agent_id区分logger
    logger.setLevel(logging.INFO)

    os.makedirs(LOG_BASE_PATH, exist_ok=True)
    log_time = datetime.now().strftime('%Y%m%d_%H')
    # workflow使用agent id命名，追加模式
    file_handler = logging.FileHandler(f'{LOG_BASE_PATH}/agent_{meta_agent_id}_{log_time}.log', mode='a')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

class Workflow(object):
    #fix
    def __init__(self, meta_agent_id, policy_net, global_step, device='cuda', greedy=False, first_mode=0, save_image=False):
        # metaAgentID：该工作流所属的远程 worker 标识
        # local_policy_net：本地拷贝的 Actor–Critic 策略网络，用于推理
        # global_step：当前全局训练步数，可用于调度或日志目录
        # param_actor：传给环境的可配置参数，保持与训练时一致
        # greedy：是否始终选择最高概率动作
        # episode_buffer：存储 (state, action, reward, done) 等序列，用于后续策略更新
        # perf_metrics：一局结束时的性能指标字典，如步数、最终奖励
        
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        # self.param_actor = param_actor
        self.save_image = save_image
        self.first_mode = first_mode

        self.env = BattleEnv() 
        self.local_policy_net = policy_net.to(self.device)
        self.local_policy_net.eval()

        self.episode_buffer = [[] for _ in range(9)]  # 0: state, 1: action, 2: reward, 3: next_state, 4: done, 5: value, 6: logit, 7: adv, 8: return
        self.perf_metrics = {}
        self.frame_history = []

        self.logger = set_logger(meta_agent_id)

        
    def get_observations(self):
        # get observations
        obs = self.env._get_observation()
        
        grid = np.array(obs['grid'], dtype=np.float32)  # (17, 17)
        round_n = obs['round'][0]
        
        current_frame = grid / 30.0  # (17, 17)
        current_frame_flat = current_frame.flatten()  # 289
        
        self.frame_history.append(current_frame_flat)

        if len(self.frame_history) > 3:
            self.frame_history.pop(0)

        while len(self.frame_history) < 3:
            self.frame_history.insert(0, current_frame_flat)#重复

        frames_feature = np.concatenate(self.frame_history, axis=0)
    
        # round
        round_info = np.array([round_n / 300.0], dtype=np.float32)
        
        # 拼接成868维特征：867 + 1 = 868维
        full_input = np.concatenate([frames_feature, round_info], axis=0)
        
        feature_tensor = torch.tensor(full_input, dtype=torch.float32, device=self.device)  # (868)
        
        return feature_tensor


    def work(self, currEpisode):
        self.run_episode(currEpisode)

    def get_action(self, observations):
        feature = observations.unsqueeze(0)
        with torch.no_grad():
            logits, value = self.local_policy_net(feature)  # logits: (1, 15)
        value = value.squeeze(1)
        action_logits = logits.view(1, 3, 5)  # reshape
        
        actions = []
        logps = []
        for step in range(3):
            step_logits = action_logits[0, step]  

            if self.greedy:
                a = int(step_logits.argmax().item())
                logp = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            else:
                dist = torch.distributions.Categorical(logits=step_logits)
                a = dist.sample()
                logp = dist.log_prob(a)
                # self.logger.info(f'step:{step}, a:{a}, logp:{logp}, prob:{dist.probs}')
            actions.append(a)
            logps.append(logp)
        
        # # dirs: 上(0),下(1),左(2),右(3),停(4)
        # dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        # curr = list(self.last_obs['player_pos']) 
        # for a in actions:
        #     dx, dy = dirs[a]
        #     curr[0] = max(0, min(16, curr[0] + dx))
        #     curr[1] = max(0, min(16, curr[1] + dy))

        # next_positions = [curr]  
        return actions, value, logps
    
       
    
    def save_observations(self, observations):
        self.episode_buffer[0].append(observations.detach().clone())    


    def save_action(self, actions):
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        self.episode_buffer[1].append(action_tensor)

    def save_value(self, value):
        self.episode_buffer[5].append(value)

    def save_logp(self, logps):
        logp_tensor = torch.stack(logps, dim=0)
        self.episode_buffer[6].append(logp_tensor)

    def save_reward_done(self, reward, done):
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.episode_buffer[2].append(r.detach())
        d = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        self.episode_buffer[4].append(d.detach())

    def save_next_observations(self, observations):
        self.episode_buffer[3].append(observations.detach().clone())   
        
    def save_adv_and_return(self, advantages, returns):
        self.episode_buffer[7] = [torch.tensor(adv, dtype=torch.float32, device=self.device) for adv in advantages]
        self.episode_buffer[8] = [torch.tensor(ret, dtype=torch.float32, device=self.device) for ret in returns]

     # def compute_gae(self):
    #     rewards = self.episode_buffer[2]
    #     dones = self.episode_buffer[4]
    #     values = self.episode_buffer[5]

    #     advantages = []
    #     gae = 0
    #     values = values + [0]
    #     for t in reversed(range(len(rewards))):
    #         delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
    #         gae = delta + GAMMA * LAM * (1 - dones[t]) * gae
    #         advantages.insert(0, gae)
    #     returns = [a + v for a, v in zip(advantages, values[:-1])]
    #     return returns, advantages

    def compute_gae(self):
        rewards = self.episode_buffer[2]
        dones = self.episode_buffer[4] 
        values = self.episode_buffer[5]


        rewards = [r.item() if torch.is_tensor(r) else r for r in rewards]
        dones = [d.item() if torch.is_tensor(d) else d for d in dones]
        values = [v.item() if torch.is_tensor(v) else v for v in values]
        
        print(f"GAE计算: rewards范围[{min(rewards):.2f}, {max(rewards):.2f}], values范围[{min(values):.2f}, {max(values):.2f}]")
        
        advantages = []
        gae = 0
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAM * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        
        # returns范围
        print(f"GAE结果: returns范围[{min(returns):.2f}, {max(returns):.2f}], advantages范围[{min(advantages):.2f}, {max(advantages):.2f}]")
        
        return returns, advantages
   
    def run_episode(self, curr_episode):
       
        obs = self.env.reset()
        self.last_obs = obs
        done = False
        step = 0

        # clear
        for buf in self.episode_buffer:
            buf.clear()

        self.frame_history = []

        # fix : 300 or other
        while not done and step < MAX_ROUNDS:
           
            feature = self.get_observations()      # (868）
            self.save_observations(feature)       
           
            actions, value, logps = self.get_action(feature)
            self.save_action(actions)
            self.save_value(value)
            self.save_logp(logps)

            # 环境交互：我方 vs 随机对手
            # opp_actions = [np.random.randint(0,5) for _ in range(3)]
            opp_actions = [4, 4, 4]

            if self.first_mode == 0:
                first_player = 0
            elif self.first_mode == 1:
                first_player= 1
            elif self.first_mode == 2: #random
                first_player = np.random.randint(0, 2)
            else:  # 'alternate'
                first_player = step % 2

            next_obs, reward, done, _ = self.env.step(
                [actions, opp_actions],
                first_player=first_player
            )
            self.save_reward_done(reward, done)           

            
            self.last_obs = next_obs
            next_feature = self.get_observations()
            self.save_next_observations(next_feature)

           
            # if self.save_image:
            #     os.makedirs(gifs_path, exist_ok=True)
            #     self.env.render()  fix gifs_path
            step += 1

        returns, advantages = self.compute_gae()
        self.save_adv_and_return(advantages, returns)
        
        self.perf_metrics['total_steps']     = int(step)
        self.perf_metrics['success_rate']    = float(done)
        self.perf_metrics['final_coin_diff'] = float(
            self.last_obs['player_coins'][0] - self.last_obs['enemy_coins'][0]
        )
        
        # 确保returns中所有元素都是标量
        if returns:
            scalar_returns = []
            for r in returns:
                if isinstance(r, torch.Tensor):
                    scalar_returns.append(r.item())
                elif isinstance(r, np.ndarray):
                    scalar_returns.append(float(r.flatten()[0]))
                else:
                    scalar_returns.append(float(r))
            self.perf_metrics['returns'] = float(np.mean(scalar_returns))
        else:
            self.perf_metrics['returns'] = 0.0
            
        self.perf_metrics['player1_coins'] = float(self.last_obs['player_coins'][0])
        self.perf_metrics['player2_coins'] = float(self.last_obs['enemy_coins'][0])

        return self.perf_metrics

    

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import ray
import os
import numpy as np
import random
import tensorboard

import logging
from datetime import datetime

from model import NetworkModelActor, NetworkModelLearner
from runner import RLRunner
from config import *
import warnings
warnings.filterwarnings('ignore')

def set_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    os.makedirs(LOG_BASE_PATH, exist_ok=True)
    log_time = datetime.now().strftime('%Y%m%d_%H')
    file_handler = logging.FileHandler(f'{LOG_BASE_PATH}/{name}_{log_time}.log', mode='w')
    file_handler.setLevel(LOG_LEVEL)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

ray.init()
logger = set_logger('train')
logger.info("Welcome to RL autonomous exploration!")


writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


def writeToTensorBoard(writer, tensorboardData, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    reward, value, policyLoss, valueLoss, entropy, policyGradNorm, success_rate, total_steps, final_coin_diff, returns, player1_coins, player2_coins = tensorboardData

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Value Loss', scalar_value=valueLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policyGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Total Steps', scalar_value=total_steps, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Final_coin_diff', scalar_value=final_coin_diff, global_step=curr_episode)  #fix
    writer.add_scalar(tag='Perf/Returns', scalar_value=returns, global_step=curr_episode)  #fix
    writer.add_scalar(tag='Perf/Player1_coins', scalar_value=player1_coins, global_step=curr_episode)  #fix
    writer.add_scalar(tag='Perf/Player2_coins', scalar_value=player2_coins, global_step=curr_episode)  #fix

def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # initialize neural networks
    global_policy_net = NetworkModelActor(INPUT_DIM, EMBEDDING_DIM).to(device)
    
    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)


    curr_episode = 0

    # load model and optimizer trained before
    if LOAD_MODEL:
        checkpoint_path = model_path + '/checkpoint.pth'
        best_checkpoint_files = [f for f in os.listdir(model_path) if f.startswith('best_checkpoint_')]
        
        # 先尝试加载最新的best checkpoint
        if best_checkpoint_files:
            best_checkpoint_files.sort()
            checkpoint_path = os.path.join(model_path, best_checkpoint_files[-1])
            logger.info(f'Trying to load best checkpoint: {checkpoint_path}')
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                global_policy_net.load_state_dict(checkpoint['policy_model'])
                global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
                curr_episode = checkpoint['episode']
                logger.info(f"Successfully loaded from {checkpoint_path}, episode: {curr_episode}")
            except Exception as e:
                logger.error(f'Error loading checkpoint: {e}')
                logger.info('Starting from scratch')
                curr_episode = 0

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))
    
    # initialize metric collector
    metric_name = ['success_rate', 'total_steps', 'final_coin_diff', 'returns', 'player1_coins','player2_coins']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(9):
        experience_buffer.append([])
    
    # collect data from worker and do training
    try:
        maxreward = 0
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)
            
            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))
            
            # start training
            newmaxreward = 0
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                logger.info("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))


                # randomly sample a batch data
                sample_indices = random.sample(indices, BATCH_SIZE)
                rollouts = []
                for i in range(len(experience_buffer)):
                    rollouts.append([experience_buffer[i][index] for index in sample_indices])
                # logger.debug(f"rollout shape: {rollouts[0][0].shape}") #868
                # stack batch data to tensors
                feature_batch = torch.stack(rollouts[0]).to(device)          # (batch, 868)
                action_batch = torch.stack(rollouts[1]).to(device)          # (batch, 3) - 3步动作
                reward_batch = torch.stack(rollouts[2]).to(device)          # (batch, 1)
                next_feature_batch = torch.stack(rollouts[3]).to(device)    # (batch, 868)
                done_batch = torch.stack(rollouts[4]).to(device)            # (batch, 1)
                value_batch = torch.stack(rollouts[5]).to(device)          # (batch, 1)
                logit_batch = torch.stack(rollouts[6]).to(device)          # (batch, 15)
                adv_batch = torch.stack(rollouts[7]).to(device)            # (batch, 1)
                return_batch = torch.stack(rollouts[8]).to(device)          # (batch, 1)

                # logger.debug(f'feature_batch: {feature_batch.shape}')
                # logger.debug(f'action_batch: {action_batch.shape}')
                # logger.debug(f'reward_batch: {reward_batch.shape}')
                # logger.debug(f'next_feature_batch: {next_feature_batch.shape}')
                # logger.debug(f'done_batch: {done_batch.shape}')
                # logger.debug(f'value_batch: {value_batch.shape}')
                # logger.debug(f'logit_batch: {logit_batch.shape}')
                # logger.debug(f'adv_batch: {adv_batch.shape}')
                # logger.debug(f'return_batch: {return_batch.shape}')
                # PPO
                # value loss
                logp_list, value = dp_policy(feature_batch)  # (batch, 15) (batch, 1)
                value_loss_unclipped = F.mse_loss(value, return_batch)

                #MSE
                diff = value - return_batch
                # logger.debug(f'diff: {diff[:200].tolist()}')

                logger.debug(f'(value - return_batch): diff mean: {diff.mean().item():.4f}, diff std: {diff.std().item():.4f}')
                logger.debug(f'(value - return_batch): diff min: {diff.min().item():.4f}, diff max: {diff.max().item():.4f}')

                # logger.debug(f'value: {value.mean()}, return_batch: {return_batch.mean()}), value_loss_unclipped: {value_loss_unclipped},value_batch: {value_batch.mean()}')

                values_clipped = value_batch + (value - value_batch).clamp(-0.2, 0.2)
                value_loss_clipped = F.mse_loss(values_clipped, return_batch)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                # logger.debug(f'value_batch: {value_batch}, value - value_batch: {value - value_batch}, clamp:{(value - value_batch).clamp(-0.2, 0.2)},  return_batch: {return_batch},value - value_diff: {(value - value_batch).mean()}, value_clipped: {values_clipped}, value_loss_clipped: {value_loss_clipped}, value_loss: {value_loss}')

                # policy loss & entropy loss
                logp_list = logp_list.reshape([-1, 3, 5])
                policy_loss = torch.zeros(1, device=device)
                entropy_loss = torch.zeros(1, device=device)
                for i in range(3):
                    logp_reshaped = logp_list[:, i, :]
                    action_reshaped = action_batch[:, i:i+1].long()

                    # policy loss
                    dist = torch.distributions.Categorical(logits=logp_reshaped)
                    logps = dist.log_prob(action_reshaped)
                    # logger.debug(f'action_reshaped: {action_reshaped}')
                    old_logps = logit_batch[:, i]
                    ratio = torch.exp(logps - old_logps)
                    # logger.debug(f'ratio: {ratio}, logps:{logps}, logit_batch:{old_logps}')
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_batch
                    policy_loss += -torch.min(surr1, surr2).mean()

                    # entropy loss
                    entropy_loss += dist.entropy().mean()
                logger.info(f'Episode {curr_episode} - Total: value_loss: {value_loss.item():.4f}, policy_loss: {policy_loss.item():.4f}, entropy_loss: {entropy_loss.item():.4f}, Complete 3-step actions sample: {action_batch[:3].cpu().numpy().tolist()}')
                loss = 0.5 * value_loss + policy_loss + 0.01 * entropy_loss
                global_policy_optimizer.zero_grad()
                loss.backward()
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=0.5)
                global_policy_optimizer.step()

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value.mean().item(), policy_loss.item(), value_loss.item(),
                        entropy_loss.mean().item(), policy_grad_norm.item(), *perf_data]
                training_data.append(data)
                newmaxreward = max(maxreward, data[0])

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW and curr_episode % 15 == 0:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)
            

            # save the model
            if curr_episode % 32 == 0 or newmaxreward >= maxreward:
                # logger.info('Saving model')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                                "policy_optimizer": global_policy_optimizer.state_dict(),
                                "episode": curr_episode
                               
                        }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                if newmaxreward > maxreward:
                    path_checkpoint = "./" + model_path + "/best_checkpoint_{}.pth".format(curr_episode)
                    torch.save(checkpoint, path_checkpoint)
                maxreward = newmaxreward
                # logger.info('Saving model')
        for a in meta_agents:
            logger.info('Saving model')
            ray.kill(a)                
    
    except KeyboardInterrupt:
        logger.info("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()

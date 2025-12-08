import torch
import ray
from workflow import Workflow
from model import NetworkModelActor
from config import set_logger
from config import *

import logging
from datetime import datetime

def set_logger(meta_agent_id):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    os.makedirs(LOG_BASE_PATH, exist_ok=True)
    log_time = datetime.now().strftime('%Y%m%d_%H')
    file_handler = logging.FileHandler(f'{LOG_BASE_PATH}/runner_{meta_agent_id}_{log_time}.log', mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device("cuda" if USE_GPU else "cpu")
        self.local_network = NetworkModelActor(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)
        self.logger = set_logger(meta_agent_id)

    def get_weight(self):
        return self.local_network.state_dict()
    
    def set_policy_net_weight(self, policy_weight):
        self.local_network.load_state_dict(policy_weight)

    def do_job(self, episode_number):
        #fix params
        workflow = Workflow(self.meta_agent_id, self.local_network, episode_number, device=self.device)
        workflow.work(episode_number)

        job_results = workflow.episode_buffer
        perf_metrics = workflow.perf_metrics
        return job_results, perf_metrics
    
    def job(self, policy_weight, episode_number):
        # print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.logger.info(f"starting episode {episode_number} on metaAgent {self.meta_agent_id}")
        self.set_policy_net_weight(policy_weight[0])

        job_results, perf_metrics = self.do_job(episode_number)
        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }
        return job_results, perf_metrics, info
    
@ray.remote(num_cpus=1, num_gpus=0)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)

if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    for buffer in out[0]:
        print(buffer[0])
    # print(out[0][5])

import torch
import torch.nn as nn
from typing import List
import logging
from datetime import datetime
import os
from config import *

def set_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    os.makedirs(LOG_BASE_PATH, exist_ok=True)
    log_time = datetime.now().strftime('%Y%m%d_%H')
    file_handler = logging.FileHandler(f'{LOG_BASE_PATH}/{name}_{log_time}.log', mode='w')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

logger = set_logger('model')


class BaseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(BaseNetwork, self).__init__()
        self.feature_len = input_dim  # 868维 17 * 17 * 3  + 1

        logger.info(f"Initializing BaseNetwork with input_dim={input_dim}, embedding_dim={embedding_dim}")

        # 修改网络结构：868 → 256 → 128
        self.main_fc_dim_list = [self.feature_len, 256, embedding_dim]
        self.main_mlp_net = MLP(self.main_fc_dim_list, "main_mlp_net", non_linearity_last=True)
        
        # Actor输出：3 * 128 → 64  5
        self.label_mlp_0 = MLP([embedding_dim, 64, 5], "label_mlp")
        self.label_mlp_1 = MLP([embedding_dim, 64, 5], "label_mlp")
        self.label_mlp_2 = MLP([embedding_dim, 64,5], "label_mlp")
        
        # Critic输出：128 64 → 1
        self.value_mlp = MLP([embedding_dim, 64, 1], "value_mlp")
        logger.info(f"Network structure: {self.feature_len} → 256 → {embedding_dim}")
    
    def forward(self, feature):
        fc_mlp_out = self.main_mlp_net(feature)  # (batch, 128)

        label_mlp_out_0 = self.label_mlp_0(fc_mlp_out)  # (batch, 5)
        label_mlp_out_1 = self.label_mlp_1(fc_mlp_out)  # (batch, 5)
        label_mlp_out_2 = self.label_mlp_2(fc_mlp_out)  # (batch, 5)
        label_mlp_out = torch.concat([label_mlp_out_0, label_mlp_out_1, label_mlp_out_2], dim=1)  # (batch, 15)
        
        value = self.value_mlp(fc_mlp_out)  # (batch, 1)

        return label_mlp_out, value
    
    
class NetworkModelActor(BaseNetwork):
    def format_data(self, obs):
        return torch.tensor(obs).to(torch.float32)


class NetworkModelLearner(BaseNetwork):
    def format_data(self, datas):
        return datas.view(-1, self.feature_len).float()

    def forward(self, data_list, inference=False):
        feature = data_list[0] 
        return super().forward(feature)

    
def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer

class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list,
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)

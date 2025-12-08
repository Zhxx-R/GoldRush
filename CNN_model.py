import torch
import torch.nn as nn
from typing import List


class BaseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(BaseNetwork, self).__init__()
        self.feature_len = input_dim


        # CNN特征提取部分 - 替换原来的main_mlp_net
        # 输入: 5通道 17x17 -> 8通道 9x9 -> 16通道 5x5 = 400维
        self.conv1 = nn.Conv2d(5, 8, kernel_size=3, stride=2, padding=1)  # 17x17 -> 9x9
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 9x9 -> 5x5
        self.relu = nn.ReLU()

        
        # 计算卷积后的特征维度: 16 * 5 * 5 = 400
        conv_output_dim = 16 * 5 * 5

        # 可选的标量输入（回合信息、双方金币）3维
        total_input_dim = conv_output_dim + 3  #  403

        # 主特征提取网络：403 -> 256
        self.main_mlp_net = MLP([total_input_dim, embedding_dim], "main_mlp_net", non_linearity_last=True)

        self.label_mlp = MLP([embedding_dim, 64, 5], "label_mlp") # 单步5
        self.value_mlp = MLP([embedding_dim, 64, 1], "value_mlp")  #1

    def forward(self, grid, scalars=None):
        # CNN特征提取
        x = self.relu(self.conv1(grid))  # (batch, 8, 9, 9)
        x = self.relu(self.conv2(x))     # (batch, 16, 5, 5)
        conv_features = x.view(x.size(0), -1)  # (batch, 400)
        
        # 拼接标量特征
        if scalars is not None:
            feature = torch.cat([conv_features, scalars], dim=1)  # (batch, 403)
        else:
            batch_size = conv_features.size(0)
            zeros = torch.zeros(batch_size, 3, device=conv_features.device)
            feature = torch.cat([conv_features, zeros], dim=1)
        
        fc_mlp_out = self.main_mlp_net(feature)
        
        # 单步输出：每次只输出一个动作的分布
        label_mlp_out = self.label_mlp(fc_mlp_out)  # (batch, 5)
        prob = torch.nn.functional.softmax(label_mlp_out, dim=1)  # (batch, 5)
        value = self.value_mlp(fc_mlp_out)

        return prob, value
    
    
class NetworkModelActor(BaseNetwork):
    def format_data(self, obs, legal_action):
        return (
            torch.tensor(obs).to(torch.float32),
            torch.tensor(legal_action).to(torch.float32),
        )


class NetworkModelLearner(BaseNetwork):
    def format_data(self, datas):
        return datas.view(-1, self.data_len).float().split(self.data_split_shape, dim=1)

    def forward(self, data_list, inference=False):
        grid = data_list[0]
        scalars = data_list[1] if len(data_list) > 1 else None
        return super().forward(grid, scalars)

def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal_(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer

class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
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

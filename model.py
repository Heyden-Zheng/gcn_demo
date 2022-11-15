import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# 构造模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=2, out_channels=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)  # 降维，第二个维度小于第一个维度
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, features, edge_index, batch_size):
        x, edge_index, batch_size = features, edge_index, batch_size

        x = self.conv1(x, edge_index)  # 第一层运算，输入为节点特征和边的稀疏矩阵
        x = torch.relu(x)  # 激活函数
        # x = torch.tanh(x)  # 激活函数
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)  # 第二层运算，输入为节点特征和边的稀疏矩阵
        x = torch.log_softmax(x, dim=1)  # softmax可以得到每个节点的概率分布，dim=1保证所有分类概率和为1
        return x
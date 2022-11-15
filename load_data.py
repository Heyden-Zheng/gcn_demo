import numpy as np
import torch
from torch_geometric.data import Data

# 读取数据集

# 1.知识点特征
skill_features = np.loadtxt(fname='./dataset/node_prediction/skill_features.csv', dtype=float, delimiter=',',
                            skiprows=1, usecols=(1, 2, 3))
# 2.节点关系
skill_relation = np.loadtxt(fname='./dataset/node_prediction/skill_relation.csv', dtype=float, delimiter=',',
                            skiprows=1)  # 知识点邻接矩阵
# 3.标签
skill_labels = np.loadtxt(fname='./dataset/node_prediction/skill_labels.csv', dtype=int, delimiter=',', skiprows=1)
# 4.生成图对象
# pyG框架要求节点关系矩阵是COO格式的矩阵(如skill_relation.csv)，shape=[2,num_edges]，若你的节点关系矩阵是N*N的邻接矩阵形式(如k_relation.txt)，需要进行以下处理
# tmp_coo = sp.coo_matrix(adj_matrix)  # tmp_coo用来记录邻接矩阵中的非零值和它的索引，tmp_coo.data就是非零值
# indices = np.vstack((tmp_coo.row, tmp_coo.col))  # 根据非零值的行列索引生成一个索引矩阵，大小为2*边数
# edge_index = torch.LongTensor(indices)
x = torch.FloatTensor(skill_features)
edge_index = torch.t(torch.LongTensor(skill_relation))  # 边索引/节点关系

# 将节点标签转为Tensor类型
y = torch.LongTensor(skill_labels)

# 构造图对象Data
graph = Data(x=x, edge_index=edge_index, y=y)
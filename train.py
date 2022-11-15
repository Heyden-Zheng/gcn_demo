import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import GCN
import torch
from load_data import graph

# 1.加载数据集
dataset = graph
data_loader = DataLoader(dataset=dataset, batch_size=20, shuffle=True)  # train_loader的数量=train_dataset的数量除以batch_size

# 2.数据集划分
train_data = data_loader.dataset.x  # 全部数据都拿来训练。原因：由于是伪造的数据，若只取前200行，则对应的边关系也需要修改（剔除那些和非训练节点相关的边），否则gcn卷积操作时，根据边索引去找相应的节点时会报越界异常的错
# train_data = data_loader.dataset.x[:200]  # 前200行用来训练
# test_data = data_loader.dataset.x[200:260]  # 第201-260行用来测试
# val_data = data_loader.dataset.x[260:]  # 第261之后的数据用来校验

# train_label = data_loader.dataset.y[:200]
# test_label = data_loader.dataset.y[200:260]
# val_label = data_loader.dataset.y[260:]

# 3.构建模型实例
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 优化器，优化参数计算

# 4.训练模型
model.train()

for epoch in range(100):  # 训练所有的训练数据集10次
    loss_all = 0
    optimizer.zero_grad()  # 梯度清零
    output = model(train_data, data_loader.dataset.edge_index, data_loader.batch_size)  # 前向传播，把一批训练数据集导入模型并返回输出结果
    label = graph.y  # 将标签的行向量转为列向量
    train_loss = F.nll_loss(output, label)  # 计算损失，原理是对应元素求和取均值
    train_loss.backward()  # 反向传播
    loss_all += train_loss.item()  # 将最后的损失值汇总
    optimizer.step()  # 更新模型参数

    # 计算准确率
    train_acc = torch.eq(output.argmax(dim=1), label).float().mean()  # output.argmax(dim=1)表示找到单个样本的二分类预测结果中最大的元素

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(train_loss.data.item()),
          'acc_train: {:.4f}'.format(train_acc.data.item())
          )
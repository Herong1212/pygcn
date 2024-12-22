import torch.nn as nn

# 下面和 torch.nn 不同的是，nn 定义的是「层」类，而 F 定义的是「操作函数」
# 例如，nn.ReLU 是一个激活层，而 F.relu 是直接调用激活函数
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


# TODO 完整的两层图卷积网络
# 继承 nn.Module： 这是 PyTorch 中的神经网络模型基类，定义【自定义模型】时必须继承它
class GCN(nn.Module):
    def __init__(
        self,
        nfeat,  # 输入特征的维度（每个节点的特征向量长度）
        nhid,  # 表示隐藏层的维度，即图卷积网络中第一层输出的特征维度
        nclass,  # 表示最终的输出类别数，通常用于节点分类任务
        dropout,  # 表示训练时随机丢弃部分节点特征的比例，常用于防止过拟合
    ):
        super(GCN, self).__init__()

        # 定义两层图卷积层：gc1 和 gc2
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        # self.dropout 用于控制在训练过程中随机丢弃一部分节点特征，从而减少模型对特定节点的依赖，提升泛化能力。
        # ps：不是丢掉整行的特征向量（即整个节点的特征），而是对每一行的特征向量的各个元素进行随机“屏蔽”（丢弃部分元素），也即：将每个节点的特征向量中的某些元素（列）设置为 0，表示这些特征在本次训练中不参与计算。
        self.dropout = dropout

    def forward(self, x, adj):
        # Step 1: 第一层图卷积 + ReLU 激活函数
        x = F.relu(self.gc1(x, adj))

        # Step 2: Dropout 防止过拟合
        # self.training：判断模型当前是否处于训练模式。只有在训练模式时，才使用 Dropout 对特征进行随机丢弃和缩放；如果是测试模式，则不会进行 Dropout 操作
        # 假如 Dropout 的概率设置为 p，表示每个特征元素有 p 的概率被丢弃。Dropout 作用步骤：
        #   1、生成一个随机掩码矩阵（Mask），掩码矩阵的形状与 x 相同，并且每一个元素都是 0 或 1；
        #   2、应用掩码矩阵；
        #       2.1、将 x 中的元素与 mask 相乘，注意是【对应位相乘】
        #   3、缩放保留的特征；
        #       3.1、每个保留的元素乘以 1 / (1 - p)
        x = F.dropout(x, self.dropout, training=self.training)

        # Step 3: 第二层图卷积（不加激活函数，因为最终的输出需要直接用作分类概率分布）
        x = self.gc2(x, adj)

        # Step 4: Log-Softmax 归一化
        return F.log_softmax(x, dim=1)

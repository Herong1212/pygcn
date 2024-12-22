import math

import torch

from torch.nn.parameter import Parameter  # 用于定义可训练的参数（例如权重和偏置）
from torch.nn.modules.module import Module  # 用于定义一个模型或模块的基类


# 继承自 PyTorch 的 Module 类，代表一个神经网络层（这里是图卷积层），它的功能是对图数据进行卷积操作，更新图节点的特征。
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self,
        in_features,  # in_features: 输入特征维度（即每个节点的特征长度），int 类型
        out_features,  # out_features: 输出特征维度（即每个节点卷积后的特征长度），int 类型
        bias=True,  # 是否使用偏置
    ):
        # 调用父类的构造函数，以确保父类 Module 的初始化逻辑能够正常执行。在 PyTorch 中，所有神经网络模块都继承自 torch.nn.Module。
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ps1：权重矩阵，形状为 (in_features, out_features)，数据类型是 torch.FloatTensor（默认数据类型是 float32），一个二维张量（矩阵），其大小由输入和输出特征的维度决定，用于将输入特征映射到输出特征。
        # 每一行对应输入特征的一个维度（in_features=16）；每一列对应输出特征的一个维度（out_features=32）
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # ps2：偏置向量，形状为 (out_features)，数据类型是 torch.FloatTensor，一个 1D 张量（向量），表示每个输出特征都有一个对应的偏置值。偏置是在输出结果上加的额外自由度，主要用于解决数据的偏移问题，偏置值可以调整每个输出特征的结果，使其更灵活
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            # 在 PyTorch 中，register_parameter() 用于显式地声明某个属性为模型的参数，但这里不需要真正初始化偏置
            self.register_parameter("bias", None)  # 如果 bias=False，则不使用偏置

        # 使用 reset_parameters() 初始化权重和偏置
        # ps：初始化的规则通常根据经验设置，这里使用均匀分布初始化权重和偏置
        self.reset_parameters()

    # TODO 初始化【权重 w】和【偏置 b】
    # 在深度学习中，权重和偏置需要在训练开始前被初始化为一些合适的值，初始化得当能够加速训练并避免梯度消失或梯度爆炸等问题。
    def reset_parameters(self):
        # 计算初始化的标准差范围，这是 Xavier 均匀分布初始化的一种变体，适用于确保神经网络中的权重分布合理，信号不会在前向传播中过大或过小。
        stdv = 1.0 / math.sqrt(self.weight.size(1))

        # 使用均匀分布在 [-stdv, stdv] 范围内随机初始化权重
        self.weight.data.uniform_(-stdv, stdv)

        # 如果使用偏置，同样使用均匀分布初始化偏置
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # note：前向传播！！！
    def forward(
        self,
        input,  # input: 节点特征矩阵，形状为 (num_nodes, in_features)，每行表示一个节点的特征向量，每列表示一个特征
        adj,  # adj: 图的邻接矩阵，形状为 (num_nodes, num_nodes)，表示节点之间的连接关系
    ):
        # step1：节点特征矩阵与权重矩阵相乘，表示每个节点经过特征变换后的结果，但还未加入邻接矩阵的信息。生成【更新后的节点特征矩阵】，形状为 (num_nodes, out_features)
        support = torch.mm(input, self.weight)

        # step2：【邻接矩阵】与【线性变换后的节点特征矩阵】相乘，完成图卷积操作，表示每个节点的最终输出特征。这一步聚合了相邻节点的特征，结果形状为 (num_nodes, out_features)
        output = torch.spmm(adj, support)

        # step3：如果有偏置，则加上偏置，返回最终结果
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 返回这个类的字符串表示，方便打印查看
    def __repr__(self):
        # 输出: 类名 + 输入输出特征维度，例如：GraphConvolution (16 -> 32)
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


# ps：在 C++ 中，可以将这个类看作是一个封装了参数和前向传播逻辑的 "神经网络层"，像这样：
# class GraphConvolution
# {
# private:
# 	Eigen::MatrixXf weight; // 权重矩阵
# 	Eigen::VectorXf bias;	// 偏置
# public:
# 	GraphConvolution(int in_features, int out_features);
# 	Eigen::MatrixXf forward(Eigen::MatrixXf input, Eigen::MatrixXf adj);
# 	void reset_parameters();
# };

from hashlib import sha1
import numpy as np
import scipy.sparse as sp
import torch


# 1、将类别标签编码为独热向量
# 输入：['A', 'B', 'C']
def encode_onehot(labels):

    # 获取所有类别的集合【set() 里不允许存在重复元素，所以这里就是把 labels 里的元素去重】
    classes = set(labels)

    # 用 np.identity(len(classes) 生成单位矩阵，每行表示一个类别的独热编码
    # 如：
    # {
    #   'A':[1,0,0],
    #   'B':[0,1,0],
    #   'C':[0,0,1]
    # }
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}

    # 首先用字典 classes_dict 映射类别到独热向量，再使用 map 将 labels 转换为对应的独热向量，并返回结果
    #
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

    # 返回一个二维数组，其中每个类别对应一个独热向量
    # 输出：
    #   [[1, 0, 0],
    #    [0, 1, 0],
    #    [1, 0, 0]]
    return labels_onehot


# 2、加载指定的数据集并返回图的特征矩阵、标签、邻接矩阵及训练/验证/测试索引
def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))

    # 使用 np.genfromtxt 加载 cora.content 文件
    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str)
    )

    # ps：从文件中提取特征矩阵和标签
    # 提取特征：[:, 1:-1]表示取第2列到倒数第2列（中间的特征列），-1表示最后一个元素，取不到；-2表示倒数第二个元素，也取不到
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 提取标签：[:, -1]表示最后一列（标签），标签通过 encode_onehot 转为独热编码
    labels = encode_onehot(idx_features_labels[:, -1])

    # notice：构建图！
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # 使用 np.genfromtxt 加载 cora.cites 文件
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)

    # 然后用稀疏矩阵存储连接关系
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix --- 将邻接矩阵变成【对称矩阵（无向图）】，确保每条边双向连接
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 归一化：将特征矩阵按行归一化，让所有特征值的范围保持一致，避免数值过大或过小影响训练效果
    features = normalize(features)
    # 添加自环并归一化邻接矩阵：给每个节点加一个自环（自己和自己相连，就是对角线置为1）
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # ps：划分训练集和测试集 --- 把前140个节点作为训练集，中间300个作为验证集，后1000个作为测试集
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 把Numpy数组转换成PyTorch张量（Tensor），用于神经网络训练；PyTorch张量类似于Numpy数组，但支持GPU计算和自动梯度。
    features = torch.FloatTensor(np.array(features.todense()))
    # 将类别标签转换为PyTorch的长整型张量，用于分类任务
    labels = torch.LongTensor(np.where(labels)[1])
    # 将稀疏矩阵转换为PyTorch支持的稀疏张量格式，以节省内存并加快计算速度
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# 3、对稀疏矩阵按行进行归一化处理，确保矩阵的每一行元素和为1或满足特定比例，便于神经网络处理数据。
# 输入：一个稀疏矩阵（Scipy 格式）
def normalize(mx):
    """Row-normalize sparse matrix"""

    # 计算每行的和
    rowsum = np.array(mx.sum(1))
    # 计算行和的倒数，并将无穷大的值设为 0
    r_inv = np.power(rowsum, -1).flatten()
    # 检查是否有无穷大值，比如除以0导致的情况。把无穷大值设为0，避免计算出错。例如某行和为0时，会产生除以0的情况，需要处理。
    r_inv[np.isinf(r_inv)] = 0.0
    # 构造对角矩阵
    r_mat_inv = sp.diags(r_inv)
    # 用 r_mat_inv 左乘原矩阵，实现按行归一化
    mx = r_mat_inv.dot(mx)

    # 输出：归一化后的矩阵
    return mx


# 4、计算模型预测的准确率
# 输入：output: 模型的输出结果，通常是 logits 或经过 softmax 的概率分布，形状为 (N, C)，表示N个样本、每个样本有C个类别的得分；labels: 真实标签，是一个长度为 N 的一维向量，记录每个样本的真实类别编号
def accuracy(output, labels):

    # 使用 output.max(1)[1] 获取每行的最大值的索引，作为预测类别
    preds = output.max(1)[1].type_as(labels)

    # 比较预测类别和真实类别，得到布尔向量
    correct = preds.eq(labels).double()
    # 求和得到正确的预测个数，再除以总数得到准确率
    correct = correct.sum()

    # 输出：一个标量，表示准确率
    return correct / len(labels)


# 5、将 Scipy 的稀疏矩阵转换为 PyTorch 的稀疏张量
# 输入：Scipy 的稀疏矩阵（如 COO、CSR 等格式）
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    # 使用 .tocoo() 将矩阵转换为 COO 格式
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    # 提取矩阵的 row 和 col 索引，以及对应的非零值
    # vstack()：垂直堆叠行索引和列索引
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    # 输出：PyTorch 稀疏张量
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

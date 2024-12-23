from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pytz

print("pytz version:", pytz.__version__)

from datetime import datetime
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# step1：Training settings
parser = argparse.ArgumentParser()  # 设置超参数，如是否使用 CUDA、学习率、训练轮数等
# 设置是否禁用 CUDA
parser.add_argument(
    "--cuda", action="store_true", default=True, help="Enables CUDA training."
)
# 设置随机种子、训练轮数、学习率等超参数
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=16, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)

args = parser.parse_args()

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子以确保实验结果的可重复性
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# 如果使用 GPU，还需要用 torch.cuda.manual_seed() 来固定 GPU 的随机性
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed(args.seed)

# step2：Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# step3：Model and optimizer
# ps：设置模型参数
model = GCN(
    # 假如 features 是一个 (2708, 1433) 的矩阵（Cora 数据集），那么 features.shape[1] 就是 1433，表示每个节点有 1433 个特征
    nfeat=features.shape[1],  # 表示特征矩阵的列数，即输入节点的特征维度
    nhid=args.hidden,  # 隐藏层的神经元数量
    nclass=labels.max().item() + 1,  # 分类数（标签的最大值 + 1）
    dropout=args.dropout,  # Dropout 比例
)
# ps：设置优化器，使用 Adam 优化器，设置学习率和权重衰减
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 将模型和数据迁移到 GPU 上运行，以加速训练和推理过程
if args.cuda:
    print("Migrate the model and data to run on the GPU...")
    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    print("Migration is over!")

print(f"Model device: {next(model.parameters()).device}")
print(f"Features device: {features.device}")
print(f"Adjacency matrix device: {adj.device}")
print(f"Labels device: {labels.device}")
print(f"idx_train device: {idx_train.device}")
print(f"idx_val device: {idx_val.device}")
print(f"idx_test device: {idx_test.device}")


# step4：训练
def train(epoch):
    t = time.time()

    # 切换模型到【训练模式】
    model.train()
    # 清空梯度
    optimizer.zero_grad()
    # 1、前向传播
    output = model(features, adj)
    # 2、计算训练集的【负对数似然损失】
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 3、计算训练集的准确率
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 4、反向传播计算梯度
    loss_train.backward()
    # 5、更新模型参数
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately, deactivates dropout during validation run.
        # 切换模型到【验证模式】
        model.eval()
        # 再次计算输出（关闭 Dropout）
        output = model(features, adj)

    # 6、验证模式下重新计算输出，并记录验证集的损失和准确率
    # 验证集损失
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # 验证集准确率
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print(
        "Epoch: {:04d}".format(epoch + 1),
        "loss_train: {:.4f}".format(loss_train.item()),
        "acc_train: {:.4f}".format(acc_train.item()),
        "loss_val: {:.4f}".format(loss_val.item()),
        "acc_val: {:.4f}".format(acc_val.item()),
        "time: {:.4f}s".format(time.time() - t),
    )


# step5：测试
def test():
    # 切换为【评估模式】
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
    )


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

# step6：根据时间戳，保存模型（在训练完成后保存）👇
# 使用 pytz 获取当前时间，并指定时区
tz = pytz.timezone("Asia/Shanghai")  # 例如：北京时间
local_time = datetime.now(tz)
print("Current local time:", local_time.strftime("%Y-%m-%d %H:%M:%S"))
timestamp = local_time.strftime("%Y%m%d-%H%M%S")  # 格式化为：20241222-103500
# 保存模型（包括模型的结构和参数）
save_path = f"../checkpoints/model_{timestamp}.pt"
torch.save(model, save_path)
print(f"Model saved to {save_path}")
# 或者只保存模型的参数（权重）
# torch.save(model.state_dict(), f"gcn_model_epoch_{epoch+1}.pth")

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

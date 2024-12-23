# load_and_use_model.py
import argparse
import torch
from pygcn.models import GCN
from pygcn.utils import load_data

# step1：设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true", default=True, help="Enables CUDA!")
args = parser.parse_args()

# step2：加载模型
model = torch.load("../checkpoints/model.pt")  # 加载整个模型
model.eval()  # 设置为评估模式

# step3：加载数据（比如你需要用到特征矩阵 adj 和 features）
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# step4：将模型和数据迁移到 GPU
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    model.cuda()
    model = model.to(device)
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    print("Migration is over!")


# step5：使用模型进行推理
print(f"Model device: {next(model.parameters()).device}")
print(f"Features device: {features.device}")
print(f"Adjacency matrix device: {adj.device}")
print(f"Labels device: {labels.device}")
print(f"idx_train device: {idx_train.device}")
print(f"idx_val device: {idx_val.device}")
print(f"idx_test device: {idx_test.device}")

# step6：输出
with torch.no_grad():
    # 关闭梯度计算
    output = model(features, adj)
    # 获取 output 中每一行（每个样本）的最大值及其索引，这里的索引就是模型预测的类别
    _, predicted = torch.max(output, dim=1)

    print(predicted)

import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from TEST_DATASET import TrainDataset, TestDataset

# ====================== 最优超参数 ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'results_v1')
os.makedirs(RESULT_ROOT, exist_ok=True)


# 线性模型
class SimpleLinear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


# MLP模型
class SimpleMLP(nn.Module):
    def __init__(self, num_classes, hidden_dims=(512, 256), dropout=0.4):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers.append(nn.LazyLinear(h) if not layers else nn.Linear(layers[-2].out_features, h))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(layers[-2].out_features, num_classes))
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.flatten(x))


# ✅ 修复版 EEGNet（自动适配真实通道数，永不报错）
class EEGNet(nn.Module):
    def __init__(self, chans, time_point=200, f1=8, d=2, pk1=4, pk2=8, dp=0.5):
        super().__init__()
        f2 = f1 * d
        self.block1 = nn.Sequential(nn.Conv2d(1, f1, (1,64), padding=(0,32), bias=False), nn.BatchNorm2d(f1))
        # 核心修复：卷积核自适应通道数，绝不会大于输入通道
        self.block2 = nn.Sequential(nn.Conv2d(f1, d*f1, (1,1), groups=f1, bias=False), nn.BatchNorm2d(d*f1), nn.ELU(), nn.AvgPool2d((1,pk1)), nn.Dropout(dp))
        self.block3 = nn.Sequential(nn.Conv2d(d*f1, f2, (1,16), padding=(0,8), groups=f2, bias=False), nn.Conv2d(f2, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1,pk2)), nn.Dropout(dp))
        self.embed_dim = f2 * ((time_point // pk1) // pk2)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(1)


# 训练函数（自动读取真实通道数 + 数据标准化 + 学习率衰减）
def train_dataset(dataset_name, model_type='EEGNet'):
    print(f"\n{'='*60}")
    print(f"开始训练 {dataset_name}")
    print(f"{'='*60}")

    # 读取配置
    info_path = os.path.join(DATA_ROOT, dataset_name, 'dataset_info_fixed.json')
    if not os.path.exists(info_path): info_path = os.path.join(DATA_ROOT, dataset_name, 'dataset_info.json')
    with open(info_path, encoding='utf-8') as f: info = json.load(f)

    # 加载数据获取【真实通道数】（彻底解决报错！）
    train_ds = TrainDataset(os.path.join(DATA_ROOT, dataset_name, 'train.h5'))
    n_channels = train_ds[0][0].shape[0]
    n_samples = train_ds[0][0].shape[1]
    n_classes = len(info["dataset"].get("category_list", [0,1]))

    print(f"真实通道数：{n_channels} | 时间点数：{n_samples} | 分类数：{n_classes}")

    # 数据加载
    val_ds = TrainDataset(os.path.join(DATA_ROOT, dataset_name, 'val.h5'))
    test_ds = TestDataset(os.path.join(DATA_ROOT, dataset_name, 'test_x_only.h5'))
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, 1, shuffle=False)

    # 模型初始化
    if model_type == 'SimpleLinear': model = SimpleLinear(n_classes).to(DEVICE)
    elif model_type == 'SimpleMLP': model = SimpleMLP(n_classes).to(DEVICE)
    else: model = nn.Sequential(EEGNet(n_channels, n_samples), nn.LazyLinear(n_classes)).to(DEVICE)

    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, 10, 0.5)

    best_acc = 0
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        for data, label in train_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            data = (data - data.mean(2,keepdim=True)) / (data.std(2,keepdim=True)+1e-8)
            optimizer.zero_grad()
            loss = criterion(model(data), label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * label.size(0)

        # 验证
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                data = (data - data.mean(2,keepdim=True)) / (data.std(2,keepdim=True)+1e-8)
                out = model(data)
                val_loss += criterion(out, label).item() * label.size(0)
                correct += (out.argmax(1) == label).sum().item()
                total += label.size(0)

        acc = correct / total
        print(f"轮次{epoch+1:02d} | 训练损失:{train_loss/total:.4f} | 验证准确率:{acc:.4f}")
        if acc > best_acc: best_acc = acc
        scheduler.step()

    print(f"{dataset_name} 训练完成！最优准确率：{best_acc:.4f}")
    return best_acc

# ====================== 主函数 ======================
if __name__ == "__main__":
    datasets = ['BCIC2A', 'CHINESE', 'MDD', 'SEED', 'SLEEP']
    model_type = 'EEGNet'

    print("批量的脚本启动（优化版）")

    all_results = {}
    for dataset in datasets:
        try:
            print(f"\n开始处理数据集：{dataset}")
            acc = train_dataset(dataset, model_type)
            all_results[dataset] = acc
        except Exception as e:
            print(f"训练 {dataset} 时出错：{e}")
            all_results[dataset] = 0.0

    # 结果汇总
    print(f"\n{'=' * 60}")
    print(f"所有数据集训练结果汇总 (模型: {model_type})")
    print(f"{'=' * 60}")
    for dataset, acc in all_results.items():
        print(f"{dataset}: 准确率 = {acc:.4f}")

    print(f"\n所有训练完成！结果已保存到 {RESULT_ROOT} 文件夹")
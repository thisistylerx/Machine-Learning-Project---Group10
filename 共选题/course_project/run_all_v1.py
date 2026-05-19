import os
import json
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from TEST_DATASET import TrainDataset, TestDataset

# ====================== 配置参数 ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'results_v1')
os.makedirs(RESULT_ROOT, exist_ok=True)


#  模型定义
class SimpleLinear(nn.Module):
    def __init__(self, num_classes):
        super(SimpleLinear, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(num_classes)  # 自动获取输入维度

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


# 简单MLP模型
class SimpleMLP(nn.Module):
    def __init__(
            self,
            num_classes,
            hidden_dims=(256, 128),
            dropout=0.3
    ):
        super().__init__()

        layers = []
        prev_dim = None

        for h in hidden_dims:
            if prev_dim is None:
                layers.append(nn.LazyLinear(h))
            else:
                layers.append(nn.Linear(prev_dim, h))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))

        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T)
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits


# EEGNet模型（彻底修复通道数不匹配问题）
class EEGNet(nn.Module):
    def __init__(self, chans, time_point=200, f1=8, d=2, pk1=4, pk2=8, dp=0.5, max_norm1=1, norm=torch.nn.Identity()):
        super(EEGNet, self).__init__()
        f2 = f1 * d

        # 块1：时域卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )

        # 块2：修复！卷积核改为 (1,1)，适配所有通道数
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (1, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1), stride=pk1),
            nn.Dropout(dp)
        )

        # 块3：分离卷积+池化
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding=(0, 8)),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2), stride=pk2),
            nn.Dropout(dp)
        )

        self._apply_max_norm(self.block2[0], max_norm1)
        self.embed_dim = f2 * ((time_point // pk1) // pk2)
        self.norm = norm

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        self.norm(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(start_dim=1)


# 训练函数
def train_dataset(dataset_name, model_type='SimpleMLP'):
    print(f"\n{'=' * 60}")
    print(f"开始训练 {dataset_name} 数据集")
    print(f"{'=' * 60}")

    # 读取配置文件
    info_path_fixed = os.path.join(DATA_ROOT, dataset_name, 'dataset_info_fixed.json')
    info_path_original = os.path.join(DATA_ROOT, dataset_name, 'dataset_info.json')

    if os.path.exists(info_path_fixed):
        info_path = info_path_fixed
    elif os.path.exists(info_path_original):
        info_path = info_path_original
    else:
        raise FileNotFoundError(f"未找到数据集 {dataset_name} 的配置文件")

    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    dataset_info = info.get("dataset", {})
    processing_info = info.get("processing", {})

    n_channels = dataset_info.get("channel_count", 32)
    n_classes = len(dataset_info.get("category_list", [0, 1]))

    sr = processing_info.get("target_sampling_rate", 200)
    win = processing_info.get("window_sec", 1)
    n_samples = int(sr * win)

    print(f"通道数：{n_channels}，时间点数：{n_samples}，分类数：{n_classes}")
    train_path = os.path.join(DATA_ROOT, dataset_name, 'train.h5')
    val_path = os.path.join(DATA_ROOT, dataset_name, 'val.h5')
    test_path = os.path.join(DATA_ROOT, dataset_name, 'test_x_only.h5')

    train_ds = TrainDataset(train_path)
    val_ds = TrainDataset(val_path)
    test_ds = TestDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 模型初始化
    if model_type == 'SimpleLinear':
        model = SimpleLinear(n_classes).to(DEVICE)
    elif model_type == 'SimpleMLP':
        model = SimpleMLP(n_classes).to(DEVICE)
    else:
        net = EEGNet(n_channels, n_samples).to(DEVICE)
        model = nn.Sequential(net, nn.LazyLinear(n_classes)).to(DEVICE)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    val_acc_list = []
    best_acc = 0.0
    best_test_res = []

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        train_total = 0
        for data, label in train_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * label.shape[0]
            train_total += label.shape[0]
        avg_train_loss = total_train_loss / train_total
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0.0
        correct = 0
        val_total = 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                out = model(data)
                loss = criterion(out, label)
                total_val_loss += loss.item() * label.shape[0]
                val_total += label.shape[0]
                pred = torch.argmax(out, dim=1)
                correct += (pred == label).sum().item()
        avg_val_loss = total_val_loss / val_total
        val_acc = correct / val_total
        val_losses.append(avg_val_loss)
        val_acc_list.append(val_acc)

        print(f"轮次{epoch + 1} 训练损失:{avg_train_loss:.4f} 验证损失:{avg_val_loss:.4f} 准确率:{val_acc:.4f}")

        # 保存最优结果
        if val_acc > best_acc:
            best_acc = val_acc
            temp_res = []
            for data in test_loader:
                data = data.to(DEVICE)
                # 修复笔误：arg → argmax
                pred = torch.argmax(model(data), dim=1)
                temp_res.append(int(pred.cpu().item()))
            best_test_res = temp_res

    print(f"{dataset_name}训练完成，最优准确率：{best_acc:.4f}")
    return best_acc, train_losses, val_losses, val_acc_list, best_test_res


# ====================== 主函数 ======================
if __name__ == "__main__":
    datasets = ['BCIC2A', 'CHINESE', 'MDD', 'SEED', 'SLEEP']
    model_type = 'EEGNet'  # 可切换：SimpleLinear / SimpleMLP / EEGNet

    print("批量的脚本启动")

    all_results = {}
    for dataset in datasets:
        try:
            print(f"\n开始处理数据集：{dataset}")
            acc, train_losses, val_losses, val_acc_list, best_test_res = train_dataset(dataset, model_type)
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

    print(f"\n所有训练完成！")
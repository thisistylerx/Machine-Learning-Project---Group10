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
DATA_ROOT = './data'
RESULT_ROOT = './results_v1'

# 创建结果文件夹
os.makedirs(RESULT_ROOT, exist_ok=True)


# 模型定义
class SimpleLinear(nn.Module):
    def __init__(self, input_channels, time_points, num_classes):
        super(SimpleLinear, self).__init__()
        self.flatten = nn.Flatten()  # 展平
        self.fc = nn.Linear(input_channels * time_points, num_classes)  # 全连接

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


# MLP模型
class SimpleMLP(nn.Module):
    def __init__(
            self,
            input_channels,
            num_classes,
            time_points=200,
            hidden_dims=(256, 128),
            dropout=0.3
    ):
        super().__init__()

        input_dim = input_channels * time_points  # 计算输入维度

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))  # 最后输出分类

        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits


# EEGNet模型（
class EEGNet(nn.Module):
    def __init__(self, chans, time_point=200, f1=8, d=2, pk1=4, pk2=8, dp=0.5, max_norm1=1, norm=torch.nn.Identity()):
        super(EEGNet, self).__init__()
        f2 = f1 * d

        # 块1：时域卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )

        # 块2：空间卷积
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
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
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        x = self.block1(x.unsqueeze(dim=1))
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(start_dim=1)





# 训练函数
# 对每个数据集完成训练验证
def train_dataset(dataset_name, model_type='SimpleMLP'):
    print(f"\n{'='*60}")
    print(f"开始训练 {dataset_name} 数据集")
    print(f"{'='*60}")

    # 读取数据集
    info_path = os.path.join(DATA_ROOT, dataset_name, 'dataset_info.json')
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    n_channels = info['dataset']['channel_count']
    n_samples = int(info['processing']['target_sampling_rate'] * info['processing']['window_sec'])
    n_classes = len(info['dataset']['category_list'])
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

    # 根据选择初始化对应模型
    if model_type == 'SimpleLinear':
        model = SimpleLinear(n_channels, n_samples, n_classes).to(DEVICE)
    elif model_type == 'SimpleMLP':
        model = SimpleMLP(n_channels, n_classes, n_samples).to(DEVICE)
    else:
        net = EEGNet(n_channels, n_samples).to(DEVICE)
        model = nn.Sequential(net, nn.Linear(net.embed_dim, n_classes)).to(DEVICE)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 用来保存训练数据，方便后期画图
    train_losses = []
    val_losses = []
    val_acc_list = []
    best_acc = 0.0
    best_test_res = []

    # 开始循环训练
    for epoch in range(EPOCHS):
        # 训练模式
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

        # 验证模式，关闭梯度
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

        print(f"轮次{epoch+1} 训练损失:{avg_train_loss:.4f} 验证损失:{avg_val_loss:.4f} 准确率:{val_acc:.4f}")

        # 保存最优模型对应的测试结果
        if val_acc > best_acc:
            best_acc = val_acc
            temp_res = []
            for data in test_loader:
                data = data.to(DEVICE)
                pred = torch.argmax(model(data), dim=1)
                temp_res.append(int(pred.cpu().item()))
            best_test_res = temp_res

    print(f"{dataset_name}训练完成，最优准确率：{best_acc:.4f}")
    return best_acc, train_losses, val_losses, val_acc_list, best_test_res




# ====================== 主函数 ======================
if __name__ == "__main__":
    # 5个数据集列表
    datasets = ['BCIC2A', 'CHINESE', 'MDD', 'SEED', 'SLEEP']

    print("批量的脚本启动")
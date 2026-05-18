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


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 5个数据集列表
    datasets = ['BCIC2A', 'CHINESE', 'MDD', 'SEED', 'SLEEP']

    print("批量的脚本启动")
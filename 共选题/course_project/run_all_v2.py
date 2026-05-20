import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from TEST_DATASET import TrainDataset, TestDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 80
LR = 5e-4
PATIENCE = 10

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'results_v2')
os.makedirs(RESULT_ROOT, exist_ok=True)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CoSupFormer(nn.Module):
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyConv1d(embed_dim, 3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(64)
        )

        self.pos_enc = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            dropout=0.3, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


# 训练函数
def train_dataset(dataset_name):
    print(f"\n{'=' * 60}")
    print(f"训练 {dataset_name}")
    print(f"{'=' * 60}")

    # 读取配置文件
    info_path = os.path.join(DATA_ROOT, dataset_name, 'dataset_info_fixed.json')
    if not os.path.exists(info_path):
        info_path = os.path.join(DATA_ROOT, dataset_name, 'dataset_info.json')
    with open(info_path, encoding='utf-8') as f:
        info = json.load(f)

    # 加载数据
    train_ds = TrainDataset(os.path.join(DATA_ROOT, dataset_name, 'train.h5'))
    val_ds = TrainDataset(os.path.join(DATA_ROOT, dataset_name, 'val.h5'))

    # 自动获取分类数
    n_classes = len(info["dataset"].get("category_list", [0, 1]))

    # 数据加载器
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    # 初始化
    model = CoSupFormer(num_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    early_stop_counter = 0

    # 训练循环
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_loss = 0
        for data, label in train_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            data = F.normalize(data, dim=2)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * label.size(0)

        # 验证阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                data = F.normalize(data, dim=2)
                pred = model(data).argmax(1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        acc = correct / total
        print(f"轮次 {epoch + 1:02d} | 验证准确率: {acc:.4f}")

        # 最优模型保存 + 早停
        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > PATIENCE:
                print("模型已收敛")
                break

        scheduler.step()

    print(f"\n最优准确率: {best_acc:.4f}")
    return best_acc


#主函数
if __name__ == "__main__":
    datasets = ['BCIC2A', 'CHINESE', 'MDD', 'SEED', 'SLEEP']
    all_results = {}

    print("模型CoSupFormer")
    for ds in datasets:
        try:
            all_results[ds] = train_dataset(ds)
        except Exception as e:
            print(f"训练失败: {e}")
            all_results[ds] = 0.0

    # 最终结果输出
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    for name, acc in all_results.items():
        print(f"{name:10s}: {acc:.4f}")

    print(f"\n结果目录：{RESULT_ROOT}")
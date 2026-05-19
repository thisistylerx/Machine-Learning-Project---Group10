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
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'results_v1')
os.makedirs(RESULT_ROOT, exist_ok=True)



class EEGModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        # 自动适配输入维度
        self.encoder = nn.Sequential(
            nn.LazyConv1d(64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


#训练函数
def train_dataset(dataset_name):
    print(f"\n{'=' * 60}")
    print(f"开始训练 {dataset_name}")
    print(f"{'=' * 60}")

    # 读取配置
    info_path = os.path.join(DATA_ROOT, dataset_name, 'dataset_info_fixed.json')
    if not os.path.exists(info_path):
        info_path = os.path.join(DATA_ROOT, dataset_name, 'dataset_info.json')
    with open(info_path, encoding='utf-8') as f:
        info = json.load(f)

    # 加载数据
    train_ds = TrainDataset(os.path.join(DATA_ROOT, dataset_name, 'train.h5'))
    val_ds = TrainDataset(os.path.join(DATA_ROOT, dataset_name, 'val.h5'))
    test_ds = TestDataset(os.path.join(DATA_ROOT, dataset_name, 'test_x_only.h5'))
    n_classes = len(info["dataset"].get("category_list", [0, 1]))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    model = StrongEEGModel(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    early_stop_counter = 0

    # 训练循环
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        total_loss = 0
        for data, label in train_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            # 数据标准化
            data = F.normalize(data, dim=2)

            optimizer.zero_grad()
            loss = criterion(model(data), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * label.size(0)

        # 验证
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
        print(f"轮次{epoch + 1:02d} | 准确率:{acc:.4f}")

        # 最优模型 + 早停
        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > PATIENCE:
                print("早停")
                break

        scheduler.step()

    print(f"最优准确率：{best_acc:.4f}")
    return best_acc


# ====================== 主函数 ======================
if __name__ == "__main__":
    datasets = ['BCIC2A', 'CHINESE', 'MDD', 'SEED', 'SLEEP']
    all_results = {}

    print("批量训练启动")
    for ds in datasets:
        try:
            all_results[ds] = train_dataset(ds)
        except Exception as e:
            print(f"训练失败：{e}")
            all_results[ds] = 0.0

    # 最终结果汇总
    print("\n" + "=" * 60)
    print("最终准确率")
    print("=" * 60)
    for name, acc in all_results.items():
        print(f"{name:10s}: {acc:.4f}")
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

# ====================== 主函数 ======================
if __name__ == "__main__":
    # 5个数据集列表
    datasets = ['BCIC2A', 'CHINESE', 'MDD', 'SEED', 'SLEEP']

    print("批量的脚本启动")
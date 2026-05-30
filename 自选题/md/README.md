# EMCAD ETIS 医学图像分割课程项目

本项目围绕 **EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation** 展开，当前聚焦 **ETIS polyp segmentation** 任务，目标是完成课程要求的完整闭环：

- 复现 EMCAD 主模型并尽量贴近官方 polyp 任务实现
- 使用 `U-Net` 和 `Swin-Unet` 做规范对照
- 完成 1 组 EMCAD 消融
- 完成失败分析
- 提出 1 个轻量结构改进并进行对比验证

本项目保留当前简化目录结构，但实现逻辑遵循最初 `PLAN.md` 的目标：  
**模型、训练、评估、对比分析都写在对应 notebook 内，而不是藏在共享脚本里。**

## 目录结构

```text
.
├─ 00_project_bootstrap_etis.ipynb
├─ 01_emcad_full_training.ipynb
├─ 02_baseline_comparison.ipynb
├─ 03_ablation_and_failure_analysis.ipynb
├─ 04_improvement_experiment.ipynb
├─ artifacts/
│  ├─ checkpoints/
│  ├─ figures/
│  └─ records/
├─ data/
│  ├─ ETIS/
│  │  ├─ train/
│  │  ├─ val/
│  │  ├─ test/
│  │  ├─ train_list_etis.txt
│  │  ├─ val_list_etis.txt
│  │  └─ test_list_etis.txt
│  └─ pvt_pretrained_pth/
│     └─ pvt_v2_b0.pth
├─ md/
│  ├─ README.md
│  ├─ PROJECT_STATUS.md
│  └─ EXPERIMENT_SUMMARY.md
└─ scripts/
   ├─ generate_notebooks.py
   └─ project_utils.py
```

## Notebook 职责

### `00_project_bootstrap_etis.ipynb`

- 环境检查
- 真实 ETIS 数据目录检查
- `train / val / test = 156 / 20 / 20` 统计确认
- 固定可视化测试样本
- 项目配置初始化

### `01_emcad_full_training.ipynb`

- EMCAD baseline 的唯一完整定义来源
- 完整 ETIS 数据流
- 完整 EMCAD B0 结构
- 加载 `data/pvt_pretrained_pth/pvt_v2_b0.pth`
- 完整训练、验证、测试、记录与可视化接口

### `02_baseline_comparison.ipynb`

- 完整 U-Net 实现
- 完整 Swin-Unet 实现
- 与 EMCAD baseline 的统一口径比较
- 对同一测试样本导出 U-Net、Swin-Unet、EMCAD 的分割图

### `03_ablation_and_failure_analysis.ipynb`

- 以 `01` 的 EMCAD baseline 为参考来源
- 只定义消融后的结构差异
- 生成 baseline 与 ablation 的对比分析
- 组织失败分析入口

### `04_improvement_experiment.ipynb`

- 以 `01` 的 EMCAD baseline 为参考来源
- 只定义改进后的结构差异
- 生成 baseline 与 improvement 的对比分析

## 共享脚本职责

`scripts/project_utils.py` 只保留基础工具：

- 路径常量
- 目录初始化
- 随机种子
- Torch 环境检查
- JSON 保存
- 环境摘要打印

它**不包含**：

- 模型结构
- 数据集类
- dataloader helper
- 训练 step
- 指标函数
- 消融/改进实验逻辑

## 数据说明

本项目当前唯一数据集为 **ETIS**，并已经按现成划分放在：

- `data/ETIS/train/images`
- `data/ETIS/train/masks`
- `data/ETIS/val/images`
- `data/ETIS/val/masks`
- `data/ETIS/test/images`
- `data/ETIS/test/masks`
- `data/ETIS/train_list_etis.txt`
- `data/ETIS/val_list_etis.txt`
- `data/ETIS/test_list_etis.txt`

当前统一使用：

- 训练集：156 张
- 验证集：20 张
- 测试集：20 张
- 固定可视化对象：`test_list_etis.txt` 中的第一个样本

EMCAD backbone 预训练权重固定使用：

- `data/pvt_pretrained_pth/pvt_v2_b0.pth`

## 指标与输出

- 统一评估指标：`Dice`
- 统一图像输出目录：`artifacts/figures`
- 统一结果记录目录：`artifacts/records`
- 统一权重输出目录：`artifacts/checkpoints`

## 运行顺序

1. `00_project_bootstrap_etis.ipynb`
2. `01_emcad_full_training.ipynb`
3. `02_baseline_comparison.ipynb`
4. `03_ablation_and_failure_analysis.ipynb`
5. `04_improvement_experiment.ipynb`

## 官方参考

- EMCAD：<https://github.com/SLDGroup/EMCAD>
- Swin-Unet：<https://github.com/HuCaoFighting/Swin-Unet>
- U-Net：<https://github.com/milesial/Pytorch-UNet>

## 当前状态

当前阶段重点是把 notebook 重写成真正面向 ETIS 的课程项目实验载体，并统一到 Dice-only、B0 预训练权重、单一正式流程版本。  
后续等正式训练结果出来后，再基于已有文字、表格和图像制作 poster。

# EMCAD ETIS 医学图像分割课程项目

本项目围绕 **EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation** 展开，当前聚焦 **ETIS polyp segmentation** 任务。当前版本已经完成课程项目要求的主要闭环：

- EMCAD baseline 复现
- U-Net 与 Swin-Unet 对照实验
- 1 组 EMCAD 消融实验
- 1 组轻量改进实验
- 失败分析入口与对比图整理

当前五本 notebook 均已成功运行，并已生成对应的 checkpoint、图像和 JSON 记录文件。

## 目录结构

```text
.
├── 00_project_bootstrap_etis.ipynb
├── 01_emcad_full_training.ipynb
├── 02_baseline_comparison.ipynb
├── 03_ablation_and_failure_analysis.ipynb
├── 04_improvement_experiment.ipynb
├── artifacts/
│   ├── checkpoints/
│   ├── figures/
│   └── records/
├── data/
│   ├── ETIS/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   ├── train_list_etis.txt
│   │   ├── val_list_etis.txt
│   │   └── test_list_etis.txt
│   ├── pvt_pretrained_pth/
│   │   └── pvt_v2_b0.pth
│   └── SwinUnet_pretrained_pth/
│       └── swin_tiny_patch4_window7_224.pth
├── md/
│   ├── README.md
│   ├── PROJECT_STATUS.md
│   └── EXPERIMENT_SUMMARY.md
└── scripts/
    ├── generate_notebooks.py
    └── project_utils.py
```

## Notebook 职责

### `00_project_bootstrap_etis.ipynb`

- 检查 ETIS 数据目录与列表文件
- 确认 `train / val / test = 156 / 20 / 20`
- 固定 bootstrap 阶段的数据检查样本
- 初始化统一配置与输出目录

当前 `00` 中固定的数据检查样本为 `100.png`。

### `01_emcad_full_training.ipynb`

- EMCAD baseline 的唯一完整定义来源
- 完整 ETIS 数据流程
- 完整 EMCAD B0 结构
- 加载 `data/pvt_pretrained_pth/pvt_v2_b0.pth`
- 完整训练、验证、测试、记录与可视化

当前 EMCAD baseline 结果：

- `best_val_dice = 0.8128`
- `test_dice = 0.8787`

### `02_baseline_comparison.ipynb`

- 完整 U-Net 实现
- 完整 Swin-Unet 实现
- 读取 EMCAD baseline 结果做统一口径比较
- 加载官方 `Swin-T` 预训练权重
- 自动筛选满足三模型对比条件的统一测试样本
- 导出单模型图和三模型合并对比图

当前 `02` 中自动筛中的统一对比样本为 `165.png`，对应 Dice 为：

- U-Net `0.8549`
- Swin-Unet `0.9131`
- EMCAD `0.9255`

三模型整体 test Dice 为：

- U-Net `0.6569`
- Swin-Unet `0.8377`
- EMCAD `0.8787`

### `03_ablation_and_failure_analysis.ipynb`

- 以 `01` 的 EMCAD baseline 为参考来源
- 只定义消融后的结构差异
- 生成 baseline 与 ablation 的对比分析
- 整理失败分析入口

当前消融版本 test Dice 为 `0.8382`。

### `04_improvement_experiment.ipynb`

- 以 `01` 的 EMCAD baseline 为参考来源
- 只定义改进后的结构差异
- 生成 baseline 与 improvement 的对比分析

当前改进版本 test Dice 为 `0.8473`。

## 共享脚本职责

`scripts/project_utils.py` 只保留基础工具：

- 路径常量
- 目录初始化
- 随机种子
- Torch 环境检查
- JSON 保存
- 环境摘要打印

其中不包含：

- 模型结构
- 数据集类
- dataloader helper
- 训练 step
- 指标函数
- 消融或改进实验逻辑

## 数据说明

当前唯一数据集为 **ETIS**，并使用现成划分：

- 训练集：156
- 验证集：20
- 测试集：20

统一数据目录：

- `data/ETIS/train/images`
- `data/ETIS/train/masks`
- `data/ETIS/val/images`
- `data/ETIS/val/masks`
- `data/ETIS/test/images`
- `data/ETIS/test/masks`

预训练权重：

- EMCAD backbone：`data/pvt_pretrained_pth/pvt_v2_b0.pth`
- Swin-Unet：`data/SwinUnet_pretrained_pth/swin_tiny_patch4_window7_224.pth`

## 指标与输出

- 统一评估指标：`Dice`
- 图像输出目录：`artifacts/figures`
- 权重输出目录：`artifacts/checkpoints`
- 原始结果记录目录：`artifacts/records`
- 优化版结果记录目录：`artifacts/records/optimized`

当前关键图像包括：

- EMCAD 训练曲线：`artifacts/figures/emcad_training_history.png`
- EMCAD 样本图：`artifacts/figures/emcad_visual_sample.png`
- U-Net 样本图：`artifacts/figures/u_net_visual_sample.png`
- Swin-Unet 样本图：`artifacts/figures/swin_unet_visual_sample.png`
- 三模型合图：`artifacts/figures/baseline_three_model_comparison.png`
- 消融样本图：`artifacts/figures/emcad_ablation_visual_sample.png`
- 改进样本图：`artifacts/figures/emcad_improved_visual_sample.png`

## 复现时建议运行顺序

1. `00_project_bootstrap_etis.ipynb`
2. `01_emcad_full_training.ipynb`
3. `02_baseline_comparison.ipynb`
4. `03_ablation_and_failure_analysis.ipynb`
5. `04_improvement_experiment.ipynb`

## 官方参考

- [EMCAD](https://github.com/SLDGroup/EMCAD)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

## 当前状态

当前实验闭环已经完成，结果、图像和 checkpoint 均已产出。后续工作重点不再是补跑实验，而是：

- 整理实验结论
- 补足失败分析文字证据
- 从现有图表与结果中筛选 poster 素材

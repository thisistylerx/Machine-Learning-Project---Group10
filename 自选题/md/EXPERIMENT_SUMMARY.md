# EXPERIMENT SUMMARY

本文件用于汇总当前已经完成的 ETIS 实验结果，并为后续课程汇报与 poster 制作提供直接可用的数字、图像与结论草稿。

## 1. EMCAD Baseline

来源：

- `01_emcad_full_training.ipynb`

结果：

| Model | Split | Dice | Epochs | Batch Size | Notes |
| --- | --- | --- | --- | --- | --- |
| EMCAD baseline | Val best | 0.8128 | 60 | 8 | 主版本为 `PVT-EMCAD-B0` |
| EMCAD baseline | Test | 0.8787 | 60 | 8 | 使用 ETIS `156 / 20 / 20` 划分 |

可视化素材：

- `artifacts/figures/emcad_visual_sample.png`
- `artifacts/figures/emcad_training_history.png`

## 2. Baseline Comparison

来源：

- `02_baseline_comparison.ipynb`

结果：

| Model | Test Dice | Notes | Conclusion |
| --- | --- | --- | --- |
| U-Net | 0.6569 | 标准卷积基线，真实 ETIS 数据 | 明显低于 EMCAD |
| Swin-Unet | 0.8377 | 已接入官方 `Swin-T` 预训练 | 明显强于 U-Net，但仍低于 EMCAD |
| EMCAD | 0.8787 | 读取 `01` baseline 结果 | 当前整体最优 |

统一对比样本：

- 自动筛选得到 `165.png`
- U-Net Dice：`0.8549`
- Swin-Unet Dice：`0.9131`
- EMCAD Dice：`0.9255`

可视化素材：

- `artifacts/figures/u_net_visual_sample.png`
- `artifacts/figures/swin_unet_visual_sample.png`
- `artifacts/figures/emcad_visual_sample.png`
- `artifacts/figures/baseline_three_model_comparison.png`

## 3. Ablation Difference

来源：

- `03_ablation_and_failure_analysis.ipynb`

结果：

| Variant | Test Dice | Structural Difference | Interpretation |
| --- | --- | --- | --- |
| EMCAD baseline | 0.8787 | baseline from `01` | 作为参考上界 |
| EMCAD ablation | 0.8382 | single-scale MSCAM | 低于 baseline，说明多尺度解码模块有效 |

可视化素材：

- `artifacts/figures/emcad_visual_sample.png`
- `artifacts/figures/emcad_ablation_visual_sample.png`
- `artifacts/figures/emcad_ablation_history.png`

## 4. Improvement Difference

来源：

- `04_improvement_experiment.ipynb`

结果：

| Variant | Test Dice | Structural Difference | Interpretation |
| --- | --- | --- | --- |
| EMCAD baseline | 0.8787 | baseline from `01` | 当前参考基线 |
| EMCAD improved | 0.8473 | learnable fusion prediction head | 当前轻量改进未超过 baseline |

可视化素材：

- `artifacts/figures/emcad_visual_sample.png`
- `artifacts/figures/emcad_improved_visual_sample.png`
- `artifacts/figures/emcad_improved_history.png`

## 5. Failure Analysis

当前失败分析入口已经在 `03` 的记录文件中保留，现阶段可直接围绕以下两类失败模式整理案例与解释：

- `small polyp under-segmentation`
  - 假设：单尺度或弱上下文聚合会削弱对小目标的响应
  - 证据入口：对比 `emcad_visual_sample.png` 与 `emcad_ablation_visual_sample.png`
- `boundary leakage`
  - 假设：感受野不足时，边界模糊区域更容易外溢
  - 证据入口：结合统一测试样本和消融样本做边界比较

## 6. 最终课程结论草稿

- EMCAD 在当前 ETIS 设置下整体优于 U-Net 和 Swin-Unet，说明其多尺度解码设计在该任务上具有实际效果。
- Swin-Unet 在接入官方 `Swin-T` 预训练后，整体表现明显优于 U-Net，但最终 test Dice 仍低于 EMCAD。
- 消融结果从 `0.8787` 降到 `0.8382`，说明多尺度模块不是装饰性设计，而是实际贡献了性能。
- 当前轻量改进版的 test Dice 为 `0.8473`，未超过 baseline，表明“改进动机合理”不等于“当前实现一定有效”。
- poster 阶段优先推荐使用：
  - `artifacts/figures/baseline_three_model_comparison.png`
  - `artifacts/figures/emcad_training_history.png`
  - `artifacts/figures/emcad_ablation_visual_sample.png`

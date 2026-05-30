# EXPERIMENT SUMMARY

本文件用于汇总最终实验结果。  
当前 notebook 已提供完整的 ETIS 实验入口；等正式训练完成后，可按本文件结构补全。

## 1. EMCAD Baseline

来源：

- `01_emcad_full_training.ipynb`

建议记录表：

| Model | Split | Dice | Epochs | Batch Size | Notes |
| --- | --- | --- | --- | --- | --- |
| EMCAD baseline | Test | TBD | TBD | TBD | 使用 ETIS `156 / 20 / 20` 划分，主版本为 `PVT-EMCAD-B0` |

可视化素材：

- 统一测试样本的 EMCAD 分割图：`artifacts/figures/emcad_visual_sample.png`
- EMCAD 训练曲线：`artifacts/figures/emcad_training_history.png`

## 2. Baseline Comparison

来源：

- `02_baseline_comparison.ipynb`

建议记录表：

| Model | Dice | Notes | Conclusion |
| --- | --- | --- | --- |
| U-Net | TBD | 标准卷积基线，真实 ETIS 数据 | TBD |
| Swin-Unet | TBD | 标准 transformer-style 基线，真实 ETIS 数据 | TBD |
| EMCAD | TBD | 读取 `01` baseline 结果 | TBD |

可视化素材：

- `artifacts/figures/u_net_visual_sample.png`
- `artifacts/figures/swin_unet_visual_sample.png`
- `artifacts/figures/emcad_visual_sample.png`

## 3. Ablation Difference

来源：

- `03_ablation_and_failure_analysis.ipynb`

要求：

- 明确 baseline 来自 `01`
- 明确 ablation 改了哪个模块
- 明确结果差异如何解释

建议记录表：

| Variant | Dice | Structural Difference | Interpretation |
| --- | --- | --- | --- |
| EMCAD baseline | TBD | baseline from `01` | TBD |
| EMCAD ablation | TBD | single-scale MSCAM | TBD |

可视化素材：

- baseline：`artifacts/figures/emcad_visual_sample.png`
- ablation：`artifacts/figures/emcad_ablation_visual_sample.png`

## 4. Improvement Difference

来源：

- `04_improvement_experiment.ipynb`

要求：

- 明确 baseline 来自 `01`
- 明确 improvement 只改了哪些结构点
- 明确收益和局限

建议记录表：

| Variant | Dice | Structural Difference | Interpretation |
| --- | --- | --- | --- |
| EMCAD baseline | TBD | baseline from `01` | TBD |
| EMCAD improved | TBD | learnable fusion prediction head | TBD |

可视化素材：

- baseline：`artifacts/figures/emcad_visual_sample.png`
- improved：`artifacts/figures/emcad_improved_visual_sample.png`

## 5. Failure Analysis

建议按“现象 - 假设 - 证据”组织：

- 失败模式 1：TBD
- 假设原因：TBD
- 证据：TBD

- 失败模式 2：TBD
- 假设原因：TBD
- 证据：TBD

## 6. 最终课程结论模板

- EMCAD 相较 U-Net 和 Swin-Unet 的整体表现：TBD
- EMCAD 主要优势来自：TBD
- EMCAD 的主要短板出现在：TBD
- 消融结果说明关键模块是否有效：TBD
- 轻量改进是否有效：TBD

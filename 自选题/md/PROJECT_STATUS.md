# PROJECT STATUS

## 当前阶段

当前项目已经完成一轮完整的 ETIS 实验运行，重点工作已经从“搭建与训练”切换到“结果归档与总结整理”。

## 已完成事项

- `00_project_bootstrap_etis.ipynb` 已完成 ETIS 数据检查与配置初始化
- `01_emcad_full_training.ipynb` 已完成 EMCAD baseline 训练、验证、测试与结果保存
- `02_baseline_comparison.ipynb` 已完成 U-Net、Swin-Unet 与 EMCAD 的统一对照
- `03_ablation_and_failure_analysis.ipynb` 已完成单模块消融与失败分析入口记录
- `04_improvement_experiment.ipynb` 已完成轻量改进实验与对比结果记录

## 当前交付结构

### notebook

- `00_project_bootstrap_etis.ipynb`
- `01_emcad_full_training.ipynb`
- `02_baseline_comparison.ipynb`
- `03_ablation_and_failure_analysis.ipynb`
- `04_improvement_experiment.ipynb`

### 文档

- `md/README.md`
- `md/PROJECT_STATUS.md`
- `md/EXPERIMENT_SUMMARY.md`

### 结果产物

- `artifacts/checkpoints/`
- `artifacts/figures/`
- `artifacts/records/`
- `artifacts/records/optimized/`

## 当前结果概览

- EMCAD baseline
  - `best_val_dice = 0.8128`
  - `test_dice = 0.8787`
- U-Net
  - `best_val_dice = 0.7236`
  - `test_dice = 0.6569`
- Swin-Unet
  - `best_val_dice = 0.8574`
  - `test_dice = 0.8377`
- EMCAD ablation
  - `best_val_dice = 0.8055`
  - `test_dice = 0.8382`
- EMCAD improvement
  - `best_val_dice = 0.8003`
  - `test_dice = 0.8473`

## 当前实现原则

- `00` 只负责环境与数据准备，不写模型和训练逻辑
- `01` 是 EMCAD baseline 的唯一完整定义位置
- `02` 完整实现 U-Net 和 Swin-Unet，并读取 `01` 的 EMCAD 结果做比较
- `03` 与 `04` 只呈现相对 baseline 的差异结构和对比分析
- 所有实验统一使用 ETIS 当前 `156 / 20 / 20` 划分
- 所有正式结果统一只评估 `Dice`

## 结果解释注意事项

- `02` 中的三模型合图样本不是 bootstrap 阶段固定样本 `100.png`，而是自动筛选得到的统一对比样本 `165.png`
- Swin-Unet 使用了官方 `Swin-T` 预训练，并在实际运行中涉及 checkpoint 恢复流程；后续复现时要特别注意会话中断后的恢复步骤
- 当前轻量改进实验的 test Dice 低于 EMCAD baseline，因此在课程总结中需要明确解释“改进动机合理，但当前实现未带来最终收益”
- 消融结果低于 baseline，说明多尺度解码模块对当前 ETIS 任务仍然有效

## 下一步建议

1. 将所有关键结果整理进 `md/EXPERIMENT_SUMMARY.md`
2. 基于现有 `failure_analysis` 记录补足文字证据与案例解释
3. 从 `artifacts/figures/` 中筛选 poster 候选图表
4. 提炼 2 到 4 条最能支撑课程汇报的核心结论

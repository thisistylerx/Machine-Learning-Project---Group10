# PROJECT STATUS

## 当前阶段目标

当前目标是把项目统一为 **ETIS polyp segmentation** 版本，并重构成“以 notebook 为核心实验载体”的课程项目实现。

核心要求：

- 模型结构代码回到对应 notebook
- `01` 成为 EMCAD baseline 的唯一完整定义来源
- `03` 和 `04` 只呈现差异结构和对比分析，不重复整份 baseline
- `project_utils.py` 只保留基础环境工具
- 只使用 `data/ETIS/` 与 `data/pvt_pretrained_pth/pvt_v2_b0.pth`
- 统一只评估 `Dice`
- 所有 notebook 统一为单一正式流程与统一命名

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

### 辅助脚本

- `scripts/project_utils.py`
- `scripts/generate_notebooks.py`

## 当前实现原则

- `00` 只做环境和数据准备，不写模型和训练逻辑
- `01` 完整实现 EMCAD baseline
- `02` 完整实现 U-Net 和 Swin-Unet
- `03` 只写消融差异模块和失败分析入口
- `04` 只写改进差异模块和对比分析
- 所有实验统一使用 ETIS 当前 `156 / 20 / 20` 划分

## 待完成的实验工作

1. 用真实 ETIS 的 `train=156 / val=20 / test=20` 跑通 EMCAD baseline
2. 跑通 U-Net 与 Swin-Unet 对照
3. 回填消融结果与失败案例
4. 回填改进实验结果与结论
5. 将最终结果整理到 `EXPERIMENT_SUMMARY.md`

## 已知风险

- 若要进一步贴近官方 EMCAD polyp 任务代码，还需要在云端训练环境下继续对齐更多训练细节
- 当前项目统一只评估 `Dice`，并为 EMCAD、U-Net、Swin-Unet、消融版和改进版保留同一测试样本的分割图导出接口，后续需要在正式训练完成后回填这些图和结果解释
- EMCAD notebook 依赖 `data/pvt_pretrained_pth/pvt_v2_b0.pth`，若后续移动位置需要同步修改配置

## 下一步建议

1. 先运行 `00_project_bootstrap_etis.ipynb` 检查 `data/ETIS/` 和预训练权重路径
2. 运行 `01_emcad_full_training.ipynb`，确认 EMCAD ETIS 主流程可执行
3. 再完成 `02` 到 `04` 的正式实验，并把结果写回文档

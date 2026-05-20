# 共选题提交说明

本目录包含共选题 5 个测试集的预测结果文件：

- `MDD.txt`
- `SLEEP.txt`
- `SEED.txt`
- `CHINESE.txt`
- `BCIC2A.txt`

## 格式说明

- 每个 `.txt` 文件均为纯文本。
- 每行一个整数类别标签。
- 无表头、无文件名字段、无额外注释。
- 预测顺序与测试集 `DataLoader(shuffle=False)` 读取顺序一致。

## 结果文件行数校验

- `MDD.txt`: 800 行
- `SLEEP.txt`: 1945 行
- `SEED.txt`: 450 行
- `CHINESE.txt`: 200 行
- `BCIC2A.txt`: 360 行

## 生成方式（复现实验）

使用 `course_project/run_course_project.py` 生成，命令如下：

```powershell
py -3.14 run_course_project.py --root-dir . --output-dir submission --model mlp --epochs 3 --batch-size 128 --lr 1e-3 --datasets MDD SLEEP SEED CHINESE BCIC2A
```

生成后的 5 个 `.txt` 文件复制到本目录用于提交。

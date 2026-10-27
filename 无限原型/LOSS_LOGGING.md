# 完整 loss 日志与离线曲线

工程版本 `Share-v2-loss` / schema `loss-log-v2`，任务 TASK-20261027-010。记录从这版源码的后续运行生效；不回填旧实验缺失数据，不修改冻结 EXP001—007 或运行中的进程。本模块可作为后续已授权新实验的观察模板。

## 记录与统计口径

每折 `<work>/runs/fold_NN/loss_logs/segments/<segment>/` 包含增量 `batches.jsonl/csv`、`epochs.jsonl/csv`、来源 `segment.json`。每次真实 optimizer step 都记录，无抽样。原稀疏 `progress.json` 继续用于看进度，完整图不从它重建。

| 字段/曲线 | 精确口径 |
| --- | --- |
| train CE | 已有 train 前向的 mean CE，原张量 detach |
| raw HHSW | 当批各唯一域的 HHSW **求和**；每域内部 1000 投影取均值，不再除域数 |
| weighted HHSW | 实际参与目标的 `.01 * raw HHSW` 原张量 detach |
| train total | 实际反向目标 `CE + weighted HHSW`，不由日志重构 |
| epoch 主值 | 所有 optimizer steps 等权均值，总量和分量同口径；组合恒等式允许浮点舍入误差 |
| CE 辅值 | `ce_sample_weighted` 按 batch 样本数加权，单列，不混入主 total |
| train_eval / sourceval CE、BA | 复用原 eval 结果；CE 为原 test() 的批均值，BA 为拼接标签口径 |
| eval HHSW / total | 原流程未计算，JSON 为 null / not_computed，CSV 留空；绝不记 0 |

epoch HHSW 是训练批次距离的均值，**不是**将整轮特征合成点云后重新计算的距离。记录还包括 0-based epoch、global_step、sample_count、缓存整数 ID、域 ID/域数、各组实际更新前 LR/参数张量数/可训练张量数。整数 ID 指向冻结缓存 `ids.json`，来源 hash 位于 segment binding。空组明确记 0；非空不表示每个参数都有梯度，不把两组解释为主干/分类头。

观察仅对已经计算的张量 detach，不增加前向、HHSW 参考分布/投影采样、梯度计算或 RNG 调用。乘法→加法→反向→step→zero_grad 顺序保持。日志增加 IO 和 CPU/GPU 标量同步开销，不承诺速度不变。

## 增量写入与恢复去重

每条 JSONL/CSV 写后 flush；epoch 边界 fsync。**JSONL 为权威源**，崩溃在两种格式之间时 CSV 可能落后，离线导出从 JSONL 重建。写失败明确中止运行，不能静默丢记录继续。CSV 的列表/对象使用 JSON 字符串，保留完整 ID 和组信息。

完整 epoch 的 `last.pt` 存入 `loss_logging_cursor`；保存完成后才更新小型 `loss_logs/committed.json`。每次恢复新开 segment，只继承 checkpoint 指定的祖先范围，重放 step 只计新分支一次。旧尾段/旧分支原样保留，`selection.json` 列明排除来源，不删除旧日志来假装未发生。

无换行的 JSONL 末行视为未完成片段，保留原字节并报告偏移；带换行或中间损坏行明确报错。缺失/重复已提交 step、首轮未提交、缺游标均不补造数据。v1 checkpoint 的 manifest/身份与 v2 不同，不能直接跨版本恢复。科学恢复仍只在完整 epoch 边界。

如果进程刚保存 `last.pt`、尚未更新提交标记就退出，离线视图会暂少最后一轮；标准 `--resume` 读取 checkpoint 游标后纠正。离线绘图不读取 checkpoint/模型。这一极短窗口明确保留，不能把标记落后解释成训练未执行。

## 运行方式

从 `Share/无限原型` 执行：

```bash
# 仅画图无需训练环境和 torch
python -m pip install -r requirements-plot.txt
python share.py plot-loss --run-dir /results/D1/runs/fold_00 --output /results/plots/D1_F0
# 无模型的工程行为测试
python -B tests/test_loss_logging.py
```

输出 `loss_epoch.png/pdf`、`loss_step.png/pdf`。每张四面板依次是 train total/CE/weighted HHSW、单独尺度 raw HHSW、原 eval CE、非空组实际 LR。epoch 图主值按 step 等权；step 图完整显示每步。epoch 横轴显示为 1-based，原日志保持 0-based。英文图例避免依赖额外中文字体。

同时导出 `batches.csv/jsonl`、`epochs.csv/jsonl`、`selection.json`、`plot_status.json`。所有原值保留，缺字段报告 missing，曲线不补零。输出应选择独立目录；可以覆盖同一派生目录重新画图，原 segments 不改。

正常 `train` 完成训练、REFIT 和冻结预测后，会启动 **CUDA_VISIBLE_DEVICES 为空的独立 CPU 子进程**，图写到该折 `loss_plots/`。父进程仅使用标准库，不导入 matplotlib 或统计库，不扰动科学进程 RNG；调度器随后继续独立评分。

绘图异常、依赖缺失或 300 秒超时单独写 `automatic_plot_status.json` / `plot_process.log`，不修改成功 train/score 的退出结果。状态文件也无法写入时 stderr 明确报告。手动 `plot-loss` 失败返回非零；缺 matplotlib 时 CSV/JSONL 已导出，错误说明安装方法。绘图成功不是模型验证或分类完成证明。

## 验证范围与版本来源

`tests/test_loss_logging.py` 抽取 v1/v2 的实际 D1/D2 train_epoch 方法，使用相同 stub 前向/HHSW 和解析梯度，对比目标分量、梯度、更新、调用次数及 Python RNG；另验变 batch 聚合、恢复/回滚分支/重放/半行、损坏拒绝、写失败、空 LR 组、eval 缺项和 CPU 子进程隔离。它不导入 torch，不代表真实 CUDA/完整模型回归。

本次实际渲染使用合成 24 epoch / 96 step 日志，检查缺依赖失败、缺字段注记以及真实自动 CPU 子进程。合成图明确标注 NOT_EXPERIMENT；没有读取 EEG 数据或模型。`STATIC_CHECKS.json` 记录实际范围，v1 原记录和原引擎文本保留在 `provenance/loss-v1/`。v2 引擎只增加观察/游标，不再声称 engine.py 与 v1 字节相同；model.py、author/、hbnn_author/、bmlr_head.py、科学 config 继续按 SHA 对照。

# D1 / D2 完整源码与原百轮启动入口

**当前工程版本：Share-v2-loss（TASK-20261027-010）。** 新增每个优化步的完整 loss、恢复去重与独立 CPU 曲线。见[完整 loss 日志与绘图说明](LOSS_LOGGING.md)。v1 和既有服务器实验未被热补，旧运行未保存的逐批数据不能还原。

本包只交付原 D1（EXP-20261027-001，Lorentz）和原 D2（EXP-20261027-002，等价 Poincaré）的 **max_epochs=100、lr=0.001** 版本。保留原早停，因此“百轮版”表示最多 100 轮，不保证跑满 100 轮。没有 200 轮、十倍学习率或已取消实验入口。

两者使用 HEEGNet 主干和官方 HBNN BMLR 分类头，D2 是原 Lorentz 运算的等价球坐标实现。它不是后续其他原生球方案，也不是用训练好的 D1 权重转换得到的模型。

**交付状态：** v1 静态检查保留在 `provenance/loss-v1/`；v2 另做实际训练方法的 stub 目标/解析梯度对照、日志恢复/聚合/失败检查及合成数据 PNG/PDF 渲染。未运行真实 PyTorch 模型、EA、REFIT 或新训练；stub 对照不代表真实模型数值回归。数据和权重不包含在包中。搬走整个 `无限原型` 目录即可；以下命令在该目录执行，不依赖原项目 tools 或服务器固定目录。

## 文件入口

```text
Share/无限原型/
  share.py                       统一跨平台入口；--help 不导入 torch
  scripts/run_D1_100.py           D1 完整百轮流水线（默认十折）
  scripts/run_D2_100.py           D2 完整百轮流水线（默认十折）
  hyperparameters/D1.json         冻结配置 + 实际生效参数 + 源码依据
  hyperparameters/D2.json
  超参数说明.md                   参数、张量形状、早停与取值说明
  requirements.txt               实际 D1/D2 链条依赖
  data_builder.py                 原数据 -> 原 EA -> 固定十折缓存
  data_protocol.json
  stage.py / portable_runtime.py  可移植 IO、运行身份、独立阶段
  delivery.py                    SHA、依赖清单、独立评分结果汇总
  loss_logging.py / loss_plot.py  完整增量记录、恢复去重、离线图与 CPU 子进程
  requirements-plot.txt           仅离线绘图依赖，无需 torch
  LOSS_LOGGING.md                 记录口径、恢复规则、使用命令与限制
  tests/test_loss_logging.py      无模型的行为测试
  variants/D1/                   原 D1 模型、完整训练/恢复/REFIT/评分/几何导出
  variants/D2/                   原 D2 + ball.py/ball_bn.py/ball_hhsw.py
    author/                      完整 HEEGNet 作者快照，105 文件
    hbnn_author/                 完整 HBNN 源码/附带文档，不含 .git 和编译缓存
    config.json                  原配置逐字节保留
    execution_compat.py          已验证的 ScriptFunction deepcopy / NumPy 比较桥
    trainer_metric_compat.py     sklearn 标量返回类型桥
    manifest.json
  provenance/                    原清单/原路径证据、版本映射、变更 diff、环境锁
  STATIC_CHECKS.json              本次实际轻量检查结果及限制
  MANIFEST.json                  本交付所有固定文件 SHA256
```

`author/demo.py` 和上游示例仅供出处核查。请从 `share.py` 或两个启动脚本运行 D1/D2；上游其他数据集示例可能需要额外依赖，不是本包的实验入口。

## 安装与核对

推荐独立 **Python 3.10** 环境。原数值环境是 Linux、PyTorch 2.5.1/CUDA 12.4、NumPy 1.26.4、SciPy 1.12.0、scikit-learn 1.7.2、geoopt 0.5.1。原完整环境锁见 `provenance/environment.original.lock.txt`；其中有与本实验无关及平台专用的依赖，不能整份直接当 Windows 安装清单。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python share.py verify
python share.py dependencies
python share.py params --variant D1
python share.py --help
```

CPU 验收可用对应 CPU torch wheel；正式训练入口要求 CUDA。驱动必须支持所选 wheel。Windows 的路径、锁与 CLI 已作轻量检查，但没有在 Windows 上重做这套模型的数值验证。不同平台或 GPU 的逐位结果不作保证。不要使用 `python -O`，原科学断言需要保持生效。

## 输入数据与 EA 缓存

原始目录须包含 `sub000.pkl` 到 `sub122.pkl`，每个是可信来源的 NumPy `float64` 数组，形状 **[28,32,7500]**，分别为 trial、电极、时间采样点。不得提前把数组改为窗口样本或 float32。标签由 28 个视频的固定顺序生成：9 类的连续计数 `[3,3,3,3,4,3,3,3,3]`；视频顺序必须与该协议一致。

这里的“原数据”是项目已有整 trial pickle 输入，不是自动从原始设备文件开始的完整滤波流程。作者 250 Hz 对应 30 秒；现有文件没有采样率/单位/滤波元数据，本包不补猜或重新滤波，原始预处理与作者数据是否完全一致仍需数据提供者确认。

```bash
python share.py prepare --raw /data/FACED_trials --cache /data/FACED_EA_cache
```

PowerShell 将路径换成实际本地路径，例如 `--raw "D:\data\FACED_trials" --cache "D:\data\FACED_EA_cache"`；其余命令相同。

构建顺序：按作者字符串顺序排列 `subject_0, subject_1, subject_10, ...` → 每被试 28 trial 的协方差均值 → 作者 `inv(sqrtm(mean_cov)) @ X` → 直接赋值到 float64 → `GroupKFold(10)` 按被试分外折 → 每外折源集合用标签分层 `StratifiedShuffleSplit(test_size=.2)`。内划分使用同一个 `RandomState(42)` 连续生成十折，**不是每折重新 seed42，也不是按被试×标签联合分层**。

EA 使用整被试/会话数据，包含目标被试的无标签 trial；它在划分前进行，属于原冻结协议。本包保留作者可能产生复数再直接写入 float64 的操作，并逐被试记录虚部及赋值位检查，不新增协方差正则或删样本。

缓存包含 `X.npy [3444,32,7500]`、`ids.json`、`splits.json`、`prepared.json`、逐被试 EA 记录。X 约 **6.16 GiB**，另需原始数据和训练检查点空间。构建只接受新的空缓存目录，失败半成品保留；重新构建请选新目录。已有兼容缓存可直接传给后续命令，入口检查真实 SHA、行身份、标签、完整十折及重建的小型划分。两模型共用一套只读缓存。

## 单折完整运行

以下命令需由使用者在准备运行时执行；交付过程没有执行这些训练命令。物理 GPU 只接受 0 或 3，程序用本机 `nvidia-smi` 解析 UUID，再将目标卡作为唯一可见 `cuda:0`。不使用原服务器的固定 UUID。

```bash
python share.py run --variant D1 --mode preflight --cache /data/FACED_EA_cache --work /results/D1
python share.py run --variant D1 --mode train --fold 0 --gpu 0 --cache /data/FACED_EA_cache --work /results/D1
python share.py run --variant D1 --mode score --fold 0 --cache /data/FACED_EA_cache --work /results/D1
```

D2 只需换成 `--variant D2 --work /results/D2`。`train` 会完成从头训练 → 源验证 CE 选中状态 → 每个目标被试整批 28 trial 的无标签 REFIT → 冻结预测与几何。`score` 是独立 CPU 进程，之后才用目标标签计算 BA、Accuracy、真 macro-F1、Kappa 与 9×9 混淆矩阵。

两个超参数启动脚本会先打印完整生效参数，再依次完成训练和独立评分。单折例子：

```bash
python scripts/run_D1_100.py --cache /data/FACED_EA_cache --output-root /results/fresh100 --folds 0 --gpus 0
python scripts/run_D2_100.py --cache /data/FACED_EA_cache --output-root /results/fresh100 --folds 1 --gpus 3
```

省略 `--folds` 即运行 0–9 十折。`--gpus 0` 为单卡串行，`--gpus 0 3` 为每卡至多一条完整流水线。没有超参数覆盖开关；修改 JSON 也不能绕过封包校验或改变 `engine.py` 中实际冻结的取值。

## 两模型十折、日志与汇总

```bash
# 先查看将执行的 20 条流水线及 40 条阶段命令，不创建进程或目录
python share.py schedule --cache /data/FACED_EA_cache --output-root /results/fresh100 --dry-run
# 准备好 GPU 后实际执行；默认 D1/D2、0..9 折、物理卡 0/3
python share.py schedule --cache /data/FACED_EA_cache --output-root /results/fresh100
python share.py aggregate --work /results/fresh100/D1 --output /results/fresh100/D1_summary.json
python share.py aggregate --work /results/fresh100/D2 --output /results/fresh100/D2_summary.json
```

调度器默认每卡一条完整流水线，每折训练/REFIT完成后启动独立 CPU 评分，成功后才接该卡下一折。日志及真实返回码在 `<output-root>/<variant>/logs/`；某卡失败会停止该卡后继，另一卡已分配工作继续。共同输出根的 GPU 锁防止本调度器重复占用；它不管理其他目录或外部任务，也不做服务器调度、SSH、进程终止或自动释放显存。需要使用者先安排可用 GPU；显存不足会保留错误，不静默减 batch、分块验证或改精度。

汇总检查预测 SHA、选中状态 SHA、混淆矩阵指标与跨折身份，默认要求十折齐全、123 个互斥目标域/3444 trial。主口径为 **123 域等权均值和样本 SD（ddof=1）**；十折等权和合并混淆矩阵另列，指标为 0–1 小数。若明确传 `--folds 0 1` 则标为部分折结果，不冒充十折。汇总不调用模型、不重新 REFIT。

## 恢复、验证与几何导出

同一 Share 包、同一缓存、同一折/attempt/GPU UUID 可从完整 `last.pt` 恢复：

```bash
python share.py run --variant D1 --mode train --fold 0 --gpu 0 --resume --cache /data/FACED_EA_cache --work /results/D1
```

保留模型/优化器全状态、Python/NumPy/Torch/CUDA RNG、BN 额外统计/别名关系、早停 best/counter、BN 调度器、epoch/step、采样 ID；恢复边界为已保存的完整 epoch。中途失败从上个完整边界继续，首轮尚无 `last.pt` 时需新输出目录重新开始。已冻结预测只需另跑 `score`；已评分结果禁止覆盖。移动整个输出目录后仍可评分，预测位置使用相对文件名。旧服务器包的 checkpoint 不能直接冒充 Share 同一身份恢复；两种 manifest 的区别会被明确拒绝。

原科学验收入口也完整保留，使用独立 `--work`，避免与正式输出混用：

```bash
python share.py run --variant D1 --mode algebra --cache /data/FACED_EA_cache --work /results/check_D1
python share.py run --variant D1 --mode smoke --cache /data/FACED_EA_cache --work /results/check_D1
python share.py run --variant D1 --mode evaluation-score --check-mode smoke --cache /data/FACED_EA_cache --work /results/check_D1
python share.py run --variant D1 --mode capacity --gpu 0 --cache /data/FACED_EA_cache --work /results/check_D1
python share.py run --variant D1 --mode evaluation-score --check-mode capacity --cache /data/FACED_EA_cache --work /results/check_D1
```

这些会真实创建模型及执行数值更新，**本次仅交付入口，未运行**。原 CUDA 续步检查曾出现约 1e-13 的 BN 参数重算差异，原失败/人工分项接受不等于新平台全部自动通过；本包不放宽容差或伪造通过回执。

导出指定 selected 的实际分类前几何（默认全部缓存 ID，可用 JSON 整数索引数组缩小范围）：

```bash
python share.py run --variant D2 --mode export --fold 0 --cache /data/FACED_EA_cache --work /results/D2 --checkpoint /results/D2/runs/fold_00/selected.pt --output /results/D2/export.pt --indices-json /data/export_indices.json
```

`export` 恢复已存状态，不新增 REFIT，目标适应后的状态另存于 `domain_*_adapted.pt`。训练目录还有 `initial.pt`、`first_update.pt`、每 epoch/last/selected、原生几何 JSONL 与 ID、目标适应前后状态、冻结 logits/预测/几何等。恢复/导出只读自己信任的 pickle/PT 文件。

## 来源与改动边界

HEEGNet 作者 commit `5ad0df23f802db8402b734ff2cd4f2a4ab4c93b9`；HBNN 官方 commit `d5c79c8eed36a0b7c2f15e9a8fbcd0318216e5b0`。两个 `author/` 的 105 文件、完整 HBNN 源文件、D1/D2 数学模型/头/损失及两份原配置逐字节对照。v2 的 `engine.py` 新增观察/游标，不再声称引擎字节未改；目标运算顺序和原选模、BN 调度保持。原引擎文本、v1 检查、版本映射及 diff 在 `provenance/loss-v1/` 和 `provenance/loss_logging_v2.json`。

新增的是本地路径和进程入口、可跨平台锁、缓存真实身份验证、相对预测路径、CPU 汇总与启动脚本；兼容桥复制自已验证实现，只复用不可变 ScriptFunction，参数/缓冲区仍独立深复制；NumPy RNG 比较保持 dtype/shape/value 精确相等。原作者的数学表达式、头初始化、优化器算法、早停/BN公式与数值阈值未改。

HBNN 在该 commit **没有 LICENSE/COPYING 文件**，本交付不为其补造许可证；其 README 明确仅发布核心层，官方完整训练代码尚未发布。本包的完整训练流程来自项目已冻结的 HEEGNet+BMLR 实验适配，不能称作 HBNN 官方训练复现。

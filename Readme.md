#一行代码不要分开成多行（我指的一行代码不是说功能太多了你把他精简成一行，而是在本身这段代码就是精简的情况下不要将有括号的一行代码拆成三行）


# 实现代码时要注上必要的注释，要求如下
做上必要的注释：
1. 在每个def开头表上注释，标明这个函数的功能
2. 要在代码中做好分隔与注释，描述清楚每段的作用（想图中的这样）


# 代码排版
代码排版采用紧凑风格：函数参数、条件判断、简单赋值和函数调用尽量写在一行，不要仅因行长而拆成多行。不同逻辑段之间用空行和中文注释分隔，只有逻辑确实复杂时才换行。调整排版时不要改变代码逻辑。




# WorkFlows相关实现需求
明白。当前先确定 Diffusion Policy workflow 的大范围，不查看 ACT 代码，也不开始实现。

整体目标分为两条独立入口：

1. `train_diffusion_policy.py`
   - 启动一次完整训练。
   - 未指定恢复点时，按照当前时间创建新的实验目录，例如：
     `checkpoints/diffusion_policy/pick__diffusion_policy__20260930_120000/`
   - 明确指定 checkpoint 时，从该 checkpoint 恢复训练，而不是创建无关的新实验。
   - 保存周期性 checkpoint。
   - 记录并上传训练/评估指标到 W&B。
   - 每隔可配置的若干 epoch，调用一次统一的评估逻辑，统计成功率。
   - 保存本次实验的完整配置，保证实验可复现。

2. `eval_diffusion_policy.py`
   - 单独运行一次评估，不进行训练。
   - 默认自动寻找最新实验中的最新 checkpoint。
   - 命令行可以指定：
     - 某个实验目录；
     - 某个具体 checkpoint。
   - 加载模型和配置，在环境中执行评估并输出成功率等指标。
   - 训练期间的周期性评估也复用这里的评估函数，避免出现两套评估实现。

每个新实验目录预计保持图二所示的结构：

```text
checkpoints/diffusion_policy/<experiment_name__timestamp>/
├── wandb/
├── epoch_0200.ckpt
├── epoch_0400.ckpt
└── run_config.json
```

配置范围包括：

- 实验名称和 checkpoint 根目录
- 新训练或恢复训练
- 恢复 checkpoint 路径
- 评估 checkpoint 路径
- 数据集与环境参数
- Diffusion Policy 模型参数
- 优化器、学习率和 EMA 参数
- batch size、epoch 数
- checkpoint 保存间隔
- 训练期间的评估间隔
- 单次评估 episode 数
- 随机种子、设备和 W&B 参数
- 允许通过命令行覆盖关键配置

有两点概念上需要提前固定：

- 训练过程中“调用 eval”应当是复用 `eval_diffusion_policy.py` 中的评估函数，而不是重新启动一次命令行程序。
- W&B 通常上传训练损失、成功率、学习率等指标；模型权重主要保存在 checkpoint。是否额外把 checkpoint 上传为 W&B Artifact，后续再单独决定。

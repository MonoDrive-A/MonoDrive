# train/losses.py

## 1. 文件职责

`train/losses.py` 汇总 MonoDrive 训练 loss，覆盖轨迹词表分数、轨迹残差、Agent 分类/状态/mode/future 和 Map 分类/点回归。该文件不执行数据读取、Hungarian matching、优化器更新或 checkpoint 保存。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingLossOutput` | NamedTuple | 单个 batch 的总 loss 和分项 loss。 |
| `MonoDriveTrainingLoss` | class | 汇总全部训练 loss 的 `nn.Module`。 |

## 3. 关键类和函数

### `MonoDriveTrainingLoss`

- 功能：根据模型输出和训练标签计算加权总 loss。
- 输入：`MonoDriveBackboneOutput` 和 `TrainingBatchLabels`。
- 输出：`TrainingLossOutput`。
- Shape：总 loss 为标量；分项 loss 均为标量。
- 关键参数：`LossWeights`，来自 `config/training.toml`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_output.logits` | `[B, V]` | 轨迹词表 raw logits。 |
| `trajectory.soft_labels` | `[B, V]` | 和为 1 的轨迹词表 soft label。 |
| `trajectory_output.residuals` | `[B, V, 6, 2]` | Symlog 空间残差预测。 |
| `agent_class_logits` | `[B, 48, C_agent + 1]` | Agent 分类 raw logits。 |
| `agent_future_trajectories` | `[B, 48, 4, 6, 2]` | Agent future Symlog 空间预测。 |
| `agent.future_mask` | `[B, 48, 4, 6]` | winner mode 和有效未来点 mask。 |
| `map_points` | `[B, 48, 100, 2]` | Map 点 Symlog 空间预测。 |

## 5. 关键实现逻辑

轨迹词表分数使用 soft cross entropy，输入是模型 raw logits，目标是和为 1 的 soft label。实现对模型输出执行 `log_softmax`，再与 soft label 做逐类加权求和。

Agent / Map 分类和 Agent mode 使用 PyTorch `cross_entropy`。该函数内部包含 `log_softmax`，调用方必须传入 raw logits。Agent mode 只在存在有效 future 点的匹配 query 上监督。

连续回归项使用 mask MSE。Agent future 的 mask 是 `[B, Q, M, K]`，只监督匹配 query 的 winner mode 和有效未来点；Map 点只监督匹配 query。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `loss_weights.trajectory_logit_soft_ce` | 见 `config/training.toml` | 轨迹词表分数 soft CE 权重。 |
| `loss_weights.trajectory_residual_mse` | 见 `config/training.toml` | winner 轨迹残差 MSE 权重。 |
| `loss_weights.agent_*` | 见 `config/training.toml` | Agent 分类、状态、mode、future 权重。 |
| `loss_weights.map_*` | 见 `config/training.toml` | Map 分类和点回归权重。 |

## 7. 依赖关系

- 上游：`model/backbone.py`、`train/data_processing.py`、`train/training_config.py`。
- 下游：`train/trainer.py`。
- 第三方：`torch`。

## 8. 注意事项

- 所有 loss 计算都转为 FP32。
- 轨迹词表概率分数使用 soft CE，模型输出在 loss 内部做 `log_softmax`。
- 修改 mask shape 或监督空间时，需要同步本文件和 `train/data_processing.py` 文档。
- Agent mode CE 依赖 `future_mask.any(dim=(2, 3))`，没有有效 future 点的匹配 Agent 不参与 mode 监督。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：轨迹词表分数 loss 从 BCEWithLogits 改为 soft CE。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练 loss 汇总模块。 |

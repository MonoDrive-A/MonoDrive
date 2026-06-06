# model/swiglu.py

## 1. 文件职责

`model/swiglu.py` 提供通用 SwiGLU 激活函数和 `nn.Module` 封装。该文件只负责把输入特征沿指定维度拆分为 value 与 gate，并计算 `value * SiLU(gate)`。

该文件不负责 FFN 线性层、隐藏维度配置、模型结构选择或任何配置文件加载。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `swiglu` | function | 对任意浮点张量沿指定维度执行 SwiGLU 激活。 |
| `SwiGLU` | class | `nn.Module` 形式的 SwiGLU 激活层。 |

## 3. 关键类和函数

### `swiglu`

- 功能：沿 `dim` 将输入特征二等分为 value 和 gate，并返回 `value * SiLU(gate)`。
- 输入：`features` 和拆分维度 `dim`。
- 输出：指定维度长度减半后的张量。
- Shape：`[..., 2 * C, ...] -> [..., C, ...]`。
- 关键参数：`dim` 仅决定拆分维度，不代表模型配置项。

### `SwiGLU`

- 功能：把 `swiglu` 包装为可插入 `nn.Module` 网络结构的激活层。
- 输入：构造参数 `dim`，前向输入 `features`。
- 输出：指定维度长度减半后的张量。
- Shape：`[..., 2 * C, ...] -> [..., C, ...]`。
- 关键参数：`dim` 必须为整数。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `features` | `[..., 2 * C, ...]` | 指定维度必须能二等分。 |
| `value_features` | `[..., C, ...]` | 指定维度前半部分。 |
| `gate_features` | `[..., C, ...]` | 指定维度后半部分，经 `SiLU` 后作为门控。 |
| 输出 | `[..., C, ...]` | 与 `value_features` shape 一致。 |

## 5. 关键实现逻辑

`swiglu` 先校验 `dim` 是整数、输入至少为 1 维、拆分维度未越界且长度可以二等分。随后使用 `torch.chunk` 沿指定维度拆分为两半，返回 value 与 `F.silu(gate)` 的逐元素乘积。

`SwiGLU` 只保存拆分维度，并在 `forward` 中调用函数式 `swiglu`。这样轨迹词表 FFN 或后续 Transformer FFN 可以复用同一个激活实现。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 无 | 无 | 该文件不读取配置文件，也不保存模型隐藏维度默认值。 |

## 7. 依赖关系

- 上游：需要 SwiGLU 激活的 FFN 或嵌入模块。
- 下游：`model/trajectory_vocab/trajectory_vocab.py` 中的轨迹词表嵌入层。
- 第三方依赖：`torch`。
- 标准库依赖：无。

## 8. 注意事项

- 数值稳定性：SwiGLU 本身不包含除法、归一化或 softmax，不需要 epsilon。
- Shape：指定维度长度必须为偶数，否则无法拆分 value 和 gate。
- 配置：FFN 的中间维度应由调用方或对应配置管理，不应写入本文件。
- 兼容性：`dim` 支持负数索引，会按 PyTorch 维度规则解析。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-06 | 1os3_Codex | AI 完成：新增通用 SwiGLU 激活函数和 `nn.Module` 封装。 |

# model/swiglu.py 摘要

## 1. 文件基本功能

提供通用 SwiGLU 激活函数和 `nn.Module` 封装。输入沿指定维度拆分为 value 与 gate，输出 `value * SiLU(gate)`，不负责 FFN 线性层或隐藏维度配置。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `swiglu` | function | 沿指定维度执行函数式 SwiGLU。 |
| `SwiGLU` | class | 可插入 `nn.Module` 网络结构的 SwiGLU 激活层。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `features` | `[..., 2 * C, ...]` | 指定维度必须能二等分。 |
| 输出 | `[..., C, ...]` | 指定维度长度变为输入的一半。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `swiglu(features, dim=-1)` | `dim` 必须为整数，且 `features.shape[dim]` 必须为偶数。 |
| `SwiGLU(dim=-1)` | 构造后在 `forward` 中复用 `swiglu`，适合作为 FFN 激活层。 |

## 5. 最小使用示例

在项目根目录执行：

```python
import torch

from model.swiglu import SwiGLU

activation = SwiGLU()
features = torch.randn(2, 4, 768)
output = activation(features)
print(output.shape)  # torch.Size([2, 4, 384])
```

## 6. 维护注意事项

- 不要在该文件加入模型隐藏维度、FFN 扩展比例或配置读取逻辑。
- 修改拆分规则时，需要同步检查 `model/trajectory_vocab/trajectory_vocab.py` 的嵌入层。
- 指定维度长度不是偶数时必须显式报错，避免静默截断。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-06 | 1os3_Codex | AI 完成：新增通用 SwiGLU 摘要文档。 |

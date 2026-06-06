# model/rope_3d.py 摘要

## 1. 文件基本功能

提供通用 3D RoPE 旋转位置编码。模块只消费调用方传入的三维位置坐标、轴通道划分和 RoPE 基频，对特征张量最后一维的前若干通道执行三轴旋转。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `apply_rope_3d` | function | 对单个特征张量应用 3D RoPE。 |
| `RoPE3D` | class | 对 query 和 key 应用相同的 3D RoPE。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `features` | `[..., N, C]` | 待旋转特征。 |
| `query` / `key` | `[..., N, C]` | `RoPE3D.forward` 的输入，shape 必须一致。 |
| `positions` | `[N, 3]` 或 `[..., N, 3]` | 上游已准备好的三维坐标。 |
| 输出 | `[..., N, C]` | 与输入特征 shape 一致。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `apply_rope_3d(features, positions, axis_dims, theta)` | `axis_dims` 必须包含 3 个正偶数，且总和不能超过 `features.shape[-1]`。 |
| `RoPE3D(axis_dims, theta)` | 构造时显式传入轴通道划分和基频；前向时 `query`、`key` shape 必须一致。 |

## 5. 最小使用示例

在项目根目录执行：

```python
import torch

from model.rope_3d import RoPE3D

rope = RoPE3D(axis_dims=(16, 16, 16), theta=100.0)
query = torch.randn(2, 6, 2304, 48)
key = torch.randn(2, 6, 2304, 48)
positions = torch.randn(2304, 3)
rotated_query, rotated_key = rope(query, key, positions)
print(rotated_query.shape, rotated_key.shape)
```

## 6. 维护注意事项

- 不要在本文件加入坐标归一化、视觉网格生成、注意力头选择或 TOML 配置读取。
- 若调用方只让部分注意力头使用 RoPE，应在调用本模块前切分对应头。
- 修改旋转公式或广播规则时，需要补充 shape-sensitive smoke test。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-06 | 1os3_Codex | AI 完成：新增通用 3D RoPE 摘要文档。 |

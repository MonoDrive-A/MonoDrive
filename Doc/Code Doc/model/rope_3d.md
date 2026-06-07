# model/rope_3d.py

## 1. 文件职责

`model/rope_3d.py` 提供通用 3D RoPE 旋转位置编码实现。它只接收调用方已经准备好的三维位置坐标和 rotary 通道划分，对特征张量最后一维的前若干通道执行三轴旋转，并强制以 FP32 输出。

该文件不生成视觉网格，不做坐标中心化、归一化、`[H, W, T]` 排列转换、注意力头选择或配置文件加载。这些策略由 Transformer Block、位置构造模块或配置加载模块负责。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `apply_rope_3d` | function | 对单个特征张量应用 3D RoPE。 |
| `RoPE3D` | class | 对 query 和 key 应用同一组 3D RoPE 参数的 `nn.Module`。 |

## 3. 关键类和函数

### `apply_rope_3d`

- 功能：按三个坐标轴分别构造旋转角，对 `features[..., N, C]` 的前 `sum(axis_dims)` 个通道应用 RoPE。
- 输入：`features`、`positions`、`axis_dims`、`theta`。
- 输出：shape 与输入一致的 FP32 特征张量。
- Shape：`features` 为 `[..., N, C]`，`positions` 为 `[N, 3]` 或 `[..., N, 3]`。
- 关键参数：`axis_dims` 三项必须为正偶数，`theta` 必须为正数。

### `RoPE3D`

- 功能：保存 `axis_dims` 和 `theta`，并对 query / key 应用同一套 3D RoPE。
- 输入：`query`、`key`、`positions`。
- 输出：FP32 的 `rotated_query` 和 `rotated_key`。
- Shape：`query`、`key` 均为 `[..., N, C]`，两者 shape 必须一致。
- 关键参数：`axis_dims` 和 `theta` 由调用方显式传入。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `features` | `[..., N, C]` | 待旋转特征，最后两维为 token 和通道。 |
| `positions` | `[N, 3]` 或 `[..., N, 3]` | 已由上游准备好的三维坐标，最后一维顺序由调用方约定。 |
| `axis_dims` | `[3]` | 三个坐标轴对应的 rotary 通道数。 |
| rotary 部分 | `[..., N, sum(axis_dims)]` | 被 3D RoPE 旋转的通道。 |
| tail 部分 | `[..., N, C - sum(axis_dims)]` | 未参与 RoPE 的剩余通道，原样保留。 |
| 输出 | `[..., N, C]` | 与输入特征 shape 一致，dtype 固定为 `torch.float32`。 |

## 5. 关键实现逻辑

`apply_rope_3d` 首先校验输入为浮点张量、`features` 和 `positions` 位于同一设备、token 数一致、`axis_dims` 为 3 个正偶数且总和不超过特征通道数。随后将特征和位置统一转换为 FP32，按轴切分特征通道，并对每个轴独立执行 1D RoPE。

每个轴的频率使用如下形式：

$$
\omega_i = \theta^{-2i / d}
$$

其中 $d$ 是该轴占用的 rotary 通道数，$i$ 是通道对索引。角度由调用方传入的位置坐标与频率相乘得到：

$$
\alpha_i = p \cdot \omega_i
$$

每个通道对按标准二维旋转更新：

$$
\begin{aligned}
x'_{2i} &= x_{2i}\cos(\alpha_i) - x_{2i+1}\sin(\alpha_i), \\
x'_{2i+1} &= x_{2i}\sin(\alpha_i) + x_{2i+1}\cos(\alpha_i).
\end{aligned}
$$

若 `positions` 只有 `[N, 3]`，角度会自动扩展到 `features` 的前缀维度；若 `positions` 已带批量前缀，则该前缀必须能与 `features` 的前缀维度广播。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 无 | 无 | 该文件不读取 TOML，也不写入模型层数、注意力头数、坐标归一化范围或 RoPE 基频默认值。 |

## 7. 依赖关系

- 上游：后续 Transformer 注意力模块、位置坐标构造模块或配置加载模块。
- 下游：自注意力中的 query / key 旋转。
- 第三方依赖：`torch`。
- 标准库依赖：`collections.abc`。

## 8. 注意事项

- 坐标口径：调用方必须在进入本文件前完成坐标构造和归一化；本文件不假设坐标范围。
- 轴顺序：`positions[..., 0:3]` 的含义由调用方约定；视觉 Token 可按 `[H, W, T]` 传入。
- 头选择：若只希望部分注意力头使用 RoPE，应由调用方先切片或只对指定头调用本模块。
- 精度：特征、位置、频率和三角函数均强制使用 FP32；输出不会转换回输入 dtype。
- Shape：`sum(axis_dims)` 可以小于 `C`，未覆盖的通道会原样保留。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：将 3D RoPE 旋转和输出强制为 FP32。 |
| 2026-06-06 | 1os3_Codex | AI 完成：新增通用 3D RoPE 旋转位置编码实现文档。 |

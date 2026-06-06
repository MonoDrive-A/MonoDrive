"""通用 3D RoPE 旋转位置编码。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


__all__ = ["RoPE3D", "apply_rope_3d"]


def apply_rope_3d(
    features: torch.Tensor,
    positions: torch.Tensor,
    axis_dims: Sequence[int],
    theta: float,
) -> torch.Tensor:
    """对特征张量应用 3D RoPE。

    本函数只消费调用方传入的三维位置坐标，不生成网格坐标，也不做中心化、
    归一化或头选择。位置坐标最后一维的 3 个值按调用方约定解释，例如视觉
    Token 可传入 `[H, W, T]`。

    Args:
        features: 待旋转特征，最后两维为 token 和通道。
        positions: 已由调用方准备好的三维位置坐标。
        axis_dims: 三个坐标轴各自占用的 rotary 通道数，每项必须为正偶数。
        theta: RoPE 基频。

    Returns:
        应用 3D RoPE 后的特征，shape 与 `features` 相同。

    Shape:
        `features`: `[..., N, C]`
        `positions`: `[N, 3]` 或 `[..., N, 3]`
        输出: `[..., N, C]`

    Raises:
        TypeError: `theta` 不是数值。
        ValueError: 输入 shape、轴通道或基频不满足 3D RoPE 要求。
    """

    if not torch.is_floating_point(features):
        raise TypeError(f"features 必须为浮点张量，实际 dtype 为 {features.dtype}。")
    if not torch.is_floating_point(positions):
        raise TypeError(f"positions 必须为浮点张量，实际 dtype 为 {positions.dtype}。")
    if features.ndim < 2:
        raise ValueError(
            f"features 期望 shape 为 [..., N, C]，实际 shape 为 {tuple(features.shape)}。"
        )
    if positions.ndim < 2 or int(positions.shape[-1]) != 3:
        raise ValueError(
            f"positions 期望 shape 为 [N, 3] 或 [..., N, 3]，实际为 {tuple(positions.shape)}。"
        )
    if int(features.shape[-2]) != int(positions.shape[-2]):
        raise ValueError(
            "features 和 positions 的 token 数必须一致，"
            f"实际分别为 {features.shape[-2]} 和 {positions.shape[-2]}。"
        )

    validated_axis_dims = _validate_axis_dims(axis_dims)
    rotary_dim = sum(validated_axis_dims)
    feature_dim = int(features.shape[-1])
    if rotary_dim > feature_dim:
        raise ValueError(
            f"axis_dims 总和不能超过 features 最后一维，实际为 {rotary_dim} > {feature_dim}。"
        )
    _validate_theta(theta)

    rotated_parts = []
    cursor = 0
    for axis_index, axis_dim in enumerate(validated_axis_dims):
        next_cursor = cursor + axis_dim
        axis_features = features[..., cursor:next_cursor]
        axis_positions = positions[..., axis_index]
        rotated_parts.append(_apply_1d_rope(axis_features, axis_positions, axis_dim, theta))
        cursor = next_cursor

    if cursor < feature_dim:
        rotated_parts.append(features[..., cursor:])
    return torch.cat(rotated_parts, dim=-1)


class RoPE3D(nn.Module):
    """通用 3D RoPE 模块。

    Args:
        axis_dims: 三个坐标轴各自占用的 rotary 通道数，每项必须为正偶数。
        theta: RoPE 基频。

    Shape:
        `query`: `[..., N, C]`
        `key`: `[..., N, C]`
        `positions`: `[N, 3]` 或 `[..., N, 3]`
        输出: 两个 shape 与输入一致的张量。
    """

    def __init__(self, axis_dims: Sequence[int], theta: float) -> None:
        super().__init__()
        self.axis_dims = _validate_axis_dims(axis_dims)
        _validate_theta(theta)
        self.theta = float(theta)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 query 和 key 应用相同的 3D RoPE。"""

        if query.shape != key.shape:
            raise ValueError(
                f"query 和 key 的 shape 必须一致，实际为 {tuple(query.shape)} 和 {tuple(key.shape)}。"
            )

        rotated_query = apply_rope_3d(query, positions, self.axis_dims, self.theta)
        rotated_key = apply_rope_3d(key, positions, self.axis_dims, self.theta)
        return rotated_query, rotated_key


def _apply_1d_rope(
    axis_features: torch.Tensor,
    axis_positions: torch.Tensor,
    axis_dim: int,
    theta: float,
) -> torch.Tensor:
    pair_count = axis_dim // 2
    frequency_indices = torch.arange(
        pair_count,
        device=axis_positions.device,
        dtype=torch.float32,
    )
    inv_frequencies = torch.pow(
        torch.tensor(float(theta), device=axis_positions.device, dtype=torch.float32),
        -2.0 * frequency_indices / float(axis_dim),
    )
    angles = axis_positions.to(dtype=torch.float32)[..., None] * inv_frequencies
    angles = _align_angles_to_features(angles, axis_features)
    cos_angles = torch.cos(angles).to(dtype=axis_features.dtype)
    sin_angles = torch.sin(angles).to(dtype=axis_features.dtype)

    even_features = axis_features[..., 0::2]
    odd_features = axis_features[..., 1::2]
    rotated_even = even_features * cos_angles - odd_features * sin_angles
    rotated_odd = even_features * sin_angles + odd_features * cos_angles
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _align_angles_to_features(angles: torch.Tensor, axis_features: torch.Tensor) -> torch.Tensor:
    feature_prefix_ndim = axis_features.ndim - 2
    angle_prefix_ndim = angles.ndim - 2
    missing_prefix_ndim = feature_prefix_ndim - angle_prefix_ndim
    if missing_prefix_ndim < 0:
        raise ValueError(
            "positions 的前缀维度不能多于 features 的前缀维度，"
            f"实际分别为 {angle_prefix_ndim} 和 {feature_prefix_ndim}。"
        )
    if missing_prefix_ndim == 0:
        return angles

    prefix_shape = tuple(angles.shape[:-2])
    feature_prefix_shape = tuple(axis_features.shape[:-2])
    token_and_pair_shape = tuple(angles.shape[-2:])
    batch_aligned_prefix = (*prefix_shape, *((1,) * missing_prefix_ndim))
    if _is_broadcastable(batch_aligned_prefix, feature_prefix_shape):
        return angles.reshape(*batch_aligned_prefix, *token_and_pair_shape)

    trailing_aligned_prefix = (*((1,) * missing_prefix_ndim), *prefix_shape)
    if _is_broadcastable(trailing_aligned_prefix, feature_prefix_shape):
        return angles.reshape(*trailing_aligned_prefix, *token_and_pair_shape)

    raise ValueError(
        "positions 的前缀维度无法与 features 的前缀维度广播，"
        f"实际分别为 {prefix_shape} 和 {feature_prefix_shape}。"
    )


def _validate_axis_dims(axis_dims: Sequence[int]) -> tuple[int, int, int]:
    try:
        raw_axis_dims = tuple(axis_dims)
    except TypeError as exc:
        raise TypeError(f"axis_dims 必须是包含 3 个整数的序列，实际为 {axis_dims!r}。") from exc
    if len(raw_axis_dims) != 3:
        raise ValueError(f"axis_dims 必须包含 3 个整数，实际为 {raw_axis_dims}。")

    validated_dims = []
    for axis_index, raw_axis_dim in enumerate(raw_axis_dims):
        if not isinstance(raw_axis_dim, int) or isinstance(raw_axis_dim, bool):
            raise TypeError(
                f"axis_dims[{axis_index}] 必须为整数，实际为 {raw_axis_dim!r}。"
            )
        axis_dim = int(raw_axis_dim)
        if axis_dim <= 0:
            raise ValueError(
                f"axis_dims[{axis_index}] 必须为正偶数，实际为 {axis_dim}。"
            )
        if axis_dim % 2 != 0:
            raise ValueError(
                f"axis_dims[{axis_index}] 必须为偶数，实际为 {axis_dim}。"
            )
        validated_dims.append(axis_dim)
    return tuple(validated_dims)


def _is_broadcastable(source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> bool:
    return all(
        source_dim == 1 or source_dim == target_dim
        for source_dim, target_dim in zip(source_shape, target_shape)
    )


def _validate_theta(theta: float) -> None:
    if not isinstance(theta, (int, float)) or isinstance(theta, bool):
        raise TypeError(f"theta 必须为数值，实际为 {type(theta).__name__}。")
    if float(theta) <= 0.0:
        raise ValueError(f"theta 必须为正数，实际为 {theta}。")

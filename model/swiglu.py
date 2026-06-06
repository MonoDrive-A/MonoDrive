"""通用 SwiGLU 激活模块。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SwiGLU", "swiglu"]


def swiglu(features: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿指定维度执行 SwiGLU 激活。

    Args:
        features: 待激活特征，指定维度会被二等分为 value 和 gate。
        dim: 拆分维度。

    Returns:
        激活后的特征，指定维度长度为输入的一半。

    Shape:
        输入: `[..., 2 * C, ...]`
        输出: `[..., C, ...]`

    Raises:
        TypeError: `dim` 不是整数。
        ValueError: `features` 为空维张量，或指定维度越界 / 不能二等分。
    """

    if not isinstance(dim, int):
        raise TypeError(f"dim 必须为整数，实际为 {type(dim).__name__}。")
    if features.ndim == 0:
        raise ValueError("features 必须至少包含 1 个维度，实际为 0 维张量。")

    normalized_dim = dim + features.ndim if dim < 0 else dim
    if normalized_dim < 0 or normalized_dim >= features.ndim:
        raise ValueError(
            f"dim 必须位于 [{-features.ndim}, {features.ndim - 1}]，实际为 {dim}。"
        )

    split_size = int(features.shape[normalized_dim])
    if split_size % 2 != 0:
        raise ValueError(
            "features 在指定维度上的长度必须能二等分，"
            f"dim={dim} 的实际长度为 {split_size}。"
        )

    value_features, gate_features = features.chunk(2, dim=normalized_dim)
    return value_features * F.silu(gate_features)


class SwiGLU(nn.Module):
    """SwiGLU 激活层。

    Args:
        dim: 拆分 value 和 gate 的维度。

    Shape:
        输入: `[..., 2 * C, ...]`
        输出: `[..., C, ...]`
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        if not isinstance(dim, int):
            raise TypeError(f"dim 必须为整数，实际为 {type(dim).__name__}。")
        self.dim = dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """沿配置维度执行 SwiGLU 激活。"""

        return swiglu(features, dim=self.dim)

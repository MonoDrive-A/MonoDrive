"""检测查询初始化和检测解码头。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn


__all__ = [
    "DetectionDecoderOutput",
    "DetectionHeadConfig",
    "DetectionHeadDecoder",
    "DetectionQueryEmbedding",
    "load_detection_head_config",
]


SUPPORTED_TOKEN_ORDERS = {"agent_then_map"}
SUPPORTED_ANCHOR_FEATURES = {
    "x_symlog",
    "y_symlog",
    "radius_normalized",
    "angle_normalized",
    "sin_angle",
    "cos_angle",
    "is_agent",
    "is_map",
    "query_progress",
}
SUPPORTED_SPATIAL_ORDERS = {"radius_major_angle_minor", "angle_major_radius_minor"}
SUPPORTED_POSITION_SOURCES = {"query_anchor_symlog"}
SUPPORTED_YAW_SOURCES = {"query_angle"}
SUPPORTED_POINT_SOURCES = {"query_anchor_symlog"}
SUPPORTED_CONTINUOUS_TRANSFORMS = {"symlog"}
SUPPORTED_SIZE_TRANSFORMS = {"log1p"}
SUPPORTED_DTYPE_NAMES = {"float32"}


@dataclass(frozen=True)
class DetectionHeadConfig:
    """检测头配置。

    Args:
        hidden_dim: 检测 Token 特征维度。
        agent_query_count: Agent 检测查询数量。
        map_query_count: Map 检测查询数量。
        token_order: 检测 Token 在统一序列中的排列顺序。
        anchor_feature_order: 写入查询初值前若干 hidden 通道的 anchor 特征顺序。
        query_unfilled_value: 未被 anchor 特征占用的查询初值。
        agent_class_names: Agent 前景类别名，不包含“无”类别。
        agent_none_class_name: Agent “无”类别名。
        agent_state_order: Agent 连续状态输出字段顺序。
        agent_future_mode_count: Agent future mode 数。
        agent_future_points: 每个 mode 的未来点数。
        agent_trajectory_dim: 每个 future 点坐标维度。
        agent_angle_min_deg: Agent 查询初始化最小角度，0 度为 ego 前向。
        agent_angle_max_deg: Agent 查询初始化最大角度。
        agent_radial_min_m: Agent 查询初始化最近半径，单位 meter。
        agent_radial_max_m: Agent 查询初始化最远半径，单位 meter。
        agent_radial_count: Agent 查询初始化半径采样数。
        agent_angle_count: Agent 查询初始化角度采样数。
        agent_spatial_order: Agent 查询空间 anchor 展平顺序。
        agent_position_source: Agent `x/y` 初始输出来源。
        agent_yaw_source: Agent `sin_yaw/cos_yaw` 初始输出来源。
        agent_size_lwh_m: Agent 长宽高初始物理尺寸，单位 meter。
        agent_velocity_xy_mps: Agent 平面速度初值，单位 meter/second。
        agent_acceleration_xy_mps2: Agent 平面加速度初值，单位 meter/second^2。
        agent_continuous_transform: 位置、速度、加速度和 future 的输出空间变换。
        agent_size_transform: 尺寸输出空间变换。
        agent_class_logit_init_value: Agent 前景类别 logit 初值。
        agent_none_logit_init_value: Agent “无”类别 logit 初值。
        agent_mode_logit_init_value: Agent mode logit 初值。
        agent_mode_angles_deg: 每个 Agent mode 的初始方向角。
        agent_mode_future_distances_m: Agent mode 每个未来点的初始位移距离。
        agent_mode_future_transform: Agent future 初始输出空间变换。
        map_class_names: Map 前景类别名，不包含“无”类别。
        map_none_class_name: Map “无”类别名。
        map_point_count: 每条 Map 元素点数。
        map_point_dim: 每个 Map 点坐标维度。
        map_angle_min_deg: Map 查询初始化最小角度。
        map_angle_max_deg: Map 查询初始化最大角度。
        map_radial_min_m: Map 查询初始化最近半径，单位 meter。
        map_radial_max_m: Map 查询初始化最远半径，单位 meter。
        map_radial_count: Map 查询初始化半径采样数。
        map_angle_count: Map 查询初始化角度采样数。
        map_spatial_order: Map 查询空间 anchor 展平顺序。
        map_point_source: Map 点初始输出来源。
        map_point_transform: Map 点输出空间变换。
        map_class_logit_init_value: Map 前景类别 logit 初值。
        map_none_logit_init_value: Map “无”类别 logit 初值。
        decoder_dtype: 解码线性层强制运行精度。
    """

    hidden_dim: int
    agent_query_count: int
    map_query_count: int
    token_order: str
    anchor_feature_order: tuple[str, ...]
    query_unfilled_value: float
    agent_class_names: tuple[str, ...]
    agent_none_class_name: str
    agent_state_order: tuple[str, ...]
    agent_future_mode_count: int
    agent_future_points: int
    agent_trajectory_dim: int
    agent_angle_min_deg: float
    agent_angle_max_deg: float
    agent_radial_min_m: float
    agent_radial_max_m: float
    agent_radial_count: int
    agent_angle_count: int
    agent_spatial_order: str
    agent_position_source: str
    agent_yaw_source: str
    agent_size_lwh_m: tuple[float, float, float]
    agent_velocity_xy_mps: tuple[float, float]
    agent_acceleration_xy_mps2: tuple[float, float]
    agent_continuous_transform: str
    agent_size_transform: str
    agent_class_logit_init_value: float
    agent_none_logit_init_value: float
    agent_mode_logit_init_value: float
    agent_mode_angles_deg: tuple[float, ...]
    agent_mode_future_distances_m: tuple[float, ...]
    agent_mode_future_transform: str
    map_class_names: tuple[str, ...]
    map_none_class_name: str
    map_point_count: int
    map_point_dim: int
    map_angle_min_deg: float
    map_angle_max_deg: float
    map_radial_min_m: float
    map_radial_max_m: float
    map_radial_count: int
    map_angle_count: int
    map_spatial_order: str
    map_point_source: str
    map_point_transform: str
    map_class_logit_init_value: float
    map_none_logit_init_value: float
    decoder_dtype: str

    def __post_init__(self) -> None:
        _validate_positive_int(self.hidden_dim, "hidden_dim")
        _validate_positive_int(self.agent_query_count, "agent_query_count")
        _validate_positive_int(self.map_query_count, "map_query_count")
        if self.token_order not in SUPPORTED_TOKEN_ORDERS:
            raise ValueError(
                f"token_order 仅支持 {sorted(SUPPORTED_TOKEN_ORDERS)}，实际为 {self.token_order!r}。"
            )
        if len(self.anchor_feature_order) == 0:
            raise ValueError("anchor_feature_order 不能为空。")
        if len(set(self.anchor_feature_order)) != len(self.anchor_feature_order):
            raise ValueError(f"anchor_feature_order 不能包含重复项，实际为 {self.anchor_feature_order}。")
        unsupported_features = set(self.anchor_feature_order) - SUPPORTED_ANCHOR_FEATURES
        if unsupported_features:
            raise ValueError(
                "anchor_feature_order 包含不支持的特征："
                f"{sorted(unsupported_features)}，支持 {sorted(SUPPORTED_ANCHOR_FEATURES)}。"
            )
        if self.hidden_dim < len(self.anchor_feature_order):
            raise ValueError(
                "hidden_dim 必须不小于 anchor_feature_order 长度，"
                f"实际为 {self.hidden_dim} 和 {len(self.anchor_feature_order)}。"
            )

        _validate_class_names(self.agent_class_names, self.agent_none_class_name, "agent")
        _validate_class_names(self.map_class_names, self.map_none_class_name, "map")
        _validate_state_order(self.agent_state_order)
        _validate_positive_int(self.agent_future_mode_count, "agent_future_mode_count")
        _validate_positive_int(self.agent_future_points, "agent_future_points")
        _validate_positive_int(self.agent_trajectory_dim, "agent_trajectory_dim")
        if self.agent_trajectory_dim != 2:
            raise ValueError(
                "agent_trajectory_dim 必须为 2，以表示 ego XY future 位移，"
                f"实际为 {self.agent_trajectory_dim}。"
            )
        _validate_positive_int(self.map_point_count, "map_point_count")
        _validate_positive_int(self.map_point_dim, "map_point_dim")
        if self.map_point_dim != 2:
            raise ValueError(f"map_point_dim 必须为 2，以表示 ego XY 点，实际为 {self.map_point_dim}。")

        _validate_spatial_config(
            self.agent_query_count,
            self.agent_angle_min_deg,
            self.agent_angle_max_deg,
            self.agent_radial_min_m,
            self.agent_radial_max_m,
            self.agent_radial_count,
            self.agent_angle_count,
            self.agent_spatial_order,
            "agent",
        )
        _validate_spatial_config(
            self.map_query_count,
            self.map_angle_min_deg,
            self.map_angle_max_deg,
            self.map_radial_min_m,
            self.map_radial_max_m,
            self.map_radial_count,
            self.map_angle_count,
            self.map_spatial_order,
            "map",
        )

        if self.agent_position_source not in SUPPORTED_POSITION_SOURCES:
            raise ValueError(
                "agent_position_source 仅支持 "
                f"{sorted(SUPPORTED_POSITION_SOURCES)}，实际为 {self.agent_position_source!r}。"
            )
        if self.agent_yaw_source not in SUPPORTED_YAW_SOURCES:
            raise ValueError(
                f"agent_yaw_source 仅支持 {sorted(SUPPORTED_YAW_SOURCES)}，"
                f"实际为 {self.agent_yaw_source!r}。"
            )
        if self.map_point_source not in SUPPORTED_POINT_SOURCES:
            raise ValueError(
                f"map_point_source 仅支持 {sorted(SUPPORTED_POINT_SOURCES)}，"
                f"实际为 {self.map_point_source!r}。"
            )
        _validate_transform(
            self.agent_continuous_transform,
            SUPPORTED_CONTINUOUS_TRANSFORMS,
            "agent_continuous_transform",
        )
        _validate_transform(self.agent_size_transform, SUPPORTED_SIZE_TRANSFORMS, "agent_size_transform")
        _validate_transform(
            self.agent_mode_future_transform,
            SUPPORTED_CONTINUOUS_TRANSFORMS,
            "agent_mode_future_transform",
        )
        _validate_transform(self.map_point_transform, SUPPORTED_CONTINUOUS_TRANSFORMS, "map_point_transform")
        _validate_float_tuple(self.agent_size_lwh_m, 3, "agent_size_lwh_m")
        if any(size <= 0.0 for size in self.agent_size_lwh_m):
            raise ValueError(f"agent_size_lwh_m 每一项都必须为正数，实际为 {self.agent_size_lwh_m}。")
        _validate_float_tuple(self.agent_velocity_xy_mps, 2, "agent_velocity_xy_mps")
        _validate_float_tuple(self.agent_acceleration_xy_mps2, 2, "agent_acceleration_xy_mps2")
        if len(self.agent_mode_angles_deg) != self.agent_future_mode_count:
            raise ValueError(
                "agent_mode_angles_deg 长度必须等于 agent_future_mode_count，"
                f"实际为 {len(self.agent_mode_angles_deg)} 和 {self.agent_future_mode_count}。"
            )
        if len(self.agent_mode_future_distances_m) != self.agent_future_points:
            raise ValueError(
                "agent_mode_future_distances_m 长度必须等于 agent_future_points，"
                f"实际为 {len(self.agent_mode_future_distances_m)} 和 {self.agent_future_points}。"
            )
        if any(distance < 0.0 for distance in self.agent_mode_future_distances_m):
            raise ValueError(
                "agent_mode_future_distances_m 不能包含负数，"
                f"实际为 {self.agent_mode_future_distances_m}。"
            )
        for mode_angle in self.agent_mode_angles_deg:
            if mode_angle < self.agent_angle_min_deg or mode_angle > self.agent_angle_max_deg:
                raise ValueError(
                    "agent_mode_angles_deg 必须落在 Agent 查询角度范围内，"
                    f"范围为 [{self.agent_angle_min_deg}, {self.agent_angle_max_deg}]，"
                    f"实际包含 {mode_angle}。"
                )
        if not math.isclose(
            self.agent_mode_angles_deg[0],
            self.agent_angle_min_deg,
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            self.agent_mode_angles_deg[-1],
            self.agent_angle_max_deg,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "agent_mode_angles_deg 首尾必须对齐 Agent 查询角度范围，"
                f"期望首尾为 {self.agent_angle_min_deg}/{self.agent_angle_max_deg}，"
                f"实际为 {self.agent_mode_angles_deg[0]}/{self.agent_mode_angles_deg[-1]}。"
            )
        _validate_uniform_angles(self.agent_mode_angles_deg, "agent_mode_angles_deg")
        if self.decoder_dtype not in SUPPORTED_DTYPE_NAMES:
            raise ValueError(
                f"decoder_dtype 仅支持 {sorted(SUPPORTED_DTYPE_NAMES)}，实际为 {self.decoder_dtype!r}。"
            )

    @property
    def total_query_count(self) -> int:
        """检测查询总数。"""

        return self.agent_query_count + self.map_query_count

    @property
    def agent_class_count_with_none(self) -> int:
        """Agent 分类输出通道数，包含“无”类别。"""

        return len(self.agent_class_names) + 1

    @property
    def map_class_count_with_none(self) -> int:
        """Map 分类输出通道数，包含“无”类别。"""

        return len(self.map_class_names) + 1

    @property
    def agent_state_dim(self) -> int:
        """Agent 连续状态输出维度。"""

        return len(self.agent_state_order)

    @property
    def agent_future_dim(self) -> int:
        """Agent future 输出展平维度。"""

        return self.agent_future_mode_count * self.agent_future_points * self.agent_trajectory_dim

    @property
    def agent_output_dim(self) -> int:
        """每个 Agent 查询的解码输出维度。"""

        return (
            self.agent_class_count_with_none
            + self.agent_state_dim
            + self.agent_future_mode_count
            + self.agent_future_dim
        )

    @property
    def map_output_dim(self) -> int:
        """每个 Map 查询的解码输出维度。"""

        return self.map_class_count_with_none + self.map_point_count * self.map_point_dim


class DetectionDecoderOutput(NamedTuple):
    """检测解码结果。

    Shape:
        `agent_class_logits`: `[B, A, C_agent + 1]`。
        `agent_states`: `[B, A, state_dim]`。
        `agent_mode_logits`: `[B, A, M]`。
        `agent_future_trajectories`: `[B, A, M, K, 2]`。
        `map_class_logits`: `[B, Q_map, C_map + 1]`。
        `map_points`: `[B, Q_map, P, 2]`。
    """

    agent_class_logits: torch.Tensor
    agent_states: torch.Tensor
    agent_mode_logits: torch.Tensor
    agent_future_trajectories: torch.Tensor
    map_class_logits: torch.Tensor
    map_points: torch.Tensor


class DetectionQueryEmbedding(nn.Module):
    """生成检测查询 Token 初值。

    Args:
        config: `load_detection_head_config` 读取并校验后的检测头配置。

    Shape:
        输出: `[agent_query_count + map_query_count, hidden_dim]`。
    """

    def __init__(self, config: DetectionHeadConfig) -> None:
        super().__init__()
        self.config = config
        agent_anchor_xy_m, agent_anchor_angle_rad = _build_spatial_anchors(
            angle_min_deg=config.agent_angle_min_deg,
            angle_max_deg=config.agent_angle_max_deg,
            radial_min_m=config.agent_radial_min_m,
            radial_max_m=config.agent_radial_max_m,
            radial_count=config.agent_radial_count,
            angle_count=config.agent_angle_count,
            spatial_order=config.agent_spatial_order,
        )
        map_anchor_xy_m, map_anchor_angle_rad = _build_spatial_anchors(
            angle_min_deg=config.map_angle_min_deg,
            angle_max_deg=config.map_angle_max_deg,
            radial_min_m=config.map_radial_min_m,
            radial_max_m=config.map_radial_max_m,
            radial_count=config.map_radial_count,
            angle_count=config.map_angle_count,
            spatial_order=config.map_spatial_order,
        )
        self.register_buffer("agent_anchor_xy_m", agent_anchor_xy_m)
        self.register_buffer("agent_anchor_angle_rad", agent_anchor_angle_rad)
        self.register_buffer("map_anchor_xy_m", map_anchor_xy_m)
        self.register_buffer("map_anchor_angle_rad", map_anchor_angle_rad)
        initial_queries = _build_initial_query_tokens(config, agent_anchor_xy_m, agent_anchor_angle_rad, map_anchor_xy_m, map_anchor_angle_rad)
        self.query_tokens = nn.Parameter(initial_queries)
        _force_floating_tensors_to_float32(self)

    def _apply(self, fn: Any) -> "DetectionQueryEmbedding":
        super()._apply(fn)
        _force_floating_tensors_to_float32(self)
        return self

    def forward(self) -> torch.Tensor:
        """返回检测查询 Token。"""

        with _disabled_autocast(self.query_tokens):
            return self.query_tokens.to(dtype=torch.float32)


class DetectionHeadDecoder(nn.Module):
    """从检测 Token 特征解码 Agent 和 Map 输出。

    Args:
        config: `load_detection_head_config` 读取并校验后的检测头配置。

    Shape:
        输入: `[B, agent_query_count + map_query_count, hidden_dim]`。
        输出: `DetectionDecoderOutput`，所有张量均为 FP32。
    """

    def __init__(self, config: DetectionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.agent_output_linear = nn.Linear(config.hidden_dim, config.agent_output_dim)
        self.map_output_linear = nn.Linear(config.hidden_dim, config.map_output_dim)
        self._reset_output_initialization()
        _force_floating_tensors_to_float32(self)

    def _apply(self, fn: Any) -> "DetectionHeadDecoder":
        super()._apply(fn)
        _force_floating_tensors_to_float32(self)
        return self

    def forward(self, detection_features: torch.Tensor) -> DetectionDecoderOutput:
        """解码检测输出。

        本函数只输出模型空间预测，不执行 Softmax、Sigmoid、反 Symlog 或其他
        物理空间反变换。分类概率、Hungarian matching、loss 和推理后处理由
        下游流程负责。
        """

        self._validate_detection_features(detection_features)
        with _disabled_autocast(detection_features):
            detection_features_fp32 = detection_features.to(dtype=torch.float32)
            agent_features, map_features = self._split_detection_features(detection_features_fp32)
            agent_raw = self.agent_output_linear(agent_features)
            map_raw = self.map_output_linear(map_features)
            return self._parse_raw_outputs(agent_raw, map_raw)

    def _validate_detection_features(self, detection_features: torch.Tensor) -> None:
        if not torch.is_floating_point(detection_features):
            raise TypeError(
                f"detection_features 必须为浮点张量，实际 dtype 为 {detection_features.dtype}。"
            )
        if detection_features.ndim != 3:
            raise ValueError(
                "detection_features 期望 shape 为 [B, Q, hidden_dim]，"
                f"实际为 {tuple(detection_features.shape)}。"
            )
        if int(detection_features.shape[1]) != self.config.total_query_count:
            raise ValueError(
                "detection_features 的查询数量与配置不一致："
                f"期望 {self.config.total_query_count}，实际为 {detection_features.shape[1]}。"
            )
        if int(detection_features.shape[2]) != self.config.hidden_dim:
            raise ValueError(
                "detection_features 的特征维度与配置不一致："
                f"期望 {self.config.hidden_dim}，实际为 {detection_features.shape[2]}。"
            )

    def _split_detection_features(self, detection_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.token_order != "agent_then_map":
            raise ValueError(f"不支持的 token_order：{self.config.token_order!r}。")
        agent_end = self.config.agent_query_count
        return detection_features[:, :agent_end, :], detection_features[:, agent_end:, :]

    def _parse_raw_outputs(
        self,
        agent_raw: torch.Tensor,
        map_raw: torch.Tensor,
    ) -> DetectionDecoderOutput:
        agent_offset = 0
        agent_class_logits = agent_raw[
            ...,
            agent_offset : agent_offset + self.config.agent_class_count_with_none,
        ]
        agent_offset += self.config.agent_class_count_with_none
        agent_states = agent_raw[..., agent_offset : agent_offset + self.config.agent_state_dim]
        agent_offset += self.config.agent_state_dim
        agent_mode_logits = agent_raw[
            ...,
            agent_offset : agent_offset + self.config.agent_future_mode_count,
        ]
        agent_offset += self.config.agent_future_mode_count
        agent_future_trajectories = agent_raw[..., agent_offset:].reshape(
            int(agent_raw.shape[0]),
            self.config.agent_query_count,
            self.config.agent_future_mode_count,
            self.config.agent_future_points,
            self.config.agent_trajectory_dim,
        )

        map_offset = 0
        map_class_logits = map_raw[..., map_offset : map_offset + self.config.map_class_count_with_none]
        map_offset += self.config.map_class_count_with_none
        map_points = map_raw[..., map_offset:].reshape(
            int(map_raw.shape[0]),
            self.config.map_query_count,
            self.config.map_point_count,
            self.config.map_point_dim,
        )
        return DetectionDecoderOutput(
            agent_class_logits=agent_class_logits,
            agent_states=agent_states,
            agent_mode_logits=agent_mode_logits,
            agent_future_trajectories=agent_future_trajectories,
            map_class_logits=map_class_logits,
            map_points=map_points,
        )

    def _reset_output_initialization(self) -> None:
        with torch.no_grad():
            self.agent_output_linear.weight.zero_()
            self.agent_output_linear.bias.zero_()
            self.map_output_linear.weight.zero_()
            self.map_output_linear.bias.zero_()
            self._initialize_agent_decoder()
            self._initialize_map_decoder()

    def _initialize_agent_decoder(self) -> None:
        agent_bias = self.agent_output_linear.bias
        agent_weight = self.agent_output_linear.weight
        class_start = 0
        class_end = self.config.agent_class_count_with_none
        agent_bias[class_start : class_end - 1].fill_(self.config.agent_class_logit_init_value)
        agent_bias[class_end - 1].fill_(self.config.agent_none_logit_init_value)

        state_start = class_end
        state_indices = _index_by_name(self.config.agent_state_order)
        if self.config.agent_position_source == "query_anchor_symlog":
            agent_weight[state_start + state_indices["x"], self._anchor_feature_index("x_symlog")] = 1.0
            agent_weight[state_start + state_indices["y"], self._anchor_feature_index("y_symlog")] = 1.0
        if self.config.agent_yaw_source == "query_angle":
            agent_weight[state_start + state_indices["sin_yaw"], self._anchor_feature_index("sin_angle")] = 1.0
            agent_weight[state_start + state_indices["cos_yaw"], self._anchor_feature_index("cos_angle")] = 1.0

        length_m, width_m, height_m = self.config.agent_size_lwh_m
        agent_bias[state_start + state_indices["length_log1p"]] = _transform_size(length_m, self.config.agent_size_transform)
        agent_bias[state_start + state_indices["width_log1p"]] = _transform_size(width_m, self.config.agent_size_transform)
        agent_bias[state_start + state_indices["height_log1p"]] = _transform_size(height_m, self.config.agent_size_transform)
        velocity_xy = _transform_continuous(
            torch.tensor(self.config.agent_velocity_xy_mps, dtype=torch.float32),
            self.config.agent_continuous_transform,
        )
        acceleration_xy = _transform_continuous(
            torch.tensor(self.config.agent_acceleration_xy_mps2, dtype=torch.float32),
            self.config.agent_continuous_transform,
        )
        agent_bias[state_start + state_indices["vx"]] = velocity_xy[0]
        agent_bias[state_start + state_indices["vy"]] = velocity_xy[1]
        agent_bias[state_start + state_indices["ax"]] = acceleration_xy[0]
        agent_bias[state_start + state_indices["ay"]] = acceleration_xy[1]

        mode_start = state_start + self.config.agent_state_dim
        mode_end = mode_start + self.config.agent_future_mode_count
        agent_bias[mode_start:mode_end].fill_(self.config.agent_mode_logit_init_value)

        future_template = _build_agent_mode_future_template(self.config)
        agent_bias[mode_end:].copy_(future_template.reshape(-1))

    def _initialize_map_decoder(self) -> None:
        map_bias = self.map_output_linear.bias
        map_weight = self.map_output_linear.weight
        class_start = 0
        class_end = self.config.map_class_count_with_none
        map_bias[class_start : class_end - 1].fill_(self.config.map_class_logit_init_value)
        map_bias[class_end - 1].fill_(self.config.map_none_logit_init_value)

        point_start = class_end
        if self.config.map_point_source == "query_anchor_symlog":
            for point_index in range(self.config.map_point_count):
                output_offset = point_start + point_index * self.config.map_point_dim
                map_weight[output_offset, self._anchor_feature_index("x_symlog")] = 1.0
                map_weight[output_offset + 1, self._anchor_feature_index("y_symlog")] = 1.0

    def _anchor_feature_index(self, feature_name: str) -> int:
        try:
            return self.config.anchor_feature_order.index(feature_name)
        except ValueError as exc:
            raise ValueError(
                f"anchor_feature_order 必须包含 {feature_name!r}，"
                f"实际为 {self.config.anchor_feature_order}。"
            ) from exc


def load_detection_head_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> DetectionHeadConfig:
    """读取检测头 TOML 配置。"""

    resolved_config_path = Path(config_path).resolve()
    resolved_project_root = (
        Path(project_root).resolve() if project_root is not None else resolved_config_path.parent.parent
    )
    _ensure_project_relative_path(resolved_config_path, resolved_project_root, "config_path")
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    query_config = _require_table(raw_config, "query")
    query_embedding_config = _require_table(raw_config, "query_embedding")
    agent_config = _require_table(raw_config, "agent")
    agent_query_config = _require_table(raw_config, "agent_query_initialization")
    agent_state_config = _require_table(raw_config, "agent_state_initialization")
    agent_decoder_config = _require_table(raw_config, "agent_decoder_initialization")
    agent_mode_config = _require_table(raw_config, "agent_mode_initialization")
    map_config = _require_table(raw_config, "map")
    map_query_config = _require_table(raw_config, "map_query_initialization")
    map_point_config = _require_table(raw_config, "map_point_initialization")
    map_decoder_config = _require_table(raw_config, "map_decoder_initialization")
    precision_config = _require_table(raw_config, "precision")

    return DetectionHeadConfig(
        hidden_dim=_require_int(query_config, "hidden_dim"),
        agent_query_count=_require_int(query_config, "agent_query_count"),
        map_query_count=_require_int(query_config, "map_query_count"),
        token_order=_require_string(query_config, "token_order"),
        anchor_feature_order=_require_string_tuple(query_embedding_config, "anchor_feature_order"),
        query_unfilled_value=_require_float(query_embedding_config, "unfilled_value"),
        agent_class_names=_require_string_tuple(agent_config, "class_names"),
        agent_none_class_name=_require_string(agent_config, "none_class_name"),
        agent_state_order=_require_string_tuple(agent_config, "state_order"),
        agent_future_mode_count=_require_int(agent_config, "future_mode_count"),
        agent_future_points=_require_int(agent_config, "future_points"),
        agent_trajectory_dim=_require_int(agent_config, "trajectory_dim"),
        agent_angle_min_deg=_require_float(agent_query_config, "angle_min_deg"),
        agent_angle_max_deg=_require_float(agent_query_config, "angle_max_deg"),
        agent_radial_min_m=_require_float(agent_query_config, "radial_min_m"),
        agent_radial_max_m=_require_float(agent_query_config, "radial_max_m"),
        agent_radial_count=_require_int(agent_query_config, "radial_count"),
        agent_angle_count=_require_int(agent_query_config, "angle_count"),
        agent_spatial_order=_require_string(agent_query_config, "spatial_order"),
        agent_position_source=_require_string(agent_state_config, "position_source"),
        agent_yaw_source=_require_string(agent_state_config, "yaw_source"),
        agent_size_lwh_m=_require_3d_float_tuple(agent_state_config, "size_lwh_m"),
        agent_velocity_xy_mps=_require_2d_float_tuple(agent_state_config, "velocity_xy_mps"),
        agent_acceleration_xy_mps2=_require_2d_float_tuple(agent_state_config, "acceleration_xy_mps2"),
        agent_continuous_transform=_require_string(agent_state_config, "continuous_transform"),
        agent_size_transform=_require_string(agent_state_config, "size_transform"),
        agent_class_logit_init_value=_require_float(agent_decoder_config, "class_logit_init_value"),
        agent_none_logit_init_value=_require_float(agent_decoder_config, "none_logit_init_value"),
        agent_mode_logit_init_value=_require_float(agent_decoder_config, "mode_logit_init_value"),
        agent_mode_angles_deg=_require_float_tuple(agent_mode_config, "mode_angles_deg"),
        agent_mode_future_distances_m=_require_float_tuple(agent_mode_config, "future_distances_m"),
        agent_mode_future_transform=_require_string(agent_mode_config, "future_transform"),
        map_class_names=_require_string_tuple(map_config, "class_names"),
        map_none_class_name=_require_string(map_config, "none_class_name"),
        map_point_count=_require_int(map_config, "point_count"),
        map_point_dim=_require_int(map_config, "point_dim"),
        map_angle_min_deg=_require_float(map_query_config, "angle_min_deg"),
        map_angle_max_deg=_require_float(map_query_config, "angle_max_deg"),
        map_radial_min_m=_require_float(map_query_config, "radial_min_m"),
        map_radial_max_m=_require_float(map_query_config, "radial_max_m"),
        map_radial_count=_require_int(map_query_config, "radial_count"),
        map_angle_count=_require_int(map_query_config, "angle_count"),
        map_spatial_order=_require_string(map_query_config, "spatial_order"),
        map_point_source=_require_string(map_point_config, "point_source"),
        map_point_transform=_require_string(map_point_config, "point_transform"),
        map_class_logit_init_value=_require_float(map_decoder_config, "class_logit_init_value"),
        map_none_logit_init_value=_require_float(map_decoder_config, "none_logit_init_value"),
        decoder_dtype=_require_string(precision_config, "decoder_dtype"),
    )


def _build_spatial_anchors(
    angle_min_deg: float,
    angle_max_deg: float,
    radial_min_m: float,
    radial_max_m: float,
    radial_count: int,
    angle_count: int,
    spatial_order: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    radii = torch.linspace(radial_min_m, radial_max_m, radial_count, dtype=torch.float32)
    angles_deg = torch.linspace(angle_min_deg, angle_max_deg, angle_count, dtype=torch.float32)
    angles_rad = torch.deg2rad(angles_deg)
    if spatial_order == "radius_major_angle_minor":
        radius_grid = radii[:, None].expand(radial_count, angle_count)
        angle_grid = angles_rad[None, :].expand(radial_count, angle_count)
    elif spatial_order == "angle_major_radius_minor":
        radius_grid = radii[None, :].expand(angle_count, radial_count)
        angle_grid = angles_rad[:, None].expand(angle_count, radial_count)
    else:
        raise ValueError(f"不支持的 spatial_order：{spatial_order!r}。")

    flat_radius = radius_grid.reshape(-1)
    flat_angle = angle_grid.reshape(-1)
    anchor_x_m = flat_radius * torch.cos(flat_angle)
    anchor_y_m = flat_radius * torch.sin(flat_angle)
    return torch.stack((anchor_x_m, anchor_y_m), dim=-1), flat_angle


def _build_initial_query_tokens(
    config: DetectionHeadConfig,
    agent_anchor_xy_m: torch.Tensor,
    agent_anchor_angle_rad: torch.Tensor,
    map_anchor_xy_m: torch.Tensor,
    map_anchor_angle_rad: torch.Tensor,
) -> torch.Tensor:
    initial_queries = torch.full(
        (config.total_query_count, config.hidden_dim),
        config.query_unfilled_value,
        dtype=torch.float32,
    )
    agent_features = _build_anchor_feature_matrix(
        config.anchor_feature_order,
        agent_anchor_xy_m,
        agent_anchor_angle_rad,
        angle_min_deg=config.agent_angle_min_deg,
        angle_max_deg=config.agent_angle_max_deg,
        radial_min_m=config.agent_radial_min_m,
        radial_max_m=config.agent_radial_max_m,
        is_agent=True,
    )
    map_features = _build_anchor_feature_matrix(
        config.anchor_feature_order,
        map_anchor_xy_m,
        map_anchor_angle_rad,
        angle_min_deg=config.map_angle_min_deg,
        angle_max_deg=config.map_angle_max_deg,
        radial_min_m=config.map_radial_min_m,
        radial_max_m=config.map_radial_max_m,
        is_agent=False,
    )
    if config.token_order != "agent_then_map":
        raise ValueError(f"不支持的 token_order：{config.token_order!r}。")
    initial_queries[: config.agent_query_count, : len(config.anchor_feature_order)] = agent_features
    initial_queries[
        config.agent_query_count :,
        : len(config.anchor_feature_order),
    ] = map_features
    return initial_queries


def _build_anchor_feature_matrix(
    feature_order: tuple[str, ...],
    anchor_xy_m: torch.Tensor,
    anchor_angle_rad: torch.Tensor,
    angle_min_deg: float,
    angle_max_deg: float,
    radial_min_m: float,
    radial_max_m: float,
    is_agent: bool,
) -> torch.Tensor:
    query_count = int(anchor_xy_m.shape[0])
    features = []
    anchor_radius_m = torch.linalg.norm(anchor_xy_m, dim=-1)
    angle_min_rad = math.radians(angle_min_deg)
    angle_max_rad = math.radians(angle_max_deg)
    angle_span_rad = angle_max_rad - angle_min_rad
    radius_span_m = radial_max_m - radial_min_m
    for feature_name in feature_order:
        if feature_name == "x_symlog":
            feature = _symlog(anchor_xy_m[:, 0])
        elif feature_name == "y_symlog":
            feature = _symlog(anchor_xy_m[:, 1])
        elif feature_name == "radius_normalized":
            feature = (anchor_radius_m - radial_min_m) / radius_span_m
        elif feature_name == "angle_normalized":
            feature = ((anchor_angle_rad - angle_min_rad) / angle_span_rad) * 2.0 - 1.0
        elif feature_name == "sin_angle":
            feature = torch.sin(anchor_angle_rad)
        elif feature_name == "cos_angle":
            feature = torch.cos(anchor_angle_rad)
        elif feature_name == "is_agent":
            feature = torch.ones(query_count, dtype=torch.float32) if is_agent else torch.zeros(query_count, dtype=torch.float32)
        elif feature_name == "is_map":
            feature = torch.zeros(query_count, dtype=torch.float32) if is_agent else torch.ones(query_count, dtype=torch.float32)
        elif feature_name == "query_progress":
            feature = torch.linspace(-1.0, 1.0, query_count, dtype=torch.float32)
        else:
            raise ValueError(f"不支持的 anchor feature：{feature_name!r}。")
        features.append(feature)
    return torch.stack(features, dim=-1)


def _build_agent_mode_future_template(config: DetectionHeadConfig) -> torch.Tensor:
    mode_angles_rad = torch.deg2rad(torch.tensor(config.agent_mode_angles_deg, dtype=torch.float32))
    future_distances_m = torch.tensor(config.agent_mode_future_distances_m, dtype=torch.float32)
    future_x_m = future_distances_m[None, :] * torch.cos(mode_angles_rad)[:, None]
    future_y_m = future_distances_m[None, :] * torch.sin(mode_angles_rad)[:, None]
    future_xy_m = torch.stack((future_x_m, future_y_m), dim=-1)
    return _transform_continuous(future_xy_m, config.agent_mode_future_transform)


def _transform_size(value: float, transform_name: str) -> torch.Tensor:
    if transform_name == "log1p":
        return torch.log1p(torch.tensor(float(value), dtype=torch.float32))
    raise ValueError(f"不支持的 size transform：{transform_name!r}。")


def _transform_continuous(values: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "symlog":
        return _symlog(values.to(dtype=torch.float32))
    raise ValueError(f"不支持的 continuous transform：{transform_name!r}。")


def _symlog(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.log1p(torch.abs(values))


def _disabled_autocast(reference_tensor: torch.Tensor) -> Any:
    """根据参考张量设备构造禁用 autocast 的上下文。"""

    if reference_tensor.device.type == "meta":
        return nullcontext()
    try:
        return torch.autocast(device_type=reference_tensor.device.type, enabled=False)
    except (RuntimeError, ValueError):
        return nullcontext()


def _force_floating_tensors_to_float32(module: nn.Module) -> None:
    """将模块内所有浮点参数、buffer 和已有梯度恢复为 FP32。"""

    with torch.no_grad():
        for parameter in module.parameters(recurse=True):
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(dtype=torch.float32)
            if parameter.grad is not None and parameter.grad.is_floating_point():
                parameter.grad.data = parameter.grad.data.to(dtype=torch.float32)

        for buffer in module.buffers(recurse=True):
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                buffer.data = buffer.data.to(dtype=torch.float32)


def _index_by_name(names: tuple[str, ...]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} 必须为整数，实际为 {value!r}。")
    if value <= 0:
        raise ValueError(f"{field_name} 必须为正整数，实际为 {value}。")


def _validate_class_names(class_names: tuple[str, ...], none_class_name: str, prefix: str) -> None:
    if len(class_names) == 0:
        raise ValueError(f"{prefix}_class_names 不能为空。")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"{prefix}_class_names 不能包含重复项，实际为 {class_names}。")
    if none_class_name in class_names:
        raise ValueError(f"{prefix}_none_class_name 不能与前景类别重复，实际为 {none_class_name!r}。")


def _validate_state_order(state_order: tuple[str, ...]) -> None:
    required_names = {
        "x",
        "y",
        "length_log1p",
        "width_log1p",
        "height_log1p",
        "sin_yaw",
        "cos_yaw",
        "vx",
        "vy",
        "ax",
        "ay",
    }
    if set(state_order) != required_names:
        raise ValueError(
            "agent_state_order 必须且只能包含 "
            f"{sorted(required_names)}，实际为 {state_order}。"
        )


def _validate_spatial_config(
    query_count: int,
    angle_min_deg: float,
    angle_max_deg: float,
    radial_min_m: float,
    radial_max_m: float,
    radial_count: int,
    angle_count: int,
    spatial_order: str,
    prefix: str,
) -> None:
    if angle_min_deg >= angle_max_deg:
        raise ValueError(
            f"{prefix}_angle_min_deg 必须小于 {prefix}_angle_max_deg，"
            f"实际为 {angle_min_deg} 和 {angle_max_deg}。"
        )
    if radial_min_m >= radial_max_m:
        raise ValueError(
            f"{prefix}_radial_min_m 必须小于 {prefix}_radial_max_m，"
            f"实际为 {radial_min_m} 和 {radial_max_m}。"
        )
    _validate_positive_int(radial_count, f"{prefix}_radial_count")
    _validate_positive_int(angle_count, f"{prefix}_angle_count")
    if radial_count * angle_count != query_count:
        raise ValueError(
            f"{prefix} 查询空间采样数量必须等于 query_count，"
            f"实际为 {radial_count} * {angle_count} != {query_count}。"
        )
    if spatial_order not in SUPPORTED_SPATIAL_ORDERS:
        raise ValueError(
            f"{prefix}_spatial_order 仅支持 {sorted(SUPPORTED_SPATIAL_ORDERS)}，"
            f"实际为 {spatial_order!r}。"
        )


def _validate_uniform_angles(values: tuple[float, ...], field_name: str) -> None:
    if len(values) <= 2:
        return
    step = values[1] - values[0]
    for index in range(2, len(values)):
        current_step = values[index] - values[index - 1]
        if not math.isclose(current_step, step, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"{field_name} 必须等间隔，以保证 mode 在 120 度空间内均匀散布，"
                f"实际为 {values}。"
            )


def _validate_transform(value: str, supported_values: set[str], field_name: str) -> None:
    if value not in supported_values:
        raise ValueError(f"{field_name} 仅支持 {sorted(supported_values)}，实际为 {value!r}。")


def _validate_float_tuple(values: tuple[float, ...], expected_length: int, field_name: str) -> None:
    if len(values) != expected_length:
        raise ValueError(f"{field_name} 必须包含 {expected_length} 个数值，实际为 {values}。")
    for index, value in enumerate(values):
        if not isinstance(value, float):
            raise TypeError(f"{field_name}[{index}] 必须为浮点数，实际为 {value!r}。")
        if not math.isfinite(value):
            raise ValueError(f"{field_name}[{index}] 必须为有限数，实际为 {value}。")


def _require_table(raw_config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少 [{key}] 表。")
    return value


def _require_string(table: Mapping[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"配置项 {key} 必须为非空字符串，实际为 {value!r}。")
    return value


def _require_int(table: Mapping[str, Any], key: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为整数，实际为 {value!r}。")
    return value


def _require_float(table: Mapping[str, Any], key: str) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为数值，实际为 {value!r}。")
    return float(value)


def _require_string_tuple(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须为字符串列表，实际为 {value!r}。")
    converted_values = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"配置项 {key}[{index}] 必须为非空字符串，实际为 {item!r}。")
        converted_values.append(item)
    return tuple(converted_values)


def _require_float_tuple(table: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须为数值列表，实际为 {value!r}。")
    converted_values = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"配置项 {key}[{index}] 必须为数值，实际为 {item!r}。")
        converted_values.append(float(item))
    return tuple(converted_values)


def _require_2d_float_tuple(table: Mapping[str, Any], key: str) -> tuple[float, float]:
    values = _require_float_tuple(table, key)
    _validate_float_tuple(values, 2, key)
    return (values[0], values[1])


def _require_3d_float_tuple(table: Mapping[str, Any], key: str) -> tuple[float, float, float]:
    values = _require_float_tuple(table, key)
    _validate_float_tuple(values, 3, key)
    return (values[0], values[1], values[2])


def _ensure_project_relative_path(path: Path, project_root: Path, config_key: str) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"{config_key} 必须解析到项目目录内，项目根目录为 {project_root}，实际为 {path}。"
        ) from exc

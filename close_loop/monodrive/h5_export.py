"""闭环推理导出与开环 ``B2DH5Dataset`` 兼容的 H5 样本。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from close_loop.monodrive.inputs import FINAL_HW, MODEL_INPUT_FPS, PAST_FRAMES, TRAJECTORY_DT

logger = logging.getLogger("monodrive_h5_export")

SCHEMA_VERSION = "b2d_h5_v5"
IMAGE_HEIGHT, IMAGE_WIDTH = FINAL_HW
INPUT_FRAME_COUNT = PAST_FRAMES
FUTURE_POINTS = 6
FUTURE_SECONDS = 3.0
TRAJECTORY_FPS = int(round(1.0 / TRAJECTORY_DT))
MAX_AGENTS = 194
MAX_MAP_ELEMENTS = 60
MAP_POINT_COUNT = 100
AGENT_STATE_DIM = 10
MAP_POINT_DIM = 2
MAX_TARGET_POINTS = 32
TRAFFIC_LIGHT_NONE_CLASS = 3


@dataclass(frozen=True)
class ClosedLoopH5Sample:
    """单条闭环样本，字段与 ``B2DH5Dataset.__getitem__`` 对齐。"""

    tick: int
    images_uint8: np.ndarray
    ego_motion: np.ndarray
    target_point: np.ndarray


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "闭环 H5 导出需要 h5py。请安装："
            ".\\.venv\\Scripts\\python.exe -m pip install h5py"
        ) from exc
    return h5py


def _frames_tensor_to_uint8_rgb(frames_past: torch.Tensor) -> np.ndarray:
    """``[T, 3, H, W]`` float [0,1] → ``[T, H, W, 3]`` uint8。"""
    frames_cpu = frames_past.detach().cpu().float().clamp(0.0, 1.0)
    if frames_cpu.ndim != 4 or int(frames_cpu.shape[1]) != 3:
        raise ValueError(
            f"frames_past 期望 shape [T, 3, H, W]，实际为 {tuple(frames_cpu.shape)}。"
        )
    if int(frames_cpu.shape[0]) != INPUT_FRAME_COUNT:
        raise ValueError(
            f"frames_past 时间维应为 {INPUT_FRAME_COUNT}，实际为 {int(frames_cpu.shape[0])}。"
        )
    height, width = int(frames_cpu.shape[2]), int(frames_cpu.shape[3])
    if (height, width) != FINAL_HW:
        raise ValueError(
            f"frames_past 分辨率应为 {FINAL_HW}，实际为 ({height}, {width})。"
        )
    rgb = frames_cpu.permute(0, 2, 3, 1).numpy()
    return (rgb * 255.0).round().astype(np.uint8, copy=False)


def _tensor_to_numpy_1d(value: torch.Tensor, expected_shape: tuple[int, ...], field_name: str) -> np.ndarray:
    array = value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    if tuple(array.shape) != expected_shape:
        raise ValueError(f"{field_name} 形状应为 {expected_shape}，实际为 {tuple(array.shape)}。")
    return array


def build_closed_loop_h5_sample(
    tick: int,
    frames_past: torch.Tensor,
    ego_motion: torch.Tensor,
    target_point: torch.Tensor,
) -> ClosedLoopH5Sample:
    """把闭环推理输入打包为单条 H5 样本。"""
    return ClosedLoopH5Sample(
        tick=int(tick),
        images_uint8=_frames_tensor_to_uint8_rgb(frames_past),
        ego_motion=_tensor_to_numpy_1d(ego_motion, (3,), "ego_motion"),
        target_point=_tensor_to_numpy_1d(target_point, (2,), "target_point"),
    )


class ClosedLoopH5Exporter:
    """在闭环运行期间累积样本，结束时写出 ``b2d_h5_v5`` H5。"""

    def __init__(
        self,
        output_path: str | Path,
        scene_name: str,
        goal_min_dist_m: float = 24.0,
        goal_max_dist_m: float = 30.0,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.scene_name = str(scene_name)
        self.goal_min_dist_m = float(goal_min_dist_m)
        self.goal_max_dist_m = float(goal_max_dist_m)
        self._samples: list[ClosedLoopH5Sample] = []
        self._finalized = False

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def append_sample(
        self,
        tick: int,
        frames_past: torch.Tensor,
        ego_motion: torch.Tensor,
        target_point: torch.Tensor,
    ) -> None:
        if self._finalized:
            raise RuntimeError("ClosedLoopH5Exporter 已 finalize，不能再追加样本。")
        self._samples.append(
            build_closed_loop_h5_sample(
                tick=tick,
                frames_past=frames_past,
                ego_motion=ego_motion,
                target_point=target_point,
            )
        )

    def finalize(self, overwrite: bool = True) -> Path | None:
        if self._finalized:
            return self.output_path if self.output_path.is_file() else None
        self._finalized = True
        if not self._samples:
            logger.warning("闭环 H5 导出跳过：未收集到任何样本 → %s", self.output_path)
            return None
        write_closed_loop_h5_file(
            output_path=self.output_path,
            samples=self._samples,
            scene_name=self.scene_name,
            goal_min_dist_m=self.goal_min_dist_m,
            goal_max_dist_m=self.goal_max_dist_m,
            overwrite=overwrite,
        )
        logger.info(
            "闭环 H5 导出完成：%s，samples=%d",
            self.output_path,
            len(self._samples),
        )
        return self.output_path


def write_closed_loop_h5_file(
    output_path: str | Path,
    samples: list[ClosedLoopH5Sample],
    scene_name: str,
    goal_min_dist_m: float = 24.0,
    goal_max_dist_m: float = 30.0,
    overwrite: bool = True,
) -> Path:
    """把闭环样本列表写成 ``b2d_h5_v5`` H5，供 ``B2DH5Dataset`` 与开环可视化读取。"""
    if not samples:
        raise ValueError("samples 不能为空。")

    h5py = _require_h5py()
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"H5 输出已存在：{output_path}")

    sample_count = len(samples)
    frame_count = sample_count * INPUT_FRAME_COUNT
    rgb_front = np.empty((frame_count, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image_frame_ids = np.empty((frame_count,), dtype=np.int32)

    current_frame_ids = np.empty((sample_count,), dtype=np.int32)
    input_frame_indices = np.empty((sample_count, INPUT_FRAME_COUNT), dtype=np.int32)
    input_frame_ids = np.empty((sample_count, INPUT_FRAME_COUNT), dtype=np.int32)
    future_frame_ids = np.zeros((sample_count, FUTURE_POINTS), dtype=np.int32)

    current_pose = np.zeros((sample_count, 3), dtype=np.float32)
    ego_motion = np.empty((sample_count, 3), dtype=np.float32)
    target_point = np.empty((sample_count, 2), dtype=np.float32)
    target_points = np.zeros((sample_count, MAX_TARGET_POINTS, 2), dtype=np.float32)
    target_valid = np.zeros((sample_count, MAX_TARGET_POINTS), dtype=np.bool_)
    commands = np.zeros((sample_count, 3), dtype=np.int16)
    control = np.zeros((sample_count, 3), dtype=np.float32)
    future_trajectory = np.zeros((sample_count, FUTURE_POINTS, 2), dtype=np.float32)
    agent_boxes = np.zeros((sample_count, MAX_AGENTS, AGENT_STATE_DIM), dtype=np.float32)
    agent_classes = np.full((sample_count, MAX_AGENTS), -1, dtype=np.int16)
    agent_valid = np.zeros((sample_count, MAX_AGENTS), dtype=np.bool_)
    agent_future_trajectory = np.zeros(
        (sample_count, MAX_AGENTS, FUTURE_POINTS, 2),
        dtype=np.float32,
    )
    agent_future_valid = np.zeros((sample_count, MAX_AGENTS, FUTURE_POINTS), dtype=np.bool_)
    map_points = np.zeros(
        (sample_count, MAX_MAP_ELEMENTS, MAP_POINT_COUNT, MAP_POINT_DIM),
        dtype=np.float32,
    )
    map_classes = np.full((sample_count, MAX_MAP_ELEMENTS), -1, dtype=np.int16)
    map_valid = np.zeros((sample_count, MAX_MAP_ELEMENTS), dtype=np.bool_)
    traffic_light_state = np.full((sample_count,), TRAFFIC_LIGHT_NONE_CLASS, dtype=np.int16)
    traffic_light_xy = np.zeros((sample_count, 2), dtype=np.float32)
    traffic_light_valid = np.zeros((sample_count,), dtype=np.bool_)
    stop_sign_state = np.zeros((sample_count,), dtype=np.int16)
    stop_sign_xy = np.zeros((sample_count, 2), dtype=np.float32)
    stop_sign_valid = np.zeros((sample_count,), dtype=np.bool_)

    for sample_index, sample in enumerate(samples):
        base_frame_index = sample_index * INPUT_FRAME_COUNT
        rgb_front[base_frame_index : base_frame_index + INPUT_FRAME_COUNT] = sample.images_uint8
        frame_id_base = int(sample.tick) - (INPUT_FRAME_COUNT - 1)
        for frame_offset in range(INPUT_FRAME_COUNT):
            global_frame_index = base_frame_index + frame_offset
            frame_id = frame_id_base + frame_offset
            image_frame_ids[global_frame_index] = frame_id
            input_frame_indices[sample_index, frame_offset] = global_frame_index
            input_frame_ids[sample_index, frame_offset] = frame_id

        current_frame_ids[sample_index] = int(sample.tick)
        ego_motion[sample_index] = sample.ego_motion
        target_point[sample_index] = sample.target_point
        target_points[sample_index, 0] = sample.target_point
        target_valid[sample_index, 0] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression_kwargs = {"compression": "gzip", "compression_opts": 4}
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["schema_version"] = SCHEMA_VERSION
        h5_file.attrs["scene_name"] = scene_name
        h5_file.attrs["scene_root"] = "carla_closed_loop"
        h5_file.attrs["sample_count"] = sample_count
        h5_file.attrs["frame_count"] = frame_count
        h5_file.attrs["raw_fps"] = int(MODEL_INPUT_FPS)
        h5_file.attrs["model_fps"] = int(MODEL_INPUT_FPS)
        h5_file.attrs["trajectory_fps"] = TRAJECTORY_FPS
        h5_file.attrs["future_seconds"] = FUTURE_SECONDS
        h5_file.attrs["future_points"] = FUTURE_POINTS
        h5_file.attrs["input_frame_count"] = INPUT_FRAME_COUNT
        h5_file.attrs["max_agents"] = MAX_AGENTS
        h5_file.attrs["max_map_elements"] = MAX_MAP_ELEMENTS
        h5_file.attrs["map_point_count"] = MAP_POINT_COUNT
        h5_file.attrs["raw_to_model_stride"] = 1
        h5_file.attrs["trajectory_stride"] = int(MODEL_INPUT_FPS / TRAJECTORY_FPS)
        h5_file.attrs["window_stride"] = 1
        h5_file.attrs["window_stride_raw"] = 1
        h5_file.attrs["image_height"] = IMAGE_HEIGHT
        h5_file.attrs["image_width"] = IMAGE_WIDTH
        h5_file.attrs["coordinate_system"] = "ego: x forward, y left, unit meter"
        h5_file.attrs["camera_sensor_name"] = "CAM_FRONT"
        h5_file.attrs["detection_forward_range"] = 32.0
        h5_file.attrs["detection_lateral_range"] = 32.0
        h5_file.attrs["min_visible_agent_vertices"] = 2
        h5_file.attrs["min_visible_agent_history_frames"] = 2
        h5_file.attrs["map_min_visible_points"] = 2
        h5_file.attrs["hd_map_root"] = ""
        h5_file.attrs["map_cache_dir"] = ""
        h5_file.attrs["hd_map_min_point_spacing"] = 0.5
        h5_file.attrs["target_min_distance"] = goal_min_dist_m
        h5_file.attrs["target_max_distance"] = goal_max_dist_m
        h5_file.attrs["max_target_points"] = MAX_TARGET_POINTS
        h5_file.attrs["target_search_seconds"] = "all_future"
        h5_file.attrs["smooth_future_trajectory"] = False
        h5_file.attrs["trajectory_smoothing_iterations"] = 1
        h5_file.attrs["source"] = "carla_closed_loop"

        frames_group = h5_file.create_group("frames")
        frames_group.create_dataset("frame_ids", data=image_frame_ids)
        frames_group.create_dataset(
            "rgb_front_path",
            data=np.asarray([""] * frame_count, dtype=object),
            dtype=string_dtype,
        )
        frames_group.create_dataset(
            "rgb_front",
            data=rgb_front,
            chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            **compression_kwargs,
        )

        samples_group = h5_file.create_group("samples")
        samples_group.create_dataset("current_frame_id", data=current_frame_ids)
        samples_group.create_dataset("input_frame_indices", data=input_frame_indices)
        samples_group.create_dataset("input_frame_ids", data=input_frame_ids)
        samples_group.create_dataset("future_frame_ids", data=future_frame_ids)

        labels_group = h5_file.create_group("labels")
        labels_group.create_dataset("current_pose", data=current_pose)
        labels_group.create_dataset("ego_motion", data=ego_motion)
        labels_group.create_dataset("target_point", data=target_point)
        labels_group.create_dataset("target_points", data=target_points)
        labels_group.create_dataset("target_valid", data=target_valid)
        labels_group.create_dataset("commands", data=commands)
        labels_group.create_dataset("control", data=control)
        labels_group.create_dataset(
            "future_trajectory",
            data=future_trajectory,
            chunks=(1, FUTURE_POINTS, 2),
            **compression_kwargs,
        )
        labels_group.create_dataset(
            "agent_boxes",
            data=agent_boxes,
            chunks=(1, MAX_AGENTS, AGENT_STATE_DIM),
            **compression_kwargs,
        )
        labels_group.create_dataset("agent_classes", data=agent_classes)
        labels_group.create_dataset("agent_valid", data=agent_valid)
        labels_group.create_dataset(
            "agent_future_trajectory",
            data=agent_future_trajectory,
            chunks=(1, MAX_AGENTS, FUTURE_POINTS, 2),
            **compression_kwargs,
        )
        labels_group.create_dataset("agent_future_valid", data=agent_future_valid)
        labels_group.create_dataset(
            "map_points",
            data=map_points,
            chunks=(1, MAX_MAP_ELEMENTS, MAP_POINT_COUNT, MAP_POINT_DIM),
            **compression_kwargs,
        )
        labels_group.create_dataset("map_classes", data=map_classes)
        labels_group.create_dataset("map_valid", data=map_valid)
        labels_group.create_dataset("traffic_light_state", data=traffic_light_state)
        labels_group.create_dataset("traffic_light_xy", data=traffic_light_xy)
        labels_group.create_dataset("traffic_light_valid", data=traffic_light_valid)
        labels_group.create_dataset("stop_sign_state", data=stop_sign_state)
        labels_group.create_dataset("stop_sign_xy", data=stop_sign_xy)
        labels_group.create_dataset("stop_sign_valid", data=stop_sign_valid)

    return output_path


def dump_openloop_h5_snapshot(
    out_dir: Path,
    tick: int,
    frames_past: torch.Tensor,
    ego_motion: torch.Tensor,
    target_point: torch.Tensor,
    scene_name: str = "carla_closed_loop",
    goal_min_dist_m: float = 24.0,
    goal_max_dist_m: float = 30.0,
) -> Path:
    """落盘单样本 ``b2d_h5_v5`` H5，可直接喂给 ``backbone_feature_pca_viewer --h5``。"""
    sample = build_closed_loop_h5_sample(
        tick=tick,
        frames_past=frames_past,
        ego_motion=ego_motion,
        target_point=target_point,
    )
    h5_path = Path(out_dir) / f"snapshot_tick{tick:05d}.h5"
    write_closed_loop_h5_file(
        output_path=h5_path,
        samples=[sample],
        scene_name=scene_name,
        goal_min_dist_m=goal_min_dist_m,
        goal_max_dist_m=goal_max_dist_m,
        overwrite=True,
    )
    logger.info("dumped open-loop-compatible H5 snapshot %s", h5_path)
    return h5_path

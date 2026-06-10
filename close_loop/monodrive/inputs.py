"""Carla 仿真到 MonoDriveBackbone 输入张量的转换。



负责两个 ring buffer 与一个坐标变换工具：



- ``FrameBuffer``：默认 **5Hz** 仿真下每 tick 缓存一帧；``stack()`` 取最近 8 帧（或按 0.2s 重采样）。

- ``EgoBuffer``：缓存足够长的 ego 物理状态历史，供 anchor 帧 ``ego_motion`` /

  ``target_point`` 与图像 anchor 对齐。

- ``build_ego_motion`` / ``build_target_point``：anchor 帧 ego-local ``(vx, vy, w)``

  与 ``(target_x, target_y)``；**不**向模型传入航向角 yaw，仅 ``w`` 为角速度。



**全部保持物理值**——模型内部目标点嵌入与 ego_motion 编码会自动 Symlog。

"""



from __future__ import annotations



import math

from collections import deque

from dataclasses import dataclass

from typing import Deque, Tuple



import numpy as np

import torch

import torch.nn.functional as F





# ─────────────────────────────────────────────────────────────

# 图像 / ego buffer 常量（与训练对齐）

# ─────────────────────────────────────────────────────────────

SOURCE_HW = (900, 1600)  # B2D CAM_FRONT 采集分辨率 (H, W)，见 camera_config.py

DEFAULT_CAMERA_HW = (450, 800)  # 闭环默认 Carla 相机 (H, W)，FOV 仍为 70°

MODEL_HW = (288, 512)    # 模型输入分辨率 (H, W)，与 config/vision_embedding.toml 一致

FINAL_HW = MODEL_HW

PAST_FRAMES = 8

TRAJECTORY_DT = 0.5  # 2Hz 未来轨迹标签间隔 (s)

MODEL_INPUT_FPS = 5.0  # B2D model_fps；图像与 ego_motion 差分窗口 (s⁻¹)

EGO_MOTION_DIFF_DT = 1.0 / MODEL_INPUT_FPS  # 0.2 s

MODEL_INPUT_SPAN_S = (PAST_FRAMES - 1) * EGO_MOTION_DIFF_DT  # 1.4 s

# 闭环默认 Carla 同步 tick 周期：与 model_fps 一致时每 tick 一帧，无需跨 tick 重采样。
DEFAULT_SIM_DT = EGO_MOTION_DIFF_DT  # 0.2 s → 5 Hz
DEFAULT_SIM_FPS = MODEL_INPUT_FPS





def resize_frame_chw(frame_chw: torch.Tensor, out_height: int, out_width: int) -> torch.Tensor:

    """双线性下采样单帧 ``(3, H, W)`` FP32 图像到 ``(3, out_height, out_width)``。"""

    if frame_chw.ndim != 3:

        raise ValueError(

            f"frame_chw 期望 shape 为 (3, H, W)，实际为 {tuple(frame_chw.shape)}。"

        )

    if int(frame_chw.shape[0]) != 3:

        raise ValueError(f"frame_chw 通道数必须为 3，实际为 {frame_chw.shape[0]}。")

    resized = F.interpolate(

        frame_chw.unsqueeze(0),

        size=(int(out_height), int(out_width)),

        mode="bilinear",

        align_corners=False,

    )

    return resized.squeeze(0).contiguous()





def wrap_pi(x: float | np.ndarray) -> float | np.ndarray:

    """把角度（弧度）规整到 ``[-π, π)``。"""

    return (x + np.pi) % (2.0 * np.pi) - np.pi





def model_input_tick_backs(sim_dt: float) -> tuple[int, ...]:

    """相对 anchor（buffer 最新帧）向回的 tick 数，共 ``PAST_FRAMES`` 个，时间间隔约 0.2s。



    返回顺序为从旧到新，例如 ``(..., 4, 2, 0)``。

    """

    sim_dt = max(float(sim_dt), 1e-6)

    backs: list[int] = []

    for frame_index in range(PAST_FRAMES):

        offset_s = (PAST_FRAMES - 1 - frame_index) * EGO_MOTION_DIFF_DT

        backs.append(int(round(offset_s / sim_dt)))

    return tuple(backs)





def required_buffer_capacity(sim_dt: float) -> int:

    """图像 / ego ring buffer 容量：覆盖 5Hz 重采样所需的最长回看 tick。"""

    return max(model_input_tick_backs(sim_dt)) + 1





def effective_yaw(yaw_rad: float, flip_y: bool) -> float:

    """``--flip-y`` 时取反航向角，供 ego-local 变换与控制侧 world 投影使用。



    模型本身不接收 yaw，只接收 ``ego_motion`` 中的角速度 ``w``。

    """

    yaw = float(yaw_rad)

    if flip_y:

        yaw = float(wrap_pi(-yaw))

    return yaw





# ─────────────────────────────────────────────────────────────

# Frame buffer

# ─────────────────────────────────────────────────────────────

class FrameBuffer:

    """前视图像 ring buffer；``stack()`` 按 5Hz 重采样最近 ``PAST_FRAMES`` 帧。"""



    def __init__(

        self,

        maxlen: int | None = None,

        source_hw: tuple[int, int] | None = DEFAULT_CAMERA_HW,

        sim_dt: float = DEFAULT_SIM_DT,

    ) -> None:

        self._sim_dt = max(float(sim_dt), 1e-6)

        self.maxlen = int(maxlen) if maxlen is not None else required_buffer_capacity(self._sim_dt)

        self.source_hw = source_hw

        self._buf: Deque[torch.Tensor] = deque(maxlen=self.maxlen)



    def set_dt(self, dt: float) -> None:

        """更新仿真 tick 间隔，并同步重算 buffer 容量。"""

        self._sim_dt = max(float(dt), 1e-6)

        needed = required_buffer_capacity(self._sim_dt)

        if needed > self.maxlen:

            self.maxlen = needed

            self._buf = deque(list(self._buf), maxlen=self.maxlen)



    @property

    def sim_dt(self) -> float:

        return self._sim_dt



    def __len__(self) -> int:

        return len(self._buf)



    def is_full(self) -> bool:

        return len(self._buf) >= self.maxlen



    def clear(self) -> None:

        self._buf.clear()



    def push_bgra_uint8(self, bgra: np.ndarray) -> None:

        """``bgra``: ``(H, W, 4)`` uint8（Carla ``sensor.camera.rgb`` 原始数据）。"""

        if bgra.ndim == 3 and bgra.shape[-1] == 4:

            rgb = np.ascontiguousarray(bgra[..., [2, 1, 0]])

        elif bgra.ndim == 3 and bgra.shape[-1] == 3:

            rgb = np.ascontiguousarray(bgra)

        else:

            raise ValueError(f"期望 (H,W,3|4)，当前 shape={bgra.shape}")

        if self.source_hw is not None and rgb.shape[:2] != self.source_hw:

            raise ValueError(

                f"摄像头输出分辨率 {rgb.shape[:2]} != 期望 {self.source_hw}; "

                f"请与 attach_front_camera 的 height/width 一致"

            )

        t = torch.from_numpy(rgb).to(torch.float32).div_(255.0).permute(2, 0, 1).contiguous()

        t = resize_frame_chw(t, FINAL_HW[0], FINAL_HW[1])

        self._buf.append(t)



    def stack(self) -> torch.Tensor:

        """返回 ``(8, 3, 288, 512)`` FP32。

        当 ``sim_dt == EGO_MOTION_DIFF_DT``（默认 5Hz）时，buffer 每 tick 一帧，直接取最近 8 帧；
        否则按 0.2s 间隔在 ring buffer 内重采样。
        """

        if not self.is_full():

            raise RuntimeError(

                f"FrameBuffer 未填满: 当前 {len(self._buf)}/{self.maxlen}"

            )

        if abs(self._sim_dt - EGO_MOTION_DIFF_DT) < 1e-9 and len(self._buf) >= PAST_FRAMES:

            frames = list(self._buf)[-PAST_FRAMES:]

            return torch.stack(frames, dim=0).contiguous()

        tick_backs = model_input_tick_backs(self._sim_dt)

        buf_len = len(self._buf)

        frames: list[torch.Tensor] = []

        for tick_back in tick_backs:

            index = buf_len - 1 - int(tick_back)

            if index < 0:

                raise RuntimeError(

                    f"5Hz 重采样索引越界: tick_back={tick_back}, buf_len={buf_len}"

                )

            frames.append(self._buf[index])

        return torch.stack(frames, dim=0).contiguous()





# ─────────────────────────────────────────────────────────────

# Ego buffer

# ─────────────────────────────────────────────────────────────

@dataclass

class EgoSnapshot:

    """一帧 ego 状态（全部世界系，FP64 numpy 标量）。"""



    x: float

    y: float

    yaw: float          # 弧度，Carla 世界系航向

    vx: float

    vy: float

    ax: float

    ay: float

    yaw_rate: float = 0.0  # 保留字段；``build_ego_motion`` 在 buffer 上按 5Hz 重算 w





class EgoBuffer:

    """ego 状态 ring buffer；容量与 ``FrameBuffer`` 对齐。"""



    def __init__(self, maxlen: int | None = None, sim_dt: float = DEFAULT_SIM_DT) -> None:

        self._sim_dt = max(float(sim_dt), 1e-6)

        self.maxlen = int(maxlen) if maxlen is not None else required_buffer_capacity(self._sim_dt)

        self._buf: Deque[EgoSnapshot] = deque(maxlen=self.maxlen)



    def set_dt(self, dt: float) -> None:

        self._sim_dt = max(float(dt), 1e-6)

        needed = required_buffer_capacity(self._sim_dt)

        if needed > self.maxlen:

            self.maxlen = needed

            self._buf = deque(list(self._buf), maxlen=self.maxlen)



    @property

    def sim_dt(self) -> float:

        return self._sim_dt



    def __len__(self) -> int:

        return len(self._buf)



    def is_full(self) -> bool:

        return len(self._buf) >= self.maxlen



    def clear(self) -> None:

        self._buf.clear()



    def latest(self) -> EgoSnapshot:

        if not self._buf:

            raise RuntimeError("EgoBuffer 为空")

        return self._buf[-1]



    def push_from_vehicle(self, vehicle) -> None:

        """从 ``carla.Vehicle`` 抓一帧。



        模型 ``ego_motion`` 由 ``build_ego_motion`` 按 5Hz 世界 xy / yaw 差分重算，

        **不**直接使用下方速度 / 加速度。

        """

        tf = vehicle.get_transform()

        loc = tf.location

        rot = tf.rotation

        vel = vehicle.get_velocity()

        acc = vehicle.get_acceleration()



        x = float(loc.x)

        y = float(loc.y)

        yaw = math.radians(float(rot.yaw))

        vx = float(vel.x)

        vy = float(vel.y)

        ax = float(acc.x)

        ay = float(acc.y)



        self._buf.append(EgoSnapshot(x=x, y=y, yaw=yaw, vx=vx, vy=vy, ax=ax, ay=ay))



    def world_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """返回 ``(xy, yaw, v, a, yaw_rate)``，全部世界系。"""

        t_count = len(self._buf)

        xy = np.zeros((t_count, 2), dtype=np.float64)

        yaw = np.zeros((t_count,), dtype=np.float64)

        v = np.zeros((t_count, 2), dtype=np.float64)

        a = np.zeros((t_count, 2), dtype=np.float64)

        yr = np.zeros((t_count,), dtype=np.float64)

        for i, s in enumerate(self._buf):

            xy[i, 0] = s.x

            xy[i, 1] = s.y

            yaw[i] = s.yaw

            v[i, 0] = s.vx

            v[i, 1] = s.vy

            a[i, 0] = s.ax

            a[i, 1] = s.ay

            yr[i] = s.yaw_rate

        return xy, yaw, v, a, yr





# ─────────────────────────────────────────────────────────────

# 世界系 -> ego-local 系

# ─────────────────────────────────────────────────────────────

def to_ego_local_xy(

    xy_world: np.ndarray, yaw_ref: float, p_ref: np.ndarray

) -> np.ndarray:

    """把若干个世界系 xy 点变换到 ego-local 系（以 ``p_ref`` 为原点、``yaw_ref`` 为前向）。"""

    c, s = math.cos(-yaw_ref), math.sin(-yaw_ref)

    r_inv = np.array([[c, -s], [s, c]], dtype=np.float64)

    arr = np.asarray(xy_world, dtype=np.float64)

    flat = arr.reshape(-1, 2)

    out = (flat - np.asarray(p_ref, dtype=np.float64)) @ r_inv.T

    return out.reshape(arr.shape)





def _world_xy_rate_to_ego_vx_vy(

    dx_dt: float, dy_dt: float, yaw_ref: float

) -> tuple[float, float]:

    """世界系 xy 标量导数 → anchor 帧 ego-local ``(vx, vy)``。"""

    c = math.cos(yaw_ref)

    s = math.sin(yaw_ref)

    vx = dx_dt * c + dy_dt * s

    vy = -dx_dt * s + dy_dt * c

    return vx, vy





def build_ego_motion(ego_buf: EgoBuffer, *, flip_y: bool = False) -> torch.Tensor:

    """anchor 帧 ego-local ``(vx, vy, w)`` → ``(3,)`` 物理值张量。



    在约 ``EGO_MOTION_DIFF_DT`` (5Hz) 窗口上对世界系 xy / yaw 差分，再转到 ego-local。

    ``flip_y`` 时对 anchor / 历史 yaw 取反，使 ``w`` 与 ``(vx, vy)`` 与 B2D 约定一致。

    """

    if not ego_buf.is_full():

        raise RuntimeError(

            f"EgoBuffer 未填满: 当前 {len(ego_buf)}/{ego_buf.maxlen}"

        )

    xy_w, yaw_w, _v_w, _a_w, _yr = ego_buf.world_arrays()

    ref = len(yaw_w) - 1

    sim_dt = max(float(ego_buf.sim_dt), 1e-6)

    stride = max(1, int(round(EGO_MOTION_DIFF_DT / sim_dt)))

    prev = max(0, ref - stride)

    dt = (ref - prev) * sim_dt

    if dt <= 0.0:

        return torch.zeros(3, dtype=torch.float32)



    yaw_ref = effective_yaw(float(yaw_w[ref]), flip_y)

    yaw_prev = effective_yaw(float(yaw_w[prev]), flip_y)

    dx_dt = (float(xy_w[ref, 0]) - float(xy_w[prev, 0])) / dt

    dy_dt = (float(xy_w[ref, 1]) - float(xy_w[prev, 1])) / dt

    vx, vy = _world_xy_rate_to_ego_vx_vy(dx_dt, dy_dt, yaw_ref)

    w = float(wrap_pi(yaw_ref - yaw_prev)) / dt

    return torch.tensor([vx, vy, w], dtype=torch.float32)





def build_target_point(

    ego_buf: EgoBuffer,

    goal_world_xy: np.ndarray,

    *,

    flip_y: bool = False,

) -> torch.Tensor:

    """anchor 帧 ego-local 目标点 ``(x, y)`` → ``(2,)`` 物理值张量。"""

    if not ego_buf.is_full():

        raise RuntimeError(

            f"EgoBuffer 未填满: 当前 {len(ego_buf)}/{ego_buf.maxlen}"

        )

    xy_w, yaw_w, _v_w, _a_w, _yr = ego_buf.world_arrays()

    ref = len(yaw_w) - 1

    p_ref = xy_w[ref]

    yaw_ref = effective_yaw(float(yaw_w[ref]), flip_y)

    goal_local = to_ego_local_xy(

        np.asarray(goal_world_xy, dtype=np.float64), yaw_ref, p_ref

    )

    return torch.tensor([goal_local[0], goal_local[1]], dtype=torch.float32)





def build_goal_dxy(

    ego_buf: EgoBuffer,

    goal_world_xy: np.ndarray,

    *,

    flip_y: bool = False,

) -> torch.Tensor:

    """``build_target_point`` 的兼容别名。"""

    return build_target_point(ego_buf, goal_world_xy, flip_y=flip_y)





def ego_local_to_world(

    xy_local: np.ndarray, yaw_ref: float, p_ref: np.ndarray

) -> np.ndarray:

    """ego-local 系 -> 世界系，``R(yaw_ref) @ xy + p_ref``。"""

    c, s = math.cos(yaw_ref), math.sin(yaw_ref)

    r_mat = np.array([[c, -s], [s, c]], dtype=np.float64)

    arr = np.asarray(xy_local, dtype=np.float64)

    flat = arr.reshape(-1, 2)

    out = flat @ r_mat.T + np.asarray(p_ref, dtype=np.float64)

    return out.reshape(arr.shape)



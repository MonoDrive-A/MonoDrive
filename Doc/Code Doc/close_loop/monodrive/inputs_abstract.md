# close_loop/monodrive/inputs.py 摘要



## 1. 文件基本功能



将 Carla 仿真状态转换为 `MonoDriveBackbone` 所需张量：ring buffer 缓存、**5Hz** 重采样 8 帧图像，以及 anchor 帧 `ego_motion` / `target_point` 构造与坐标变换。



## 2. 主要公开接口



| 名称 | 类型 | 说明 |

| --- | --- | --- |

| `PAST_FRAMES` | constant | 模型输入帧数 8。 |

| `MODEL_INPUT_FPS` | constant | B2D model_fps（5 Hz）。 |

| `DEFAULT_SIM_DT` | constant | 闭环默认 tick 周期 0.2 s（5 Hz）。 |

| `FrameBuffer` | class | `stack()` 输出 5Hz 采样的 `[8,3,288,512]`。 |

| `EgoBuffer` | class | 缓存 ego 世界系状态。 |

| `effective_yaw` | function | `--flip-y` 时对 Carla yaw 取反。 |

| `build_ego_motion` | function | 输出 `(3,)` 物理 `[Vx,Vy,W]`（无 yaw 角）。 |

| `build_target_point` | function | 输出 `(2,)` ego-local 目标点。 |

| `ego_local_to_world` | function | 轨迹点 ego-local → 世界系。 |



## 3. 输入输出 Shape 概览



| 接口 | 输入 | 输出 Shape |

| --- | --- | --- |

| `FrameBuffer.stack` | buffer 满（约 1.4s 回看） | `[8, 3, 288, 512]` |

| `build_ego_motion` | 满 `EgoBuffer` | `(3,)` |

| `build_target_point` | 满 buffer + goal 世界坐标 | `(2,)` |



## 4. 公开接口使用规范



- 闭环默认 **5Hz** 仿真：每 tick push 一帧，`stack()` 直接取最近 8 帧；非 5Hz 时内部按 0.2s 重采样。

- `ego_motion` / `target_point` 为物理米制量；Symlog 在模型内完成。

- `--flip-y` 须在 `build_*` 与 `ego_local_to_world` 间统一使用 `effective_yaw`。



## 5. 维护记录



| 日期 | 修改人 | 说明 |

| --- | --- | --- |

| 2026-06-10 | FuZiR_Cursor | 默认 5Hz 仿真；`DEFAULT_SIM_DT`；`stack()` 快路径。 |

| 2026-06-10 | FuZiR_Cursor | 5Hz 图像重采样；`flip_y` 作用于 yaw。 |

| 2026-06-09 | FuZiR_Cursor | 自 JEPA 16 帧输入迁移为 MonoDrive 8 帧与 `target_point` 命名。 |



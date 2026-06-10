# close_loop/monodrive/inputs.py



## 1. 文件职责



负责 Carla 闭环中模型输入的构造：前视图像下采样与缓存、ego 状态缓存、anchor 帧 `ego_motion` 与 `target_point` 计算，以及 ego-local / 世界系 xy 变换。字段语义与 `data/b2d_dataset.py` 对齐。



**模型不接收航向角 yaw**，只接收 `ego_motion` 的 `[Vx, Vy, W]`（`W` 为角速度）。yaw 仅用于构造 `target_point` 与轨迹 world 投影。



## 2. 公开接口



| 名称 | 类型 | 说明 |

| --- | --- | --- |

| `PAST_FRAMES` | int | 模型输入历史帧数 8。 |

| `MODEL_INPUT_FPS` | float | B2D model_fps；图像与 ego_motion 差分基准 5 Hz。 |

| `DEFAULT_SIM_DT` | float | 闭环默认 Carla tick 周期 0.2 s（5 Hz）。 |

| `DEFAULT_SIM_FPS` | float | 与 `MODEL_INPUT_FPS` 相同。 |

| `TRAJECTORY_DT` | float | 2Hz 轨迹点间隔 0.5 s，供控制差分。 |

| `model_input_tick_backs` | function | 给定 `sim_dt`，返回 5Hz 重采样的 tick 回看表。 |

| `required_buffer_capacity` | function | ring buffer 最小容量。 |

| `effective_yaw` | function | `--flip-y` 时对 Carla yaw 取反。 |

| `FrameBuffer` | class | RGB 帧 ring buffer；`stack()` 输出 5Hz 8 帧。 |

| `EgoBuffer` | class | ego 快照 ring buffer。 |

| `build_ego_motion` | function | 构造 anchor ego-local `[Vx, Vy, W]`。 |

| `build_target_point` | function | 构造 anchor ego-local 目标点。 |

| `ego_local_to_world` | function | 批量 xy 变换到世界系。 |



## 3. 关键类和函数



### `FrameBuffer`



- 每个仿真 tick `push` 一帧；内部按 `sim_dt` 扩容至覆盖约 1.4s 回看。

- `stack()`：默认 5Hz 仿真下每 tick 一帧，直接取最近 8 帧 → `[8, 3, 288, 512]` FP32；其它 `sim_dt` 时按 0.2s 间隔重采样。

- 默认 5Hz 下约需 **8 tick**（1.6 s）填满 buffer。



### `EgoBuffer`



- 容量与 `FrameBuffer` 一致；anchor 为最新 tick。

- `build_ego_motion` 在 anchor 与约 0.2s 前快照间差分世界 xy / yaw。



### `effective_yaw` / `flip_y`



- `build_ego_motion(..., flip_y=True)` 与 `build_target_point(..., flip_y=True)` 对 anchor / 历史 yaw 取反。

- 控制侧 `ego_local_to_world` 使用同一 `effective_yaw`，避免只翻 traj_y 不翻 yaw。



## 4. 配置与常量



| 常量 | 值 | 说明 |

| --- | --- | --- |

| `PAST_FRAMES` | 8 | 与 `config/vision_embedding.toml` 一致。 |

| `MODEL_INPUT_FPS` | 5.0 | 与 B2D `model_fps` 一致。 |

| `DEFAULT_SIM_DT` | 0.2 | 闭环 `fixed_delta_seconds` 默认值 (s)。 |

| `DEFAULT_SIM_FPS` | 5.0 | 闭环仿真 tick 频率 (Hz)。 |

| `MODEL_INPUT_SPAN_S` | 1.4 | `(PAST_FRAMES-1) × 0.2` (s)。 |

| `TRAJECTORY_DT` | 0.5 | 与 B2D 2Hz 未来轨迹标签一致。 |

| `FINAL_HW` | (288, 512) | 与 `config/vision_embedding.toml` 一致。 |



## 5. 依赖关系



- `torch.nn.functional.interpolate`（双线性下采样）

- `numpy`, `torch`



## 6. 维护记录



| 日期 | 修改人 | 说明 |

| --- | --- | --- |

| 2026-06-10 | FuZiR_Cursor | 默认仿真改为 5Hz；`DEFAULT_SIM_DT`；5Hz 下 `stack()` 直接取连续 8 帧。 |

| 2026-06-10 | FuZiR_Cursor | 图像 `stack()` 改 5Hz 重采样；`flip_y` 统一作用于 yaw 与 ego_motion/target_point。 |

| 2026-06-10 | FuZiR_Cursor | `build_ego_motion` 改为 5Hz 世界 xy/yaw 差分，对齐 B2D。 |

| 2026-06-09 | FuZiR_Cursor | 迁移至 MonoDrive 8 帧输入契约。 |



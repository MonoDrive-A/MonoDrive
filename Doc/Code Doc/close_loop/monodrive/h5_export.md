# close_loop/monodrive/h5_export.py

## 1. 文件职责

把 Carla 闭环推理时的模型输入（8 帧图像、`ego_motion`、`target_point`）写成与 `data/b2d_preprocess.py` 相同 schema 的 `b2d_h5_v5` H5，供 `B2DH5Dataset` 与 `visualization/backbone_feature_pca_viewer.py` 直接读取。闭环没有 GT 标签，因此 `future_trajectory`、Agent、Map 等字段以零或无效占位。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `ClosedLoopH5Sample` | dataclass | 单条闭环样本。 |
| `build_closed_loop_h5_sample` | function | 把推理张量打包为样本。 |
| `ClosedLoopH5Exporter` | class | 运行期间累积样本，结束时写出会话级 H5。 |
| `write_closed_loop_h5_file` | function | 把样本列表写成 `b2d_h5_v5` H5。 |
| `dump_openloop_h5_snapshot` | function | 写出单样本 H5（`sample_index=0`）。 |

## 3. 输入输出与 Shape

| 名称 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `frames_past` | `[8, 3, 288, 512] float` | 值域 `[0, 1]`，与 `FrameBuffer.stack()` 一致。 |
| `ego_motion` | `[3]` | `[Vx, Vy, W]`，anchor ego-local。 |
| `target_point` | `[2]` | anchor ego-local 目标点 (m)。 |
| H5 `frames/rgb_front` | `[F, 288, 512, 3] uint8` | 每个样本追加 8 帧。 |
| H5 `labels/ego_motion` | `[S, 3]` | 与 B2D 预处理一致。 |
| H5 `labels/target_point` | `[S, 2]` | 与 B2D 预处理一致。 |

## 4. 关键实现逻辑

- 图像由 float `[0,1]` 量化为 uint8 后写入 `frames/rgb_front`。
- 每个样本使用独立 8 帧切片，`input_frame_indices` 为连续全局索引。
- `target_points[0]` 与 `target_valid[0]=True`，保证 `B2DH5Dataset(random_target_point=False)` 稳定选中闭环目标点。
- attrs 中 `source=carla_closed_loop`，`scene_root=carla_closed_loop`。

## 5. 依赖关系

- `close_loop.monodrive.inputs`（`FINAL_HW`、`PAST_FRAMES` 等常量）
- `h5py`（运行时依赖）
- 下游：`data.b2d_dataset.B2DH5Dataset`、`visualization/backbone_feature_pca_viewer.py`

## 6. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-10 | 1os3_Composer | AI 完成：新增闭环 `b2d_h5_v5` 导出，支持会话级累积与单样本 snapshot。 |

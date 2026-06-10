# close_loop/monodrive/h5_export.py 摘要

## 1. 文件基本功能

`close_loop/monodrive/h5_export.py` 把 Carla 闭环推理输入写成 `b2d_h5_v5` H5，字段布局与 B2D 预处理一致，可直接被 `B2DH5Dataset` 和 `backbone_feature_pca_viewer.py` 读取。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `ClosedLoopH5Exporter` | class | 累积样本并在运行结束时写出会话级 H5。 |
| `write_closed_loop_h5_file` | function | 低层 H5 写入。 |
| `dump_openloop_h5_snapshot` | function | 单样本 H5 snapshot。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 Shape | 输出 |
| --- | --- | --- |
| `ClosedLoopH5Exporter.append_sample` | `images [8,3,288,512]`, `ego_motion [3]`, `target_point [2]` | 内存累积 |
| `write_closed_loop_h5_file` | 样本列表 | `b2d_h5_v5` H5 文件 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `ClosedLoopH5Exporter` | 通过 `MonoDriveAgent(export_h5_path=...)` 或 CLI `--export-h5` 启用；必须在运行结束时调用 `finalize()`。 |
| `dump_openloop_h5_snapshot` | 由 `diagnostic.dump_openloop_snapshot` 在 `--diagnostic-dir` 模式下按 tick 写出；`sample_index` 固定为 0。 |

## 5. 维护说明

- 修改 H5 字段或 attrs 时必须同步检查 `B2DH5Dataset` 与 `backbone_feature_pca_viewer.py`。
- Agent/Map 等无 Carla GT 的字段保持零占位，不要伪造标签。

## 6. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-10 | 1os3_Composer | AI 完成：新增闭环开环兼容 H5 导出模块。 |

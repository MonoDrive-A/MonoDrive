# Carla 闭环推理

## 前置条件

1. 本机已安装 Carla 仿真端（UE4/UE5）与匹配的 Python API（`carla` 包）。
2. MonoDrive 虚拟环境与训练 checkpoint（含 `model_state` 字段）。
3. 项目根目录下执行命令，确保 `model/`、`train/` 与 `close_loop/` 可被导入。

## 启动仿真

```powershell
# Carla 服务器（示例）
cd <Carla安装目录>
./CarlaUE4.sh -RenderOffScreen -nosound -fps=5
```

## 运行闭环

```powershell
cd F:\MonoDrive
.\.venv\Scripts\python.exe -m close_loop.monodrive.run_closed_loop `
    --host 127.0.0.1 --port 2000 `
    --checkpoint .\checkpoints\step_000100.pt `
    --town Town10HD `
    --n-ticks 960 `
    --mp4 .\viz_out\closed_loop.mp4
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--backbone-config` | 主干 TOML，默认 `config/backbone.toml` |
| `--viz-top-k` | 可视化 top-k 候选轨迹，默认 8 |
| `--goal-min-dist-m` | 目标点最小直线距离 (m)，与 B2D 预处理一致，默认 24 |
| `--goal-max-dist-m` | 目标点最大直线距离 (m)，默认 30 |
| `--camera-fov` | 前视水平 FOV (°)，与 B2D 一致，默认 70 |
| `--camera-full-res` | 使用 B2D 采集分辨率 1600×900 |
| `--no-residual` | 关闭轨迹 Tanh 残差修正，仅用词表轨迹 |
| `--legacy-tracking` | 每 tick 推理 + ego-local 预瞄（默认 committed 模式） |
| `--flip-y` | Carla→B2D 坐标对齐：取反 yaw，并修正 `ego_motion` / `target_point`（模型不接收 yaw） |
| `--diagnostic-dir` | 诊断目录：PNG/NPZ + 单样本 `b2d_h5_v5` H5（`snapshot_tickXXXXX.h5`） |
| `--export-h5` | 会话级开环兼容 H5；运行结束后可用 `backbone_feature_pca_viewer --h5` 读取 |

## 开环可视化对接

闭环可导出与 B2D 预处理相同 schema 的 H5（`b2d_h5_v5`），直接喂给开环诊断工具：

```powershell
# 1) 闭环运行时写出会话级 H5
.\.venv\Scripts\python.exe -m close_loop.monodrive.run_closed_loop `
    --checkpoint .\checkpoints\step_00023800.pt `
    --export-h5 .\viz_out\closed_loop_session.h5 `
    --diagnostic-every 4

# 2) 用开环 backbone PCA 工具读取（sample_index 对应第 N 次被导出的推理）
.\.venv\Scripts\python.exe visualization\backbone_feature_pca_viewer.py `
    --h5 .\viz_out\closed_loop_session.h5 `
    --sample-index 0 `
    --checkpoint .\checkpoints\step_00023800.pt
```

说明：

- `--diagnostic-dir` 会额外按 tick 写出 `snapshot_tickXXXXX.h5`（每个文件 1 个样本，`--sample-index 0`）。
- `--export-h5` 在运行结束时写出包含全部采样推理的会话 H5；采样频率由 `--diagnostic-every` 控制（与诊断 dump 同步）。
- 闭环 H5 不含 GT `future_trajectory` / Agent / Map 标签，BEV 面板的 GT 轨迹为空，但模型输入与 PCA 诊断仍可用。

## 输入对齐

闭环 agent 与 [`data/b2d_dataset.py`](../data/b2d_dataset.py) 使用相同约定：

- 图像：最近 **8** 帧、**5Hz** 间隔（跨度约 1.4s），B2D **CAM_FRONT** 对齐——水平 **FOV 70°**，外参 **cam2ego (0.8, 0, 1.6) m**；默认 **800×450** Carla 相机（`--camera-full-res` → 1600×900）→ 288×512，值域 `[0, 1]`。
- `ego_motion`：`[Vx, Vy, W]`，anchor 帧 ego-local 物理量。
- `target_point`：anchor 帧 ego-local 目标点 (m)。
- 轨迹输出：256 路词表 + 残差，6 点 @ 2Hz（相邻点间隔 0.5 s）。

## 目录结构

```
close_loop/
  agents/navigation/   # Carla 官方 navigation  vendored 副本
  agents/tools/
  monodrive/           # MonoDriveAgent 与 CLI
```

## 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-10 | 1os3_Composer | 新增 `--export-h5` 与诊断目录单样本 H5，对接 `backbone_feature_pca_viewer`。 |
| 2026-06-10 | FuZiR_Cursor | Carla 服务端建议 `-fps=5`，与模型 5Hz 输入对齐。 |
| 2026-06-09 | FuZiR_Cursor | 相机 FOV/外参与 B2D CAM_FRONT 对齐（70°，cam2ego 0.8/0/1.6 m）。 |
| 2026-06-09 | FuZiR_Cursor | 自 JEPA 闭环迁移至 MonoDriveBackbone，新增本文档。 |

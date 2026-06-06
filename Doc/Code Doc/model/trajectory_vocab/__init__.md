# model/trajectory_vocab/__init__.py

## 1. 文件职责

`model/trajectory_vocab/__init__.py` 是轨迹词表模型包入口，负责从 `model/trajectory_vocab/trajectory_vocab.py` 重新导出公开接口，保留 `from model.trajectory_vocab import ...` 的调用方式。

该文件不实现词表加载、嵌入、解码或配置解析逻辑。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrajectoryDecoderOutput` | NamedTuple | 轨迹词表解码输出。 |
| `TrajectoryVocabData` | dataclass | 加载后的轨迹词表数据。 |
| `TrajectoryVocabModelConfig` | dataclass | 模型侧轨迹词表配置。 |
| `TrajectoryVocabularyDecoder` | class | 轨迹词表解码层。 |
| `TrajectoryVocabularyEmbedding` | class | 轨迹词表嵌入层。 |
| `load_trajectory_vocab_config` | function | 读取 TOML 配置。 |
| `load_trajectory_vocabulary` | function | 加载 `.npz` 词表。 |

## 3. 关键类和函数

### 包级导出

- 功能：将核心模块公开接口重新导出到 `model.trajectory_vocab` 包级命名空间。
- 输入：无。
- 输出：包级可导入的类和函数。
- Shape：不直接处理张量。
- 关键参数：无。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 包级导出 | 无 | 只转发接口，不改变张量 shape。 |

## 5. 关键实现逻辑

该文件直接从 `.trajectory_vocab` 导入模型侧轨迹词表公开接口，并通过 `__all__` 限定包级导出列表。这样源码文件移动到 `model/trajectory_vocab/trajectory_vocab.py` 后，外部仍可以使用 `from model.trajectory_vocab import TrajectoryVocabularyEmbedding`。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 无 | 无 | 本文件不读取配置。 |

## 7. 依赖关系

- 上游：`model/trajectory_vocab/trajectory_vocab.py`。
- 下游：所有使用包级导入的模型代码或实验脚本。

## 8. 注意事项

- 数值稳定性：本文件不执行数值计算。
- 性能：包导入时会加载核心模块。
- 兼容性：新增或删除核心模块公开接口时，需要同步更新本文件和代码文档。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-06 | 1os3_Codex | AI 完成：新增轨迹词表包级导出入口，保留原包级导入方式。 |

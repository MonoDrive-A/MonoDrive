# model/trajectory_vocab/__init__.py 摘要

## 1. 文件基本功能

作为 `model.trajectory_vocab` 包入口，从 `model/trajectory_vocab/trajectory_vocab.py` 重新导出轨迹词表配置、加载、嵌入和解码公开接口，保留包级导入方式。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `TrajectoryDecoderOutput` | NamedTuple | 轨迹词表解码输出。 |
| `TrajectoryVocabData` | dataclass | 加载后的词表数据。 |
| `TrajectoryVocabModelConfig` | dataclass | 模型侧词表配置。 |
| `TrajectoryVocabularyDecoder` | class | 解码轨迹 logit 和 Tanh 残差。 |
| `TrajectoryVocabularyEmbedding` | class | 生成 384 维轨迹查询。 |
| `load_trajectory_vocab_config` | function | 读取 TOML 配置。 |
| `load_trajectory_vocabulary` | function | 加载 `.npz` 词表。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| 包级导出 | 无 | 该文件只导出接口，不直接处理张量。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| 包级导入 | 可使用 `from model.trajectory_vocab import TrajectoryVocabularyEmbedding`。 |
| 核心模块导入 | 也可使用 `from model.trajectory_vocab.trajectory_vocab import TrajectoryVocabularyEmbedding`。 |

## 5. 最小使用示例

在项目根目录执行：

```python
from model.trajectory_vocab import load_trajectory_vocab_config

config = load_trajectory_vocab_config("config/trajectory_vocab.toml")
```

## 6. 维护注意事项

- 新增或删除 `trajectory_vocab.py` 的公开接口时，需要同步更新本文件的 `__all__`。
- 本文件不应加入配置默认值或模型逻辑。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-06 | 1os3_Codex | AI 完成：新增轨迹词表包入口摘要。 |

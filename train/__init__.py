"""MonoDrive 训练辅助模块。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentMatchingTargets",
    "MapMatchingTargets",
    "TrainingBatchLabels",
    "TrainingDataConfig",
    "ValidatedTrainingDataset",
    "build_agent_matching_targets",
    "build_map_matching_targets",
    "build_training_batch_labels",
    "build_training_dataset",
    "build_trajectory_vocab_labels",
    "load_training_data_config",
    "training_collate",
]

_LAZY_EXPORTS = {
    "AgentMatchingTargets": "train.data_processing",
    "MapMatchingTargets": "train.data_processing",
    "TrainingBatchLabels": "train.data_processing",
    "TrainingDataConfig": "train.data_processing",
    "ValidatedTrainingDataset": "train.data_processing",
    "build_agent_matching_targets": "train.data_processing",
    "build_map_matching_targets": "train.data_processing",
    "build_training_batch_labels": "train.data_processing",
    "build_training_dataset": "train.data_processing",
    "build_trajectory_vocab_labels": "train.data_processing",
    "load_training_data_config": "train.data_processing",
    "training_collate": "train.data_processing",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

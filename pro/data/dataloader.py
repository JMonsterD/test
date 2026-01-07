"""
数据加载封装：返回符合规范的 DataLoader。
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import torch


def _clamp_collate(batch):
    collated = default_collate(batch)
    if isinstance(collated, dict):
        for key in ("Y_lr", "Z_hr", "X_gt"):
            if key in collated and torch.is_tensor(collated[key]):
                collated[key] = collated[key].clamp(0.0, 1.0)
    return collated


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_clamp_collate,
    )

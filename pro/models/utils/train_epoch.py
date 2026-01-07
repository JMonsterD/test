"""
Epoch runner helper (pcDEQ-style training scaffold).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch.nn as nn


def run_epoch(
    loader: Iterable,
    model: nn.Module,
    step_fn: Callable[[object], None],
    opt: Optional[object] = None,
) -> None:
    """
    Run one epoch over loader. If opt is None -> eval, else train.
    """
    if opt is None:
        model.eval()
    else:
        model.train()
    for batch in loader:
        step_fn(batch)

"""
LogBus: single logging sink for stdout + jsonl + meters.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Optional


def _as_number(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if not math.isfinite(float(val)):
            return None
        return float(val)
    if hasattr(val, "detach"):
        try:
            val = val.detach()
        except Exception:
            return None
    if hasattr(val, "mean"):
        try:
            return float(val.mean().item())
        except Exception:
            pass
    if hasattr(val, "item"):
        try:
            return float(val.item())
        except Exception:
            return None
    return None


class MeterStore:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.last: Dict[str, float] = {}

    def update(self, metrics: Dict[str, Any]) -> None:
        for key, val in metrics.items():
            if key in {"type", "epoch", "step", "stage"}:
                continue
            num = _as_number(val)
            if num is None:
                continue
            self.totals[key] = self.totals.get(key, 0.0) + num
            self.counts[key] = self.counts.get(key, 0) + 1
            self.last[key] = num

    def mean(self) -> Dict[str, float]:
        return {k: self.totals[k] / max(1, self.counts.get(k, 1)) for k in self.totals}

    def snapshot(self) -> Dict[str, float]:
        out = self.mean()
        for k, v in self.last.items():
            out[f"{k}_last"] = v
        return out


class LogBus:
    def __init__(self, outdir: str, log_file: str, jsonl: bool = True) -> None:
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.log_path = log_file if os.path.isabs(log_file) else os.path.join(outdir, log_file)
        self.jsonl_path = os.path.splitext(self.log_path)[0] + ".jsonl"
        self.jsonl_enabled = bool(jsonl)
        self.logger = self._init_logger(self.log_path)
        self.meters = {
            "step": MeterStore(),
            "eval": MeterStore(),
            "deq": MeterStore(),
        }

    def _init_logger(self, log_path: str) -> logging.Logger:
        logger = logging.getLogger("logbus")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def write_jsonl(self, record: Dict[str, Any]) -> None:
        if not self.jsonl_enabled:
            return
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def step(self, record: Dict[str, Any], line: Optional[str] = None) -> None:
        if line is not None:
            self.logger.info(line)
        self.meters["step"].update(record)
        self.write_jsonl(record)

    def eval(self, record: Dict[str, Any], line: Optional[str] = None) -> None:
        if line is not None:
            self.logger.info(line)
        self.meters["eval"].update(record)
        self.write_jsonl(record)

    def deq(self, record: Dict[str, Any], line: Optional[str] = None) -> None:
        if line is not None:
            self.logger.info(line)
        self.meters["deq"].update(record)
        self.write_jsonl(record)

    def meter_snapshot(self, kind: str) -> Dict[str, float]:
        store = self.meters.get(kind)
        return store.snapshot() if store is not None else {}


__all__ = ["LogBus", "MeterStore"]

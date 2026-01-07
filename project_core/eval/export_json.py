"""
JSON 导出工具：metrics / solver_stats / theta_stats。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def export_json(records: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

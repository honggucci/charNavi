from __future__ import annotations
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from wpcn._03_common._01_core.types import RunConfig, Theta

def make_run_id(exchange_id: str, symbol: str, timeframe: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_symbol = symbol.replace("/", "-").replace(":", "_")
    return f"{ts}_{exchange_id}_{safe_symbol}_{timeframe}"

def ensure_run_dir(root: str, run_id: str) -> Path:
    p = Path(root) / run_id
    p.mkdir(parents=True, exist_ok=True)
    (p / "charts").mkdir(exist_ok=True)
    (p / "reports").mkdir(exist_ok=True)
    (p / "review_packet" / "prompts").mkdir(parents=True, exist_ok=True)
    return p

def dump_resolved_config(run_dir: Path, resolved: Dict[str, Any]) -> None:
    import yaml
    with open(run_dir / "config.resolved.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index.name = "time"
    out.to_parquet(path, index=True)

def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_review_packet(run_dir: Path, summary_md: str, key_metrics: Dict[str, Any], prompt_text: str) -> None:
    (run_dir / "review_packet" / "summary.md").write_text(summary_md, encoding="utf-8")
    save_json(key_metrics, run_dir / "review_packet" / "key_metrics.json")
    (run_dir / "review_packet" / "prompts" / "gemini_review_prompt.txt").write_text(prompt_text, encoding="utf-8")

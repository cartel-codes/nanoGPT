"""
Train Logger for nanoGPT UI
-----------------------------
Reads and parses the newline-delimited metrics.json file written by train.py.
Provides helpers for the Streamlit dashboard to consume live training data.

Log format (one JSON object per line):
    {"step": N, "train_loss": X, "val_loss": Y, "lr": W, "mfu": Z, "timestamp": T}
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


DEFAULT_LOG_PATH = "logs/metrics.json"


def load_metrics(log_path: str = DEFAULT_LOG_PATH) -> list[dict]:
    """
    Read all metrics entries from *log_path*.

    Returns:
        List of metric dicts, ordered by step (ascending). Empty list if file
        does not exist yet.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines

    # Sort by step in case entries are out of order
    entries.sort(key=lambda e: e.get("step", 0))
    return entries


def get_latest_metrics(log_path: str = DEFAULT_LOG_PATH) -> Optional[dict]:
    """
    Return only the most recent metrics entry, or None if no data yet.
    """
    entries = load_metrics(log_path)
    return entries[-1] if entries else None


def get_metric_series(
    log_path: str = DEFAULT_LOG_PATH,
    key: str = "val_loss",
) -> tuple[list[int], list[float]]:
    """
    Extract a single metric series as (steps, values) lists for plotting.

    Args:
        log_path: Path to the metrics JSON file.
        key:      Metric key to extract, e.g. "train_loss", "val_loss", "lr", "mfu".

    Returns:
        (steps, values) — two parallel lists of equal length.
        Returns ([], []) if no data.
    """
    entries = load_metrics(log_path)
    steps = []
    values = []
    for e in entries:
        if key in e and e[key] is not None:
            steps.append(e["step"])
            values.append(e[key])
    return steps, values


def get_all_series(log_path: str = DEFAULT_LOG_PATH) -> dict[str, Any]:
    """
    Return all series in a single pass as a dict of lists, convenient for
    building Plotly figures.

    Returns:
        {
          "steps":       [int, ...],
          "train_loss":  [float, ...],
          "val_loss":    [float, ...],
          "lr":          [float, ...],
          "mfu":         [float, ...],
          "timestamps":  [float, ...],
        }
    """
    entries = load_metrics(log_path)
    result: dict[str, list] = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "mfu": [],
        "timestamps": [],
    }
    for e in entries:
        result["steps"].append(e.get("step", 0))
        result["train_loss"].append(e.get("train_loss"))
        result["val_loss"].append(e.get("val_loss"))
        result["lr"].append(e.get("lr"))
        result["mfu"].append(e.get("mfu"))
        result["timestamps"].append(e.get("timestamp"))
    return result


def get_log_file_mtime(log_path: str = DEFAULT_LOG_PATH) -> Optional[float]:
    """
    Return the last modification timestamp of the log file, or None if missing.
    Useful for detecting whether new data has been written since last read.
    """
    path = Path(log_path)
    if path.exists():
        return os.path.getmtime(path)
    return None


def describe_training(log_path: str = DEFAULT_LOG_PATH) -> dict:
    """
    Summarize the training run captured in *log_path*.

    Returns a dict with:
        total_steps, best_val_loss, best_val_step,
        final_train_loss, final_val_loss,
        duration_seconds (wall-clock from first to last entry),
        num_entries.
    """
    entries = load_metrics(log_path)
    if not entries:
        return {}

    best_val = min(entries, key=lambda e: e.get("val_loss", float("inf")))
    first_ts = entries[0].get("timestamp")
    last_ts = entries[-1].get("timestamp")
    duration = (last_ts - first_ts) if (first_ts and last_ts) else None

    return {
        "num_entries": len(entries),
        "total_steps": entries[-1].get("step"),
        "best_val_loss": best_val.get("val_loss"),
        "best_val_step": best_val.get("step"),
        "final_train_loss": entries[-1].get("train_loss"),
        "final_val_loss": entries[-1].get("val_loss"),
        "duration_seconds": round(duration, 1) if duration else None,
    }

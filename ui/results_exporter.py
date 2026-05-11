"""
Results Exporter for nanoGPT
------------------------------
Computes final validation perplexity, collects training stats from a
checkpoint, and exports them to JSON, CSV, or a Markdown summary report.

CLI usage:
    # Export to JSON (default)
    python ui/results_exporter.py --checkpoint out/ckpt.pt

    # Export to CSV
    python ui/results_exporter.py --checkpoint out/ckpt.pt --format csv --output results.csv

    # Export Markdown report
    python ui/results_exporter.py --checkpoint out/ckpt.pt --format markdown --output report.md

    # Include metrics log for full training history summary
    python ui/results_exporter.py --checkpoint out/ckpt.pt --log logs/metrics.json --format markdown
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_perplexity(val_loss: float) -> float:
    """
    Compute perplexity from cross-entropy validation loss.
    perplexity = e^(val_loss)
    """
    return math.exp(val_loss)


def collect_stats(checkpoint_path: str, log_path: str | None = None) -> dict:
    """
    Build a comprehensive stats dict for *checkpoint_path*.

    Loads only metadata from the checkpoint (model weights are not moved to GPU).
    Optionally reads training history from *log_path*.

    Returns:
        dict with keys: checkpoint, model_args, training, metrics_summary
    """
    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # ----- Model metadata -----
    model_args = ckpt.get("model_args", {})
    iter_num = ckpt.get("iter_num", 0)
    best_val_loss = ckpt.get("best_val_loss", None)
    config = ckpt.get("config", {})

    # Estimate parameter count without allocating full model on GPU
    try:
        from model import GPT, GPTConfig
        gptconf = GPTConfig(**model_args)
        dummy = GPT(gptconf)
        num_params = dummy.get_num_params()
        del dummy
    except Exception:
        num_params = None

    # ----- Checkpoint file info -----
    file_size_bytes = ckpt_path.stat().st_size
    file_mtime = datetime.fromtimestamp(ckpt_path.stat().st_mtime, tz=timezone.utc).isoformat()

    stats = {
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(ckpt_path),
            "file_size_mb": round(file_size_bytes / 1_048_576, 2),
            "last_modified": file_mtime,
        },
        "model": {
            "num_params": num_params,
            "num_params_millions": round(num_params / 1e6, 2) if num_params else None,
            **model_args,
        },
        "training": {
            "iter_num": iter_num,
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
            "best_val_perplexity": compute_perplexity(float(best_val_loss)) if best_val_loss else None,
            "dataset": config.get("dataset"),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "max_iters": config.get("max_iters"),
            "block_size": config.get("block_size"),
        },
    }

    # ----- Optional: metrics log summary -----
    if log_path:
        try:
            from ui.train_logger import describe_training
            summary = describe_training(log_path)
            stats["metrics_summary"] = summary
        except Exception as e:
            stats["metrics_summary"] = {"error": str(e)}

    return stats


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_json(stats: dict, output_path: str):
    """Write stats dict as pretty-printed JSON."""
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ JSON exported → {output_path}")


def export_csv(stats: dict, output_path: str):
    """Flatten stats dict into a single-row CSV."""
    flat: dict = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, prefix=f"{key}.")
            else:
                flat[key] = v

    _flatten(stats)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    print(f"✓ CSV exported → {output_path}")


def export_markdown(stats: dict, output_path: str):
    """Generate a human-readable Markdown summary report."""
    m = stats.get("model", {})
    t = stats.get("training", {})
    ckpt = stats.get("checkpoint", {})
    ms = stats.get("metrics_summary", {})

    lines = [
        "# nanoGPT Training Results Report",
        "",
        f"*Generated: {stats.get('exported_at', '')}*",
        "",
        "---",
        "",
        "## Checkpoint",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Path | `{ckpt.get('path')}` |",
        f"| File Size | {ckpt.get('file_size_mb')} MB |",
        f"| Last Modified | {ckpt.get('last_modified')} |",
        "",
        "## Model Architecture",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Parameters | {m.get('num_params_millions')}M ({m.get('num_params'):,} total) |" if m.get('num_params') else "| Parameters | N/A |",
        f"| Layers | {m.get('n_layer')} |",
        f"| Attention Heads | {m.get('n_head')} |",
        f"| Embedding Dim | {m.get('n_embd')} |",
        f"| Block Size | {m.get('block_size')} |",
        f"| Vocab Size | {m.get('vocab_size')} |",
        f"| Bias | {m.get('bias')} |",
        "",
        "## Training Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Dataset | {t.get('dataset')} |",
        f"| Training Steps | {t.get('iter_num'):,} |" if t.get('iter_num') else "| Training Steps | N/A |",
        f"| Max Iters | {t.get('max_iters'):,} |" if t.get('max_iters') else "| Max Iters | N/A |",
        f"| Batch Size | {t.get('batch_size')} |",
        f"| Learning Rate | {t.get('learning_rate')} |",
        f"| Best Val Loss | **{t.get('best_val_loss'):.4f}** |" if t.get('best_val_loss') else "| Best Val Loss | N/A |",
        f"| Best Val Perplexity | **{t.get('best_val_perplexity'):.2f}** |" if t.get('best_val_perplexity') else "| Best Val Perplexity | N/A |",
    ]

    if ms and "error" not in ms:
        lines += [
            "",
            "## Metrics Log Summary",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Eval Steps Logged | {ms.get('num_entries')} |",
            f"| Total Steps | {ms.get('total_steps'):,} |" if ms.get('total_steps') else "",
            f"| Best Val Loss | {ms.get('best_val_loss'):.4f} @ step {ms.get('best_val_step')} |" if ms.get('best_val_loss') else "",
            f"| Final Train Loss | {ms.get('final_train_loss'):.4f} |" if ms.get('final_train_loss') else "",
            f"| Duration | {ms.get('duration_seconds')}s ({round(ms.get('duration_seconds', 0)/3600, 2)}h) |" if ms.get('duration_seconds') else "",
        ]

    lines += ["", "---", "*Generated by nanoGPT Results Exporter*", ""]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Markdown report exported → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export nanoGPT training results to JSON, CSV, or Markdown."
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to the checkpoint .pt file (e.g. out/ckpt.pt)",
    )
    parser.add_argument(
        "--log", "-l",
        default=None,
        help="Optional path to metrics.json log for training history (e.g. logs/metrics.json)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format: json | csv | markdown  (default: json)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path. Defaults to results.json / results.csv / results.md",
    )
    args = parser.parse_args()

    # Default output filename
    if args.output is None:
        ext_map = {"json": "json", "csv": "csv", "markdown": "md"}
        args.output = f"results.{ext_map[args.format]}"

    print(f"Collecting stats from: {args.checkpoint}")
    stats = collect_stats(args.checkpoint, log_path=args.log)

    if args.format == "json":
        export_json(stats, args.output)
    elif args.format == "csv":
        export_csv(stats, args.output)
    elif args.format == "markdown":
        export_markdown(stats, args.output)

    # Always print a quick summary to stdout
    t = stats.get("training", {})
    m = stats.get("model", {})
    print()
    print("=== Quick Summary ===")
    if m.get("num_params_millions"):
        print(f"  Model:          {m['num_params_millions']}M parameters")
    if t.get("iter_num") is not None:
        print(f"  Steps trained:  {t['iter_num']:,}")
    if t.get("best_val_loss") is not None:
        print(f"  Best val loss:  {t['best_val_loss']:.4f}")
    if t.get("best_val_perplexity") is not None:
        print(f"  Perplexity:     {t['best_val_perplexity']:.2f}")


if __name__ == "__main__":
    main()

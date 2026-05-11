# nanoGPT UI — Usage Guide

> **Prerequisites**: Install dependencies before running any UI app.
> ```bash
> pip install gradio streamlit plotly tiktoken
> ```

---

## Quick Start

```bash
# From the nanoGPT root directory:
cd nanoGPT

# 1. Text generation (Gradio)
python ui/app_inference.py
#  → http://localhost:7860

# 2. Training dashboard (Streamlit)
streamlit run ui/app_dashboard.py
#  → http://localhost:8501

# 3. Model comparison (Streamlit)
streamlit run ui/app_compare.py
#  → http://localhost:8501

# 4. Export results
python ui/results_exporter.py --checkpoint out/ckpt.pt
python ui/results_exporter.py --checkpoint out/ckpt.pt --format markdown --output report.md
python ui/results_exporter.py --checkpoint out/ckpt.pt --log logs/metrics.json --format markdown
```

---

## File Overview

| File | Purpose | URL |
|---|---|---|
| `ui/app_inference.py` | Gradio inference UI | `localhost:7860` |
| `ui/app_dashboard.py` | Streamlit training dashboard | `localhost:8501` |
| `ui/app_compare.py` | Streamlit comparison tool | `localhost:8501` |
| `ui/results_exporter.py` | CLI results export | — |
| `ui/checkpoint_manager.py` | Load/cache checkpoints | — |
| `ui/train_logger.py` | Read metrics.json | — |

---

## Metrics Logging

`train.py` now automatically writes to `logs/metrics.json` at every eval step.
Each line is a JSON object:

```json
{"step": 2000, "train_loss": 3.12, "val_loss": 3.25, "lr": 0.0006, "mfu": 12.4, "timestamp": 1715000000.0}
```

The dashboard reads this file live and auto-refreshes every 10 seconds.

---

## Inference UI (`app_inference.py`)

- **Checkpoint selector** — auto-populated from `out/*.pt` + GPT-2 variants
- **Temperature** (0.1–2.0): lower = more focused, higher = more creative
- **Top-K** (1–500): sample from the top K most likely tokens
- **Max new tokens**: how many tokens to generate per sample
- **Num samples**: generate multiple independent continuations
- **Copy button**: copy generated text to clipboard

No GPU required — runs on CPU automatically.

---

## Training Dashboard (`app_dashboard.py`)

Shows live metrics while training is running:

- **Metric cards**: step, train loss, val loss, learning rate, MFU
- **Loss curves**: train vs val loss over steps
- **LR schedule**: cosine warmup/decay visualization
- **MFU trend**: model efficiency percentage
- **Perplexity**: val perplexity (= e^val_loss)
- **Summary table**: best val loss, duration, final losses
- **Auto-refresh**: configurable 5–60 second refresh interval

---

## Model Comparison (`app_compare.py`)

- Select 2–4 checkpoints from the sidebar
- Enter a shared prompt
- All models generate simultaneously
- Side-by-side output columns with timing stats
- Model info cards: params, steps, val loss, architecture
- **Export** button: saves full comparison to `comparison_TIMESTAMP.txt`

---

## Results Exporter (`results_exporter.py`)

```bash
# JSON output (default)
python ui/results_exporter.py -c out/ckpt.pt -o results.json

# CSV (single-row, flat)
python ui/results_exporter.py -c out/ckpt.pt -f csv -o results.csv

# Markdown report with training history
python ui/results_exporter.py -c out/ckpt.pt -l logs/metrics.json -f markdown -o report.md
```

Exported fields: model architecture, parameter count, training steps,
best val loss, perplexity, dataset, batch size, learning rate, and
(if `--log` provided) full training history summary.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NANOGPT_OUT_DIR` | `out` | Directory scanned for `.pt` checkpoints |
| `NANOGPT_DEVICE` | auto | Device for inference (`cpu`, `cuda`, `cuda:0`) |

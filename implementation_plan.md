# nanoGPT UI & Results Measurement System

## Overview

Build a full-featured web UI ecosystem for nanoGPT that covers:
- **Real-time training monitoring** (loss, LR, MFU dashboards)
- **Inference interface** (text generation with model checkpoint selection)
- **Model comparison tool** (side-by-side checkpoint outputs)
- **Results exporter** (perplexity, stats, markdown/JSON reports)

This requires modifying `train.py` to emit metrics, plus creating a `ui/` directory with 6 new Python modules.

---

## User Review Required

> [!IMPORTANT]
> **Dependency install required** — The following packages will be needed:
> ```bash
> pip install gradio streamlit plotly tiktoken
> ```
> These are not installed automatically by this plan. You'll need to run this before starting the UIs.

> [!WARNING]
> **train.py will be modified** — A metrics logging hook will be added to the eval block. It is backward-compatible (new code only runs on `master_process` and only appends a `logs/metrics.json` file). Existing behavior is unaffected.

> [!NOTE]
> **No GPU required** for UI apps — The inference app works on CPU (`device=cpu`). For CUDA users it will auto-detect and use GPU. The dashboard/compare tools only read checkpoint files + metrics JSON.

---

## Open Questions

> [!IMPORTANT]
> **Checkpoint location**: The plan assumes checkpoints live in `out/ckpt.pt` (nanoGPT default). Do you want support for multiple named checkpoint files in `out/` (e.g. `out/ckpt_step1000.pt`, `out/ckpt_best.pt`)?

> [!NOTE]
> **Gradio vs pure HTML**: The inference UI uses Gradio (Python web framework). This creates a clean web UI with ~50 lines. Do you want a pure HTML/JS alternative instead, or is Gradio acceptable?

---

## Proposed Changes

### Component 1: `train.py` — Metrics Logging Hook

#### [MODIFY] [train.py](file:///home/aghilas/Desktop/test/test/nanoGPT/train.py)

At the top, add `import json` and a `logs/` directory setup. Inside the eval block (line ~263-288), after printing losses, append a metrics entry to `logs/metrics.json`:

```python
# New: log metrics to JSON
if master_process:
    os.makedirs('logs', exist_ok=True)
    metrics_entry = {
        "step": iter_num,
        "train_loss": losses['train'].item(),
        "val_loss": losses['val'].item(),
        "lr": lr,
        "mfu": running_mfu * 100,
        "timestamp": time.time()
    }
    with open(os.path.join('logs', 'metrics.json'), 'a') as f:
        f.write(json.dumps(metrics_entry) + '\n')
```

---

### Component 2: `ui/` Directory (6 new files)

#### [NEW] `ui/__init__.py`
Empty init to make `ui` a Python package.

---

#### [NEW] [checkpoint_manager.py](file:///home/aghilas/Desktop/test/test/nanoGPT/ui/checkpoint_manager.py)

Responsibilities:
- Scan `out/` directory for `*.pt` files
- Lazy-load and cache models in a global dict
- Return model + encode/decode functions
- Support both custom checkpoints and `gpt2*` pretrained variants

Key functions:
- `list_checkpoints(out_dir)` → list of checkpoint paths
- `load_model(checkpoint_path, device)` → `(model, encode_fn, decode_fn, meta)`
- `get_model_info(checkpoint_path)` → dict with params, steps, val_loss, config

---

#### [NEW] [train_logger.py](file:///home/aghilas/Desktop/test/test/nanoGPT/ui/train_logger.py)

Responsibilities:
- Read `logs/metrics.json` (newline-delimited JSON)
- Parse into pandas-friendly dicts
- Return latest metrics for dashboard header
- Support incremental reads (only new lines since last read)

Key functions:
- `load_metrics(log_path)` → list of metric dicts
- `get_latest_metrics(log_path)` → single latest entry dict
- `get_metric_series(log_path, key)` → `(steps[], values[])` for plotting

---

#### [NEW] [results_exporter.py](file:///home/aghilas/Desktop/test/test/nanoGPT/ui/results_exporter.py)

Responsibilities:
- Load a checkpoint and compute validation perplexity
- Collect training stats from checkpoint metadata
- Export to JSON or CSV
- Generate markdown summary report

CLI usage:
```bash
python ui/results_exporter.py --checkpoint out/ckpt.pt --output results.json
python ui/results_exporter.py --checkpoint out/ckpt.pt --format markdown --output report.md
```

Key functions:
- `compute_perplexity(val_loss)` → float
- `collect_stats(checkpoint_path)` → dict
- `export_results(checkpoint_path, output_path, format)` → None

---

#### [NEW] [app_inference.py](file:///home/aghilas/Desktop/test/test/nanoGPT/ui/app_inference.py) — Gradio

**Gradio web UI** for text generation:

| Input | Control |
|---|---|
| Prompt text | Textarea |
| Checkpoint | Dropdown (auto-populated from `out/`) |
| Temperature | Slider (0.1 → 2.0, default 0.8) |
| Top-K | Slider (1 → 500, default 200) |
| Max tokens | Number input (default 200) |
| Num samples | Number input (default 1) |

Outputs: Generated text, generation time (ms), token count

Starts server at `http://localhost:7860`

---

#### [NEW] [app_dashboard.py](file:///home/aghilas/Desktop/test/test/nanoGPT/ui/app_dashboard.py) — Streamlit

**Streamlit real-time training dashboard**:

Sections:
1. **Header metrics row** — Current step, train loss, val loss, LR, MFU (live cards)
2. **Loss curves** — Plotly line chart: train_loss + val_loss over steps
3. **Learning rate schedule** — Plotly line chart
4. **MFU trend** — Plotly line chart (model efficiency %)
5. **Checkpoint info** — Best val loss, checkpoint path, last saved time
6. **Auto-refresh** — `st.rerun()` timer (every 10 seconds during active training)

Reads from `logs/metrics.json`. Falls back gracefully if file doesn't exist yet.

---

#### [NEW] [app_compare.py](file:///home/aghilas/Desktop/test/test/nanoGPT/ui/app_compare.py) — Streamlit

**Streamlit model comparison tool**:

Layout:
- Shared prompt input at top
- Side-by-side columns per checkpoint (up to 4)
- Each column shows: generated text, model config (layers, heads, params), best val loss, checkpoint size
- "Export Comparison" button → saves to `comparison_TIMESTAMP.txt`
- Temperature/top-k controls apply to all models simultaneously

---

### Component 3: Documentation

#### [NEW] [README_UI.md](file:///home/aghilas/Desktop/test/test/nanoGPT/README_UI.md)

Quick-start guide covering:
- Installation (`pip install ...`)
- How to enable metrics logging (just run train.py normally)
- How to launch each app
- Screenshots (post-build)

---

## Verification Plan

### Automated Tests
- Run `python -c "from ui.checkpoint_manager import list_checkpoints; print(list_checkpoints('out'))"` to verify imports work
- Run `python ui/results_exporter.py --help` to verify CLI is valid
- Smoke-test Gradio app starts: `timeout 10 python ui/app_inference.py` (check no import errors)
- Smoke-test Streamlit app starts: `timeout 10 streamlit run ui/app_dashboard.py --server.headless true`

### Manual Verification
- Open browser to `http://localhost:7860` and verify Gradio UI renders
- Open browser to `http://localhost:8501` and verify Streamlit dashboard renders (with or without a live training run)
- Verify `logs/metrics.json` is created during a short training run (if GPU available)
- Verify `results_exporter.py` produces valid JSON/markdown

---

## File Structure After Changes

```
nanoGPT/
├── ui/
│   ├── __init__.py
│   ├── checkpoint_manager.py
│   ├── train_logger.py
│   ├── results_exporter.py
│   ├── app_inference.py        # gradio: http://localhost:7860
│   ├── app_dashboard.py        # streamlit: http://localhost:8501
│   └── app_compare.py          # streamlit: http://localhost:8501
├── logs/
│   └── metrics.json            # auto-created by train.py
├── train.py                    # MODIFIED: +metrics logging
└── README_UI.md                # new usage guide
```

## Quick Start (After Implementation)

```bash
# Install deps
pip install gradio streamlit plotly tiktoken

# Inference UI
cd nanoGPT && python ui/app_inference.py

# Training dashboard (run while training or after)
cd nanoGPT && streamlit run ui/app_dashboard.py

# Model comparison
cd nanoGPT && streamlit run ui/app_compare.py

# Export results
cd nanoGPT && python ui/results_exporter.py --checkpoint out/ckpt.pt --output results.json
```

"""
Streamlit Training Dashboard for nanoGPT
-----------------------------------------
Real-time monitoring of training progress.

Run:
    cd nanoGPT
    streamlit run ui/app_dashboard.py

Reads logs/metrics.json (written by the modified train.py).
Auto-refreshes every 10 seconds while training is active.
"""

import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.chdir(_ROOT)

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.train_logger import get_all_series, get_latest_metrics, describe_training
from ui.checkpoint_manager import list_checkpoints, get_model_info

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="nanoGPT Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOG_PATH = "logs/metrics.json"
OUT_DIR = "out"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    log_path = st.text_input("Metrics log path", value=LOG_PATH)
    out_dir = st.text_input("Checkpoint directory", value=OUT_DIR)
    refresh_secs = st.slider("Auto-refresh (seconds)", 5, 60, 10)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    st.divider()

    st.markdown("### Available Checkpoints")
    ckpts = list_checkpoints(out_dir)
    file_ckpts = [c for c in ckpts if c.endswith(".pt")]
    if file_ckpts:
        for c in file_ckpts:
            try:
                info = get_model_info(c)
                bvl = info.get("best_val_loss")
                step = info.get("iter_num")
                label = Path(c).name
                bvl_str = f"val={bvl:.4f}" if bvl is not None else "val=?"
                step_str = f"step={step:,}" if step is not None else ""
                st.markdown(f"- **{label}** — {bvl_str} {step_str}")
            except Exception:
                st.markdown(f"- {Path(c).name}")
    else:
        st.markdown("*No checkpoints found in `out/`*")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 nanoGPT Training Dashboard")
st.caption("Real-time monitoring of training loss, learning rate, and MFU.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
series = get_all_series(log_path)
latest = get_latest_metrics(log_path)
summary = describe_training(log_path)
has_data = bool(series["steps"])

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

def _fmt(val, fmt=".4f", fallback="—"):
    return format(val, fmt) if val is not None else fallback

with c1:
    st.metric("📍 Current Step",
              f"{latest['step']:,}" if latest else "—",
              delta=None)
with c2:
    st.metric("📉 Train Loss",
              _fmt(latest.get("train_loss") if latest else None))
with c3:
    st.metric("✅ Val Loss",
              _fmt(latest.get("val_loss") if latest else None))
with c4:
    lr_val = latest.get("lr") if latest else None
    st.metric("📈 Learning Rate",
              f"{lr_val:.2e}" if lr_val else "—")
with c5:
    mfu_val = latest.get("mfu") if latest else None
    st.metric("⚡ MFU",
              f"{mfu_val:.2f}%" if mfu_val and mfu_val > 0 else "—")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
if not has_data:
    st.info(
        "⏳ No training data yet. Start training with the modified `train.py` "
        "and data will appear here automatically.",
        icon="⏳",
    )
else:
    PLOT_HEIGHT = 320
    COLORS = {
        "train": "#7c6af7",
        "val":   "#22d3ee",
        "lr":    "#f59e0b",
        "mfu":   "#34d399",
    }

    # ---- Row 1: Loss curves ----
    col_loss, col_lr = st.columns(2)

    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=series["steps"], y=series["train_loss"],
            name="Train Loss", line=dict(color=COLORS["train"], width=2),
            mode="lines",
        ))
        fig_loss.add_trace(go.Scatter(
            x=series["steps"], y=series["val_loss"],
            name="Val Loss", line=dict(color=COLORS["val"], width=2, dash="dot"),
            mode="lines",
        ))
        fig_loss.update_layout(
            title="Loss Curves",
            xaxis_title="Step", yaxis_title="Loss",
            height=PLOT_HEIGHT,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig_loss.update_xaxes(gridcolor="#1e293b")
        fig_loss.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_lr:
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(
            x=series["steps"], y=series["lr"],
            name="Learning Rate",
            line=dict(color=COLORS["lr"], width=2),
            mode="lines", fill="tozeroy",
            fillcolor="rgba(245,158,11,0.1)",
        ))
        fig_lr.update_layout(
            title="Learning Rate Schedule",
            xaxis_title="Step", yaxis_title="LR",
            height=PLOT_HEIGHT,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig_lr.update_xaxes(gridcolor="#1e293b")
        fig_lr.update_yaxes(gridcolor="#1e293b", tickformat=".2e")
        st.plotly_chart(fig_lr, use_container_width=True)

    # ---- Row 2: MFU + Perplexity ----
    col_mfu, col_ppl = st.columns(2)

    with col_mfu:
        mfu_vals = [v for v in series["mfu"] if v is not None and v > 0]
        mfu_steps = [series["steps"][i] for i, v in enumerate(series["mfu"])
                     if v is not None and v > 0]
        fig_mfu = go.Figure()
        fig_mfu.add_trace(go.Scatter(
            x=mfu_steps, y=mfu_vals,
            name="MFU %",
            line=dict(color=COLORS["mfu"], width=2),
            mode="lines", fill="tozeroy",
            fillcolor="rgba(52,211,153,0.1)",
        ))
        fig_mfu.update_layout(
            title="Model FLOPs Utilization (MFU %)",
            xaxis_title="Step", yaxis_title="MFU %",
            height=PLOT_HEIGHT,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig_mfu.update_xaxes(gridcolor="#1e293b")
        fig_mfu.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_mfu, use_container_width=True)

    with col_ppl:
        ppl_vals = [math.exp(v) for v in series["val_loss"] if v is not None]
        ppl_steps = [series["steps"][i] for i, v in enumerate(series["val_loss"])
                     if v is not None]
        fig_ppl = go.Figure()
        fig_ppl.add_trace(go.Scatter(
            x=ppl_steps, y=ppl_vals,
            name="Perplexity",
            line=dict(color="#f472b6", width=2),
            mode="lines",
        ))
        fig_ppl.update_layout(
            title="Validation Perplexity",
            xaxis_title="Step", yaxis_title="Perplexity",
            height=PLOT_HEIGHT,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig_ppl.update_xaxes(gridcolor="#1e293b")
        fig_ppl.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_ppl, use_container_width=True)

    # ---- Summary table ----
    if summary:
        st.divider()
        st.subheader("📋 Training Summary")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.metric("Eval Steps Logged", summary.get("num_entries", "—"))
        with s_col2:
            bvl = summary.get("best_val_loss")
            bvs = summary.get("best_val_step")
            st.metric("Best Val Loss", f"{bvl:.4f}" if bvl else "—",
                      delta=f"@ step {bvs:,}" if bvs else None, delta_color="off")
        with s_col3:
            dur = summary.get("duration_seconds")
            st.metric("Training Duration",
                      f"{dur/3600:.1f}h" if dur else "—")
        with s_col4:
            ftl = summary.get("final_train_loss")
            st.metric("Final Train Loss", f"{ftl:.4f}" if ftl else "—")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()

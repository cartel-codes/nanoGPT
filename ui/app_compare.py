"""
Streamlit Model Comparison Tool for nanoGPT
---------------------------------------------
Load 2-4 checkpoints side-by-side, generate from the same prompt,
and compare outputs with model stats.

Run:
    cd nanoGPT
    streamlit run ui/app_compare.py
"""

import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.chdir(_ROOT)

import torch
import streamlit as st

from ui.checkpoint_manager import list_checkpoints, load_model, get_model_info

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="nanoGPT — Model Comparison",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUT_DIR = "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Comparison Settings")

    all_ckpts = list_checkpoints(OUT_DIR)
    num_models = st.slider("Number of models to compare", 2, 4, 2)

    selected_ckpts = []
    for i in range(num_models):
        default = all_ckpts[i] if i < len(all_ckpts) else all_ckpts[0] if all_ckpts else None
        ckpt = st.selectbox(
            f"Model {i+1}",
            options=all_ckpts,
            index=i if i < len(all_ckpts) else 0,
            key=f"ckpt_{i}",
        )
        selected_ckpts.append(ckpt)

    st.divider()
    st.markdown("### Generation Settings")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05, key="cmp_temp")
    top_k = st.slider("Top-K", 1, 500, 200, key="cmp_topk")
    max_tokens = st.slider("Max New Tokens", 10, 1000, 200, key="cmp_maxtokens")
    seed = st.number_input("Random Seed", value=42, min_value=0, key="cmp_seed")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚖️ nanoGPT — Model Comparison")
st.caption(
    "Enter a prompt below and compare outputs from multiple checkpoints side-by-side."
)

# ---------------------------------------------------------------------------
# Prompt input
# ---------------------------------------------------------------------------
prompt = st.text_area(
    "Shared Prompt",
    value="Once upon a time",
    height=100,
    placeholder="Enter your prompt here...",
    key="cmp_prompt",
)

col_gen, col_clr = st.columns([3, 1])
with col_gen:
    run_btn = st.button("🚀 Generate from All Models", type="primary", use_container_width=True)
with col_clr:
    clear_btn = st.button("🗑 Clear Results", use_container_width=True)

if clear_btn:
    if "cmp_results" in st.session_state:
        del st.session_state["cmp_results"]

# ---------------------------------------------------------------------------
# Model info cards (always visible)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📊 Model Info")

info_cols = st.columns(num_models)
for i, (col, ckpt) in enumerate(zip(info_cols, selected_ckpts)):
    with col:
        try:
            info = get_model_info(ckpt)
        except Exception as e:
            st.error(f"Cannot load info: {e}")
            continue

        label = Path(ckpt).name if ckpt.endswith(".pt") else ckpt
        st.markdown(f"**Model {i+1}: `{label}`**")

        if info.get("type") == "pretrained":
            st.markdown(f"- Type: OpenAI pretrained")
        else:
            st.markdown(f"- File: {info.get('file_size_mb')} MB")

        if info.get("num_params"):
            st.markdown(f"- Params: {info['num_params']/1e6:.1f}M")
        if info.get("iter_num") is not None:
            st.markdown(f"- Steps: {info['iter_num']:,}")
        if info.get("best_val_loss") is not None:
            bvl = info["best_val_loss"]
            st.markdown(
                f"- Best val loss: `{bvl:.4f}` "
                f"(ppl ≈ {math.exp(bvl):.1f})"
            )
        ma = info.get("model_args", {})
        if ma:
            st.markdown(
                f"- Arch: {ma.get('n_layer')}L · "
                f"{ma.get('n_head')}H · "
                f"embd={ma.get('n_embd')}"
            )

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
if run_btn:
    torch.manual_seed(seed)

    results = []
    progress = st.progress(0, text="Loading models and generating...")

    for i, ckpt in enumerate(selected_ckpts):
        progress.progress((i) / num_models, text=f"Generating from model {i+1}/{num_models}...")
        try:
            model, encode_fn, decode_fn, info, ctx = load_model(ckpt, device=DEVICE)
            ids = encode_fn(prompt.strip() or "\n")
            x = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None, ...]

            t0 = time.time()
            with torch.no_grad():
                with ctx:
                    y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)
            elapsed_ms = (time.time() - t0) * 1000

            generated = decode_fn(y[0].tolist())
            results.append({
                "ckpt": ckpt,
                "label": Path(ckpt).name if ckpt.endswith(".pt") else ckpt,
                "text": generated,
                "elapsed_ms": elapsed_ms,
                "tokens": max_tokens,
                "error": None,
            })
        except Exception as e:
            results.append({
                "ckpt": ckpt,
                "label": Path(ckpt).name if ckpt.endswith(".pt") else ckpt,
                "text": "",
                "elapsed_ms": 0,
                "tokens": 0,
                "error": str(e),
            })

    progress.progress(1.0, text="Done!")
    st.session_state["cmp_results"] = results

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "cmp_results" in st.session_state:
    results = st.session_state["cmp_results"]
    st.divider()
    st.subheader("📝 Generated Outputs")

    out_cols = st.columns(len(results))
    for col, res in zip(out_cols, results):
        with col:
            st.markdown(f"**{res['label']}**")
            if res["error"]:
                st.error(f"❌ {res['error']}")
            else:
                st.text_area(
                    label=f"Output ({res['elapsed_ms']:.0f}ms)",
                    value=res["text"],
                    height=400,
                    key=f"out_{res['label']}_{id(res)}",
                )
                st.caption(f"⏱ {res['elapsed_ms']:.0f}ms · {res['tokens']} tokens")

    # ---- Export button ----
    st.divider()
    if st.button("💾 Export Comparison to Text File"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"comparison_{ts}.txt"
        lines = [
            f"nanoGPT Model Comparison — {ts}",
            f"Prompt: {prompt!r}",
            f"Temperature: {temperature}  Top-K: {top_k}  Max tokens: {max_tokens}",
            "=" * 70,
            "",
        ]
        for res in results:
            lines += [
                f"=== {res['label']} ===",
                f"Generation time: {res['elapsed_ms']:.0f}ms",
                "",
                res["text"] if not res["error"] else f"ERROR: {res['error']}",
                "",
                "-" * 70,
                "",
            ]
        with open(out_file, "w") as f:
            f.write("\n".join(lines))
        st.success(f"✓ Saved to `{out_file}`")

"""
Gradio Inference UI for nanoGPT
---------------------------------
Interactive web UI for text generation.

Run:
    cd nanoGPT
    python ui/app_inference.py
    -> http://localhost:7860

Env vars:
    NANOGPT_OUT_DIR   Directory to scan for checkpoints (default: "out")
    NANOGPT_DEVICE    Device for inference (default: auto-detect)
"""

import os
import sys
import math
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.chdir(_ROOT)

import torch
import gradio as gr

from ui.checkpoint_manager import list_checkpoints, load_model, get_model_info, GPT2_VARIANTS

OUT_DIR = os.environ.get("NANOGPT_OUT_DIR", "out")
DEVICE = os.environ.get("NANOGPT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def generate_text(checkpoint, prompt, max_new_tokens, temperature, top_k, num_samples):
    if not checkpoint:
        return "⚠️ Please select a checkpoint.", ""
    prompt = prompt.strip() or "\n"
    try:
        model, encode_fn, decode_fn, info, ctx = load_model(checkpoint, device=DEVICE)
    except Exception as e:
        return f"❌ Failed to load model:\n{e}", ""

    t0 = time.time()
    try:
        ids = encode_fn(prompt)
        x = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None, ...]
        outputs = []
        with torch.no_grad():
            with ctx:
                for _ in range(int(num_samples)):
                    y = model.generate(x, int(max_new_tokens),
                                       temperature=float(temperature),
                                       top_k=int(top_k))
                    outputs.append(decode_fn(y[0].tolist()))
    except Exception as e:
        return f"❌ Generation error:\n{e}", ""

    elapsed_ms = (time.time() - t0) * 1000
    text_out = ("\n" + "─" * 60 + "\n").join(outputs)

    params_str = f" · {info['num_params']/1e6:.1f}M params" if info.get("num_params") else ""
    bvl = info.get("best_val_loss")
    ppl_str = f" · ppl≈{math.exp(bvl):.1f}" if bvl else ""
    stats = f"⏱ {elapsed_ms:.0f}ms · {int(max_new_tokens)*int(num_samples)} tokens{params_str}{ppl_str}"
    return text_out, stats


def show_checkpoint_info(checkpoint):
    if not checkpoint:
        return ""
    try:
        info = get_model_info(checkpoint)
    except Exception as e:
        return f"Error: {e}"
    if "error" in info:
        return f"⚠️ {info['error']}"

    lines = []
    if info.get("type") == "pretrained":
        lines.append(f"**Type:** OpenAI pretrained (`{info['variant']}`)")
    else:
        lines.append(f"**Type:** Custom checkpoint")
        lines.append(f"**File:** {info.get('file_size_mb')} MB")

    if info.get("num_params"):
        lines.append(f"**Params:** {info['num_params']/1e6:.1f}M")
    if info.get("iter_num") is not None:
        lines.append(f"**Steps trained:** {info['iter_num']:,}")
    bvl = info.get("best_val_loss")
    if bvl is not None:
        lines.append(f"**Best val loss:** `{bvl:.4f}` (ppl ≈ {math.exp(bvl):.1f})")

    ma = info.get("model_args", {})
    if ma:
        lines.append(
            f"**Arch:** {ma.get('n_layer')}L · {ma.get('n_head')}H · "
            f"embd={ma.get('n_embd')} · ctx={ma.get('block_size')}"
        )
    return "\n\n".join(lines)


def build_ui():
    checkpoints = list_checkpoints(OUT_DIR)
    default_ckpt = checkpoints[0] if checkpoints else None

    css = """
    .gradio-container { font-family: 'Inter', sans-serif; }
    #title  { text-align: center; margin-bottom: 0.25rem; }
    #sub    { text-align: center; color: #888; margin-bottom: 1.5rem; font-size: 0.9rem; }
    #stats  { font-size: 0.82rem; color: #aaa; min-height: 1.4rem; }
    #output textarea { font-family: 'Fira Code', monospace; font-size: 0.88rem; }
    """

    with gr.Blocks(
        title="nanoGPT — Inference",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css=css,
    ) as demo:

        gr.Markdown("# 🧠 nanoGPT — Text Generation", elem_id="title")
        gr.Markdown(
            "Generate text from your trained checkpoints or OpenAI GPT-2 weights.",
            elem_id="sub",
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                ckpt_dd = gr.Dropdown(
                    choices=checkpoints, value=default_ckpt,
                    label="Model Checkpoint",
                    info="Scanned from out/ + GPT-2 variants",
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")
                ckpt_info = gr.Markdown(
                    value=show_checkpoint_info(default_ckpt) if default_ckpt else ""
                )
                gr.Markdown("### Generation Settings")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature",
                                        info="Lower = focused, Higher = creative")
                top_k = gr.Slider(1, 500, value=200, step=1, label="Top-K")
                max_tokens = gr.Slider(10, 2000, value=200, step=10, label="Max New Tokens")
                num_samples = gr.Slider(1, 5, value=1, step=1, label="Samples")

            with gr.Column(scale=2):
                prompt_box = gr.Textbox(
                    label="Prompt", lines=5,
                    placeholder="Enter your prompt here…",
                )
                with gr.Row():
                    gen_btn = gr.Button("✨ Generate", variant="primary", size="lg", scale=3)
                    gr.ClearButton([prompt_box], value="🗑 Clear", size="lg", scale=1)

                output_box = gr.Textbox(
                    label="Generated Text", lines=18,
                    interactive=False, show_copy_button=True,
                    elem_id="output",
                )
                stats_box = gr.Markdown("", elem_id="stats")

        gr.Examples(
            examples=[
                ["Once upon a time in a land far away,", 0.8, 200, 200, 1],
                ["The future of AI is", 1.0, 300, 150, 2],
                ["def fibonacci(n):\n    ", 0.5, 100, 50, 1],
                ["In the beginning,", 0.9, 250, 200, 1],
            ],
            inputs=[prompt_box, temperature, max_tokens, top_k, num_samples],
        )

        gen_btn.click(
            fn=generate_text,
            inputs=[ckpt_dd, prompt_box, max_tokens, temperature, top_k, num_samples],
            outputs=[output_box, stats_box],
        )
        ckpt_dd.change(fn=show_checkpoint_info, inputs=ckpt_dd, outputs=ckpt_info)

        def refresh():
            new = list_checkpoints(OUT_DIR)
            return gr.Dropdown(choices=new, value=new[0] if new else None)

        refresh_btn.click(fn=refresh, outputs=ckpt_dd)

    return demo


if __name__ == "__main__":
    print(f"nanoGPT Inference UI  |  device={DEVICE}  |  out={OUT_DIR}")
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

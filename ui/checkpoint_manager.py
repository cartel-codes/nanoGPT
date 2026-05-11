"""
Checkpoint Manager for nanoGPT UI
-----------------------------------
Discovers, loads, and caches model checkpoints from the out/ directory.
Supports both custom-trained checkpoints and OpenAI GPT-2 pretrained variants.

Usage:
    from ui.checkpoint_manager import list_checkpoints, load_model, get_model_info
"""

import os
import sys
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch

# ---------------------------------------------------------------------------
# Path setup — allow importing model.py from the nanoGPT root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import GPT, GPTConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Global model cache  {checkpoint_path_str -> (model, encode_fn, decode_fn)}
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}

# ---------------------------------------------------------------------------
# GPT-2 pretrained variants that can be loaded without a checkpoint file
# ---------------------------------------------------------------------------
GPT2_VARIANTS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def _get_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_checkpoints(out_dir: str = "out") -> list[str]:
    """
    Scan *out_dir* for *.pt files and return a list of checkpoint labels.

    Returns:
        List of strings like "out/ckpt.pt", "out/ckpt_step1000.pt", plus
        the GPT-2 pretrained variant names.
    """
    labels = []
    out_path = Path(out_dir)
    if out_path.is_dir():
        for pt_file in sorted(out_path.glob("*.pt")):
            labels.append(str(pt_file))
    labels += GPT2_VARIANTS
    return labels


def get_model_info(checkpoint_path: str) -> dict:
    """
    Return metadata about a checkpoint without fully loading the model weights.

    For file-based checkpoints: reads model_args, iter_num, best_val_loss, config.
    For GPT-2 pretrained variants: returns fixed known stats.
    """
    if checkpoint_path in GPT2_VARIANTS:
        param_counts = {
            "gpt2": 124_000_000,
            "gpt2-medium": 350_000_000,
            "gpt2-large": 774_000_000,
            "gpt2-xl": 1_558_000_000,
        }
        return {
            "type": "pretrained",
            "variant": checkpoint_path,
            "num_params": param_counts.get(checkpoint_path, 0),
            "iter_num": None,
            "best_val_loss": None,
            "config": {},
            "file_size_mb": None,
        }

    path = Path(checkpoint_path)
    if not path.exists():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    # Load only the metadata fields (no model weights)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_args = ckpt.get("model_args", {})
    config = GPTConfig(**model_args)

    # Estimate parameter count from config
    dummy = GPT(config)
    num_params = dummy.get_num_params()
    del dummy

    return {
        "type": "custom",
        "path": checkpoint_path,
        "num_params": num_params,
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "config": ckpt.get("config", {}),
        "model_args": model_args,
        "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
    }


def load_model(
    checkpoint_path: str,
    device: Optional[str] = None,
    force_reload: bool = False,
) -> tuple:
    """
    Load (and cache) a model from *checkpoint_path*.

    Args:
        checkpoint_path: Either a file path like "out/ckpt.pt" or a GPT-2
                         variant name like "gpt2".
        device:          PyTorch device string. Defaults to best available.
        force_reload:    If True, bypass cache and reload from disk.

    Returns:
        (model, encode_fn, decode_fn, info_dict)
    """
    if device is None:
        device = _get_device()

    cache_key = f"{checkpoint_path}::{device}"
    if cache_key in _MODEL_CACHE and not force_reload:
        return _MODEL_CACHE[cache_key]

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = (
        torch.bfloat16
        if device_type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device_type == "cuda"
        else torch.float32
    )
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    encode_fn: Callable
    decode_fn: Callable

    # ---- GPT-2 pretrained ----
    if checkpoint_path in GPT2_VARIANTS:
        model = GPT.from_pretrained(checkpoint_path, dict(dropout=0.0))
        model.eval()
        model.to(device)

        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode_fn = lambda ids: enc.decode(ids)
        except ImportError:
            raise ImportError("tiktoken is required for GPT-2 pretrained models. pip install tiktoken")

        info = get_model_info(checkpoint_path)
        result = (model, encode_fn, decode_fn, info, ctx)
        _MODEL_CACHE[cache_key] = result
        return result

    # ---- Custom checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = ckpt["model"]

    # Strip unwanted prefix added by torch.compile
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Determine encode/decode: prefer dataset meta.pkl, fall back to tiktoken
    encode_fn, decode_fn = _build_encode_decode(ckpt)

    info = {
        "type": "custom",
        "path": checkpoint_path,
        "num_params": model.get_num_params(),
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "config": ckpt.get("config", {}),
        "model_args": model_args,
        "file_size_mb": round(Path(checkpoint_path).stat().st_size / 1_048_576, 2),
    }

    result = (model, encode_fn, decode_fn, info, ctx)
    _MODEL_CACHE[cache_key] = result
    return result


def clear_cache():
    """Remove all cached models from memory."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_encode_decode(ckpt: dict) -> tuple[Callable, Callable]:
    """Determine encode/decode functions from a checkpoint dict."""
    cfg = ckpt.get("config", {})
    dataset = cfg.get("dataset", None)

    if dataset:
        meta_path = _ROOT / "data" / dataset / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            stoi = meta["stoi"]
            itos = meta["itos"]
            encode_fn = lambda s: [stoi[c] for c in s]
            decode_fn = lambda ids: "".join([itos[i] for i in ids])
            return encode_fn, decode_fn

    # Fallback: GPT-2 BPE via tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode_fn = lambda ids: enc.decode(ids)
        return encode_fn, decode_fn
    except ImportError:
        raise ImportError(
            "tiktoken is required for tokenization. Install it with: pip install tiktoken"
        )

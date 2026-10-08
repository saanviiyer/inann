"""Shared pieces of the Bills et al. 2023 explain-then-simulate loop, used by explain.py (the
Tour's network) and explain_lab.py (any ONNX model). Claude writes one sentence from a unit's
top examples; a fresh request predicts activations on held-out examples from the sentence alone;
the Pearson r between prediction and truth is the explanation's score."""
import base64
import io
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

from common import ROOT

MODEL = "claude-opus-5-5"

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["explanation", "tags"], "additionalProperties": False,
}
SIM_SCHEMA = {
    "type": "object",
    "properties": {"predictions": {"type": "array", "items": {"type": "integer"}}},
    "required": ["predictions"], "additionalProperties": False,
}


def load_env():
    p = ROOT / ".env.local"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def tile(img, label=None, overlay=None, box=None, size=72):
    """img: (H, W) or (3, H, W) in 0..1. overlay: (h, w) map tinted orange. box: (y, x, h, w) outline."""
    a = np.asarray(img, dtype=np.float32)
    rgb = np.stack([a] * 3, -1) if a.ndim == 2 else np.transpose(a, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1) * 255
    H, W = rgb.shape[:2]
    if overlay is not None:
        o = np.array(Image.fromarray(overlay.astype(np.float32)).resize((W, H), Image.NEAREST))
        o = o / (o.max() or 1)
        rgb = rgb * (1 - 0.55 * o[..., None]) + np.array([255, 150, 40]) * 0.55 * o[..., None]
    im = Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.NEAREST)
    if box is not None:
        y, x, h, w = box
        ImageDraw.Draw(im).rectangle([x * size / W, y * size / H, (x + w) * size / W - 1, (y + h) * size / H - 1], outline=(255, 181, 71), width=2)
    canvas = Image.new("RGB", (size + 8, size + (22 if label is not None else 8)), (30, 34, 44))
    canvas.paste(im, (4, 4))
    if label is not None:
        ImageDraw.Draw(canvas).text((6, size + 6), label, fill=(240, 240, 240))
    return canvas


def grid(tiles, cols):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (cols * w, rows * h), (20, 22, 30))
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * w, (k // cols) * h))
    buf = io.BytesIO()
    g.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(client, text, schema, effort="medium", png_b64=None):
    """One structured request. Images are optional (token models are explained in text)."""
    content = ([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64}}] if png_b64 else []) + [{"type": "text", "text": text}]
    msg = client.beta.messages.create(
        model=MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
    )
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"refused: {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens")
    return json.loads("".join(b.text for b in msg.content if b.type == "text")), msg.model


def sim_set(acts, score_idx, seed):
    """10 top-activating + 10 random items from the scoring half, shuffled (Bills et al.)."""
    rng = random.Random(seed)
    top = list(score_idx[np.argsort(-acts[score_idx])][:10])
    rest = [i for i in score_idx if i not in top]
    idx = top + rng.sample(rest, min(10, len(rest)))
    rng.shuffle(idx)
    return idx


def pearson(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 and true.std() > 0 else 0.0

"""Shared helpers for the offline scripts: rebuild the bundled CNN from src/data/model.json
and recreate the exact train / held-out split used by train.py."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "data"
SEED = 0


def load_split():
    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    return train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)  # Xtr, Xte, ytr, yte


def load_cnn():
    m = json.loads((DATA / "model.json").read_text())
    t = lambda a: torch.tensor(np.array(a, dtype=np.float32))
    return {k: (t(m[k]["w"]), t(m[k]["b"])) for k in ("conv1", "conv2", "dense", "out")}


def forward(W, x):
    """x: (N, 8, 8) in 0..1. Returns post-ReLU activations per layer, as in src/lib/net.js."""
    x = torch.as_tensor(x, dtype=torch.float32)[:, None]
    c1 = F.relu(F.conv2d(x, *W["conv1"], padding=1))
    c2 = F.relu(F.conv2d(F.max_pool2d(c1, 2), *W["conv2"], padding=1))
    p2 = F.max_pool2d(c2, 2).flatten(1)
    d = F.relu(F.linear(p2, *W["dense"]))
    out = F.linear(d, *W["out"])
    return {"conv1": c1, "conv2": c2, "dense": d, "out": out}


def shifted(X, max_shift=1):
    """Every 1px shift (with zero fill) of every image, to give the SAE more data."""
    outs = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            Z = np.zeros_like(X)
            ys = slice(max(0, dy), 8 + min(0, dy)); yd = slice(max(0, -dy), 8 + min(0, -dy))
            xs = slice(max(0, dx), 8 + min(0, dx)); xd = slice(max(0, -dx), 8 + min(0, -dx))
            Z[:, ys, xs] = X[:, yd, xd]
            outs.append(Z)
    return np.concatenate(outs)

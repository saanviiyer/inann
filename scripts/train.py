"""Train the tiny CNN that inann renders, offline, and write its weights to src/data/model.json.

Data: scikit-learn's bundled 8x8 handwritten digits (the UCI "optdigits" set,
Alpaydin & Kaynak 1998). These are 8x8 downsampled digits with pixel values 0..16.
They are not MNIST, and the UI says so.

Architecture (channels first):
  input 1x8x8 -> conv1 6@3x3 pad1 -> ReLU -> maxpool2 -> 6x4x4
              -> conv2 12@3x3 pad1 -> ReLU -> maxpool2 -> 12x2x2 (48)
              -> dense 32 -> ReLU -> out 10

Run: python3 scripts/train.py   (deterministic, seed 0)
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT_MODEL = ROOT / "src" / "data" / "model.json"
OUT_DATA = ROOT / "src" / "data" / "digits.json"


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.dense = nn.Linear(48, 32)
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.dense(x))
        return self.out(x)


def main():
    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)  # 0..1
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-4)
    xt = torch.tensor(Xtr)[:, None]
    yt = torch.tensor(ytr)
    xv = torch.tensor(Xte)[:, None]
    yv = torch.tensor(yte)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 64):
            idx = perm[i : i + 64]
            xb = xt[idx]
            # light augmentation: random 1px shifts, so hand-drawn input is less brittle
            sh = torch.randint(-1, 2, (2,), generator=g)
            xb = torch.roll(xb, shifts=(int(sh[0]), int(sh[1])), dims=(2, 3))
            loss = F.cross_entropy(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        tr_acc = (model(xt).argmax(1) == yt).float().mean().item()
        te_logits = model(xv)
        te_acc = (te_logits.argmax(1) == yv).float().mean().item()
    print(f"train acc {tr_acc:.4f}  test acc {te_acc:.4f}  (n_test={len(yv)})")

    r = lambda t: np.round(t.detach().numpy().astype(np.float64), 5).tolist()
    model_json = {
        "arch": "conv1 6@3x3 pad1 relu pool2 | conv2 12@3x3 pad1 relu pool2 | dense 32 relu | out 10",
        "dataset": "scikit-learn load_digits (UCI optdigits), 8x8, values scaled 0..1",
        "seed": SEED,
        "train_acc": round(tr_acc, 4),
        "test_acc": round(te_acc, 4),
        "n_train": int(len(ytr)),
        "n_test": int(len(yv)),
        "conv1": {"w": r(model.conv1.weight), "b": r(model.conv1.bias)},
        "conv2": {"w": r(model.conv2.weight), "b": r(model.conv2.bias)},
        "dense": {"w": r(model.dense.weight), "b": r(model.dense.bias)},
        "out": {"w": r(model.out.weight), "b": r(model.out.bias)},
        # parity fixture: first 20 test images and their logits, checked by scripts/check_parity.mjs
        "parity": {"logits": r(te_logits[:20])},
    }
    OUT_MODEL.write_text(json.dumps(model_json, separators=(",", ":")))

    # held-out digits the browser uses for neighborhoods, top patches and sample inputs.
    # stored as integers 0..16 to keep the file small.
    data_json = {
        "source": "scikit-learn load_digits test split (seed 0), never seen in training",
        "images": (Xte * 16).round().astype(int).reshape(len(Xte), 64).tolist(),
        "labels": yte.tolist(),
    }
    OUT_DATA.write_text(json.dumps(data_json, separators=(",", ":")))
    print("wrote", OUT_MODEL, OUT_MODEL.stat().st_size, "bytes")
    print("wrote", OUT_DATA, OUT_DATA.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

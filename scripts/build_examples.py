"""Build the example models and datasets the Lab offers, so "upload a model" can be tried
without bringing one. Everything lands in public/examples/ and is served as static files.

  mnist_mlp.onnx    784-128-64-10 MLP, ReLU
  mnist_cnn.onnx    conv8-relu-pool, conv16-relu-pool, fc64-relu, fc10
  optdigits_cnn.onnx  the Tour's 8x8 network, rebuilt from src/data/model.json
  *.meta.json       optional sidecar: how to feed the model (see README "Lab model format")
  mnist_test_1000.json, optdigits_450.json   sample datasets (uint8, base64)

MNIST is trained on [0,1] pixels with no mean/std, for 2 epochs on CPU (seed 0).
Run: python3 scripts/build_examples.py
"""
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from common import ROOT, load_cnn, load_split

OUT = ROOT / "public" / "examples"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "scripts" / ".cache"
torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(784, 128), nn.Linear(128, 64), nn.Linear(64, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(16 * 7 * 7, 64), nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


class OptdigitsCNN(nn.Module):
    """The Tour network, loaded from the committed weights."""
    def __init__(self):
        super().__init__()
        W = load_cnn()
        self.c1, self.c2 = nn.Conv2d(1, 6, 3, padding=1), nn.Conv2d(6, 12, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(48, 32), nn.Linear(32, 10)
        for mod, k in ((self.c1, "conv1"), (self.c2, "conv2"), (self.fc1, "dense"), (self.fc2, "out")):
            mod.weight.data, mod.bias.data = W[k][0].clone(), W[k][1].clone()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def mnist(train):
    ds = torchvision.datasets.MNIST(CACHE, train=train, download=True)
    return ds.data.float().div(255)[:, None], ds.targets


def train(model, X, y, epochs=2):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 128):
            idx = perm[i : i + 128]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def acc(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def export(model, shape, name, meta):
    model.eval()
    torch.onnx.export(model, torch.zeros(1, *shape), OUT / f"{name}.onnx", input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", name, (OUT / f"{name}.onnx").stat().st_size, "bytes")


def dataset_json(X, y, name, source):
    u8 = (X.numpy() * 255).round().clip(0, 255).astype(np.uint8)
    (OUT / f"{name}.json").write_text(json.dumps({
        "source": source, "shape": list(u8.shape[1:]), "range": [0, 1],
        "labels": [int(v) for v in y], "classes": [str(i) for i in range(10)],
        "data": base64.b64encode(u8.tobytes()).decode(),
    }))
    print("wrote", name, len(y), "items")


def main():
    Xtr, ytr = mnist(True)
    Xte, yte = mnist(False)
    for name, model in (("mnist_mlp", MLP()), ("mnist_cnn", CNN())):
        train(model, Xtr, ytr)
        a = acc(model.eval(), Xte, yte)
        print(f"{name} test acc {a:.4f}")
        export(model, (1, 28, 28), name, {
            "name": name.replace("_", " ").upper().replace("MNIST", "MNIST"),
            "description": f"Trained here on MNIST for 2 epochs (seed 0), {100 * a:.1f}% test accuracy.",
            "input": {"kind": "image", "range": [0, 1], "preprocess": "mnist"},
            "classes": [str(i) for i in range(10)], "dataset": "mnist_test_1000.json",
            "test_acc": round(a, 4),
        })
    # 1000 MNIST test images, the first 100 of each digit
    idx = torch.cat([torch.nonzero(yte == d)[:100, 0] for d in range(10)])
    dataset_json(Xte[idx], yte[idx], "mnist_test_1000", "MNIST test set, first 100 of each digit (LeCun et al. 1998)")

    _, Xo, _, yo = load_split()
    Xo_t = torch.tensor(Xo)[:, None]
    m = OptdigitsCNN().eval()
    a = acc(m, Xo_t, torch.tensor(yo))
    print(f"optdigits_cnn held-out acc {a:.4f}")
    export(m, (1, 8, 8), "optdigits_cnn", {
        "name": "Tour 8x8 CNN", "description": f"The network from the Tour, {100 * a:.1f}% on its 450 held-out digits.",
        "input": {"kind": "image", "range": [0, 1], "preprocess": "optdigits"},
        "classes": [str(i) for i in range(10)], "dataset": "optdigits_450.json", "test_acc": round(a, 4),
    })
    dataset_json(Xo_t, torch.tensor(yo), "optdigits_450", "scikit-learn load_digits held-out split (UCI optdigits)")


if __name__ == "__main__":
    main()

"""An RGB example for the Lab: a small CNN on CIFAR-10 (Krizhevsky 2009), so colour photos,
per-class galleries and SAEs on natural images can be tried without bringing a model.

Writes public/examples/cifar_cnn.onnx, .meta.json and cifar_test_1000.json (100 per class).
Trained on CPU for a few epochs (seed 0); expect a modest accuracy, it is a demo network.
Run: python3 scripts/build_cifar_example.py
"""
import base64, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from common import ROOT

OUT = ROOT / "public" / "examples"
CACHE = ROOT / "scripts" / ".cache"
torch.manual_seed(0)
CLASSES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2, self.c3 = nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(64 * 4 * 4, 64), nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def load(train):
    ds = torchvision.datasets.CIFAR10(CACHE, train=train, download=True)
    return torch.tensor(ds.data).permute(0, 3, 1, 2).float().div(255), torch.tensor(ds.targets)


Xtr, ytr = load(True)
Xte, yte = load(False)
m = Net()
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
g = torch.Generator().manual_seed(0)
for epoch in range(6):
    perm = torch.randperm(len(Xtr), generator=g)
    for i in range(0, len(Xtr), 128):
        idx = perm[i : i + 128]
        xb = Xtr[idx]
        if torch.rand(1, generator=g) < 0.5:
            xb = xb.flip(3)  # horizontal flips
        loss = F.cross_entropy(m(xb), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(Xte[:2000]).argmax(1) == yte[:2000]).float().mean().item()
    m.train()
    print(f"epoch {epoch} test acc (first 2000) {acc:.3f}")
m.eval()
with torch.no_grad():
    acc = (torch.cat([m(Xte[i:i + 1000]).argmax(1) for i in range(0, 10000, 1000)]) == yte).float().mean().item()
print(f"test accuracy {acc:.4f}")
torch.onnx.export(m, torch.zeros(1, 3, 32, 32), OUT / "cifar_cnn.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
(OUT / "cifar_cnn.meta.json").write_text(json.dumps({
    "name": "CIFAR-10 CNN",
    "description": f"Small RGB CNN trained here on CIFAR-10 for 6 epochs (seed 0), {100 * acc:.1f}% test accuracy.",
    "input": {"kind": "image", "norm": "unit", "preprocess": "stretch"}, "classes": CLASSES,
    "dataset": "cifar_test_1000.json", "test_acc": round(acc, 4),
}, indent=1))
idx = torch.cat([torch.nonzero(yte == c)[:100, 0] for c in range(10)])
u8 = (Xte[idx].numpy() * 255).round().astype(np.uint8)
(OUT / "cifar_test_1000.json").write_text(json.dumps({
    "source": "CIFAR-10 test set, first 100 of each class (Krizhevsky 2009)", "shape": [3, 32, 32], "range": [0, 1],
    "labels": yte[idx].tolist(), "classes": CLASSES, "data": base64.b64encode(u8.tobytes()).decode(),
}))
print("wrote cifar_cnn.onnx", (OUT / "cifar_cnn.onnx").stat().st_size, "and cifar_test_1000.json")

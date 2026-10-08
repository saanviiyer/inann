"""A tiny transformer for the Lab's sequence path: read 16 parentheses, say whether they are
balanced. Written with plain ops (no fused attention) so it exports to clean ONNX, where the
attention weights (Softmax) and MLP activations (ReLU) become rooms.

Writes public/examples/paren_transformer.onnx, .meta.json (with the character vocabulary, so
the Lab can tokenize typed text) and paren_1000.json (token-id dataset).
Run: python3 scripts/build_paren_example.py   (seed 0, CPU, a few minutes)
"""
import base64, json, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from common import ROOT

OUT = ROOT / "public" / "examples"
T, D, H, L = 16, 32, 4, 2
VOCAB = ["(", ")"]
torch.manual_seed(0); random.seed(0)


def dyck(n):  # a random balanced string of length n
    s, open_ = [], 0
    for i in range(n):
        left = n - i
        if open_ == 0 or (open_ < left and random.random() < 0.5): s.append("("); open_ += 1
        else: s.append(")"); open_ -= 1
    return "".join(s)


def sample():
    if random.random() < 0.5:
        return dyck(T), 1
    s = list(dyck(T))
    for _ in range(random.choice([1, 1, 2])):  # near misses: flip one or two characters
        i = random.randrange(T); s[i] = "(" if s[i] == ")" else ")"
    s = "".join(s)
    depth, ok = 0, True
    for ch in s:
        depth += 1 if ch == "(" else -1
        ok &= depth >= 0
    return s, int(ok and depth == 0)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, 2 * D), nn.Linear(2 * D, D)

    def forward(self, x):
        B = x.shape[0]
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.reshape(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
        att = torch.softmax(q @ k.transpose(-1, -2) / (D // H) ** 0.5, dim=-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class ParenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(2, D), nn.Parameter(torch.randn(T, D) * 0.1)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, 2)

    def forward(self, ids):
        x = self.tok(ids) + self.pos
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x).mean(1))


def batch(n):
    xs, ys = zip(*(sample() for _ in range(n)))
    return torch.tensor([[VOCAB.index(c) for c in s] for s in xs]), torch.tensor(ys)


m = ParenNet()
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
for step in range(3000):
    x, y = batch(128)
    loss = F.cross_entropy(m(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(step, round(loss.item(), 4))
m.eval()
random.seed(123)
xt, yt = batch(2000)
with torch.no_grad(): acc = (m(xt).argmax(1) == yt).float().mean().item()
print(f"held-out accuracy {acc:.4f}")

torch.onnx.export(m, xt[:1], OUT / "paren_transformer.onnx", input_names=["ids"], output_names=["logits"],
                  dynamic_axes={"ids": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
(OUT / "paren_transformer.meta.json").write_text(json.dumps({
    "name": "Paren transformer",
    "description": f"2-layer transformer trained here to tell balanced from unbalanced strings of 16 parentheses, {100 * acc:.1f}% held-out accuracy.",
    "input": {"kind": "tokens", "vocab": VOCAB, "tokenizer": "char", "dims": [T]},
    "classes": ["unbalanced", "balanced"], "dataset": "paren_1000.json", "test_acc": round(acc, 4),
}, indent=1))
ids = xt[:1000].numpy().astype(np.int32)
(OUT / "paren_1000.json").write_text(json.dumps({
    "source": "random balanced strings and near misses (seed 123), never seen in training",
    "kind": "tokens", "dtype": "int32", "shape": [T], "labels": yt[:1000].tolist(), "classes": ["unbalanced", "balanced"],
    "data": base64.b64encode(ids.tobytes()).decode(),
}))
print("wrote paren_transformer.onnx", (OUT / "paren_transformer.onnx").stat().st_size)

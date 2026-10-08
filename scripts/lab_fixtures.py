"""Harder test fixtures for the Lab engine, kept out of git (scripts/.cache): a residual CNN with
BatchNorm whose flatten goes through Shape/Gather/Concat (integer tensors), and a small
transformer-style token MLP with LayerNorm and GELU. Random weights; only the plumbing is tested."""
import torch, torch.nn as nn, torch.nn.functional as F
from common import ROOT
C = ROOT / "scripts" / ".cache"
torch.manual_seed(0)

class Res(nn.Module):
    def __init__(s):
        super().__init__()
        s.c0 = nn.Conv2d(3, 8, 3, padding=1); s.b0 = nn.BatchNorm2d(8)
        s.c1 = nn.Conv2d(8, 8, 3, padding=1); s.b1 = nn.BatchNorm2d(8)
        s.fc = nn.Linear(8 * 16 * 16, 5)
    def forward(s, x):
        h = F.relu(s.b0(s.c0(x)))
        h = F.relu(h + s.b1(s.c1(h)))          # skip connection
        h = F.max_pool2d(h, 2)
        return s.fc(h.view(h.size(0), -1))      # shape arithmetic in the graph

class Tok(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Linear(6, 16); s.ln = nn.LayerNorm(16); s.ff = nn.Linear(16, 32); s.out = nn.Linear(32, 3)
    def forward(s, x):                          # x: (batch, 10 tokens, 6 features)
        h = s.ln(s.emb(x))
        h = F.gelu(s.ff(h))
        return s.out(h.mean(1))

for name, m, shape in (("fixture_res", Res(), (1, 3, 32, 32)), ("fixture_tok", Tok(), (1, 10, 6))):
    m.eval()
    torch.onnx.export(m, torch.zeros(shape), C / f"{name}.onnx", input_names=["input"], output_names=["out"],
                      dynamic_axes={"input": {0: "b"}, "out": {0: "b"}}, opset_version=17, dynamo=False)
    print("wrote", name)

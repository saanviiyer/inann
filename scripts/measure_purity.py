"""Do SAE features look more monosemantic than neurons? Measured, not assumed.
Purity of a unit = fraction of its top-N held-out activating digits that share the most
common label (only counting digits where the unit is active). Prints the comparison that
src/lib reproduces live in the browser."""
import json
import numpy as np
import torch
from common import DATA, forward, load_cnn, load_split

W = load_cnn(); _, Xte, _, yte = load_split()
sae = json.loads((DATA / "sae.json").read_text())
T = lambda k: torch.tensor(np.array(sae[k], dtype=np.float32))
with torch.no_grad():
    d = forward(W, Xte)["dense"]
    pre = torch.relu((d - T("b_dec")) @ T("W_enc").T + T("b_enc"))
    v, i = pre.topk(sae["stats"]["k"], 1)
    f = torch.zeros_like(pre).scatter(1, i, v)

def purity(acts, n):
    out = []
    for u in range(acts.shape[1]):
        a = acts[:, u].numpy()
        idx = [j for j in np.argsort(-a)[:n] if a[j] > 0]
        if len(idx) < n: continue  # needs n active examples to be comparable
        out.append(np.bincount(yte[idx], minlength=10).max() / n)
    return np.array(out)

for n in (5, 12, 20):
    pn, pf = purity(d, n), purity(f, n)
    print(f"top-{n:2d}: neurons median purity {np.median(pn):.2f} (n={len(pn)}), "
          f"SAE features median {np.median(pf):.2f} (n={len(pf)}); "
          f"fully pure: neurons {np.mean(pn==1):.0%}, features {np.mean(pf==1):.0%}")

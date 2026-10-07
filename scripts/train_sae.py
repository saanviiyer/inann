"""Train a top-k sparse autoencoder on the CNN's 32 dense-layer activations and write
src/data/sae.json. CPU only, about a minute.

Method: top-k SAE (Makhzani & Frey 2013; Gao et al. 2024), as used for dictionary learning
in Bricken et al. 2023 / Templeton et al. 2024 (who used an L1 penalty instead).
  f = TopK(ReLU(W_enc (d - b_dec) + b_enc)),  d_hat = W_dec f + b_dec
Dictionary 256 = 8x the 32 hidden neurons. Decoder columns are kept at unit norm.
Training data: training-split digits plus all 1px shifts (9x). Held-out digits are only
used for the reported metrics.

Run: python3 scripts/train_sae.py
"""
import json
import time

import numpy as np
import torch

from common import DATA, SEED, forward, load_cnn, load_split, shifted

N_HIDDEN, N_FEAT, K = 32, 256, 4
STEPS, BATCH, LR = 12000, 256, 2e-3

torch.manual_seed(SEED)
t0 = time.time()
W = load_cnn()
Xtr, Xte, ytr, yte = load_split()
with torch.no_grad():
    A = forward(W, shifted(Xtr))["dense"]
    Ate = forward(W, Xte)["dense"]
    out_te = forward(W, Xte)["out"]
print(f"training activations {tuple(A.shape)}, held-out {tuple(Ate.shape)}")

b_dec = torch.nn.Parameter(A.median(0).values.clone())
W_dec = torch.nn.Parameter(torch.randn(N_HIDDEN, N_FEAT))
with torch.no_grad():
    W_dec /= W_dec.norm(dim=0, keepdim=True)
W_enc = torch.nn.Parameter(W_dec.detach().T.clone())  # tied init (Gao et al. 2024)
b_enc = torch.nn.Parameter(torch.zeros(N_FEAT))
opt = torch.optim.Adam([W_enc, b_enc, W_dec, b_dec], lr=LR)


def encode(d, k=K):
    pre = torch.relu((d - b_dec) @ W_enc.T + b_enc)
    v, i = pre.topk(k, dim=1)
    return torch.zeros_like(pre).scatter(1, i, v)


def decode(f):
    return f @ W_dec.T + b_dec


last_fired = torch.zeros(N_FEAT)
g = torch.Generator().manual_seed(SEED)
for step in range(STEPS):
    d = A[torch.randint(0, len(A), (BATCH,), generator=g)]
    f = encode(d)
    loss = ((decode(f) - d) ** 2).sum(1).mean()
    # auxiliary loss on features that have not fired for a while, so fewer die (Gao et al. 2024)
    last_fired += 1
    last_fired[(f > 0).any(0)] = 0
    dead = last_fired > 400
    if dead.any():
        pre = torch.relu((d - b_dec) @ W_enc.T + b_enc) * dead
        v, i = pre.topk(min(16, int(dead.sum())), dim=1)
        aux = torch.zeros_like(pre).scatter(1, i, v)
        resid = (d - decode(f)).detach()
        loss = loss + (1 / 32) * ((aux @ W_dec.T - resid) ** 2).sum(1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        W_dec /= W_dec.norm(dim=0, keepdim=True)
    if step % 3000 == 0:
        print(f"step {step:5d} loss {loss.item():.4f} dead {int(dead.sum())}")

with torch.no_grad():
    f_te = encode(Ate)
    rec = decode(f_te)
    fvu = (((rec - Ate) ** 2).sum() / ((Ate - Ate.mean(0)) ** 2).sum()).item()
    logits_rec = rec.clamp(min=0) @ W["out"][0].T + W["out"][1]
    acc_orig = (out_te.argmax(1).numpy() == yte).mean()
    acc_rec = (logits_rec.argmax(1).numpy() == yte).mean()
    agree = (logits_rec.argmax(1) == out_te.argmax(1)).float().mean().item()
    f_tr = encode(A)
    alive_tr = (f_tr > 0).any(0)
    alive_te = (f_te > 0).any(0)

stats = {
    "n_hidden": N_HIDDEN, "n_features": N_FEAT, "k": K,
    "train_samples": int(len(A)), "seconds": round(time.time() - t0, 1),
    "heldout_fvu": round(fvu, 4),
    "heldout_variance_explained": round(1 - fvu, 4),
    "heldout_acc_original": round(float(acc_orig), 4),
    "heldout_acc_with_sae_reconstruction": round(float(acc_rec), 4),
    "heldout_pred_agreement": round(agree, 4),
    "alive_on_train": int(alive_tr.sum()),
    "alive_on_heldout": int(alive_te.sum()),
}
print(json.dumps(stats, indent=1))
r = lambda t: np.round(t.detach().numpy().astype(np.float64), 5).tolist()
(DATA / "sae.json").write_text(json.dumps({
    "layer": "dense", "method": "top-k SAE", "stats": stats,
    "W_enc": r(W_enc), "b_enc": r(b_enc), "W_dec": r(W_dec), "b_dec": r(b_dec),
}, separators=(",", ":")))
print("wrote", DATA / "sae.json")

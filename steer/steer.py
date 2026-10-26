"""Neuron steering for Hugging Face decoder LLMs (Qwen3, Qwen3.5, Llama-style MLPs).

A neuron is one coordinate n of the input to layer L's MLP down_proj, i.e. act(gate(x)) * up(x).
Steering adds a constant to that coordinate at every position (prompt and generated tokens).
The recipe that circulated for Qwen3.5-4B ("L14 N8949 +1, L13 N7956 -1") sets the constant to

    value = dose * median_t>=1 ||h_L(t)|| / ||W_down[:, n]||

where h_L is the residual stream leaving layer L on the current prompt, tokens 1: (token 0 is
an attention sink with an outsized norm). The added residual vector value * W_down[:, n] then has
norm dose * median ||h_L||: a dose of 1 writes a vector as large as a typical residual state.

This module holds the maths so it can be tested without a server; server.py serves it.
"""
from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass

import torch


def find_layers(model):
    """The list of decoder layers, wherever the architecture nests it."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if len(obj) and hasattr(obj[0], "mlp") and hasattr(obj[0].mlp, "down_proj"):
            return obj
    raise ValueError("could not find decoder layers with mlp.down_proj")


@dataclass
class Intervention:
    layer: int
    neuron: int
    dose: float


def residual_norms(model, input_ids):
    """||h_L(t)|| for every layer L (0-indexed, the output of decoder layer L) and position t."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states  # embeddings, then one entry per layer
    return [h[0].float().norm(dim=-1) for h in hs[1:]]


def recipe_values(model, input_ids, interventions, norms=None):
    """The value added to each neuron, and the pieces it is built from."""
    layers = find_layers(model)
    norms = norms if norms is not None else residual_norms(model, input_ids)
    out = []
    for iv in interventions:
        n_tok = norms[iv.layer].shape[0]
        med = norms[iv.layer][1:].median().item() if n_tok > 1 else norms[iv.layer][0].item()
        w = layers[iv.layer].mlp.down_proj.weight[:, iv.neuron].float().norm().item()
        out.append({"layer": iv.layer, "neuron": iv.neuron, "dose": iv.dose, "median_resid_norm": med,
                    "w_down_col_norm": w, "value": iv.dose * med / w, "added_norm": abs(iv.dose) * med})
    return out


@contextlib.contextmanager
def steering(model, values):
    """Add value to neuron n of down_proj's input at every position, for each entry of values."""
    layers = find_layers(model)
    by_layer = {}
    for v in values:
        by_layer.setdefault(v["layer"], []).append((v["neuron"], v["value"]))
    handles = []
    for L, items in by_layer.items():
        def pre(mod, args, items=items):
            x = args[0].clone()
            for n, val in items:
                x[..., n] += val
            return (x,) + tuple(args[1:])
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def recording(model, wanted):
    """Record neuron activations (down_proj inputs) for (layer, neuron) pairs, every forward pass."""
    layers = find_layers(model)
    rec = {(L, n): [] for L, n in wanted}
    handles = []
    for L in {L for L, _ in wanted}:
        ns = [n for LL, n in wanted if LL == L]
        def pre(mod, args, L=L, ns=ns):
            x = args[0]
            for n in ns:
                rec[(L, n)].append(x[0, :, n].float().detach().cpu())
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(pre))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()


def flat(rec):
    return {f"{L}:{n}": torch.cat(v).tolist() if v else [] for (L, n), v in rec.items()}


def generate(model, tok, input_ids, max_new_tokens=120, temperature=0.0, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, **kw)
    return out[0, input_ids.shape[1]:]


def random_control(interventions, d_mlp, seed=0):
    """Same layers and doses, different neurons: the null for 'these two neurons do it'."""
    rnd = random.Random(seed)
    used = {(iv.layer, iv.neuron) for iv in interventions}
    out = []
    for iv in interventions:
        n = rnd.randrange(d_mlp)
        while (iv.layer, n) in used:
            n = rnd.randrange(d_mlp)
        out.append(Intervention(iv.layer, n, iv.dose))
    return out

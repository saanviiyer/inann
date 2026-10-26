"""Checks the steering maths on tiny randomly initialised models (nothing is downloaded).
Run: python3 steer/test_steer.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from steer import Intervention, find_layers, random_control, recipe_values, recording, residual_norms, steering  # noqa: E402

fails = 0


def check(ok, msg):
    global fails
    print(("ok   " if ok else "FAIL ") + msg)
    fails += 0 if ok else 1


def tiny_models():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    yield "Qwen3", Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=4,
                                                 num_attention_heads=4, num_key_value_heads=2, head_dim=16)).eval()
    try:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        cfg = Qwen3_5TextConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=4, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16, layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16)
        yield "Qwen3.5", Qwen3_5ForCausalLM(cfg).eval()
    except Exception as e:  # the Qwen3.5 class may need options this transformers build lacks
        print(f"skip Qwen3.5 tiny model: {type(e).__name__}: {e}")


for name, model in tiny_models():
    layers = find_layers(model)
    ids = torch.randint(0, 97, (1, 12))
    ivs = [Intervention(2, 37, 1.0), Intervention(1, 5, -0.5)]
    norms = residual_norms(model, ids)
    vals = recipe_values(model, ids, ivs, norms)

    # 1. the added residual vector has norm |dose| x median ||h_L|| over tokens 1:
    for v, iv in zip(vals, ivs):
        med = norms[iv.layer][1:].median().item()
        w = layers[iv.layer].mlp.down_proj.weight[:, iv.neuron]
        check(abs(abs(v["value"]) * w.norm().item() - abs(iv.dose) * med) < 1e-4 * med, f"{name}: L{iv.layer} N{iv.neuron} adds a vector of norm |dose| x median resid ({v['added_norm']:.3f})")

    # 2. with the hook, layer L's MLP output moves by exactly value * W_down[:, n] at every position
    L, n = 2, 37
    v = [x for x in vals if x["layer"] == L][0]
    outs = {}
    def grab(key):
        return lambda m, a, o: outs.__setitem__(key, o.detach().clone())
    h = layers[L].mlp.register_forward_hook(grab("base"))
    with torch.no_grad():
        model(ids)
    h.remove()
    # steer only layer L so its own input is unchanged
    h = layers[L].mlp.register_forward_hook(grab("steer"))
    with steering(model, [v]), torch.no_grad():
        model(ids)
    h.remove()
    delta = outs["steer"] - outs["base"]
    expect = v["value"] * layers[L].mlp.down_proj.weight[:, n]
    check(torch.allclose(delta[0], expect.expand_as(delta[0]), atol=1e-4), f"{name}: the steered MLP output differs by value x W_down[:, n] at all {ids.shape[1]} positions")

    # 3. hooks are removed afterwards
    with torch.no_grad():
        again = model(ids).logits
        base = model(ids).logits
    check(torch.equal(again, base) and all(len(l.mlp.down_proj._forward_pre_hooks) == 0 for l in layers), f"{name}: no hooks left behind")

    # 4. recording sees the shifted activation
    with steering(model, [v]), recording(model, [(L, n)]) as rs, torch.no_grad():
        model(ids)
    with recording(model, [(L, n)]) as rb, torch.no_grad():
        model(ids)
    shift = (rs[(L, n)][0] - rb[(L, n)][0])
    check(torch.allclose(shift, torch.full_like(shift, v["value"]), atol=1e-4), f"{name}: recorded activation shifts by exactly the added value")

    # 5. the random control keeps layers and doses and avoids the steered neurons
    c = random_control(ivs, layers[0].mlp.down_proj.weight.shape[1], seed=3)
    check(all(a.layer == b.layer and a.dose == b.dose and a.neuron != b.neuron for a, b in zip(c, ivs)), f"{name}: random control matches layers and doses, different neurons")

print("\nall passed" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)

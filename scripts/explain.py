"""Write one plain-language explanation per unit and score it by simulation, following
Bills et al. 2023, "Language models can explain neurons in language models", adapted to images.

For every unit (conv channels, dense neurons, SAE features):
  1. EXPLAIN. Show Claude the unit's top-12 activating digits from the explain half of the
     held-out set, with activations scaled 0..10 (and, for conv channels, where on the digit
     the channel fired). Claude returns one sentence plus a few search tags.
  2. SIMULATE. In a fresh request, show Claude only the explanation and 20 unlabeled digits
     from the other half (10 top-activating, 10 random, shuffled). Claude predicts each one's
     activation 0..10.
  3. SCORE. Pearson correlation between the predicted and true activations (Bills et al.'s
     correlation scoring). The score is stored and shown in the app next to the sentence.

Output: src/data/explanations.json. Per-unit results are cached in scripts/.cache so an
interrupted run resumes where it stopped.

Credentials: ANTHROPIC_API_KEY from the environment or from inann/.env.local (gitignored).
Run:   python3 scripts/explain.py --limit 3     # smoke test on 3 units
       python3 scripts/explain.py               # everything (about 200 units, 2 calls each)
"""
import argparse
import base64
import io
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import numpy as np
import torch
from PIL import Image, ImageDraw

from common import DATA, ROOT, forward, load_cnn, load_split

MODEL = "claude-opus-5-5"
CACHE = ROOT / "scripts" / ".cache" / "explain"
MIN_ACTIVE = 5  # units active on fewer held-out digits than this are not explained

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["explanation", "tags"],
    "additionalProperties": False,
}
SIM_SCHEMA = {
    "type": "object",
    "properties": {"predictions": {"type": "array", "items": {"type": "integer"}}},
    "required": ["predictions"],
    "additionalProperties": False,
}

EXPLAIN_PROMPT = """You are explaining one unit inside a small convolutional network that classifies 8x8 handwritten digits (0 to 9).

The image shows the {n} held-out digits that activate this unit most. Each tile is labeled with the unit's activation scaled 0 to 10, where 10 is its largest activation on any held-out digit.{where}

Write ONE plain sentence, at most 20 words, saying what visual pattern this unit responds to, so that someone could predict its activation on a new digit from the sentence alone. Describe strokes, shapes and positions (for example "a horizontal bar across the top"); name specific digits only if the unit really is about those digits. Use simple words, no jargon, no dashes.

Also give 3 to 6 short search tags (lowercase, a few words each) that someone might type to find this unit, such as "horizontal line", "loop", "detects 7"."""

WHERE_CONV = " The orange tint on each tile marks where on the digit this unit fired; a conv unit applies the same small filter at every position."

SIM_PROMPT = """A unit inside a small digit-classifying network is described as:

"{explanation}"

Below are {n} handwritten 8x8 digits, numbered 1 to {n} in reading order. Using only the description, predict this unit's activation on each digit as an integer from 0 (no response) to 10 (its strongest response). Most digits produce little or no activation.

Return exactly {n} integers, in order."""


def load_env():
    p = ROOT / ".env.local"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def unit_activations():
    """Every unit's activation on every held-out digit, matching src/lib/net.js and sae.js.
    Conv channels use the max over positions, and keep the spatial map for the overlay."""
    W = load_cnn()
    _, Xte, _, yte = load_split()
    sae = json.loads((DATA / "sae.json").read_text())
    T = lambda k: torch.tensor(np.array(sae[k], dtype=np.float32))
    with torch.no_grad():
        acts = forward(W, Xte)
        d = acts["dense"]
        pre = torch.relu((d - T("b_dec")) @ T("W_enc").T + T("b_enc"))
        v, i = pre.topk(sae["stats"]["k"], 1)
        f = torch.zeros_like(pre).scatter(1, i, v)
    units = []
    for layer in ("conv1", "conv2"):
        maps = acts[layer].numpy()
        for c in range(maps.shape[1]):
            units.append({"id": f"{layer}:{c}", "layer": layer, "index": c,
                          "act": maps[:, c].reshape(len(Xte), -1).max(1), "maps": maps[:, c]})
    for j in range(d.shape[1]):
        units.append({"id": f"dense:{j}", "layer": "dense", "index": j, "act": d[:, j].numpy()})
    for j in range(f.shape[1]):
        units.append({"id": f"sae:{j}", "layer": "sae", "index": j, "act": f[:, j].numpy()})
    return units, Xte, yte


def tile(img, label=None, overlay=None, size=72):
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], -1).astype(np.float32)
    if overlay is not None:
        o = np.array(Image.fromarray(overlay.astype(np.float32)).resize((8, 8), Image.NEAREST))
        o = o / (o.max() or 1)
        rgb = rgb * (1 - 0.55 * o[..., None]) + np.array([255, 150, 40]) * 0.55 * o[..., None]
    im = Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.NEAREST)
    canvas = Image.new("RGB", (size + 8, size + (22 if label is not None else 8)), (30, 34, 44))
    canvas.paste(im, (4, 4))
    if label is not None:
        ImageDraw.Draw(canvas).text((6, size + 6), label, fill=(240, 240, 240))
    return canvas


def grid(tiles, cols):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (cols * w, rows * h), (20, 22, 30))
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * w, (k // cols) * h))
    buf = io.BytesIO()
    g.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(client, png_b64, text, schema, effort):
    msg = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64}},
            {"type": "text", "text": text},
        ]}],
    )
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"refused: {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens")
    text_out = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(text_out), msg.model


def explain_unit(client, u, X, explain_idx, score_idx):
    a = u["act"]
    amax = a.max()
    # 1. explain from the explain half
    top = [i for i in explain_idx[np.argsort(-a[explain_idx])][:12] if a[i] > 0]
    tiles = [tile(X[i], f"{10 * a[i] / amax:.1f}", u["maps"][i] if "maps" in u else None) for i in top]
    ex, served = ask(client, grid(tiles, 6),
                     EXPLAIN_PROMPT.format(n=len(top), where=WHERE_CONV if "maps" in u else ""),
                     EXPLAIN_SCHEMA, "medium")
    # 2. simulate on the other half: 10 top-activating + 10 random, shuffled (seeded per unit)
    rng = random.Random(u["id"])
    top_s = list(score_idx[np.argsort(-a[score_idx])][:10])
    rest = [i for i in score_idx if i not in top_s]
    sim_idx = top_s + rng.sample(rest, 10)
    rng.shuffle(sim_idx)
    sim, _ = ask(client, grid([tile(X[i]) for i in sim_idx], 5),
                 SIM_PROMPT.format(explanation=ex["explanation"], n=len(sim_idx)), SIM_SCHEMA, "medium")
    pred = np.array(sim["predictions"][: len(sim_idx)], dtype=float)
    if len(pred) != len(sim_idx):
        raise RuntimeError(f"simulator returned {len(pred)} predictions for {len(sim_idx)} digits")
    true = a[sim_idx]
    score = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 and true.std() > 0 else 0.0
    return {
        "id": u["id"], "layer": u["layer"], "index": u["index"],
        "explanation": ex["explanation"].strip(), "tags": [t.strip().lower() for t in ex["tags"]][:6],
        "score": round(score, 3), "n_explain_examples": len(top), "n_sim": len(sim_idx),
        "sim_digits": [int(i) for i in sim_idx], "sim_predicted": pred.astype(int).tolist(),
        "sim_true": [round(float(t), 4) for t in true], "model": served,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N units (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    load_env()
    client = anthropic.Anthropic()

    units, X, y = unit_activations()
    explain_idx, score_idx = np.arange(0, len(X), 2), np.arange(1, len(X), 2)
    todo, skipped = [], []
    for u in units:
        n_active = int((u["act"][explain_idx] > 0).sum()), int((u["act"][score_idx] > 0).sum())
        (todo if min(n_active) >= MIN_ACTIVE else skipped).append(u)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} units to explain, {len(skipped)} skipped (active on < {MIN_ACTIVE} digits in a half)")

    CACHE.mkdir(parents=True, exist_ok=True)
    results = {}
    pending = []
    for u in todo:
        p = CACHE / f"{u['id'].replace(':', '_')}.json"
        if p.exists():
            results[u["id"]] = json.loads(p.read_text())
        else:
            pending.append(u)
    print(f"{len(results)} cached, {len(pending)} to run ({2 * len(pending)} API calls)")

    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(explain_unit, client, u, X, explain_idx, score_idx): u for u in pending}
        for k, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                r = fut.result()
            except anthropic.RateLimitError as e:
                print(f"  rate limited on {u['id']}, rerun to resume: {e}")
                continue
            except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError, ValueError) as e:
                print(f"  failed {u['id']}: {e}")
                continue
            (CACHE / f"{u['id'].replace(':', '_')}.json").write_text(json.dumps(r))
            results[u["id"]] = r
            print(f"  [{k}/{len(pending)}] {u['id']:9s} score {r['score']:+.2f}  {r['explanation']}")

    if args.limit:
        print("smoke test only, src/data/explanations.json not written")
        return
    out = {
        "method": "Bills et al. 2023, adapted to images: explain from top-12 of the even-indexed held-out "
                  "digits, simulate 10 top + 10 random odd-indexed digits, score = Pearson r",
        "explainer_model": MODEL,
        "min_active": MIN_ACTIVE,
        "units": [results[u["id"]] for u in todo if u["id"] in results],
        "not_explained": [u["id"] for u in skipped] + [u["id"] for u in todo if u["id"] not in results],
    }
    (DATA / "explanations.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {DATA / 'explanations.json'}: {len(out['units'])} explained, "
          f"{len(out['not_explained'])} not explained")


if __name__ == "__main__":
    main()

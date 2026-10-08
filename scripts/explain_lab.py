"""Explain the units of ANY ONNX model for the Lab, the same way explain.py does for the Tour
(Bills et al. 2023): Claude writes one sentence per unit from its top examples, a fresh request
predicts activations on held-out examples from the sentence alone, and the Pearson r is stored as
the explanation's score. Load the output in the Lab with "Load explanations".

Units are the channels of one tensor (a conv map's channels, a sequence's features, a vector's
entries) or, with --sae, the features of an SAE exported from the Lab ("Download SAE .json").

  python3 scripts/explain_lab.py --model m.onnx --data dataset.json --tensor /Relu_2_output_0
  python3 scripts/explain_lab.py --model m.onnx --meta m.meta.json --data d.json --sae sae.json
  ... --dry-run    builds every prompt and montage without calling the API (writes previews)

Credentials: ANTHROPIC_API_KEY in the environment or inann/.env.local (gitignored).
"""
import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

from autointerp import EXPLAIN_SCHEMA, SIM_SCHEMA, MODEL, ask, grid, load_env, pearson, sim_set, tile
from common import ROOT

NORMS = {
    "unit": lambda v, c: v, "sym": lambda v, c: v * 2 - 1, "byte": lambda v, c: v * 255,
    "mnist": lambda v, c: (v - 0.1307) / 0.3081,
    "imagenet": lambda v, c: (v - [0.485, 0.456, 0.406][c % 3]) / [0.229, 0.224, 0.225][c % 3],
}

EXPLAIN_IMG = """You are explaining one unit inside a neural network ({about}).

The image shows the {n} held-out inputs that activate this unit most. Each tile is labeled with the unit's activation scaled 0 to 10 (10 = its largest activation on any held-out input).{where}

Write ONE plain sentence, at most 20 words, saying what this unit responds to, so that someone could predict its activation on a new input from the sentence alone. Describe visible content, shapes and positions; name classes only if the unit really is about them. Simple words, no jargon, no dashes.

Also give 3 to 6 short lowercase search tags someone might type to find this unit."""

EXPLAIN_TXT = """You are explaining one unit inside a neural network ({about}) that reads sequences of tokens.

Below are the {n} held-out sequences that activate this unit most, one per line, each with the unit's activation scaled 0 to 10 (10 = its largest activation on any held-out sequence). The token where it fired most is wrapped in [brackets].

{examples}

Write ONE plain sentence, at most 20 words, saying what this unit responds to, so that someone could predict its activation on a new sequence from the sentence alone. Simple words, no jargon, no dashes.

Also give 3 to 6 short lowercase search tags someone might type to find this unit."""

SIM_IMG = """A unit inside a neural network ({about}) is described as:

"{explanation}"

Below are {n} inputs, numbered 1 to {n} in reading order. Using only the description, predict this unit's activation on each as an integer from 0 (no response) to 10 (its strongest response). Most inputs produce little or no activation.

Return exactly {n} integers, in order."""

SIM_TXT = """A unit inside a neural network ({about}) that reads token sequences is described as:

"{explanation}"

Predict its activation on each sequence below as an integer from 0 (no response) to 10 (its strongest response). Most produce little or no activation.

{examples}

Return exactly {n} integers, in order."""


def load_dataset(path):
    j = json.loads(Path(path).read_text())
    raw = base64.b64decode(j["data"])
    size = int(np.prod(j["shape"]))
    dtype = j.get("dtype", "uint8")
    arr = np.frombuffer(raw, dtype={"uint8": np.uint8, "int32": np.int32, "float32": np.float32}[dtype]).reshape(-1, *j["shape"])
    return arr, dtype, np.array(j["labels"]) if j.get("labels") else None, j.get("classes")


def model_inputs(arr, dtype, norm, kind_tokens):
    if kind_tokens or dtype == "int32":
        return arr.astype(np.int64)
    if dtype == "float32":
        return arr.astype(np.float32)
    x = arr.astype(np.float32) / 255
    if x.ndim == 4:  # (N, C, H, W)
        return np.stack([NORMS[norm](x[:, c], c) for c in range(x.shape[1])], 1).astype(np.float32)
    return NORMS[norm](x, 0).astype(np.float32)


def tensor_acts(model_path, tensor, X, batch=64):
    m = onnx.load(model_path)
    if tensor not in [o.name for o in m.graph.output]:
        m.graph.output.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, None))
    s = ort.InferenceSession(m.SerializeToString())
    name = s.get_inputs()[0].name
    batchable = not isinstance(s.get_inputs()[0].shape[0], int) or s.get_inputs()[0].shape[0] != 1
    outs = []
    step = batch if batchable else 1
    for i in range(0, len(X), step):
        res = s.run([tensor], {name: X[i : i + step]})
        outs.append(res[0])
    return np.concatenate(outs)


def as_positions(T):
    """(N, ...) tensor -> (N, P, C) vectors, as the Lab's layout does."""
    if T.ndim >= 4:  # (N, C, H, W, ...)
        N, C = T.shape[:2]
        return np.moveaxis(T.reshape(N, C, -1), 1, 2), ("spatial", T.shape[-2], T.shape[-1])
    if T.ndim == 3:  # (N, T, D)
        return T, ("seq", T.shape[1], 1)
    return T[:, None, :], ("vector", 1, 1)


def sae_encode(sae, V):
    """V: (M, D) vectors -> (M, F) top-k features, matching src/lib/lab/saeCore.js."""
    We, be, bd, mu = (np.array(sae[k], dtype=np.float32) for k in ("We", "be", "bd", "mu"))
    We = We.reshape(sae["F"], sae["D"])
    pre = np.maximum(((V - mu) / sae["s"] - bd) @ We.T + be, 0)
    k = sae["k"]
    idx = np.argpartition(-pre, k - 1, axis=1)[:, :k]
    out = np.zeros_like(pre)
    np.put_along_axis(out, idx, np.take_along_axis(pre, idx, 1), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dataset .json in the Lab format")
    ap.add_argument("--meta", help="model .meta.json (norm, vocab, classes)")
    ap.add_argument("--tensor", help="tensor name (defaults to the SAE's tensor)")
    ap.add_argument("--sae", help="SAE .json downloaded from the Lab")
    ap.add_argument("--max-units", type=int, default=64)
    ap.add_argument("--min-active", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = json.loads(Path(a.meta).read_text()) if a.meta else {}
    sae = json.loads(Path(a.sae).read_text()) if a.sae else None
    tensor = a.tensor or (sae and sae.get("tensor"))
    if not tensor:
        ap.error("give --tensor, or an --sae that records its tensor")
    tokens = meta.get("input", {}).get("kind") == "tokens"
    vocab = meta.get("input", {}).get("vocab")
    norm = meta.get("input", {}).get("norm", "unit")
    arr, dtype, labels, classes = load_dataset(a.data)
    classes = classes or meta.get("classes")
    X = model_inputs(arr, dtype, norm, tokens)
    T = tensor_acts(a.model, tensor, X)
    V, (kind, H, W) = as_positions(T)
    N, P, C = V.shape
    if sae:
        if sae["D"] != C:
            raise SystemExit(f"the SAE has D={sae['D']} but {tensor} has {C} channels")
        F = sae_encode(sae, V.reshape(-1, C)).reshape(N, P, -1)
        acts_all, args_all, prefix = F.max(1), F.argmax(1), "f"
    else:
        acts_all, args_all, prefix = V.max(1), V.argmax(1), "c"
    about = meta.get("description") or meta.get("name") or Path(a.model).name
    explain_idx, score_idx = np.arange(0, N, 2), np.arange(1, N, 2)
    units = [u for u in range(acts_all.shape[1]) if min((acts_all[explain_idx, u] > 0).sum(), (acts_all[score_idx, u] > 0).sum()) >= a.min_active]
    units = sorted(units, key=lambda u: -(acts_all[:, u] > 0).sum())[: a.max_units]
    print(f"{N} items, tensor {tensor} {tuple(T.shape[1:])}, {acts_all.shape[1]} {'features' if sae else 'channels'}, {len(units)} to explain ({2 * len(units)} API calls)")

    def show_img(i, u, label=None, mark=True):
        img = arr[i].astype(np.float32) / 255 if dtype == "uint8" else (arr[i] - arr[i].min()) / (np.ptp(arr[i]) or 1)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        box = None
        if mark and kind == "spatial":
            p = int(args_all[i, u]); Hin, Win = img.shape[-2:]
            box = (p // W * Hin / H, p % W * Win / W, max(1, Hin / H), max(1, Win / W))
        return tile(img, label, box=box)

    def show_seq(i, u, mark=True):
        toks = [vocab[t] if vocab and t < len(vocab) else str(t) for t in arr[i]]
        if mark and kind == "seq":
            p = int(args_all[i, u]); toks[p] = f"[{toks[p]}]"
        return ("" if vocab else " ").join(toks)

    def run_unit(u):
        acts = acts_all[:, u]
        amax = acts.max()
        top = [i for i in explain_idx[np.argsort(-acts[explain_idx])][:12] if acts[i] > 0]
        sim_idx = sim_set(acts, score_idx, f"{tensor}:{prefix}{u}")
        if tokens:
            ex_txt = "\n".join(f"{10 * acts[i] / amax:4.1f}  {show_seq(i, u)}" for i in top)
            p1 = EXPLAIN_TXT.format(about=about, n=len(top), examples=ex_txt)
            sim_txt = "\n".join(f"{k + 1}. {show_seq(i, u, mark=False)}" for k, i in enumerate(sim_idx))
            p2 = lambda e: SIM_TXT.format(about=about, explanation=e, examples=sim_txt, n=len(sim_idx))
            img1 = img2 = None
        else:
            img1 = grid([show_img(i, u, f"{10 * acts[i] / amax:.1f}") for i in top], 6)
            img2 = grid([show_img(i, u, mark=False) for i in sim_idx], 5)
            where = " The orange box marks where on the input this unit fired most." if kind == "spatial" else ""
            p1 = EXPLAIN_IMG.format(about=about, n=len(top), where=where)
            p2 = lambda e: SIM_IMG.format(about=about, explanation=e, n=len(sim_idx))
        if a.dry_run:
            return {"id": f"{prefix}:{u}", "prompt": p1, "image": img1, "sim_prompt": p2("<explanation>"), "sim_image": img2}
        ex, served = ask(client, p1, EXPLAIN_SCHEMA, "medium", img1)
        sim, _ = ask(client, p2(ex["explanation"]), SIM_SCHEMA, "medium", img2)
        pred = np.array(sim["predictions"][: len(sim_idx)], float)
        if len(pred) != len(sim_idx):
            raise RuntimeError(f"simulator returned {len(pred)} predictions for {len(sim_idx)} items")
        return {
            "id": f"{prefix}:{u}", "unit": int(u), "explanation": ex["explanation"].strip(), "tags": [t.lower() for t in ex["tags"]][:6],
            "score": round(pearson(pred, acts[sim_idx]), 3), "n_explain_examples": len(top), "n_sim": len(sim_idx), "model": served,
        }

    out_path = Path(a.out or ROOT / "scripts" / ".cache" / f"lab_explanations_{Path(a.model).stem}.json")
    if a.dry_run:
        prev = ROOT / "scripts" / ".cache" / "explain_lab_preview"
        prev.mkdir(parents=True, exist_ok=True)
        for u in units[:3]:
            r = run_unit(u)
            (prev / f"{r['id'].replace(':', '_')}.txt").write_text(r["prompt"] + "\n\n---\n\n" + r["sim_prompt"])
            for key in ("image", "sim_image"):
                if r[key]:
                    (prev / f"{r['id'].replace(':', '_')}_{key}.png").write_bytes(base64.b64decode(r[key]))
        print(f"dry run: previews of the first {min(3, len(units))} units in {prev}")
        return

    import anthropic
    load_env()
    client = anthropic.Anthropic()
    results = []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(run_unit, u): u for u in units}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                print(f"  {r['id']:>7} r={r['score']:+.2f}  {r['explanation']}")
            except Exception as e:  # report and keep going; reruns redo failures
                print(f"  failed {prefix}:{futs[fut]}: {e}")
    out_path.write_text(json.dumps({
        "kind": "inann-lab-explanations", "model": Path(a.model).name, "tensor": tensor, "sae": bool(sae),
        "method": "Bills et al. 2023: explain from top 12 of the even items, simulate 10 top + 10 random odd items, score = Pearson r",
        "explainer_model": MODEL, "units": sorted(results, key=lambda r: r["unit"]),
    }, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

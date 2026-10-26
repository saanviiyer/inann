"""A local steering server for inann's LLM steering page.

    python3 steer/server.py --model Qwen/Qwen3.5-4B          # the model from the recipe (9.3 GB)
    python3 steer/server.py --model Qwen/Qwen3-1.7B          # smaller, same MLP layout

It binds to 127.0.0.1 only and answers JSON:
    GET  /info                       model, layers, MLP width, device
    POST /run    {prompt, system, chat, thinking, interventions: [{layer, neuron, dose}],
                  max_new_tokens, temperature, seed, control}
                 -> baseline and steered completions, the recipe's numbers, and the steered
                    neurons' activations on every token (prompt and completion, both runs);
                    with control: true, also a run with random neurons at the same layers and doses
    POST /sweep  {..., multipliers: [0, 0.5, 1, ...]} -> one steered completion per multiplier
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from steer import Intervention, find_layers, flat, generate, random_control, recipe_values, recording, residual_norms, steering  # noqa: E402

STATE = {}
LOCK = threading.Lock()


def load(model_id, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    layers = find_layers(model)
    STATE.update(model=model, tok=tok, model_id=model_id, device=device, n_layers=len(layers),
                 d_mlp=layers[0].mlp.down_proj.weight.shape[1], d_model=layers[0].mlp.down_proj.weight.shape[0])


def encode(body):
    tok = STATE["tok"]
    prompt = body.get("prompt", "")
    if body.get("chat", True) and getattr(tok, "chat_template", None):
        msgs = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(body.get("thinking", False)))
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tok(text, return_tensors="pt").input_ids.to(STATE["device"])


def parse_ivs(body, mult=1.0):
    ivs = []
    for x in body.get("interventions", []):
        L, n = int(x["layer"]), int(x["neuron"])
        if not (0 <= L < STATE["n_layers"]) or not (0 <= n < STATE["d_mlp"]):
            raise ValueError(f"layer {L} neuron {n} is outside this model ({STATE['n_layers']} layers, {STATE['d_mlp']} neurons)")
        ivs.append(Intervention(L, n, float(x["dose"]) * mult))
    return ivs


def one_run(ids, values, body, wanted):
    tok, model = STATE["tok"], STATE["model"]
    t0 = time.time()
    with steering(model, values), recording(model, wanted) as rec:
        gen = generate(model, tok, ids, int(body.get("max_new_tokens", 120)), float(body.get("temperature", 0)), int(body.get("seed", 0)))
    return {"text": tok.decode(gen, skip_special_tokens=True), "tokens": [tok.decode([t]) for t in gen.tolist()],
            "acts": flat(rec), "seconds": round(time.time() - t0, 2)}


def run(body):
    model, tok = STATE["model"], STATE["tok"]
    ids = encode(body)
    ivs = parse_ivs(body)
    norms = residual_norms(model, ids)
    values = recipe_values(model, ids, ivs, norms)
    wanted = [(iv.layer, iv.neuron) for iv in ivs]
    out = {"prompt_tokens": [tok.decode([t]) for t in ids[0].tolist()], "recipe": values,
           "median_resid_by_layer": [n[1:].median().item() if n.shape[0] > 1 else n[0].item() for n in norms]}
    out["baseline"] = one_run(ids, [], body, wanted)
    out["steered"] = one_run(ids, values, body, wanted)
    if body.get("control"):
        ctrl = random_control(ivs, STATE["d_mlp"], seed=int(body.get("seed", 0)) + 1)
        cvals = recipe_values(model, ids, ctrl, norms)
        out["control"] = {**one_run(ids, cvals, body, [(c.layer, c.neuron) for c in ctrl]), "recipe": cvals}
    return out


def sweep(body):
    model = STATE["model"]
    ids = encode(body)
    norms = residual_norms(model, ids)
    rows = []
    for m in body.get("multipliers", [0, 0.5, 1, 1.5, 2]):
        ivs = parse_ivs(body, float(m))
        values = recipe_values(model, ids, ivs, norms)
        r = one_run(ids, values, body, [(iv.layer, iv.neuron) for iv in ivs])
        rows.append({"multiplier": m, "text": r["text"], "seconds": r["seconds"]})
    return {"rows": rows}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _cors(self):
        # the page may be served from localhost or from saanviiyer.github.io; the server only listens on loopback
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/info"):
            return self._send(200, {k: STATE[k] for k in ("model_id", "device", "n_layers", "d_mlp", "d_model")})
        self._send(404, {"error": "unknown path"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            fn = {"/run": run, "/sweep": sweep}.get(self.path)
            if not fn:
                return self._send(404, {"error": "unknown path"})
            with LOCK:
                return self._send(200, fn(body))
        except Exception as e:  # report to the page instead of dropping the connection
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("[steer] " + fmt % args + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    a = ap.parse_args()
    print(f"[steer] loading {a.model} on {a.device} ({a.dtype})...", flush=True)
    load(a.model, a.device, getattr(torch, a.dtype))
    print(f"[steer] {STATE['n_layers']} layers, {STATE['d_mlp']} MLP neurons per layer; listening on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

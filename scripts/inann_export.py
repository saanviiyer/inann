"""Export a PyTorch model (and optionally a dataset) in the form the inann Lab reads.

    from inann_export import export
    export(model, example_input, "out/my_model",
           classes=["cat", "dog"], input_kind="image", norm="imagenet",
           dataset=(images, labels))        # images: (N, C, H, W) in 0..1, or token ids (N, T)

writes out/my_model.onnx, out/my_model.meta.json and, with a dataset, out/my_model_data.json.
Drop the .onnx and .meta.json on the Lab page; the meta file points at the dataset, so with a
static server (or ?model=&meta= links) the dataset loads too.
"""
import base64
import json
from pathlib import Path

import numpy as np
import torch


def export(model, example_input, out_prefix, classes=None, input_kind=None, norm="unit", preprocess="fit",
           vocab=None, dataset=None, name=None, description=None, opset=17, max_items=1000):
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    x = example_input
    torch.onnx.export(model, x, f"{out}.onnx", input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=opset, dynamo=False)
    if input_kind is None:
        input_kind = "tokens" if not torch.is_floating_point(x) else "image" if x.dim() == 4 else "vector"
    meta = {"name": name or out.name, "input": {"kind": input_kind}}
    if description:
        meta["description"] = description
    if input_kind == "image":
        meta["input"].update({"norm": norm, "preprocess": preprocess})
    if input_kind == "tokens":
        meta["input"]["dims"] = list(x.shape[1:])
        if vocab:
            meta["input"].update({"vocab": list(vocab), "tokenizer": "char"})
    if classes:
        meta["classes"] = [str(c) for c in classes]
    if dataset is not None:
        X, y = dataset if isinstance(dataset, tuple) else (dataset, None)
        X = X[:max_items].detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)[:max_items]
        entry = {"source": "exported with inann_export.py", "shape": list(X.shape[1:])}
        if input_kind == "tokens":
            entry.update({"dtype": "int32", "data": base64.b64encode(X.astype(np.int32).tobytes()).decode()})
        elif input_kind == "image":
            if X.min() < -1e-6 or X.max() > 1 + 1e-6:
                raise ValueError("images must be in 0..1 (un-normalized); set norm= to tell the Lab how the model normalizes")
            entry.update({"range": [0, 1], "data": base64.b64encode((X * 255).round().astype(np.uint8).tobytes()).decode()})
        else:
            entry.update({"dtype": "float32", "data": base64.b64encode(X.astype(np.float32).tobytes()).decode()})
        if y is not None:
            yy = y[:max_items].detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)[:max_items]
            entry["labels"] = [int(v) for v in yy]
            if classes:
                entry["classes"] = meta["classes"]
        Path(f"{out}_data.json").write_text(json.dumps(entry))
        meta["dataset"] = f"{out.name}_data.json"
    Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=1))
    return f"{out}.onnx"


if __name__ == "__main__":
    # self-test: a tiny MLP on random 4-feature vectors
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    X = torch.randn(200, 4)
    y = m(X).argmax(1)
    p = export(m, X[:1], "scripts/.cache/export_selftest/tiny", classes=["a", "b", "c"], dataset=(X, y))
    meta = json.loads(Path("scripts/.cache/export_selftest/tiny.meta.json").read_text())
    data = json.loads(Path("scripts/.cache/export_selftest/tiny_data.json").read_text())
    assert meta["input"]["kind"] == "vector" and meta["dataset"] == "tiny_data.json"
    arr = np.frombuffer(base64.b64decode(data["data"]), dtype=np.float32).reshape(-1, 4)
    assert np.allclose(arr, X.numpy()) and data["labels"][:5] == y[:5].tolist()
    print("self-test OK:", p)

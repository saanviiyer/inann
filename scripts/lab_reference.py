"""Reference values for scripts/test_lab_engine.mjs, computed independently with the official
onnx Python tooling: every intermediate tensor of each example model on one input, plus the
logits when one hidden tensor is scaled by 0.5 through an override built with onnx.helper."""
import json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from common import ROOT

EX = ROOT / "public" / "examples"
out = {}
rng = np.random.default_rng(0)
for name, shape in (("mnist_mlp", (1, 1, 28, 28)), ("mnist_cnn", (1, 1, 28, 28)), ("optdigits_cnn", (1, 1, 8, 8))):
    m = onnx.load(EX / f"{name}.onnx")
    x = rng.random(shape, dtype=np.float32)
    inter = [o for n in m.graph.node for o in n.output if n.op_type in ("Relu", "Gemm", "Conv", "MaxPool", "Flatten")]
    probe = onnx.ModelProto(); probe.CopyFrom(m)
    for t in inter:
        probe.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    s = ort.InferenceSession(probe.SerializeToString())
    vals = dict(zip([o.name for o in s.get_outputs()], s.run(None, {"input": x})))
    relus = [t for n in m.graph.node for t in n.output if n.op_type == "Relu"]
    cut = relus[-1]
    ov = onnx.ModelProto(); ov.CopyFrom(m)
    for n in ov.graph.node:
        n.output[:] = [o + "__ref_orig" if o == cut else o for o in n.output]
    ov.graph.input.append(helper.make_tensor_value_info(cut, TensorProto.FLOAT, list(vals[cut].shape)))
    s2 = ort.InferenceSession(ov.SerializeToString())
    logits_half = s2.run(None, {"input": x, cut: vals[cut] * 0.5})[0]
    out[name] = {
        "x": x.ravel().tolist(), "shape": list(shape), "cut": cut,
        "tensors": {k: {"shape": list(v.shape), "sum": float(v.sum()), "first": v.ravel()[:5].tolist()} for k, v in vals.items()},
        "logits_half": logits_half.ravel().tolist(),
    }
(ROOT / "scripts" / ".cache" / "lab_reference.json").write_text(json.dumps(out))
print({k: (len(v["tensors"]), v["cut"]) for k, v in out.items()})

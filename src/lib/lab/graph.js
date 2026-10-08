// ONNX graph surgery for the Lab, shared by the browser and the Node tests.
//  - decode / encode ModelProto with protobufjs and the vendored onnx.proto
//  - expose intermediate tensors as extra graph outputs (the "probe" model)
//  - replace one tensor by a graph input (the "override" model) so it can be clamped or
//    rewritten by an SAE while the rest of the graph, skip connections included, runs normally
import protobuf from 'protobufjs';

let T = null;
export function initProto(protoText) {
  if (T) return;
  const root = protobuf.parse(protoText, { keepCase: true }).root;
  T = {
    Model: root.lookupType('onnx.ModelProto'),
    Graph: root.lookupType('onnx.GraphProto'),
    Node: root.lookupType('onnx.NodeProto'),
    ValueInfo: root.lookupType('onnx.ValueInfoProto'),
  };
}

export const ELEM = { FLOAT: 1, UINT8: 2, INT8: 3, INT32: 6, INT64: 7, BOOL: 9, FLOAT16: 10, DOUBLE: 11 };
const num = (v) => (v == null ? null : typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : v.toNumber ? v.toNumber() : Number(v));

export const decodeModel = (bytes) => T.Model.decode(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
export const encodeModel = (m) => T.Model.encode(m).finish();

const ACTIVATIONS = new Set(['Relu', 'LeakyRelu', 'PRelu', 'Elu', 'Selu', 'Celu', 'Gelu', 'Sigmoid', 'HardSigmoid', 'Tanh', 'Softplus', 'Softsign', 'HardSwish', 'Mish', 'Clip', 'Softmax', 'LogSoftmax']);
const NON_FLOAT_OPS = new Set(['Shape', 'Size', 'Constant', 'ConstantOfShape', 'Range', 'NonZero', 'ArgMax', 'ArgMin', 'Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'Not', 'And', 'Or', 'Xor', 'IsNaN', 'IsInf', 'NonMaxSuppression']);
const LINEAR = new Set(['Conv', 'ConvTranspose', 'Gemm', 'MatMul']);

function shapeOf(vi) {
  const tt = vi?.type?.tensor_type;
  if (!tt) return null;
  return { elem: tt.elem_type, dims: (tt.shape?.dim ?? []).map((d) => (d.dim_param ? d.dim_param : num(d.dim_value))) };
}

export function summarize(model) {
  const g = model.graph;
  const inits = new Set(g.initializer.map((t) => t.name));
  const inputs = g.input.filter((i) => !inits.has(i.name)).map((i) => ({ name: i.name, ...shapeOf(i) }));
  const outputs = g.output.map((o) => ({ name: o.name, ...shapeOf(o) }));
  const opCounts = {};
  for (const n of g.node) opCounts[n.op_type] = (opCounts[n.op_type] ?? 0) + 1;
  let params = 0;
  for (const t of g.initializer) params += t.dims.reduce((a, d) => a * num(d), 1);
  const producer = {};
  g.node.forEach((n, i) => n.output.forEach((o) => (producer[o] = i)));
  const known = {};
  for (const vi of [...g.value_info, ...g.output, ...g.input]) known[vi.name] = shapeOf(vi);
  return {
    inputs, outputs, opCounts, params, nodes: g.node.length, inits, producer,
    opset: (model.opset_import ?? []).map((o) => `${o.domain || 'ai.onnx'} ${num(o.version)}`),
    producerName: model.producer_name, known,
  };
}

// Integer tensors, propagated statically: shape arithmetic starts at Shape/Constant-int/etc. and
// most ops keep the type of their first input. This keeps ORT retries rare.
function intTensors(model) {
  const ints = new Set();
  for (const t of model.graph.initializer) if (t.data_type !== ELEM.FLOAT && t.data_type !== ELEM.FLOAT16 && t.data_type !== ELEM.DOUBLE) ints.add(t.name);
  for (const n of model.graph.node) {
    let isInt;
    if (n.op_type === 'Constant') {
      const v = n.attribute.find((a) => a.name === 'value');
      isInt = v ? v.t && v.t.data_type !== ELEM.FLOAT : n.attribute.some((a) => a.name === 'value_int' || a.name === 'value_ints');
    } else if (n.op_type === 'Cast') {
      const to = num(n.attribute.find((a) => a.name === 'to')?.i);
      isInt = to !== ELEM.FLOAT && to !== ELEM.FLOAT16 && to !== ELEM.DOUBLE;
    } else if (NON_FLOAT_OPS.has(n.op_type)) isInt = true;
    else isInt = n.input.length > 0 && ints.has(n.input[0]);
    if (isInt) n.output.forEach((o) => ints.add(o));
  }
  return ints;
}

// Every node output that is plausibly a float activation, in graph order.
export function candidateTensors(model, summary, limit = 400) {
  const out = [];
  const outputNames = new Set(model.graph.output.map((o) => o.name));
  const ints = intTensors(model);
  model.graph.node.forEach((n, i) => {
    if (NON_FLOAT_OPS.has(n.op_type)) return;
    n.output.forEach((o, k) => {
      if (!o || ints.has(o)) return;
      if (n.op_type === 'TopK' && k === 1) return;
      const kn = summary.known[o];
      if (kn && kn.elem && kn.elem !== ELEM.FLOAT) return;
      out.push({ name: o, op: n.op_type, node: i, nodeName: n.name, isOutput: outputNames.has(o) });
    });
  });
  return out.slice(0, limit);
}

// Default rooms: activation outputs (including GELU that an exporter split into Erf/Add/Mul),
// linear outputs that feed no activation (e.g. logits; a MatMul followed by a bias Add counts as
// the Add), and graph outputs.
export function defaultRooms(model, cands) {
  const g = model.graph;
  const consumers = {};
  g.node.forEach((n) => n.input.forEach((x) => (consumers[x] ??= []).push(n)));
  const candSet = new Set(cands.map((c) => c.name));
  const picks = new Set();
  // decomposed GELU: Erf -> Add -> Mul -> Mul; take the last tensor of that chain
  for (const n of g.node) {
    if (n.op_type !== 'Erf') continue;
    let t = n.output[0];
    for (let hop = 0; hop < 4; hop++) {
      const c = consumers[t] ?? [];
      if (c.length !== 1 || !['Add', 'Mul'].includes(c[0].op_type)) break;
      t = c[0].output[0];
    }
    if (candSet.has(t)) picks.add(t);
  }
  const follow = (t) => {
    // a bias add right after a linear op is part of that layer
    const c = consumers[t] ?? [];
    if (c.length === 1 && c[0].op_type === 'Add' && c[0].input.some((x) => g.initializer.some((i) => i.name === x))) return c[0].output[0];
    return t;
  };
  for (const c of cands) {
    if (c.isOutput || ACTIVATIONS.has(c.op) || c.op === 'LayerNormalization') { picks.add(c.name); continue; }
    if (LINEAR.has(c.op)) {
      const t = follow(c.name);
      const next = (consumers[t] ?? []).map((n) => n.op_type);
      const feedsAct = next.some((op) => ACTIVATIONS.has(op) || op === 'BatchNormalization' || op === 'Erf' || op === 'Div' || op === 'LayerNormalization');
      if (!feedsAct && candSet.has(t)) picks.add(t);
    }
  }
  const order = new Map(cands.map((c, i) => [c.name, i]));
  let list = [...picks].filter((t) => order.has(t)).sort((a, b) => order.get(a) - order.get(b));
  // deep networks: keep the first activation, every residual block output (an activation fed
  // by an Add), and the outputs, so a ResNet opens with one room per block instead of every ReLU
  if (list.length > 14) {
    const producer = {};
    g.node.forEach((n) => n.output.forEach((o) => (producer[o] = n)));
    const blockOut = (t) => { const n = producer[t]; return n && ACTIVATIONS.has(n.op_type) && producer[n.input[0]]?.op_type === 'Add'; };
    const keep = list.filter((t, i) => i === 0 || blockOut(t) || cands[order.get(t)].isOutput);
    if (keep.length >= 3) list = keep;
    if (list.length > 14) list = list.filter((t, i) => i % Math.ceil(list.length / 12) === 0 || cands[order.get(t)].isOutput);
  }
  if (list.length >= 2) return list;
  return cands.filter((c) => LINEAR.has(c.op) || c.isOutput).map((c) => c.name);
}

function shallowModel(model, graphPatch) {
  const g = T.Graph.create({ ...model.graph, ...graphPatch });
  return T.Model.create({ ...model, graph: g });
}

const floatOut = (name, elem = ELEM.FLOAT) => T.ValueInfo.create({ name, type: { tensor_type: { elem_type: elem } } });

// The probe model: the original graph with extra outputs.
export function withOutputs(model, names) {
  const have = new Set(model.graph.output.map((o) => o.name));
  const extra = names.filter((n) => !have.has(n)).map((n) => floatOut(n));
  return shallowModel(model, { output: [...model.graph.output, ...extra] });
}

// The override model: tensor `name` is produced as `${name}__inann_orig` and read back from a new
// graph input called `name`, so whatever is fed there replaces it for every consumer.
export function withOverride(model, name, dimsNoBatch, outputs) {
  const g = model.graph;
  const node = g.node.map((n) => (n.output.includes(name) ? T.Node.create({ ...n, output: n.output.map((o) => (o === name ? `${name}__inann_orig` : o)) }) : n));
  const input = T.ValueInfo.create({
    name,
    type: { tensor_type: { elem_type: ELEM.FLOAT, shape: { dim: [{ dim_param: 'inann_batch' }, ...dimsNoBatch.map((d) => ({ dim_value: d }))] } } },
  });
  const have = new Set(g.output.map((o) => o.name));
  // a tensor that is now an input cannot also be an output
  const outs = g.output.filter((o) => o.name !== name);
  const extra = outputs.filter((n) => n !== name && !have.has(n)).map((n) => floatOut(n));
  return shallowModel(model, { node, input: [...g.input, input], output: [...outs, ...extra] });
}

// ORT rejects an extra output whose declared type is wrong. Its error names the tensor and the real
// type; we drop non-float tensors and retry. Returns the names to drop, or null if not a type error.
export function badOutputsFromError(msg) {
  const re = /output arg \(([^)]+)\)[^]*?Type \(?tensor\((\w+)\)|Type \(tensor\((\w+)\)\) of output arg \(([^)]+)\)/g;
  const names = new Set();
  let m;
  while ((m = re.exec(msg))) names.add(m[1] ?? m[4]);
  return names.size ? [...names] : null;
}

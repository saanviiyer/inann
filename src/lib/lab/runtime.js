// The Lab's model runtime: load any ONNX file in the browser with onnxruntime-web, expose its
// intermediate tensors, describe their layout, and run "override" passes that rewrite one tensor.
import * as ort from 'onnxruntime-web/wasm';
// let Vite serve (and, on build, emit) the runtime's own files, so no copy step is needed
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import protoText from './onnx.proto?raw';
import {
  initProto, decodeModel, encodeModel, summarize, candidateTensors, defaultRooms, withOutputs, withOverride, badOutputsFromError,
} from './graph.js';

initProto(protoText);
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which a static host may not give
ort.env.logLevel = 'error';

async function createWithRetry(model, names) {
  let keep = [...names];
  for (let tries = 0; tries < 60; tries++) {
    try {
      const session = await ort.InferenceSession.create(encodeModel(withOutputs(model, keep)), { executionProviders: ['wasm'] });
      return { session, kept: keep };
    } catch (e) {
      const bad = badOutputsFromError(String(e?.message ?? e));
      if (!bad) throw e;
      keep = keep.filter((n) => !bad.includes(n));
    }
  }
  throw new Error('could not expose intermediate tensors');
}

// How a tensor (without its batch dim) splits into channels and positions.
//   [C,H,W] -> C channels over an HxW map;  [T,D] -> D channels over T tokens;  [D] -> D units
export { layoutOf } from './runtime-layout.js';
import { layoutOf } from './runtime-layout.js';

export class LabModel {
  static async load(bytes, meta = {}, fileName = 'model.onnx') {
    const model = decodeModel(new Uint8Array(bytes));
    const summary = summarize(model);
    if (summary.inputs.length === 0) throw new Error('the model has no inputs');
    const input = summary.inputs[0];
    // float32 inputs (images, vectors) and integer token ids (int64 / int32) are supported
    const TYPES = { 1: 'float32', 7: 'int64', 6: 'int32' };
    if (input.elem && !TYPES[input.elem]) throw new Error(`the first input has ONNX element type ${input.elem}; supported: float32, int64 and int32 token ids`);
    if (summary.inputs.length > 1) console.warn('inann: the model has several inputs; only the first is driven (masks get ones, positions 0..T-1, others zeros)');
    const cands = candidateTensors(model, summary);
    const { session, kept } = await createWithRetry(model, cands.map((c) => c.name));
    const m = new LabModel();
    Object.assign(m, { model, summary, meta, fileName, session, input, inputType: TYPES[input.elem ?? 1] });
    m.extraInputs = summary.inputs.slice(1);
    m.cands = cands.filter((c) => kept.includes(c.name));
    m.batchable = typeof input.dims[0] === 'string' || input.dims[0] == null || input.dims[0] <= 0;
    // symbolic dims beyond the batch need a value; meta.input.dims can supply them, else 16
    m.symbolic = input.dims.slice(1).map((d) => !(typeof d === 'number' && d > 0));
    m.inputDims = input.dims.slice(1).map((d, i) => (typeof d === 'number' && d > 0 ? d : meta.input?.dims?.[i] ?? 16));
    m.inputSize = m.inputDims.reduce((a, b) => a * b, 1);
    // one pass on zeros to learn every tensor's shape
    const out = await m.run([new Float32Array(m.inputSize)]);
    m.shapes = {};
    for (const [k, v] of Object.entries(out)) m.shapes[k] = v.dims.slice(1);
    m.cands = m.cands.filter((c) => m.shapes[c.name]);
    // the first float output is the one read as the answer (an argmax/label output may come first)
    m.outputName = (summary.outputs.find((o) => out[o.name] && out[o.name].data instanceof Float32Array) ?? summary.outputs[0]).name;
    const roomNames = defaultRooms(model, m.cands);
    m.allRooms = m.cands.map((c) => m.describe(c.name));
    m.rooms = roomNames.map((n) => m.describe(n));
    m.overrides = new Map();
    return m;
  }

  describe(name) {
    const c = this.cands.find((k) => k.name === name);
    const dims = this.shapes[name];
    return { name, op: c?.op ?? '?', dims, layout: layoutOf(dims), isOutput: name === this.outputName || !!c?.isOutput };
  }

  // xs: array of Float32Array model inputs (already normalized). Returns { name: {dims, data} } for
  // every exposed tensor, with data concatenated over the batch.
  async run(xs, session = this.session, extraFeeds = {}) {
    const name = this.input.name;
    const feedFor = (arr, B) => {
      let data;
      if (this.inputType === 'int64') {
        data = new BigInt64Array(B * this.inputSize);
        arr.forEach((x, b) => x.forEach((v, j) => (data[b * this.inputSize + j] = BigInt(Math.round(v)))));
      } else {
        data = this.inputType === 'int32' ? new Int32Array(B * this.inputSize) : new Float32Array(B * this.inputSize);
        arr.forEach((x, b) => data.set(this.inputType === 'int32' ? Int32Array.from(x, Math.round) : x, b * this.inputSize));
      }
      return new ort.Tensor(this.inputType, data, [B, ...this.inputDims]);
    };
    // further inputs, filled the way Hugging Face exports expect: attention masks with ones,
    // position ids with 0..T-1, everything else (token type ids, ...) with zeros. Symbolic dims
    // take the main input's sizes where they line up.
    const extraZeros = (B) => Object.fromEntries((this.extraInputs ?? []).map((inp) => {
      const dims = inp.dims.map((d, i) => (i === 0 ? B : typeof d === 'number' && d > 0 ? d : this.inputDims[i - 1] ?? 1));
      const n = dims.reduce((a, b) => a * b, 1);
      const nm = inp.name.toLowerCase();
      const per = n / B, last = dims[dims.length - 1];
      const fill = (j) => (nm.includes('mask') ? 1 : nm.includes('position') ? j % last : 0);
      const t = inp.elem === 7
        ? new ort.Tensor('int64', BigInt64Array.from({ length: n }, (_, j) => BigInt(fill(j % per))), dims)
        : inp.elem === 6
          ? new ort.Tensor('int32', Int32Array.from({ length: n }, (_, j) => fill(j % per)), dims)
          : new ort.Tensor('float32', Float32Array.from({ length: n }, (_, j) => fill(j % per)), dims);
      return [inp.name, t];
    }));
    if (this.batchable || xs.length === 1) {
      const res = await session.run({ [name]: feedFor(xs, xs.length), ...extraZeros(xs.length), ...extraFeeds });
      return Object.fromEntries(Object.entries(res).map(([k, t]) => [k, { dims: t.dims, data: t.data }]));
    }
    // fixed batch of 1: run one at a time and stitch
    const parts = [];
    for (const x of xs) parts.push(await session.run({ [name]: feedFor([x], 1), ...extraZeros(1), ...extraFeeds }));
    const out = {};
    for (const k of Object.keys(parts[0])) {
      const per = parts[0][k].data.length;
      const data = new parts[0][k].data.constructor(per * xs.length);
      parts.forEach((p, i) => data.set(p[k].data, i * per));
      out[k] = { dims: [xs.length, ...parts[0][k].dims.slice(1)], data };
    }
    return out;
  }

  async overrideSession(name) {
    if (!this.overrides.has(name)) {
      const outs = this.cands.map((c) => c.name).filter((n) => n !== name);
      const ov = withOverride(this.model, name, this.shapes[name], outs);
      this.overrides.set(name, await ort.InferenceSession.create(encodeModel(ov), { executionProviders: ['wasm'] }));
    }
    return this.overrides.get(name);
  }

  // Run one input with tensor `name` rewritten by transform(data) (a Float32Array without batch).
  async runWith(x, name, transform) {
    const base = await this.run([x]);
    const t = base[name];
    const data = transform(Float32Array.from(t.data));
    const s = await this.overrideSession(name);
    const out = await this.run([x], s, { [name]: new ort.Tensor('float32', data, t.dims) });
    out[name] = { dims: t.dims, data };
    // tensors upstream of the cut are not outputs of the override graph; keep the base values
    for (const [k, v] of Object.entries(base)) if (!out[k]) out[k] = v;
    return out;
  }
}

export const softmax = (logits) => {
  const m = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
};

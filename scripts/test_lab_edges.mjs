// Checks the Lab's incoming-edge contributions are exact: summed over every input (no top-k cut)
// and plus the bias, they must equal the unit's pre-activation as computed by ONNX Runtime.
import fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { initProto, decodeModel, encodeModel, summarize, candidateTensors, defaultRooms, withOutputs, initializerArray } from '../src/lib/lab/graph.js';
import { unitEdges } from '../src/lib/lab/edges.js';
import { layoutOf } from '../src/lib/lab/runtime-layout.js';

initProto(fs.readFileSync(new URL('../src/lib/lab/onnx.proto', import.meta.url), 'utf8'));
let fail = 0;
for (const name of ['mnist_cnn', 'mnist_mlp', 'cifar_cnn']) {
  const model = decodeModel(fs.readFileSync(new URL(`../public/examples/${name}.onnx`, import.meta.url)));
  const sum = summarize(model);
  const cands = candidateTensors(model, sum);
  const rooms = defaultRooms(model, cands);
  const s = await ort.InferenceSession.create(encodeModel(withOutputs(model, cands.map((c) => c.name))));
  const inDims = sum.inputs[0].dims.slice(1);
  let r = 3;
  const x = Float32Array.from({ length: inDims.reduce((a, b) => a * b, 1) }, () => { r = (r * 16807) % 2147483647; return r / 2147483647; });
  const outs = await s.run({ input: new ort.Tensor('float32', x, [1, ...inDims]) });
  const lab = { model, input: { name: sum.inputs[0].name }, inputDims: inDims };
  const producer = {};
  model.graph.node.forEach((n) => n.output.forEach((o) => (producer[o] = n)));
  for (const room of rooms) {
    const dims = outs[room].dims.slice(1);
    const L = layoutOf(dims);
    const index = Math.floor(L.size / 2) + 1;
    const e = unitEdges(lab, rooms, outs, room, index, L, x, Infinity);
    if (!e) continue;
    // the linear op feeding this room, its bias, and its output (the pre-activation)
    let n = producer[room];
    while (n && !['Conv', 'Gemm', 'MatMul'].includes(n.op_type)) n = producer[n.input[0]];
    const pre = outs[n.output[0]].data[index];
    const c = L.kind === 'spatial' ? Math.floor(index / L.P) : index;
    const bias = n.input[2] ? initializerArray(model, n.input[2]).data[c] : 0;
    const total = e.edges.reduce((a, t) => a + t.v, 0) + bias;
    const ok = !e.exact || Math.abs(total - pre) < 1e-3 * (1 + Math.abs(pre));
    if (!ok) fail++;
    console.log(`${name} ${room.split('/').pop()} <- ${e.fromRoom.split('/').pop()}: sum ${total.toFixed(4)} vs pre-activation ${pre.toFixed(4)} ${e.exact ? (ok ? 'exact' : 'MISMATCH') : '(approximate, not checked)'}`);
  }
}
// Harder cases with every candidate tensor as a room: depthwise convs (MobileNetV2) and per-token
// MatMuls (a tiny BERT). Skipped when those files have not been exported into scripts/.cache.
for (const [file, feed] of [
  ['mobilenet_v2.onnx', (inDims) => ({ input: new ort.Tensor('float32', Float32Array.from({ length: 3 * 224 * 224 }, (_, i) => Math.sin(i) * 0.5), [1, 3, 224, 224]) })],
  ['tiny_bert.onnx', () => ({ input_ids: new ort.Tensor('int64', BigInt64Array.from({ length: 12 }, (_, i) => BigInt((i * 7) % 50)), [1, 12]), attention_mask: new ort.Tensor('int64', new BigInt64Array(12).fill(1n), [1, 12]) })],
]) {
  const url = new URL(`./.cache/${file}`, import.meta.url);
  if (!fs.existsSync(url)) { console.log(`SKIP ${file}: not exported`); continue; }
  const model = decodeModel(fs.readFileSync(url));
  const sum = summarize(model);
  const cands = candidateTensors(model, sum);
  let s;
  try { s = await ort.InferenceSession.create(encodeModel(withOutputs(model, cands.map((c) => c.name)))); } catch { console.log(`SKIP ${file}: could not expose all tensors`); continue; }
  const feeds = feed();
  const outs = await s.run(feeds);
  const roomNames = cands.map((c) => c.name).filter((n) => outs[n]);
  const inDims = sum.inputs[0].dims.slice(1).map((d) => (typeof d === 'number' && d > 0 ? d : 12));
  const lab = { model, input: { name: sum.inputs[0].name }, inputDims: inDims };
  const xIn = Float32Array.from(feeds[sum.inputs[0].name].data, Number);
  const producer = {};
  model.graph.node.forEach((n) => n.output.forEach((o) => (producer[o] = n)));
  let checked = 0, bad = 0;
  for (const room of roomNames) {
    const n = producer[room];
    if (!n || !['Conv', 'MatMul', 'Gemm'].includes(n.op_type)) continue; // check linear outputs directly
    const L = layoutOf(outs[room].dims.slice(1));
    for (const index of [0, Math.floor(L.size / 3) + 5, L.size - 1]) {
      const e = unitEdges(lab, roomNames, outs, room, index, L, xIn, Infinity);
      if (!e || !e.exact) continue;
      const c = L.kind === 'spatial' ? Math.floor(index / L.P) : L.kind === 'seq' ? index % L.C : index;
      const b = n.input[2] ? initializerArray(model, n.input[2]) : null;
      if (n.input[2] && !b) continue; // bias computed elsewhere in the graph; not checkable here
      const bias = b ? b.data[c] : 0;
      const total = e.edges.reduce((a, t) => a + t.v, 0) + bias;
      const pre = outs[room].data[index];
      checked++;
      if (Math.abs(total - pre) > 1e-3 * (1 + Math.abs(pre))) { bad++; if (bad < 4) console.log(`  ${file} ${room} #${index}: ${total.toFixed(4)} vs ${pre.toFixed(4)}`); }
    }
  }
  console.log(`${file}: ${checked} exact units checked, ${bad} mismatches`);
  if (bad) fail++;
}
if (fail) { console.error('FAIL'); process.exit(1); }
console.log('OK edges');

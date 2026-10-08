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
if (fail) { console.error('FAIL'); process.exit(1); }
console.log('OK edges');

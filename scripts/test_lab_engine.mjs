// Tests the Lab's graph surgery (src/lib/lab/graph.js) against references computed with the
// official onnx Python tooling (scripts/lab_reference.py), using onnxruntime-node.
import fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { initProto, decodeModel, encodeModel, summarize, candidateTensors, defaultRooms, withOutputs, withOverride } from '../src/lib/lab/graph.js';

initProto(fs.readFileSync(new URL('../src/lib/lab/onnx.proto', import.meta.url), 'utf8'));
const ref = JSON.parse(fs.readFileSync(new URL('./.cache/lab_reference.json', import.meta.url)));
let fail = 0;
const close = (a, b, tol = 1e-4) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

for (const [name, r] of Object.entries(ref)) {
  const model = decodeModel(fs.readFileSync(new URL(`../public/examples/${name}.onnx`, import.meta.url)));
  const sum = summarize(model);
  const cands = candidateTensors(model, sum);
  const rooms = defaultRooms(model, cands);
  const probe = withOutputs(model, cands.map((c) => c.name));
  const s = await ort.InferenceSession.create(encodeModel(probe));
  const x = new ort.Tensor('float32', Float32Array.from(r.x), r.shape);
  const res = await s.run({ input: x });
  let checked = 0;
  for (const [t, v] of Object.entries(r.tensors)) {
    const got = res[t];
    if (!got) { console.log(`  ${name}: missing ${t}`); fail++; continue; }
    const gsum = got.data.reduce((a, b) => a + b, 0);
    if (!close(gsum, v.sum, 1e-3) || JSON.stringify(got.dims) !== JSON.stringify(v.shape)) { console.log(`  ${name}: ${t} mismatch`, gsum, v.sum, got.dims, v.shape); fail++; }
    checked++;
  }
  // override: scale the last ReLU by 0.5 and compare logits with the Python-built override
  const cutDims = res[r.cut].dims.slice(1);
  const ov = withOverride(model, r.cut, cutDims, rooms);
  const s2 = await ort.InferenceSession.create(encodeModel(ov));
  const half = new ort.Tensor('float32', res[r.cut].data.map((v) => v * 0.5), res[r.cut].dims);
  const res2 = await s2.run({ input: x, [r.cut]: half });
  const logits = res2[sum.outputs[0].name].data;
  const okHalf = r.logits_half.every((v, i) => close(logits[i], v));
  // and feeding the tensor back unchanged must reproduce the original logits
  const res3 = await s2.run({ input: x, [r.cut]: res[r.cut] });
  const okSame = res3[sum.outputs[0].name].data.every((v, i) => close(v, res[sum.outputs[0].name].data[i]));
  if (!okHalf || !okSame) fail++;
  console.log(`${name}: ${sum.nodes} nodes, ${sum.params} params, ${cands.length} candidate tensors, rooms [${rooms.join(', ')}]; ` +
    `${checked} tensors match python; override x0.5 ${okHalf ? 'matches' : 'DIFFERS'}; identity override ${okSame ? 'matches' : 'DIFFERS'}`);
}
if (fail) { console.error(`FAIL (${fail})`); process.exit(1); }
console.log('OK');

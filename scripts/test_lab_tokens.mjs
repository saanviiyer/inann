// The token path: the paren transformer takes int64 ids. Checks the probe exposes attention
// maps and MLP activations, and that predictions match the labels on the bundled dataset.
import fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { initProto, decodeModel, encodeModel, summarize, candidateTensors, defaultRooms, withOutputs } from '../src/lib/lab/graph.js';
initProto(fs.readFileSync(new URL('../src/lib/lab/onnx.proto', import.meta.url), 'utf8'));
const model = decodeModel(fs.readFileSync(new URL('../public/examples/paren_transformer.onnx', import.meta.url)));
const sum = summarize(model);
const cands = candidateTensors(model, sum);
const s = await ort.InferenceSession.create(encodeModel(withOutputs(model, cands.map((c) => c.name))));
const ds = JSON.parse(fs.readFileSync(new URL('../public/examples/paren_1000.json', import.meta.url)));
const ids = new Int32Array(Uint8Array.from(Buffer.from(ds.data, 'base64')).buffer);
const N = 200, T = 16;
const x = new ort.Tensor('int64', BigInt64Array.from(ids.subarray(0, N * T), (v) => BigInt(v)), [N, T]);
const out = await s.run({ ids: x });
const rooms = defaultRooms(model, cands);
console.log('input', sum.inputs[0], 'rooms:', rooms.map((r) => `${r} ${JSON.stringify(out[r].dims)}`).join(' | '));
let ok = 0;
for (let i = 0; i < N; i++) { const l = out.logits.data.subarray(i * 2, i * 2 + 2); if ((l[1] > l[0] ? 1 : 0) === ds.labels[i]) ok++; }
console.log(`accuracy on ${N} bundled items: ${(ok / N).toFixed(3)}`);
if (ok / N < 0.95) process.exit(1);

import fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { initProto, decodeModel, encodeModel, summarize, candidateTensors, defaultRooms, withOutputs, withOverride, badOutputsFromError } from '../src/lib/lab/graph.js';
initProto(fs.readFileSync(new URL('../src/lib/lab/onnx.proto', import.meta.url),'utf8'));
for (const [name, shape] of [['fixture_res',[1,3,32,32]],['fixture_tok',[1,10,6]]]) {
  const model = decodeModel(fs.readFileSync(new URL(`./.cache/${name}.onnx`, import.meta.url)));
  const sum = summarize(model); let cands = candidateTensors(model, sum);
  console.log(name, sum.opCounts);
  let s, tries = 0;
  while (true) {
    try { s = await ort.InferenceSession.create(encodeModel(withOutputs(model, cands.map(c=>c.name)))); break; }
    catch (e) { const bad = badOutputsFromError(String(e.message)); console.log('  retry drop', bad, String(e.message).slice(0,160)); if (!bad || ++tries > 20) throw e; cands = cands.filter(c=>!bad.includes(c.name)); }
  }
  const x = new ort.Tensor('float32', new Float32Array(shape.reduce((a,b)=>a*b)).map((_, i) => Math.sin(i)), shape);
  const res = await s.run({ input: x });
  const rooms = defaultRooms(model, cands);
  console.log('  rooms', rooms.map(r => `${r} ${JSON.stringify(res[r].dims)}`).join(' | '));
  // override the first room, which feeds a skip connection in fixture_res
  const cut = rooms[0];
  const ov = withOverride(model, cut, res[cut].dims.slice(1), rooms);
  const s2 = await ort.InferenceSession.create(encodeModel(ov));
  const same = await s2.run({ input: x, [cut]: res[cut] });
  const d = Math.max(...same.out.data.map((v,i)=>Math.abs(v-res.out.data[i])));
  console.log('  identity override max diff', d.toExponential(2));
}

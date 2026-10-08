// Runs the Lab's SAE training worker in Node on the Tour network's dense activations and
// checks that it reconstructs held-out vectors about as well as the PyTorch SAE (92.9%).
import fs from 'node:fs';
import { loadNet, forward } from '../src/lib/net.js';
const net = loadNet(JSON.parse(fs.readFileSync(new URL('../src/data/model.json', import.meta.url))));
const data = JSON.parse(fs.readFileSync(new URL('../src/data/digits.json', import.meta.url)));
const X = new Float32Array(450 * 32);
data.images.forEach((im, i) => X.set(forward(net, im.map((v) => v / 16)).acts.dense, i * 32));

let done;
globalThis.self = { postMessage: (m) => { if (m.type === 'done') done = m; else if (m.fvu != null && m.step % 500 === 0) console.log(`step ${m.step} loss ${m.loss.toFixed(3)} fvu ${m.fvu.toFixed(3)} dead ${m.dead}`); } };
await import('../src/lib/lab/sae.worker.js');
const t0 = Date.now();
self.onmessage({ data: { X, D: 32, F: 256, k: 4, steps: 600, lr: 2e-3, batch: 128, seed: 3 } });
console.log(`trained in ${((Date.now() - t0) / 1000).toFixed(1)} s; held-out variance explained ${(100 * done.metrics.varianceExplained).toFixed(1)}% (n_test=${done.metrics.nTest})`);
if (done.metrics.varianceExplained < 0.7) { console.error('FAIL'); process.exit(1); }
console.log('OK');

// steering transforms: setting a feature to the value it already has must change nothing,
// and raising it must move the layer along that feature's decoder direction only
const { saeTransform } = await import('../src/lib/lab/saeLab.js');
const { encode } = await import('../src/lib/lab/saeCore.js');
const s = done.sae;
const L = { kind: 'vector', C: 32, P: 1, idx: (c) => c, size: 32 };
const x = X.subarray(0, 32);
const { idx, val } = encode(s, x);
const f0 = idx[0];
const same = saeTransform(s, L, 'steer', f0, val[0])(x);
const maxSame = Math.max(...same.map((v, j) => Math.abs(v - x[j])));
const up = saeTransform(s, L, 'steer', f0, val[0] + 1)(x);
const dir = Array.from({ length: 32 }, (_, j) => (up[j] - x[j]) / s.s);
const cos = dir.reduce((a, v, j) => a + v * s.Wd[f0 * 32 + j], 0) / Math.hypot(...dir);
console.log(`steer to own value: max change ${maxSame.toExponential(2)}; steer +1: cosine with decoder row ${cos.toFixed(4)}`);
if (maxSame > 1e-5 || cos < 0.999) { console.error('FAIL steering'); process.exit(1); }
console.log('OK steering');

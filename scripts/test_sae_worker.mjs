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
self.onmessage({ data: { X, D: 32, F: 256, k: 4, steps: 2000, lr: 2e-3, batch: 128, seed: 3 } });
console.log(`trained in ${((Date.now() - t0) / 1000).toFixed(1)} s; held-out variance explained ${(100 * done.metrics.varianceExplained).toFixed(1)}% (n_test=${done.metrics.nTest})`);
if (done.metrics.varianceExplained < 0.8) { console.error('FAIL'); process.exit(1); }
console.log('OK');

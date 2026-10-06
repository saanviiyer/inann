// Checks that the JS forward pass reproduces the PyTorch logits stored at training time,
// and that inputGradient matches a finite-difference estimate.
import fs from 'node:fs';
import { loadNet, forward, inputGradient } from '../src/lib/net.js';

const model = JSON.parse(fs.readFileSync(new URL('../src/data/model.json', import.meta.url)));
const data = JSON.parse(fs.readFileSync(new URL('../src/data/digits.json', import.meta.url)));
const net = loadNet(model);

let maxErr = 0, agree = 0;
const N = data.images.length;
let correct = 0;
for (let i = 0; i < N; i++) {
  const x = data.images[i].map((v) => v / 16);
  const f = forward(net, x);
  if (f.pred === data.labels[i]) correct++;
  if (i < 20) {
    const ref = model.parity.logits[i];
    for (let k = 0; k < 10; k++) maxErr = Math.max(maxErr, Math.abs(ref[k] - f.logits[k]));
    if (ref.indexOf(Math.max(...ref)) === f.pred) agree++;
  }
}
console.log(`parity: max |logit diff| = ${maxErr.toExponential(2)}, argmax agree ${agree}/20`);
console.log(`JS accuracy on bundled held-out digits: ${(correct / N).toFixed(4)} (python reported ${model.test_acc})`);

// finite-difference check of the input gradient
const x = data.images[3].map((v) => v / 16);
const f = forward(net, x);
let worst = 0;
for (const [layer, idx] of [['out', f.pred], ['dense', 5], ['conv2', 40], ['conv1', 100]]) {
  const g = inputGradient(net, f, layer, idx);
  for (const p of [10, 27, 36, 50]) {
    const e = 1e-3;
    const xp = x.slice(); xp[p] += e;
    const xm = x.slice(); xm[p] -= e;
    const fd = (forward(net, xp).pre[layer][idx] - forward(net, xm).pre[layer][idx]) / (2 * e);
    worst = Math.max(worst, Math.abs(fd - g[p]));
  }
}
console.log(`gradient check: worst |analytic - finite diff| = ${worst.toExponential(2)}`);
if (maxErr > 1e-3 || agree !== 20 || worst > 1e-2) { console.error('FAIL'); process.exit(1); }
console.log('OK');

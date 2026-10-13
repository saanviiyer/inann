// Linear probes on the Tour network: accuracy per layer, the shuffled-label control, the
// shuffled-weights network, and steering along a probed direction against a random one.
import fs from 'node:fs';
import { loadNet } from '../src/lib/net.js';
import { probeNetwork, scrambledNet, steerTest, PROBE_LAYERS } from '../src/lib/probeTour.js';
import { splitOf, trainProbe, accuracyOf } from '../src/lib/probe.js';

const model = JSON.parse(fs.readFileSync(new URL('../src/data/model.json', import.meta.url)));
const data = JSON.parse(fs.readFileSync(new URL('../src/data/digits.json', import.meta.url)));
const net = loadNet(model);
const digits = data.images.map((im) => Float32Array.from(im, (v) => v / 16));
const labels = data.labels;
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };

// a separable toy problem: the probe must solve it, and the shuffled control must not
{
  const N = 300, D = 5, X = new Float32Array(N * D), y = [];
  for (let i = 0; i < N; i++) { const c = i % 3; y.push(c); for (let j = 0; j < D; j++) X[i * D + j] = (j === c ? 3 : 0) + Math.sin(i * 7.1 + j); }
  const sp = splitOf(y);
  const p = trainProbe(X, D, y, 3, sp.train, { epochs: 40 });
  check(accuracyOf(p, X, y, sp.test) > 0.95, 'toy separable problem solved on held-out rows');
}

const split = splitOf(labels);
const real = await probeNetwork(net, digits, labels, { split });
const rand = await probeNetwork(scrambledNet(net), digits, labels, { split });
console.log(`majority-class baseline ${(100 * real.majority).toFixed(1)}%`);
for (const l of PROBE_LAYERS) {
  const r = real.results[l.key], s = rand.results[l.key];
  console.log(`${l.label.padEnd(13)} D=${String(r.D).padStart(3)}  test ${(100 * r.test).toFixed(1)}%  train ${(100 * r.train).toFixed(1)}%  | shuffled labels train ${(100 * r.ctrlTrain).toFixed(1)}% test ${(100 * r.ctrlTest).toFixed(1)}%  | shuffled weights test ${(100 * s.test).toFixed(1)}%`);
}
check(real.results.out.test > 0.9, 'output layer probe matches the network');
check(Math.abs(real.results.input.test - rand.results.input.test) < 1e-9, 'input probe identical for both networks');
for (const l of PROBE_LAYERS) check(real.results[l.key].ctrlTest < 0.25, `${l.label}: shuffled-label control near chance on held-out items`);

for (const layer of ['input', 'conv1', 'conv2', 'dense']) {
  const row = [];
  for (const alpha of [0.5, 1, 2]) {
    let hit = 0, hr = 0, n = 0;
    for (let k = 0; k < 10; k++) { const t = steerTest(net, digits, labels, real.acts, split, layer, k, alpha); hit += t.hit; hr += t.hitRand; n += t.n; }
    row.push(`a=${alpha}: ${(100 * hit / n).toFixed(1)}% vs random ${(100 * hr / n).toFixed(1)}%`);
  }
  console.log(`steer ${layer.padEnd(6)} ${row.join('  ')}`);
}
process.exit(fail ? 1 : 0);

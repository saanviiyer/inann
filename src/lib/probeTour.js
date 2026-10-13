// Probes on the Tour's bundled digits: one per layer of the trained network, the same probes
// on a network whose weights were shuffled, and steering along a probed direction.
// Takes the net and data as arguments so node tests can run it without the bundler.
import { forward } from './net.js';
import { splitOf, probeLayer, majorityOf, massMeanDirection, randomLike, scrambleArrays } from './probe.js';

export const PROBE_LAYERS = [
  { key: 'input', label: 'Input pixels' },
  { key: 'conv1', label: 'Conv 1' },
  { key: 'conv2', label: 'Conv 2' },
  { key: 'dense', label: 'Dense' },
  { key: 'out', label: 'Output' },
];

const flatOf = (rows) => {
  const D = rows[0].length, X = new Float32Array(rows.length * D);
  rows.forEach((r, i) => X.set(r, i * D));
  return { X, D };
};

export function actsOf(net, digits) {
  const A = Object.fromEntries(PROBE_LAYERS.map((l) => [l.key, []]));
  for (const x of digits) {
    const f = forward(net, x);
    // the output layer is probed on its logits, the values a linear readout would see
    for (const l of PROBE_LAYERS) A[l.key].push(l.key === 'out' ? f.logits : f.acts[l.key]);
  }
  return Object.fromEntries(Object.entries(A).map(([k, rows]) => [k, flatOf(rows)]));
}

export function scrambledNet(net, seed = 101) {
  const [c1w, c2w, dw, ow] = scrambleArrays([net.c1w, net.c2w, net.dw, net.ow], seed);
  return { ...net, c1w, c2w, dw, ow };
}

// Every layer of one network. onLayer(key, result) fires as each finishes.
export async function probeNetwork(net, digits, labels, { split = splitOf(labels), onLayer, yieldEvery } = {}) {
  const A = actsOf(net, digits), out = {};
  for (const l of PROBE_LAYERS) {
    const { X, D } = A[l.key];
    const { probe, ...r } = probeLayer(X, D, labels, 10, split);
    out[l.key] = { ...r, probe };
    onLayer?.(l.key, out[l.key]);
    if (yieldEvery) await new Promise((res) => setTimeout(res, 0));
  }
  return { results: out, acts: A, split, majority: majorityOf(labels, split) };
}

// Add alpha times the class-k mass-mean direction to one layer, on held-out items not already
// predicted as k, and count how many the network now calls k. The control adds a random
// direction of the same length.
export function steerTest(net, digits, labels, acts, split, layer, k, alpha) {
  const { X, D } = acts[layer];
  const dir = massMeanDirection(X, D, labels, k, split.train);
  const rand = randomLike(dir, 41 + k);
  const items = Array.from(split.test).filter((i) => forward(net, digits[i]).pred !== k);
  let hit = 0, hitRand = 0;
  for (const i of items) {
    if (forward(net, digits[i], { add: { layer, vec: dir.map((v) => alpha * v) } }).pred === k) hit++;
    if (forward(net, digits[i], { add: { layer, vec: rand.map((v) => alpha * v) } }).pred === k) hitRand++;
  }
  return { n: items.length, hit, hitRand, norm: Math.sqrt(dir.reduce((s, v) => s + v * v, 0)) };
}

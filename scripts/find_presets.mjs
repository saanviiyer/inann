// Search for steering demos instead of inventing them. For every conv channel, dense neuron
// and live SAE feature, clamp it to its natural ceiling (its largest held-out activation) or
// to zero, and count how correctly classified held-out digits of each class move.
// A demo only counts if it is selective: a clamp that sends every digit to the same answer is
// a sledgehammer, not evidence that this unit drives one behavior. Selectivity is the flip
// rate on the source class minus the rate at which all other digits are sent to the same target.
// Writes src/data/presets.json.
// Run: node scripts/find_presets.mjs
import fs from 'node:fs';
import { loadNet, forward } from '../src/lib/net.js';
import { loadSae } from '../src/lib/sae.js';

const rd = (p) => JSON.parse(fs.readFileSync(new URL(p, import.meta.url)));
const net = loadNet(rd('../src/data/model.json'));
const sae = loadSae(rd('../src/data/sae.json'));
const data = rd('../src/data/digits.json');
const X = data.images.map((im) => Float32Array.from(im, (v) => v / 16));
const Y = data.labels;

const base = X.map((x) => forward(net, x, { sae }));
const correct = base.map((f, i) => f.pred === Y[i]);
const nPerClass = Array.from({ length: 10 }, (_, d) => Y.filter((y, i) => y === d && correct[i]).length);

const units = [];
for (const [layer, C, per] of [['conv1', 6, 64], ['conv2', 12, 16]])
  for (let c = 0; c < C; c++) {
    let mx = 0;
    for (const f of base) for (let p = 0; p < per; p++) mx = Math.max(mx, f.acts[layer][c * per + p]);
    units.push({ layer, index: c, scope: 'channel', max: mx });
  }
for (let j = 0; j < 32; j++) units.push({ layer: 'dense', index: j, scope: 'unit', max: Math.max(...base.map((f) => f.acts.dense[j])) });
for (let j = 0; j < sae.F; j++) {
  const mx = Math.max(...base.map((f) => f.feats[j]));
  if (mx > 0) units.push({ layer: 'sae', index: j, scope: 'unit', max: mx });
}
console.log(`${units.length} units, ${correct.filter(Boolean).length} correctly classified held-out digits`);

const found = [];
for (const u of units) {
  if (u.max <= 0) continue;
  for (const mode of ['high', 'off']) {
    const value = mode === 'high' ? +u.max.toFixed(3) : 0;
    const clamp = { layer: u.layer, index: u.index, scope: u.scope, value };
    const flips = Array.from({ length: 10 }, () => new Array(10).fill(0));
    X.forEach((x, i) => {
      if (!correct[i]) return;
      const p = forward(net, x, { sae, clamp }).pred;
      if (p !== Y[i]) flips[Y[i]][p]++;
    });
    const nAll = nPerClass.reduce((x, y) => x + y, 0);
    const changedAll = flips.flat().reduce((x, y) => x + y, 0);
    for (let a = 0; a < 10; a++)
      for (let b = 0; b < 10; b++) {
        if (!flips[a][b]) continue;
        const othersToB = flips.reduce((s, row, c) => s + (c !== a ? row[b] : 0), 0);
        const othersN = nAll - nPerClass[a] - nPerClass[b];
        const rate = flips[a][b] / nPerClass[a];
        found.push({ ...clamp, mode, from: a, to: b, flipped: flips[a][b], of: nPerClass[a], rate,
          othersToTarget: othersToB, othersN, changedAll, nAll, selectivity: rate - othersToB / othersN });
      }
  }
}
found.sort((p, q) => q.selectivity - p.selectivity);
// keep the best effect per unit and per (from, to) pair, so the demos differ
const seenUnit = new Set(), seenPair = new Set(), presets = [];
for (const p of found) {
  const uk = `${p.layer}:${p.index}`, pk = `${p.from}>${p.to}`;
  if (seenUnit.has(uk) || seenPair.has(pk) || p.rate < 0.3) continue;
  seenUnit.add(uk); seenPair.add(pk); presets.push(p);
  if (presets.length >= 16) break;
}
// the best measured effect for every (from, to) pair, so the UI can answer "can anything do X?"
const bestPair = {};
for (const p of found) {
  const k = `${p.from}>${p.to}`;
  if (!bestPair[k] || bestPair[k].rate < p.rate) bestPair[k] = { layer: p.layer, index: p.index, mode: p.mode, value: p.value, flipped: p.flipped, of: p.of, rate: +p.rate.toFixed(3), othersToTarget: p.othersToTarget, othersN: p.othersN };
}
console.log('best 1>0:', bestPair['1>0']);
for (const p of presets)
  console.log(`${p.layer}:${p.index} ${p.mode.padEnd(4)} (${p.value}) ${p.flipped}/${p.of} ${p.from}s -> ${p.to}; others -> ${p.to}: ${p.othersToTarget}/${p.othersN}; any change ${p.changedAll}/${p.nAll}; sel ${p.selectivity.toFixed(2)}`);
fs.writeFileSync(new URL('../src/data/presets.json', import.meta.url), JSON.stringify({
  method: 'clamp to the max held-out activation (high) or to 0 (off); flips counted among correctly classified held-out digits; ranked by selectivity = source-class flip rate minus the rate other digits are sent to the same target; best per unit and per (from, to) pair; source rate >= 0.3',
  presets,
  bestPair,
}));

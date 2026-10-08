// Trains a top-k SAE off the main thread. Receives {X: Float32Array (N*D), D, F, k, steps, lr, batch}
// and posts {type:'progress', step, loss, fvu, dead} then {type:'done', sae, metrics}.
import { makeSae, encode, decode } from './saeCore.js';
import { featureStats } from './saeLab.js';

self.onmessage = (e) => {
  // stats-only mode: score an existing SAE on every item (used when an SAE or a session is loaded)
  if (e.data.mode === 'stats') {
    const { sae, items } = e.data;
    const stats = featureStats(sae, { items: items.items, N: items.N, P: items.P, C: sae.D }, (frac) => self.postMessage({ type: 'stats', frac }));
    self.postMessage({ type: 'done', sae, stats, metrics: null });
    return;
  }
  const { X, D, F, k, steps, lr, batch, seed } = e.data;
  const N = X.length / D;
  // hold out 10% for the reported metrics
  const order = Array.from({ length: N }, (_, i) => i);
  let r = seed || 7;
  const rnd = () => { r = (r * 16807) % 2147483647; return r / 2147483647; };
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const nTest = Math.max(1, Math.floor(N * 0.1));
  const test = order.slice(0, nTest), train = order.slice(nTest);
  const row = (i) => X.subarray(i * D, (i + 1) * D);

  const sae = makeSae(D, F, k, seed || 1);
  for (const i of train) for (let j = 0; j < D; j++) sae.mu[j] += row(i)[j] / train.length;
  let s = 0;
  for (const i of train) { let n = 0; for (let j = 0; j < D; j++) n += (row(i)[j] - sae.mu[j]) ** 2; s += Math.sqrt(n); }
  sae.s = s / train.length || 1;

  const P = [sae.We, sae.be, sae.Wd, sae.bd];
  const G = P.map((p) => new Float32Array(p.length));
  const M = P.map((p) => new Float32Array(p.length));
  const V = P.map((p) => new Float32Array(p.length));
  const lastFired = new Int32Array(F);
  const pre = new Float32Array(F);
  const xcBuf = new Float32Array(D), rec = new Float32Array(D), d = new Float32Array(D);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;

  const evalTest = () => {
    let err = 0, tot = 0;
    for (const i of test) {
      const x = row(i);
      const { idx, val } = encode(sae, x, pre);
      const xh = decode(sae, idx, val);
      for (let j = 0; j < D; j++) { err += (xh[j] - x[j]) ** 2; tot += (x[j] - sae.mu[j]) ** 2; }
    }
    return err / (tot || 1);
  };

  for (let step = 1; step <= steps; step++) {
    G.forEach((g) => g.fill(0));
    let loss = 0;
    for (let b = 0; b < batch; b++) {
      const x = row(train[Math.floor(rnd() * train.length)]);
      const { idx, val, xc } = encode(sae, x, pre, xcBuf);
      // reconstruction in standardized units (buffers reused across samples)
      rec.set(sae.bd);
      for (let t = 0; t < idx.length; t++) { const o = idx[t] * D, a = val[t]; for (let j = 0; j < D; j++) rec[j] += a * sae.Wd[o + j]; }
      for (let j = 0; j < D; j++) { const tgt = (x[j] - sae.mu[j]) / sae.s; d[j] = (rec[j] - tgt) * (2 / batch); loss += (rec[j] - tgt) ** 2 / batch; }
      for (let j = 0; j < D; j++) G[3][j] += d[j];
      idx.forEach((f, t) => {
        lastFired[f] = step;
        const o = f * D;
        let da = 0;
        for (let j = 0; j < D; j++) { G[2][o + j] += val[t] * d[j]; da += sae.Wd[o + j] * d[j]; }
        G[1][f] += da;
        for (let j = 0; j < D; j++) { G[0][o + j] += da * xc[j]; G[3][j] -= da * sae.We[o + j]; }
      });
    }
    // Adam
    const c1 = 1 - b1 ** step, c2 = 1 - b2 ** step;
    for (let q = 0; q < P.length; q++) {
      const p = P[q], g = G[q], m = M[q], v = V[q];
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
    // unit-norm decoder rows
    for (let f = 0; f < F; f++) {
      let n = 0; const o = f * D;
      for (let j = 0; j < D; j++) n += sae.Wd[o + j] ** 2;
      n = Math.sqrt(n) || 1;
      for (let j = 0; j < D; j++) sae.Wd[o + j] /= n;
    }
    // resample features that have not fired for 400 steps toward a badly reconstructed example
    if (step % 400 === 0 && step < steps * 0.8) {
      for (let f = 0; f < F; f++) {
        if (step - lastFired[f] < 400) continue;
        const x = row(train[Math.floor(rnd() * train.length)]);
        const { idx, val } = encode(sae, x, pre);
        const xh = decode(sae, idx, val);
        let n = 0;
        const dir = new Float32Array(D);
        for (let j = 0; j < D; j++) { dir[j] = (x[j] - xh[j]) / sae.s; n += dir[j] ** 2; }
        n = Math.sqrt(n) || 1;
        const o = f * D;
        for (let j = 0; j < D; j++) { sae.Wd[o + j] = dir[j] / n; sae.We[o + j] = (0.2 * dir[j]) / n; }
        sae.be[f] = 0;
        lastFired[f] = step;
        for (const arr of [M[0], V[0], M[2], V[2]]) arr.fill(0, o, o + D);
      }
    }
    if (step % 25 === 0 || step === steps) {
      const dead = Array.from(lastFired).filter((t) => step - t > Math.min(400, step)).length;
      self.postMessage({ type: 'progress', step, loss, fvu: step % 100 === 0 || step === steps ? evalTest() : null, dead });
    }
  }
  const fvu = evalTest();
  // per-item feature maxima for the rooms, if the page sent every item's vectors
  let stats = null;
  if (e.data.items) {
    const { items, N: nItems, P } = e.data.items;
    stats = featureStats(sae, { items: items.items ?? items, N: nItems, P, C: D }, (frac) => self.postMessage({ type: 'stats', frac }));
  }
  self.postMessage({ type: 'done', sae, stats, metrics: { fvu, varianceExplained: 1 - fvu, nTrain: train.length, nTest: test.length } });
};

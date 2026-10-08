// One pass of a dataset through an uploaded model, keeping what the rooms need:
// per channel, its max over positions on every item (and where), the full activations when they
// are small enough, a robust brightness scale, and the model's predictions.
import { modelInputOf } from './data.js';

const FULL_LIMIT = 30e6; // floats kept per room at most (~120 MB)

export async function collect(model, ds, kind, norm, rooms, { max = 2000, batch = 32, onProgress } = {}) {
  const N = Math.min(ds.n, max);
  const stats = {};
  for (const r of rooms) {
    const L = r.layout;
    stats[r.name] = {
      layout: L, chMax: new Float32Array(N * L.C), chArg: new Int32Array(N * L.C),
      full: L.size * N <= FULL_LIMIT ? new Float32Array(N * L.size) : null,
      sample: [], lo: Infinity, hi: -Infinity,
    };
  }
  const preds = new Int32Array(N);
  const B = model.batchable ? batch : 1;
  for (let s = 0; s < N; s += B) {
    const idx = Array.from({ length: Math.min(B, N - s) }, (_, k) => s + k);
    const out = await model.run(idx.map((i) => modelInputOf(ds, i, kind, norm)));
    for (const r of rooms) {
      const st = stats[r.name], L = st.layout, t = out[r.name];
      if (!t) continue;
      idx.forEach((i, b) => {
        const v = t.data.subarray(b * L.size, (b + 1) * L.size);
        if (st.full) st.full.set(v, i * L.size);
        for (let c = 0; c < L.C; c++) {
          let m = -Infinity, a = 0;
          for (let p = 0; p < L.P; p++) { const x = v[L.idx(c, p)]; if (x > m) { m = x; a = p; } }
          st.chMax[i * L.C + c] = m; st.chArg[i * L.C + c] = a;
        }
        for (let j = 0; j < v.length; j += Math.max(1, Math.floor(v.length / 64))) {
          st.sample.push(v[j]);
          st.lo = Math.min(st.lo, v[j]); st.hi = Math.max(st.hi, v[j]);
        }
      });
    }
    const o = out[model.outputName];
    if (o) {
      const K = o.data.length / idx.length;
      idx.forEach((i, b) => {
        let best = 0;
        for (let k = 1; k < K; k++) if (o.data[b * K + k] > o.data[b * K + best]) best = k;
        preds[i] = best;
      });
    }
    onProgress?.(Math.min(N, s + B) / N);
  }
  for (const st of Object.values(stats)) {
    const a = st.sample.map(Math.abs).sort((x, y) => x - y);
    st.scale = a[Math.floor(a.length * 0.99)] || a[a.length - 1] || 1;
    st.signed = st.lo < -1e-6;
    delete st.sample;
  }
  return { N, stats, preds, labels: ds.labels ? ds.labels.slice(0, N) : null, classes: ds.classes ?? null };
}

// Channel c's activation (max over positions) on every item.
export function channelActs(store, room, c) {
  const st = store.stats[room], C = st.layout.C;
  return Float32Array.from({ length: store.N }, (_, i) => st.chMax[i * C + c]);
}

export function topItems(acts, k = 12) {
  return Array.from(acts.keys()).sort((a, b) => acts[b] - acts[a]).slice(0, k);
}

export function histogramOf(acts, bins = 24) {
  let lo = Infinity, hi = -Infinity;
  for (const v of acts) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (lo > 0) lo = 0;
  const counts = new Array(bins).fill(0);
  for (const v of acts) counts[Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo || 1)) * bins))]++;
  return { counts, lo, hi, n: acts.length };
}

// Fraction of the top-n items that share one label (null without labels or with too few active).
export function labelPurity(acts, labels, n = 12) {
  if (!labels) return null;
  const top = topItems(acts, n).filter((i) => acts[i] > 0);
  if (top.length < n) return null;
  const c = {};
  top.forEach((i) => (c[labels[i]] = (c[labels[i]] ?? 0) + 1));
  const [lab, cnt] = Object.entries(c).sort((a, b) => b[1] - a[1])[0];
  return { label: Number(lab), count: cnt, n, value: cnt / n };
}

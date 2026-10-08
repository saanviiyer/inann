// Gluing an SAE onto one layer of an uploaded model. The SAE reads one vector per position:
// a conv map gives one C-dim vector per pixel, a sequence one D-dim vector per token, a plain
// vector just itself (Bricken et al. 2023 and Templeton et al. 2024 work per token the same way).
import { encode, decode } from './saeCore.js';

// items: Float32Array laid out [item][position][channel], built from the stored activations
export function itemVectors(store, roomName) {
  const st = store.stats[roomName], L = st.layout;
  if (!st.full) return null;
  const out = new Float32Array(store.N * L.P * L.C);
  for (let i = 0; i < store.N; i++)
    for (let p = 0; p < L.P; p++)
      for (let c = 0; c < L.C; c++) out[(i * L.P + p) * L.C + c] = st.full[i * L.size + L.idx(c, p)];
  return { items: out, N: store.N, P: L.P, C: L.C };
}

export function sampleRows(iv, max = 20000, seed = 5) {
  const rows = iv.N * iv.P;
  if (rows <= max) return iv.items;
  let r = seed;
  const out = new Float32Array(max * iv.C);
  for (let k = 0; k < max; k++) {
    r = (r * 16807) % 2147483647;
    const row = r % rows;
    out.set(iv.items.subarray(row * iv.C, (row + 1) * iv.C), k * iv.C);
  }
  return out;
}

// Per item: each feature's max over positions, and where.
export function featureStats(sae, iv, onProgress) {
  const { N, P, C } = iv;
  const featMax = new Float32Array(N * sae.F), featArg = new Int32Array(N * sae.F);
  const pre = new Float32Array(sae.F);
  for (let i = 0; i < N; i++) {
    for (let p = 0; p < P; p++) {
      const { idx, val } = encode(sae, iv.items.subarray((i * P + p) * C, (i * P + p + 1) * C), pre);
      idx.forEach((f, t) => { if (val[t] > featMax[i * sae.F + f]) { featMax[i * sae.F + f] = val[t]; featArg[i * sae.F + f] = p; } });
    }
    if (onProgress && i % 50 === 0) onProgress(i / N);
  }
  return { featMax, featArg };
}

const vecAt = (data, L, p) => Float32Array.from({ length: L.C }, (_, c) => data[L.idx(c, p)]);

// The current input's features: max over positions.
export function featuresNow(sae, data, L) {
  const max = new Float32Array(sae.F), arg = new Int32Array(sae.F);
  const pre = new Float32Array(sae.F);
  for (let p = 0; p < L.P; p++) {
    const { idx, val } = encode(sae, vecAt(data, L, p), pre);
    idx.forEach((f, t) => { if (val[t] > max[f]) { max[f] = val[t]; arg[f] = p; } });
  }
  return { max, arg };
}

// Rewrite a tensor through the SAE. mode 'recon': replace each vector by its reconstruction
// (to measure what the SAE loses). mode 'steer': set one feature to `value` at every position
// and add only the change, keeping the reconstruction error (Templeton et al. 2024).
export function saeTransform(sae, L, mode, feature, value) {
  return (data) => {
    const out = Float32Array.from(data);
    const pre = new Float32Array(sae.F);
    for (let p = 0; p < L.P; p++) {
      const x = vecAt(data, L, p);
      const { idx, val } = encode(sae, x, pre);
      let delta;
      if (mode === 'recon') {
        const xh = decode(sae, idx, val);
        delta = xh.map((v, c) => v - x[c]);
      } else {
        const before = decode(sae, idx, val);
        const i2 = [...idx], v2 = [...val];
        const at = i2.indexOf(feature);
        if (at >= 0) v2[at] = value; else { i2.push(feature); v2.push(value); }
        const after = decode(sae, i2, v2);
        delta = after.map((v, c) => v - before[c]);
      }
      for (let c = 0; c < L.C; c++) out[L.idx(c, p)] += delta[c];
    }
    return out;
  };
}

export function neuronTransform(L, c, p, value) {
  return (data) => {
    const out = Float32Array.from(data);
    if (p == null) for (let q = 0; q < L.P; q++) out[L.idx(c, q)] = value;
    else out[L.idx(c, p)] = value;
    return out;
  };
}

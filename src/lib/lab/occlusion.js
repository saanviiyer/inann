// Occlusion sensitivity for any image model (Zeiler & Fergus 2014): slide a patch of the
// background value over the input and record how much a target number drops. Forward passes
// only, so it works where gradients are not available. Batched when the model allows it.
import { toModel } from './data.js';

// pixels: display form (C,H,W). target(out) -> number, read from a run's outputs.
export async function occlusionMap(model, pixels, kind, norm, target, { patch = null, stride = null, fill = 0 } = {}) {
  const { C, H, W } = kind;
  const ps = patch ?? Math.max(2, Math.round(Math.min(H, W) / 7));
  const st = stride ?? Math.max(1, Math.round(ps / 2));
  const base = target(await model.run([toModel(pixels, kind, norm)]), 0);
  const spots = [];
  for (let y = 0; y + ps <= H; y += st) for (let x = 0; x + ps <= W; x += st) spots.push([y, x]);
  const heat = new Float32Array(H * W), cnt = new Float32Array(H * W);
  const B = model.batchable ? 32 : 1;
  for (let s = 0; s < spots.length; s += B) {
    const chunk = spots.slice(s, s + B);
    const xs = chunk.map(([y0, x0]) => {
      const q = Float32Array.from(pixels);
      for (let c = 0; c < C; c++) for (let y = y0; y < y0 + ps; y++) for (let x = x0; x < x0 + ps; x++) q[c * H * W + y * W + x] = fill;
      return toModel(q, kind, norm);
    });
    const out = await model.run(xs);
    chunk.forEach(([y0, x0], b) => {
      const drop = base - target(out, b);
      for (let y = y0; y < y0 + ps; y++) for (let x = x0; x < x0 + ps; x++) { heat[y * W + x] += drop; cnt[y * W + x]++; }
    });
  }
  let mx = 0;
  for (let i = 0; i < heat.length; i++) { heat[i] = cnt[i] ? heat[i] / cnt[i] : 0; mx = Math.max(mx, heat[i]); }
  return { heat, max: mx, base, patch: ps, stride: st, runs: spots.length };
}

// helpers to read one number per batch item from a run
export const readClassProb = (outputName, k, softmax) => (out, b) => {
  const t = out[outputName], K = t.data.length / t.dims[0];
  return softmax(t.data.subarray(b * K, (b + 1) * K))[k];
};
export const readChannelMax = (name, L, c) => (out, b) => {
  const t = out[name];
  const v = t.data.subarray(b * L.size, (b + 1) * L.size);
  let m = -Infinity;
  for (let p = 0; p < L.P; p++) m = Math.max(m, v[L.idx(c, p)]);
  return m;
};

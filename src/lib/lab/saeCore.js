// Top-k sparse autoencoder used by the Lab (Gao et al. 2024): shared by the training worker and
// the page. Inputs are standardized by a mean and one scalar scale before encoding.
//   f = TopK(ReLU(We ((x - mu)/s - bd) + be)),   x_hat = mu + s (Wd^T f + bd)
export function makeSae(D, F, k, seed = 1) {
  let r = seed;
  const rnd = () => { r = (r * 16807) % 2147483647; return r / 2147483647 - 0.5; };
  const Wd = new Float32Array(F * D);
  for (let f = 0; f < F; f++) {
    let n = 0;
    for (let j = 0; j < D; j++) { const v = rnd(); Wd[f * D + j] = v; n += v * v; }
    n = Math.sqrt(n);
    for (let j = 0; j < D; j++) Wd[f * D + j] /= n;
  }
  return { D, F, k, We: Float32Array.from(Wd), be: new Float32Array(F), Wd, bd: new Float32Array(D), mu: new Float32Array(D), s: 1 };
}

// Encode one vector; returns {idx, val} of the active features.
export function encode(sae, x, pre = new Float32Array(sae.F)) {
  const { D, F, k, We, be, bd, mu, s } = sae;
  const xc = new Float32Array(D);
  for (let j = 0; j < D; j++) xc[j] = (x[j] - mu[j]) / s - bd[j];
  for (let f = 0; f < F; f++) {
    let v = be[f];
    const o = f * D;
    for (let j = 0; j < D; j++) v += We[o + j] * xc[j];
    pre[f] = v;
  }
  return topk(pre, k, xc);
}

function topk(pre, k, xc) {
  const idx = [], val = [];
  for (let f = 0; f < pre.length; f++) {
    const v = pre[f];
    if (v <= 0) continue;
    if (idx.length < k) { idx.push(f); val.push(v); continue; }
    let mi = 0;
    for (let t = 1; t < k; t++) if (val[t] < val[mi]) mi = t;
    if (v > val[mi]) { idx[mi] = f; val[mi] = v; }
  }
  return { idx, val, xc };
}

export function decode(sae, idx, val) {
  const { D, Wd, bd, mu, s } = sae;
  const out = new Float32Array(D);
  for (let j = 0; j < D; j++) out[j] = bd[j];
  idx.forEach((f, t) => { const o = f * D; for (let j = 0; j < D; j++) out[j] += val[t] * Wd[o + j]; });
  for (let j = 0; j < D; j++) out[j] = mu[j] + s * out[j];
  return out;
}

export function toJSON(sae, extra = {}) {
  const r = (a) => Array.from(a, (v) => +v.toFixed(6));
  return { ...extra, D: sae.D, F: sae.F, k: sae.k, s: sae.s, mu: r(sae.mu), bd: r(sae.bd), be: r(sae.be), We: r(sae.We), Wd: r(sae.Wd) };
}
export function fromJSON(j) {
  return { D: j.D, F: j.F, k: j.k, s: j.s, mu: Float32Array.from(j.mu), bd: Float32Array.from(j.bd), be: Float32Array.from(j.be), We: Float32Array.from(j.We), Wd: Float32Array.from(j.Wd) };
}

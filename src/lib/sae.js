// The sparse autoencoder over the 32 dense neurons (scripts/train_sae.py), run in the browser.
// f = TopK(ReLU(W_enc (d - b_dec) + b_enc)),  d_hat = W_dec f + b_dec
export function loadSae(json) {
  const F = json.W_enc.length, H = json.W_dec.length;
  return {
    F, H, k: json.stats.k, stats: json.stats,
    We: Float32Array.from(json.W_enc.flat()), // [F][H]
    be: Float32Array.from(json.b_enc),
    Wd: Float32Array.from(json.W_dec.flat()), // [H][F]
    bd: Float32Array.from(json.b_dec),
  };
}

export function saeEncode(s, d) {
  const pre = new Float32Array(s.F);
  for (let f = 0; f < s.F; f++) {
    let v = s.be[f];
    for (let j = 0; j < s.H; j++) v += s.We[f * s.H + j] * (d[j] - s.bd[j]);
    pre[f] = v > 0 ? v : 0;
  }
  // keep the k largest
  const idx = Array.from(pre.keys()).sort((a, b) => pre[b] - pre[a]).slice(0, s.k);
  const out = new Float32Array(s.F);
  for (const i of idx) out[i] = pre[i];
  return out;
}

export function saeDecode(s, f) {
  const d = Float32Array.from(s.bd);
  for (let fi = 0; fi < s.F; fi++) {
    if (!f[fi]) continue;
    for (let j = 0; j < s.H; j++) d[j] += s.Wd[j * s.F + fi] * f[fi];
  }
  return d;
}

// Direct effect of one feature on each digit's score: W_out . W_dec[:, f]. Exact, since the
// output layer is linear in the dense activations.
export function featureLogitEffect(net, s, fi) {
  const e = new Float32Array(10);
  for (let o = 0; o < 10; o++) for (let j = 0; j < s.H; j++) e[o] += net.ow[o * 32 + j] * s.Wd[j * s.F + fi];
  return e;
}

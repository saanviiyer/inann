// Incoming edges for a selected unit of any model: which units in the room below feed it, and
// how much, on the current input. Contribution = weight x the activation it multiplies.
//  - Dense (Gemm/MatMul): exact, per input element; grouped per channel when the input was a
//    flattened conv map.
//  - Conv: exact for stride-1 "same" convolutions (sum over the kernel window at the unit's
//    position, per input channel); otherwise the summed weight times the input channel's mean.
import { linearSource, sourceRoom, preFlattenTensor, incomingWeights } from './graph.js';

export function unitEdges(model, roomNames, outs, room, index, layout, inputTensorValues, limit = 16) {
  const src = linearSource(model.model, room);
  if (!src?.input) return null;
  const inputName = model.input.name;
  const from = sourceRoom(model.model, src.input, [inputName, ...roomNames]);
  if (!from) return null;
  const fromRoom = from.room === inputName ? '__input' : from.room;
  const L = layout;
  const c = L.kind === 'spatial' ? Math.floor(index / L.P) : L.kind === 'seq' ? index % L.C : index;
  const p = L.kind === 'spatial' ? index % L.P : L.kind === 'seq' ? Math.floor(index / L.C) : 0;
  const w = incomingWeights(src, c);
  // the tensor the linear op actually reads (after pooling / flattening), for exact contributions
  const readName = src.op === 'Conv' ? src.input : preFlattenTensor(model.model, src.input);
  const read = readName === inputName ? { data: inputTensorValues, dims: [1, ...model.inputDims] } : outs[readName];
  if (!read) return null;
  const x = read.data;
  const rd = read.dims.slice(1);
  let contrib = [];
  let exact = true;
  if (src.op === 'Conv') {
    const [cin, kh, kw] = [w.cin, w.kh, w.kw];
    const [, H, W] = rd.length === 3 ? rd : [rd[0], 1, 1];
    const ph = (kh - 1) / 2, pw = (kw - 1) / 2;
    const sameRes = L.kind === 'spatial' && L.H === H && L.W === W;
    const y0 = Math.floor(p / L.W), x0 = p % L.W;
    for (let k = 0; k < cin; k++) {
      let v = 0;
      if (sameRes) {
        for (let dy = 0; dy < kh; dy++) for (let dx = 0; dx < kw; dx++) {
          const yy = y0 + dy - ph, xx = x0 + dx - pw;
          if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
          v += w.data[(k * kh + dy) * kw + dx] * x[k * H * W + yy * W + xx];
        }
      } else {
        exact = false;
        let ws = 0, m = 0;
        for (let j = 0; j < kh * kw; j++) ws += w.data[k * kh * kw + j];
        for (let j = 0; j < H * W; j++) m += x[k * H * W + j] / (H * W);
        v = ws * m;
      }
      contrib.push({ k, v });
    }
  } else {
    // dense row over the flattened input; group per channel if that input was a [C,H,W] map
    const n = w.data.length;
    if (x.length < n) return null;
    if (rd.length === 3) {
      const [C, H, W] = rd, per = H * W;
      for (let k = 0; k < C; k++) { let v = 0; for (let j = 0; j < per; j++) v += w.data[k * per + j] * x[k * per + j]; contrib.push({ k, v }); }
    } else {
      for (let j = 0; j < n; j++) contrib.push({ k: j, v: w.data[j] * x[j] });
    }
  }
  contrib = contrib.filter((e) => e.v !== 0).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, limit);
  return { fromRoom, perChannel: src.op === 'Conv' || rd.length === 3, exact, edges: contrib };
}

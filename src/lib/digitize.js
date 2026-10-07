// Turn a free drawing (or a typed character) into the 8x8 format the model was trained on,
// following the optdigits recipe: fit the ink into a 32x32 bitmap, then count ink per 4x4 block.
export function toDigit(src) {
  const S = src.width;
  const d = src.getContext('2d').getImageData(0, 0, S, S).data;
  let x0 = S, y0 = S, x1 = -1, y1 = -1;
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      if (d[(y * S + x) * 4 + 3] > 40) {
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
  const out = new Float32Array(64);
  if (x1 < 0) return out;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, k = 32 / Math.max(bw, bh);
  const tw = Math.max(1, Math.round(bw * k)), th = Math.max(1, Math.round(bh * k));
  const c = document.createElement('canvas');
  c.width = 32; c.height = 32;
  const cx = c.getContext('2d');
  cx.drawImage(src, x0, y0, bw, bh, Math.floor((32 - tw) / 2), Math.floor((32 - th) / 2), tw, th);
  const b = cx.getImageData(0, 0, 32, 32).data;
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++)
      if (b[(y * 32 + x) * 4 + 3] > 90) out[(y >> 2) * 8 + (x >> 2)] += 1 / 16;
  return out;
}

// A typed character, drawn in a plain font and then digitized like a drawing.
export function textToDigit(ch) {
  const c = document.createElement('canvas');
  c.width = 200; c.height = 200;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 170px Inter, Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(ch, 100, 108);
  return toDigit(c);
}

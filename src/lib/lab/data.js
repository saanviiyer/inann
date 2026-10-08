// Inputs and datasets for the Lab.
// A dataset item is kept in "display" form: pixels in 0..1 laid out like the model input
// (channels first for images). `toModel` applies the normalization the model expects.
import { toDigit } from '../digitize.js';

// ---------- what kind of input does the model take? ----------
// Returns { kind: 'image', C, H, W, layout: 'nchw'|'nhwc'|'flat' } or { kind: 'vector' }.
export function inputKind(dims, meta = {}, inputType = 'float32') {
  if (meta.input?.kind === 'tokens' || inputType !== 'float32') {
    const vocab = meta.input?.vocab ?? null;
    return { kind: 'tokens', T: dims.reduce((a, b) => a * b, 1), vocab, tokenizer: meta.input?.tokenizer ?? (vocab ? 'char' : 'ids') };
  }
  if (meta.input?.kind === 'vector') return { kind: 'vector' };
  if (dims.length === 3 && (dims[0] === 1 || dims[0] === 3)) return { kind: 'image', C: dims[0], H: dims[1], W: dims[2], layout: 'nchw' };
  if (dims.length === 3 && (dims[2] === 1 || dims[2] === 3)) return { kind: 'image', C: dims[2], H: dims[0], W: dims[1], layout: 'nhwc' };
  if (dims.length === 2 && dims[0] >= 4 && dims[1] >= 4 && meta.input?.kind !== 'sequence') return { kind: 'image', C: 1, H: dims[0], W: dims[1], layout: 'nchw' };
  if (dims.length === 1) {
    const s = Math.round(Math.sqrt(dims[0]));
    if (s * s === dims[0] && s >= 8 && meta.input?.kind !== 'vector') return { kind: 'image', C: 1, H: s, W: s, layout: 'flat' };
  }
  return { kind: 'vector' };
}

export const NORMS = {
  unit: { label: '0 to 1', fn: (v) => v },
  sym: { label: '-1 to 1', fn: (v) => v * 2 - 1 },
  byte: { label: '0 to 255', fn: (v) => v * 255 },
  mnist: { label: 'MNIST mean/std', fn: (v) => (v - 0.1307) / 0.3081 },
  imagenet: { label: 'ImageNet mean/std', fn: (v, c) => (v - [0.485, 0.456, 0.406][c % 3]) / [0.229, 0.224, 0.225][c % 3] },
};

export function normFromMeta(meta) {
  const r = meta?.input?.range;
  if (meta?.input?.norm && NORMS[meta.input.norm]) return meta.input.norm;
  if (r && r[0] === -1) return 'sym';
  if (r && r[1] === 255) return 'byte';
  return 'unit';
}

// display pixels (C,H,W in 0..1) -> model input
export function toModel(pixels, kind, norm) {
  const f = NORMS[norm].fn;
  if (kind.kind !== 'image') return Float32Array.from(pixels);
  const { C, H, W } = kind;
  const out = new Float32Array(C * H * W);
  for (let c = 0; c < C; c++)
    for (let p = 0; p < H * W; p++) {
      const v = f(pixels[c * H * W + p], c);
      out[kind.layout === 'nhwc' ? p * C + c : c * H * W + p] = v;
    }
  return out;
}

// ---------- drawings ----------
// MNIST recipe: fit the ink into a 20x20 box keeping aspect, then centre it by its centre of
// mass inside the 28x28 frame (LeCun et al. 1998). Scaled to any H x W.
export function drawingToPixels(canvas, kind, mode = 'mnist') {
  const { C, H, W } = kind;
  let gray;
  if (mode === 'optdigits' && H === 8 && W === 8) gray = toDigit(canvas);
  else {
    const S = canvas.width;
    const d = canvas.getContext('2d').getImageData(0, 0, S, S).data;
    let x0 = S, y0 = S, x1 = -1, y1 = -1;
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) if (d[(y * S + x) * 4 + 3] > 30) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    gray = new Float32Array(H * W);
    if (x1 >= 0) {
      const tmp = document.createElement('canvas');
      tmp.width = W; tmp.height = H;
      const ctx = tmp.getContext('2d');
      if (mode === 'stretch') ctx.drawImage(canvas, 0, 0, W, H);
      else {
        const box = mode === 'mnist' ? 20 / 28 : 0.86;
        const bw = x1 - x0 + 1, bh = y1 - y0 + 1, k = (box * Math.min(W, H)) / Math.max(bw, bh);
        const tw = bw * k, th = bh * k;
        ctx.drawImage(canvas, x0, y0, bw, bh, (W - tw) / 2, (H - th) / 2, tw, th);
      }
      const px = ctx.getImageData(0, 0, W, H).data;
      for (let i = 0; i < H * W; i++) gray[i] = px[i * 4 + 3] / 255;
      if (mode === 'mnist') gray = centreOfMass(gray, H, W);
    }
  }
  const out = new Float32Array(C * H * W);
  for (let c = 0; c < C; c++) out.set(gray, c * H * W);
  return out;
}

function centreOfMass(g, H, W) {
  let m = 0, cx = 0, cy = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const v = g[y * W + x]; m += v; cx += v * x; cy += v * y; }
  if (!m) return g;
  const dx = Math.round((W - 1) / 2 - cx / m), dy = Math.round((H - 1) / 2 - cy / m);
  const out = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const sy = y - dy, sx = x - dx;
    if (sy >= 0 && sy < H && sx >= 0 && sx < W) out[y * W + x] = g[sy * W + sx];
  }
  return out;
}

// An uploaded picture, letterboxed into H x W. `invert` turns dark-on-light into light-on-dark.
export async function imageFileToPixels(file, kind, invert = false) {
  const bmp = await createImageBitmap(file);
  const { C, H, W } = kind;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.fillStyle = invert ? '#fff' : '#000';
  ctx.fillRect(0, 0, W, H);
  const k = Math.min(W / bmp.width, H / bmp.height);
  ctx.drawImage(bmp, (W - bmp.width * k) / 2, (H - bmp.height * k) / 2, bmp.width * k, bmp.height * k);
  const px = ctx.getImageData(0, 0, W, H).data;
  const out = new Float32Array(C * H * W);
  for (let p = 0; p < H * W; p++) {
    const rgb = [px[p * 4], px[p * 4 + 1], px[p * 4 + 2]].map((v) => (invert ? 255 - v : v) / 255);
    if (C === 1) out[p] = (rgb[0] + rgb[1] + rgb[2]) / 3;
    else for (let ch = 0; ch < C; ch++) out[ch * H * W + p] = rgb[ch % 3];
  }
  return out;
}

// ---------- datasets ----------
// { name, n, size, item(i) -> display pixels or raw vector, labels?: int[], classes?: string[], source }
export function datasetFromJson(json, name) {
  const bin = atob(json.data);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const size = json.shape.reduce((a, b) => a * b, 1);
  if (json.dtype === 'int32' || json.dtype === 'float32') {
    // token ids or raw vectors, fed to the model as they are
    const arr = json.dtype === 'int32' ? new Int32Array(u8.buffer) : new Float32Array(u8.buffer);
    return {
      name, n: arr.length / size, size, shape: json.shape, labels: json.labels, classes: json.classes, source: json.source,
      raw: true, tokens: json.dtype === 'int32', item: (i) => arr.subarray(i * size, (i + 1) * size),
    };
  }
  const n = u8.length / size;
  return {
    name, n, size, shape: json.shape, labels: json.labels, classes: json.classes, source: json.source,
    item: (i) => Float32Array.from(u8.subarray(i * size, (i + 1) * size), (v) => v / 255),
  };
}

export async function datasetFromImages(files, kind, invert) {
  const imgs = [], names = [];
  for (const f of files) {
    if (!/^image\//.test(f.type)) continue;
    imgs.push(await imageFileToPixels(f, kind, invert));
    // label from the parent folder (folder upload) or a filename prefix like "7_abc.png"
    const parts = (f.webkitRelativePath || f.name).split('/');
    names.push(parts.length > 2 ? parts[parts.length - 2] : (f.name.match(/^([^_\-. ]+)[_\-]/)?.[1] ?? null));
  }
  const uniq = [...new Set(names.filter((v) => v != null))].sort((a, b) => (isNaN(a) || isNaN(b) ? a.localeCompare(b) : a - b));
  const labeled = names.every((v) => v != null) && uniq.length > 1 && uniq.length <= 1000;
  // numeric folder names ("0".."9") keep their value so they line up with model outputs
  const numeric = labeled && uniq.every((v) => /^\d+$/.test(v));
  const classes = !labeled ? null : numeric ? Array.from({ length: Math.max(...uniq.map(Number)) + 1 }, (_, k) => String(k)) : uniq;
  return {
    name: `${imgs.length} uploaded images`, n: imgs.length, size: imgs[0]?.length ?? 0,
    labels: labeled ? names.map((v) => (numeric ? Number(v) : uniq.indexOf(v))) : null, classes,
    source: 'your files', item: (i) => imgs[i],
  };
}

// Minimal .npy reader: little-endian float32/float64/uint8/int64, C order.
export function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== 'NUMPY') throw new Error('not a .npy file');
  const major = u8[6];
  const hlen = major === 1 ? u8[8] | (u8[9] << 8) : new DataView(buf).getUint32(8, true);
  const off = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.subarray(off, off + hlen));
  const descr = header.match(/'descr':\s*'([^']+)'/)[1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error('Fortran-order .npy is not supported; save with np.ascontiguousarray');
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const start = off + hlen;
  const body = buf.slice(start);
  const map = { '<f4': Float32Array, '<f8': Float64Array, '|u1': Uint8Array, '<i8': BigInt64Array, '<i4': Int32Array };
  const Ctor = map[descr];
  if (!Ctor) throw new Error(`unsupported dtype ${descr}`);
  let data = new Ctor(body);
  if (Ctor === BigInt64Array) data = Float64Array.from(data, Number);
  return { shape, data, dtype: descr };
}

// raw = true: the array already holds model inputs (e.g. normalized with a mean/std), fed as-is.
export function datasetFromNpy(x, y, kind, raw = false) {
  const n = x.shape[0], size = x.data.length / n;
  const isByte = x.dtype === '|u1';
  const image = kind.kind === 'image';
  let mx = 1;
  if (image && !isByte && !raw) for (let i = 0; i < Math.min(x.data.length, 200000); i++) mx = Math.max(mx, x.data[i]);
  // integer labels keep their value, so label k lines up with model output k
  const labels = y ? Array.from(y.data, Number) : null;
  const intLabels = labels && labels.every((v) => Number.isInteger(v) && v >= 0 && v < 10000);
  const uniq = labels ? [...new Set(labels)].sort((a, b) => a - b) : null;
  const classes = labels ? (intLabels ? Array.from({ length: Math.max(...labels) + 1 }, (_, k) => String(k)) : uniq.map(String)) : null;
  return {
    name: `${n} items from .npy`, n, size, labels: labels ? (intLabels ? labels : labels.map((v) => uniq.indexOf(v))) : null, classes,
    source: 'your .npy file', raw: raw || !image,
    item: (i) => {
      const s = x.data.subarray(i * size, (i + 1) * size);
      if (raw || !image) return Float32Array.from(s);
      const v = Float32Array.from(s, (t) => (isByte ? t / 255 : mx > 1.5 ? t / 255 : t));
      return kind.layout === 'nhwc' ? hwcToChw(v, kind) : v;
    },
  };
}

function hwcToChw(v, { C, H, W }) {
  const out = new Float32Array(v.length);
  for (let p = 0; p < H * W; p++) for (let c = 0; c < C; c++) out[c * H * W + p] = v[p * C + c];
  return out;
}

// The model input for item i, and a 0..1 picture of it for display.
export function modelInputOf(ds, i, kind, norm) {
  return ds.raw ? ds.item(i) : toModel(ds.item(i), kind, norm);
}
export function displayOf(ds, i, kind) {
  const v = ds.item(i);
  if (!ds.raw || ds.tokens || kind.kind === 'tokens') return v;
  let lo = Infinity, hi = -Infinity;
  for (const t of v) { lo = Math.min(lo, t); hi = Math.max(hi, t); }
  const s = Float32Array.from(v, (t) => (t - lo) / (hi - lo || 1));
  return kind.kind === 'image' && kind.layout === 'nhwc' ? hwcToChw(s, kind) : s;
}

// ---------- tokens ----------
// Character tokenizer over the model's vocabulary (from its meta.json). Unknown characters are
// dropped; the result is cut or padded (with the first vocabulary entry) to the model's length.
export function textToIds(text, kind) {
  const ids = [];
  if (kind.tokenizer === 'char' && kind.vocab) for (const ch of text) { const k = kind.vocab.indexOf(ch); if (k >= 0) ids.push(k); }
  else for (const t of text.split(/[\s,]+/)) if (t !== '' && !isNaN(+t)) ids.push(+t);
  const out = new Float32Array(kind.T);
  for (let i = 0; i < kind.T; i++) out[i] = ids[i] ?? 0;
  return { ids: out, used: Math.min(ids.length, kind.T), dropped: Math.max(0, ids.length - kind.T) };
}
export const idsToText = (ids, kind) => Array.from(ids, (v) => (kind.vocab ? kind.vocab[v] ?? '?' : `${v} `)).join('');

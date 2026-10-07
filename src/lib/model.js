// Loads the bundled weights and held-out digits once, and precomputes what every view shares.
import modelJson from '../data/model.json';
import digitsJson from '../data/digits.json';
import saeJson from '../data/sae.json';
import presetsJson from '../data/presets.json';
import { loadNet, forward, LAYERS, layerSize } from './net.js';
import { loadSae } from './sae.js';

export const net = loadNet(modelJson);
export const sae = loadSae(saeJson);
export const PRESETS = presetsJson;

// Explanations are written offline by scripts/explain.py. The app works without them.
const exMods = import.meta.glob('../data/explanations.json', { eager: true, import: 'default' });
export const EXPLAINED = Object.values(exMods)[0] ?? null;
export const EXPLANATIONS = Object.fromEntries((EXPLAINED?.units ?? []).map((u) => [u.id, u]));
export const META = modelJson;
export const DIGITS = digitsJson.images.map((im) => Float32Array.from(im, (v) => v / 16));
export const LABELS = digitsJson.labels;

// Activations of every held-out digit, layer by layer: ACTS[layer][imageIdx] = Float32Array
export const ACTS = Object.fromEntries(LAYERS.map((l) => [l.key, []]));
for (const x of DIGITS) {
  const f = forward(net, x, { sae });
  for (const l of LAYERS) ACTS[l.key].push(f.acts[l.key]);
}

// Brightness scale per layer: the 99th percentile of activations over the held-out set,
// so the same brightness means the same value no matter which digit is drawn.
export const SCALE = {};
for (const l of LAYERS) {
  const all = [];
  for (const a of ACTS[l.key]) for (const v of a) if (v > 0) all.push(v);
  all.sort((a, b) => a - b);
  SCALE[l.key] = l.key === 'out' ? 1 : all[Math.floor(all.length * 0.99)] || 1;
}

export function describe(layer, index) {
  if (layer === 'input') return { title: `Input pixel (row ${Math.floor(index / 8)}, col ${index % 8})`, channel: null };
  if (layer === 'conv1') {
    const c = Math.floor(index / 64), p = index % 64;
    return { title: `Conv 1, channel ${c}, at (row ${Math.floor(p / 8)}, col ${p % 8})`, channel: c, pos: p };
  }
  if (layer === 'conv2') {
    const c = Math.floor(index / 16), p = index % 16;
    return { title: `Conv 2, channel ${c}, at (row ${Math.floor(p / 4)}, col ${p % 4})`, channel: c, pos: p };
  }
  if (layer === 'dense') return { title: `Dense neuron ${index}`, channel: null };
  if (layer === 'sae') return { title: `SAE feature ${index}`, channel: null };
  return { title: `Output pod for digit ${index}`, channel: null };
}

// A representative unit for a channel: the one nearest the middle of its map.
export function centerUnit(layer, channel) {
  if (layer === 'conv1') return channel * 64 + 3 * 8 + 3;
  if (layer === 'conv2') return channel * 16 + 1 * 4 + 1;
  return channel;
}

export { layerSize };

// ---------- rooms: one per neuron, per conv channel, per SAE feature ----------
// Conv units share weights across positions, so a conv room describes its whole channel and
// the channel's activation on an image is its largest value over positions.
const PER = { conv1: 64, conv2: 16 };
export function roomOf(layer, index) {
  if (PER[layer]) return { layer, index: Math.floor(index / PER[layer]), channel: true };
  return { layer, index, channel: false };
}
export const roomId = (r) => `${r.layer}:${r.index}`;
export function roomName(r) {
  if (r.layer === 'conv1' || r.layer === 'conv2') return `${r.layer === 'conv1' ? 'Conv 1' : 'Conv 2'} channel ${r.index}`;
  if (r.layer === 'dense') return `Dense neuron ${r.index}`;
  if (r.layer === 'sae') return `SAE feature ${r.index}`;
  if (r.layer === 'out') return `Output pod ${r.index}`;
  return `Input pixel ${r.index}`;
}
export function roomActOf(acts, r) {
  const per = PER[r.layer];
  if (!per) return acts[r.layer][r.index];
  let m = 0;
  for (let p = 0; p < per; p++) m = Math.max(m, acts[r.layer][r.index * per + p]);
  return m;
}
const roomCache = new Map();
export function roomActivations(r) {
  const k = roomId(r);
  if (!roomCache.has(k)) {
    const per = PER[r.layer];
    roomCache.set(k, Float32Array.from(ACTS[r.layer], (a) => {
      if (!per) return a[r.index];
      let m = 0;
      for (let p = 0; p < per; p++) m = Math.max(m, a[r.index * per + p]);
      return m;
    }));
  }
  return roomCache.get(k);
}
// where in its channel a conv room fired hardest on one image (a unit index), for selection
export function bestUnitInRoom(acts, r) {
  const per = PER[r.layer];
  if (!per) return r.index;
  let bi = r.index * per;
  for (let p = 0; p < per; p++) if (acts[r.layer][r.index * per + p] > acts[r.layer][bi]) bi = r.index * per + p;
  return bi;
}

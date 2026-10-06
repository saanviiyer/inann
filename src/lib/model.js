// Loads the bundled weights and held-out digits once, and precomputes what every view shares.
import modelJson from '../data/model.json';
import digitsJson from '../data/digits.json';
import { loadNet, forward, LAYERS, layerSize } from './net.js';

export const net = loadNet(modelJson);
export const META = modelJson;
export const DIGITS = digitsJson.images.map((im) => Float32Array.from(im, (v) => v / 16));
export const LABELS = digitsJson.labels;

// Activations of every held-out digit, layer by layer: ACTS[layer][imageIdx] = Float32Array
export const ACTS = Object.fromEntries(LAYERS.map((l) => [l.key, []]));
for (const x of DIGITS) {
  const f = forward(net, x);
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
  return { title: `Output pod for digit ${index}`, channel: null };
}

// A representative unit for a channel: the one nearest the middle of its map.
export function centerUnit(layer, channel) {
  if (layer === 'conv1') return channel * 64 + 3 * 8 + 3;
  if (layer === 'conv2') return channel * 16 + 1 * 4 + 1;
  return channel;
}

export { layerSize };

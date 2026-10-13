// Trains a linear probe and its shuffled-label control on each layer, off the main thread.
// Receives {layers: [{name, X: Float32Array (N*D), D}], labels, K, split: {train, test}}
// and posts {type:'layer', name, result} per layer, then {type:'done'}.
import { probeLayer } from '../probe.js';

self.onmessage = (e) => {
  const { layers, labels, K, split } = e.data;
  layers.forEach(({ name, X, D }, i) => {
    self.postMessage({ type: 'start', name, i, n: layers.length });
    const { probe, ...r } = probeLayer(X, D, labels, K, split);
    self.postMessage({ type: 'layer', name, result: { ...r, probe } });
  });
  self.postMessage({ type: 'done' });
};

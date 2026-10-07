// Per-room statistics the feature rooms, search and SAE comparison share. All computed from
// the bundled model on the 450 held-out digits.
import { DIGITS, LABELS, roomActivations, roomId, EXPLANATIONS, sae } from './model.js';

export function topExamples(r, k = 12) {
  const a = roomActivations(r);
  return Array.from(a.keys()).sort((i, j) => a[j] - a[i]).slice(0, k)
    .filter((i) => a[i] > 0).map((i) => ({ i, act: a[i], label: LABELS[i], img: DIGITS[i] }));
}

export function histogram(r, bins = 24) {
  const a = roomActivations(r);
  const max = Math.max(...a) || 1;
  const counts = new Array(bins).fill(0);
  let zeros = 0;
  for (const v of a) {
    if (v <= 0) { zeros++; continue; }
    counts[Math.min(bins - 1, Math.floor((v / max) * bins))]++;
  }
  return { counts, zeros, max, n: a.length };
}

// Fraction of the top-n active examples that share the most common digit label.
// Null when the unit is active on fewer than n held-out digits.
export function purity(r, n = 12) {
  const top = topExamples(r, n);
  if (top.length < n) return null;
  const c = new Array(10).fill(0);
  top.forEach((t) => c[t.label]++);
  const best = c.indexOf(Math.max(...c));
  return { value: c[best] / n, digit: best, count: c[best], n };
}

export const explanationOf = (r) => EXPLANATIONS[roomId(r)] ?? null;

// Score bands for displaying simulation scores (Pearson r between simulated and true activations).
export function scoreBand(s) {
  if (s == null) return null;
  if (s < 0.3) return { label: 'weak explanation', cls: 'border-red-400/50 text-red-300' };
  if (s < 0.6) return { label: 'partial explanation', cls: 'border-warm/50 text-warm' };
  return { label: 'good explanation', cls: 'border-glow/50 text-glow' };
}

// All rooms of the network that have any activity on held-out digits.
export function allRooms() {
  const out = [];
  for (let c = 0; c < 6; c++) out.push({ layer: 'conv1', index: c, channel: true });
  for (let c = 0; c < 12; c++) out.push({ layer: 'conv2', index: c, channel: true });
  for (let j = 0; j < 32; j++) out.push({ layer: 'dense', index: j, channel: false });
  for (let j = 0; j < sae.F; j++) out.push({ layer: 'sae', index: j, channel: false });
  return out.filter((r) => Math.max(...roomActivations(r)) > 0);
}

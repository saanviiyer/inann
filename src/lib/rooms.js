// Per-room statistics the feature rooms, search and SAE comparison share. All computed from
// the bundled model on the 450 held-out digits.
import { DIGITS, LABELS, ACTS, roomActivations, roomId, EXPLANATIONS, sae } from './model.js';

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

// A measured one-line description, used when no written explanation exists: which digit its
// strongest held-out examples are, how often it fires, and (for conv channels) where on the
// digit it fires. Every phrase is a count from the bundled held-out set.
const GRID = { conv1: [8, 8], conv2: [4, 4] };
export function measuredSummary(r) {
  const a = roomActivations(r);
  const n = a.length, on = Array.from(a).filter((v) => v > 0).length;
  const parts = [];
  const p = purity(r, 12);
  if (p && p.value >= 0.5) parts.push(`${p.count} of its top 12 held-out digits are ${p.digit}s`);
  else if (p) parts.push(`its top 12 held-out digits mix several classes (at most ${p.count} of one)`);
  parts.push(`fires on ${Math.round((100 * on) / n)}% of held-out digits`);
  const g = GRID[r.layer];
  if (g && on) {
    const [H, W] = g, per = H * W;
    let fy = 0, fx = 0, m = 0;
    for (const t of topExamples(r, 12)) {
      const maps = ACTS[r.layer][t.i];
      let best = 0;
      for (let q = 1; q < per; q++) if (maps[r.index * per + q] > maps[r.index * per + best]) best = q;
      fy += Math.floor(best / W) / (H - 1); fx += (best % W) / (W - 1); m++;
    }
    if (m) {
      const y = fy / m, x = fx / m;
      const area = `${y < 0.34 ? 'upper' : y > 0.66 ? 'lower' : 'middle'} ${x < 0.34 ? 'left' : x > 0.66 ? 'right' : 'centre'}`.replace('middle centre', 'centre');
      parts.push(`usually strongest in the ${area} of the digit`);
    }
  }
  return parts.join(' · ');
}

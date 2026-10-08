// Measured one-line descriptions of channels and SAE features, built only from the dataset pass:
// what share of items it fires on, which label dominates its top items, and where it fires.
// No language model is involved; every phrase is a count you can check in the room.
import { topItems, labelPurity } from './store.js';

const AREA = (fy, fx) => `${fy < 0.34 ? 'upper' : fy > 0.66 ? 'lower' : 'middle'} ${fx < 0.34 ? 'left' : fx > 0.66 ? 'right' : 'centre'}`.replace('middle centre', 'centre');

// acts: per-item activation; args: per-item position of the max (or null); L: layout of the layer
export function describeUnit(acts, args, L, labels, classes) {
  const n = acts.length;
  const on = Array.from(acts).filter((v) => v > 0).length;
  const parts = [];
  const pur = labelPurity(acts, labels, 12);
  if (pur && pur.value >= 0.5) parts.push(`${pur.count} of top 12 are ${classes?.[pur.label] ?? pur.label}`);
  parts.push(`fires on ${Math.round((100 * on) / n)}% of items`);
  if (args && L.P > 1) {
    const top = topItems(acts, 12).filter((i) => acts[i] > 0);
    if (top.length >= 4) {
      if (L.kind === 'spatial') {
        let fy = 0, fx = 0;
        top.forEach((i) => { const p = args(i); fy += Math.floor(p / L.W) / Math.max(1, L.H - 1); fx += (p % L.W) / Math.max(1, L.W - 1); });
        parts.push(`usually ${AREA(fy / top.length, fx / top.length)}`);
      } else if (L.kind === 'seq') {
        const c = {};
        top.forEach((i) => (c[args(i)] = (c[args(i)] ?? 0) + 1));
        const [pos, cnt] = Object.entries(c).sort((a, b) => b[1] - a[1])[0];
        if (cnt >= top.length / 2) parts.push(`peaks at token ${pos} in ${cnt} of ${top.length}`);
      }
    }
  }
  return { text: parts.join(' · '), purity: pur, freq: on / n };
}

// Search measured summaries: a class name or label, a position word, or "rare"/"common".
export function searchUnits(query, entries, classes = null) {
  const q = query.toLowerCase().trim();
  if (!q) return [];
  const stop = ['find', 'the', 'a', 'detector', 'detectors', 'feature', 'features', 'neuron', 'channel', 'for', 'that', 'fires', 'on', 'of'];
  const words = q.split(/[\s-]+/).filter((w) => w && !stop.includes(w));
  const classSet = new Set((classes ?? []).map((c) => String(c).toLowerCase()));
  return entries
    .map((e) => {
      const t = `${e.summary.text} ${e.expl ? `${e.expl.explanation} ${e.expl.tags.join(' ')}` : ''}`.toLowerCase();
      let score = 0;
      for (const w of words) {
        // a word that names a class matches only units whose top items are that class
        if (classSet.has(w)) { if (String(e.className ?? '').toLowerCase() === w) score += 3 * e.summary.purity.value; continue; }
        if (w === 'rare') { if (e.summary.freq < 0.05) score += 1; continue; }
        if (w === 'common') { if (e.summary.freq > 0.5) score += 1; continue; }
        if (t.includes(w)) score += 1;
      }
      return { ...e, score };
    })
    .filter((e) => e.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);
}

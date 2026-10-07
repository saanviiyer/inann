// Search by concept. Two kinds of match, labeled separately in the UI:
//  1. text: the query against each unit's LLM-written explanation and tags (scripts/explain.py),
//     with a small synonym table. Keyword matching, not embeddings.
//  2. measured: if the query names a digit, units whose top-12 held-out examples are mostly that digit.
import { EXPLAINED } from './model.js';
import { allRooms, purity } from './rooms.js';

const STOP = new Set('find show me the a an of that for which unit units neuron neurons feature features detector detectors detect detects responds to is are in on with one'.split(' '));
const NUM = { zero: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9' };
const SYN = [
  ['horizontal', 'flat', 'bar', 'crossbar', 'across'],
  ['vertical', 'upright', 'stem', 'straight down', 'column'],
  ['diagonal', 'slant', 'slanted', 'slanting'],
  ['curve', 'curved', 'curvy', 'arc', 'bend', 'round', 'rounded'],
  ['loop', 'closed', 'ring', 'circle', 'hole', 'oval'],
  ['top', 'upper'],
  ['bottom', 'lower', 'base'],
  ['left'],
  ['right'],
  ['line', 'stroke', 'edge'],
  ['center', 'middle', 'centre'],
  ['corner', 'angle', 'sharp'],
];

function tokens(q) {
  return q.toLowerCase().replace(/(\d)-/g, '$1 ').split(/[^a-z0-9]+/).filter((t) => t && !STOP.has(t)).map((t) => NUM[t] ?? t);
}
function expand(t) {
  const g = SYN.find((s) => s.includes(t) || s.some((w) => t.startsWith(w) && w.length > 3));
  return g ? new Set([t, ...g]) : new Set([t]);
}

export function search(query) {
  const qs = tokens(query);
  if (!qs.length) return [];
  const results = new Map();
  const add = (r, kind, score, why) => {
    const k = `${r.layer}:${r.index}`;
    const prev = results.get(k);
    if (!prev || prev.score < score) results.set(k, { room: r, kind, score, why });
  };

  // 1. text match over explanations
  const units = EXPLAINED?.units ?? [];
  const docs = units.map((u) => ({ u, text: `${u.explanation} ${u.tags.join(' ')}`.toLowerCase() }));
  for (const q of qs) {
    const terms = expand(q);
    const df = docs.filter((d) => [...terms].some((t) => d.text.includes(t))).length;
    if (!df) continue;
    const idf = Math.log(1 + docs.length / df);
    for (const d of docs) {
      const hits = [...terms].filter((t) => d.text.includes(t)).length;
      if (!hits) continue;
      const tagHit = d.u.tags.some((tag) => [...terms].some((t) => tag.includes(t)));
      const r = { layer: d.u.layer, index: d.u.index, channel: d.u.layer.startsWith('conv') };
      const k = `${r.layer}:${r.index}`;
      const prev = results.get(k)?.kind === 'text' ? results.get(k).score : 0;
      // weight by the explanation's own simulation score, so trusted explanations rank higher
      results.set(k, { room: r, kind: 'text', score: prev + idf * (tagHit ? 2 : 1) * (0.5 + Math.max(0, d.u.score)), why: d.u.explanation, sim: d.u.score });
    }
  }

  // 2. measured digit match
  const digits = qs.filter((t) => /^\d$/.test(t)).map(Number);
  if (digits.length) {
    for (const r of allRooms()) {
      const p = purity(r, 12);
      if (p && digits.includes(p.digit) && p.value >= 0.75)
        add(r, 'measured', 1 + 4 * p.value, `${p.count} of its top 12 held-out digits are ${p.digit}s`);
    }
  }
  return [...results.values()].sort((a, b) => b.score - a.score).slice(0, 8);
}

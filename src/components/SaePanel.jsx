import React, { useMemo, useState } from 'react';
import { PixelGrid, Term, Cite, NotShown, Btn } from './ui.jsx';
import { sae, ACTS, roomName } from '../lib/model.js';
import { purity, topExamples } from '../lib/rooms.js';
import { Explanation } from './FeatureRoom.jsx';

const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN;
};

// Which SAE features carry a neuron: total decoder contribution W_dec[n, f] * f(x) over held-out digits.
function splitInto(neuron, k = 4) {
  const tot = new Float64Array(sae.F);
  for (const a of ACTS.sae) for (let f = 0; f < sae.F; f++) if (a[f]) tot[f] += sae.Wd[neuron * sae.F + f] * a[f];
  return Array.from(tot.keys()).filter((f) => tot[f] > 0).sort((p, q) => tot[q] - tot[p]).slice(0, k).map((f) => ({ f, share: tot[f] }));
}

function Mini({ room, onOpen }) {
  const p = purity(room, 12);
  const top = topExamples(room, 6);
  return (
    <div className="space-y-1 rounded-md border border-line p-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-slate-100">{roomName(room)}</span>
        <Btn onClick={() => onOpen(room)}>Open room</Btn>
      </div>
      <Explanation room={room} compact />
      <div className="flex gap-1">
        {top.map((t) => (
          <div key={t.i} className="text-center">
            <PixelGrid values={t.img} size={34} />
            <div className="text-[9px] text-dim">{t.label}</div>
          </div>
        ))}
      </div>
      <div className="text-[10px] text-dim">
        {p ? `${p.count} of its top 12 are ${p.digit}s` : `active on fewer than 12 held-out digits, purity not measured`}
      </div>
    </div>
  );
}

export default function SaePanel({ onOpen }) {
  const stats = useMemo(() => {
    const pn = [], pf = [];
    for (let j = 0; j < 32; j++) { const p = purity({ layer: 'dense', index: j }, 12); if (p) pn.push(p.value); }
    for (let j = 0; j < sae.F; j++) { const p = purity({ layer: 'sae', index: j }, 12); if (p) pf.push(p.value); }
    return { pn, pf };
  }, []);
  const neurons = useMemo(
    () => Array.from({ length: 32 }, (_, j) => ({ j, p: purity({ layer: 'dense', index: j }, 12) })).filter((n) => n.p).sort((a, b) => a.p.value - b.p.value),
    [],
  );
  const [pick, setPick] = useState(neurons[0]?.j ?? 0);
  const parts = useMemo(() => splitInto(pick), [pick]);
  const np = purity({ layer: 'dense', index: pick }, 12);
  const s = sae.stats;
  const pctPure = (a) => Math.round((100 * a.filter((v) => v === 1).length) / a.length);

  return (
    <div className="space-y-3 rounded-lg border border-line p-3">
      <div className="text-sm text-slate-100">Neurons vs features</div>
      <p className="text-xs leading-snug text-slate-400">
        A <Term k="sparse autoencoder" /> rewrites the 32 dense neurons as {s.n_features} <Term k="SAE feature">features</Term>, at most {s.k} on at a time.
        <Cite id="bricken2023" /><Cite id="templeton2024" /><Cite id="gao2024" /> Flip the header switch to <b>Features</b> to see them in the space.
      </p>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded border border-line p-2">
          <div className="text-dim">held-out variance kept</div>
          <div className="text-lg text-glow">{(100 * s.heldout_variance_explained).toFixed(1)}%</div>
        </div>
        <div className="rounded border border-line p-2">
          <div className="text-dim">accuracy if features replace neurons</div>
          <div className="text-lg text-slate-200">{(100 * s.heldout_acc_with_sae_reconstruction).toFixed(1)}% <span className="text-xs text-dim">vs {(100 * s.heldout_acc_original).toFixed(1)}%</span></div>
        </div>
      </div>
      <div className="rounded border border-line p-2 text-xs text-slate-300">
        <div className="mb-1 text-dim">How single-minded are they? Share of the top 12 held-out digits that are the same digit:</div>
        neurons, median <span className="text-glow">{median(stats.pn).toFixed(2)}</span> ({pctPure(stats.pn)}% fully one digit, n = {stats.pn.length}) ·
        features, median <span className="text-glow">{median(stats.pf).toFixed(2)}</span> ({pctPure(stats.pf)}% fully one digit, n = {stats.pf.length})
        <div className="mt-1 text-[10px] text-dim">
          Only units active on at least 12 held-out digits count, so many rarely-firing features are left out. Matching a digit label is a rough stand-in for meaning one thing.
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-slate-200">Split a neuron</span>
          <select value={pick} onChange={(e) => setPick(Number(e.target.value))} className="rounded border border-line bg-panel px-1 py-0.5 text-xs text-slate-200">
            {neurons.map((n) => (
              <option key={n.j} value={n.j}>
                Dense {n.j} ({n.p.count}/12 one digit)
              </option>
            ))}
          </select>
        </div>
        <p className="text-[11px] text-slate-400">
          Neurons are listed most mixed first. Dense {pick}'s top 12 held-out digits: {np ? `${np.count} are ${np.digit}s, the rest are other digits` : 'n/a'}. The features that carry most of its activity:
        </p>
        <Mini room={{ layer: 'dense', index: pick, channel: false }} onOpen={onOpen} />
        <div className="pl-3 border-l border-glow/30 space-y-2">
          {parts.map(({ f }) => <Mini key={f} room={{ layer: 'sae', index: f, channel: false }} onOpen={onOpen} />)}
        </div>
      </div>
      <p className="text-xs leading-snug text-slate-200">
        A mixed neuron can be the sum of several narrower features, and the dictionary makes them visible one at a time. Here that holds on average, not for every feature, so check each room's examples.
        <Cite id="lieberum2024" />
      </p>
      <NotShown>Features for the conv layers. The autoencoder is trained only on the 32 dense neurons.</NotShown>
    </div>
  );
}

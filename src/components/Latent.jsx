import React, { useEffect, useMemo, useRef, useState } from 'react';
import LatentCloud from './LatentCloud.jsx';
import { PixelGrid, Term, Cite, NotShown, Btn } from './ui.jsx';
import { LATENT_LAYERS, pca, neighbours, knnAccuracy, separation } from '../lib/latent.js';
import { net, sae, DIGITS, LABELS } from '../lib/model.js';

const CLASSES = Array.from({ length: 10 }, (_, d) => String(d));
import { forward } from '../lib/net.js';

export function LatentScene({ layer, f, highlight, trail, onPick }) {
  const P = useMemo(() => pca(layer), [layer]);
  const trailVecs = useMemo(() => trail.map((acts) => acts[layer]), [trail, layer]);
  return <LatentCloud P={P} labels={LABELS} classes={CLASSES} me={f.acts[layer]} trail={trailVecs} highlight={highlight} onPick={onPick} />;
}

export function LatentPanel({ layer, setLayer, f, highlight, setHighlight, setX, setTrail }) {
  const P = useMemo(() => pca(layer), [layer]);
  const sep = useMemo(() => LATENT_LAYERS.map((l) => ({ ...l, acc: knnAccuracy(l.key), ratio: separation(l.key) })), []);
  const maxRatio = Math.max(...sep.map((l) => l.ratio));
  const sepOf = (k) => sep.find((l) => l.key === k);
  const nb = useMemo(() => neighbours(layer, f.acts[layer], 5), [layer, f]);
  const firstOf = (d) => LABELS.indexOf(d);
  const [a, setA] = useState(1);
  const [b, setB] = useState(7);
  const [t, setT] = useState(0);
  const blend = (A, B, tt) => DIGITS[firstOf(A)].map((v, i) => (1 - tt) * v + tt * DIGITS[firstOf(B)][i]);
  const runPath = (A, B, tt) => {
    setX(blend(A, B, tt), 'latent');
    // the path through every layer at 21 steps, drawn as a trail in whichever layer is shown
    setTrail(Array.from({ length: 21 }, (_, k) => forward(net, blend(A, B, k / 20), { sae }).acts));
  };
  return (
    <div className="space-y-3 rounded-lg border border-line p-3">
      <div className="text-sm text-slate-100">Latent space</div>
      <p className="text-xs leading-snug text-slate-400">
        Each numeral is one held-out digit, placed by how a layer represents it, squeezed to 3D with <Term k="PCA" />.<Cite id="jolliffe2016" /> Switch layers and watch the same 450 digits move.
      </p>
      <div className="flex flex-wrap gap-1">
        {LATENT_LAYERS.map((l) => <Btn key={l.key} on={layer === l.key} onClick={() => setLayer(l.key)}>{l.label}</Btn>)}
      </div>
      <div className="text-xs text-slate-300">
        This view keeps <span className="text-glow">{(100 * P.explained.reduce((x, y) => x + y, 0)).toFixed(0)}%</span> of the spread in {P.D} dimensions
        (PC1 {(100 * P.explained[0]).toFixed(0)}%, PC2 {(100 * P.explained[1]).toFixed(0)}%, PC3 {(100 * P.explained[2]).toFixed(0)}%).
      </div>

      <div>
        <div className="mb-1 text-xs text-slate-200">Two ways to ask how each layer groups same digits<Cite id="alain2016" /></div>
        <div className="flex text-[10px] text-dim">
          <span className="w-28" />
          <span className="flex-1"><Term k="nearest neighbours">5-nearest-neighbour</Term> accuracy</span>
          <span className="w-24 text-right"><Term k="separation ratio">spacing ratio</Term></span>
        </div>
        {sep.map((l) => (
          <button key={l.key} onClick={() => setLayer(l.key)} className="flex w-full items-center gap-2 text-left text-[11px]">
            <span className={`w-28 ${layer === l.key ? 'text-glow' : 'text-dim'}`}>{l.label}</span>
            <span className="w-10 text-slate-300">{(l.acc * 100).toFixed(1)}%</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-line">
              <div className="h-full rounded bg-[#3987e5]" style={{ width: `${(l.ratio / maxRatio) * 100}%` }} />
            </div>
            <span className="w-10 text-right text-slate-300">{l.ratio.toFixed(2)}</span>
          </button>
        ))}
        <div className="mt-1 text-[10px] text-dim">Both use the full space, not this 3D view. Neighbour accuracy is leave-one-out.</div>
      </div>

      <div className="flex items-center gap-2 text-xs">
        <span className="text-slate-200">Highlight</span>
        <select value={highlight ?? ''} onChange={(e) => setHighlight(e.target.value === '' ? null : Number(e.target.value))} className="rounded border border-line bg-panel px-1 py-0.5 text-slate-200">
          <option value="">none</option>
          {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>digit {d}</option>)}
        </select>
        <span className="text-[10px] text-dim">click any numeral to load that digit</span>
      </div>

      <div>
        <div className="mb-1 text-xs text-slate-200">Your input's nearest held-out digits in this layer</div>
        <div className="flex gap-1.5">
          {nb.map((n) => (
            <div key={n.i} className="text-center">
              <PixelGrid values={DIGITS[n.i]} size={38} />
              <div className="text-[9px] text-dim">{n.label} · {n.d.toFixed(1)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-1.5 rounded-md border border-line p-2">
        <div className="text-xs text-slate-200">Walk between two digits</div>
        <div className="flex items-center gap-1.5 text-xs">
          <select value={a} onChange={(e) => { setA(+e.target.value); runPath(+e.target.value, b, t); }} className="rounded border border-line bg-panel px-1 text-slate-200">
            {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <input type="range" min={0} max={1} step={0.05} value={t} onChange={(e) => { setT(+e.target.value); runPath(a, b, +e.target.value); }} className="flex-1 accent-[#ffb547]" />
          <select value={b} onChange={(e) => { setB(+e.target.value); runPath(a, +e.target.value, t); }} className="rounded border border-line bg-panel px-1 text-slate-200">
            {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div className="text-[11px] text-slate-400">
          The input fades pixel by pixel from one held-out {a} to one held-out {b}. Its trail shows where the layer puts each in-between image; the model reads it as{' '}
          <span className="text-warm">{f.pred}</span> right now.
        </div>
      </div>

      <p className="text-xs leading-snug text-slate-200">
        These digits are easy, so raw pixels already put most digits next to a same-digit neighbour ({(sepOf('input').acc * 100).toFixed(1)}%), and depth barely moves that number.
        What depth changes is the spacing: different digits sit {sepOf('input').ratio.toFixed(1)} times farther apart than same digits spread in pixels, and {sepOf('out').ratio.toFixed(1)} times at the output.
      </p>
      <NotShown>
        The other {P.D - 3} dimensions. Points that look close here can be far apart in the full space; the neighbour list uses the full space. Nonlinear maps like t-SNE are left out because they can invent clusters and distances.
        <Cite id="wattenberg2016" />
      </NotShown>
    </div>
  );
}

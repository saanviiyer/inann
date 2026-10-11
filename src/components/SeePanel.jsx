import React, { useEffect, useMemo, useRef, useState } from 'react';
import { PixelGrid, Term, Cite, NotShown, Btn } from './ui.jsx';
import { net, describe, centerUnit } from '../lib/model.js';
import { activationMax, inputGradient } from '../lib/net.js';
import { featureVis, topPatches, labelCounts, unitVectors, cosineMatrix, mds2 } from '../lib/analysis.js';

// ---------- "show me what excites you": gradient ascent, one step per frame ----------
export function FeatureVis({ sel, f, paused, onFrame }) {
  const [view, setView] = useState('fv');
  const [state, setState] = useState(null);
  const gen = useRef(null);
  const raf = useRef(0);
  const ok = sel && sel.layer !== 'input';

  useEffect(() => {
    cancelAnimationFrame(raf.current);
    gen.current = null; setState(null); onFrame(null);
    return () => cancelAnimationFrame(raf.current);
  }, [sel]); // eslint-disable-line

  const start = () => {
    cancelAnimationFrame(raf.current);
    gen.current = activationMax(net, sel.layer, sel.index);
    const step = () => {
      if (!gen.current) return;
      if (!paused.current) {
        const r = gen.current.next();
        if (r.done) { gen.current = null; return; }
        setState(r.value);
        onFrame({ values: r.value.x, caption: `what excites it, step ${r.value.step}` });
      }
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
  };

  const sal = useMemo(() => {
    if (!ok) return null;
    const g = inputGradient(net, f, sel.layer, sel.index);
    const m = Math.max(...g.map(Math.abs)) || 1;
    return { abs: Array.from(g, (v) => Math.abs(v) / m), zero: m === 1 && g.every((v) => v === 0) };
  }, [sel, f, ok]);

  if (!ok) return <div className="rounded-lg border border-line p-3 text-xs text-slate-400">Select a neuron in Conv 1 or above to see what excites it.</div>;
  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-100">What excites it</div>
        <div className="flex gap-1">
          <Btn on={view === 'fv'} onClick={() => setView('fv')}>Feature vis</Btn>
          <Btn on={view === 'sal'} onClick={() => setView('sal')}>Saliency</Btn>
        </div>
      </div>
      {view === 'fv' ? (
        <>
          <p className="text-xs leading-snug text-slate-400">
            Start from a near-blank image and use <Term k="gradient ascent" /> to push its pixels toward whatever makes this neuron fire hardest.
            <Cite id="erhan2009" /><Cite id="olah2017" />
          </p>
          <div className="flex items-center gap-3">
            <PixelGrid values={state?.x ?? new Float32Array(64)} size={96} />
            <div className="space-y-1 text-xs text-slate-300">
              <Btn on onClick={start}>Show me what excites you</Btn>
              <div>step {state?.step ?? 0} / 120</div>
              <div><Term k="pre-activation" />: {state ? state.value.toFixed(2) : '...'}</div>
              <div className="text-dim">It also floats next to the neuron in the space.</div>
            </div>
          </div>
          <NotShown>
            The regularizers from Olah et al. 2017 (jitter, rotation, decorrelation). At 8x8 there is little room for them, so this is one optimum from one start.
          </NotShown>
        </>
      ) : (
        <>
          <p className="text-xs leading-snug text-slate-400">
            <Term k="saliency" /> is the 2D companion: for the digit you drew, how much each pixel would change this neuron if nudged.
            <Cite id="simonyan2013" />
          </p>
          <div className="flex items-center gap-3">
            <PixelGrid values={f.input} size={96} alpha={sal.abs} />
            <div className="text-xs text-slate-400">
              Orange marks pixels with large <Term k="gradient" />. {sal.zero && 'This neuron is off for this input, so every gradient is zero.'}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---------- top real patches: the polysemanticity check ----------
export function Patches({ sel }) {
  const ok = sel && (sel.layer === 'conv1' || sel.layer === 'conv2' || sel.layer === 'dense');
  const data = useMemo(() => (ok ? { top: topPatches(sel.layer, sel.index, 5), counts: labelCounts(sel.layer, sel.index, 20) } : null), [sel, ok]);
  if (!ok) return <div className="rounded-lg border border-line p-3 text-xs text-slate-400">Select a Conv 1, Conv 2 or Dense neuron to see its top real patches. Conv 2 is the clearest middle layer.</div>;
  const distinct = new Set(data.top.filter((t) => t.act > 0).map((t) => t.label)).size;
  const n20 = data.counts.reduce((a, b) => a + b, 0);
  const kinds = data.counts.filter((c) => c > 0).length;
  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div className="text-sm text-slate-100">Top 5 real patches</div>
      <p className="text-xs leading-snug text-slate-400">
        The 5 held-out digits, cropped to the <Term k="receptive field" />, that drive this neuron hardest. They also float above it in the space.
      </p>
      <div className="flex gap-2">
        {data.top.map((t) => (
          <div key={t.i} className="text-center">
            <PixelGrid values={t.patch} w={t.w} h={t.h} size={44} />
            <div className="text-[10px] text-dim">digit {t.label}</div>
            <div className="text-[10px] text-dim">{t.act.toFixed(2)}</div>
          </div>
        ))}
      </div>
      <div className="text-xs text-slate-300">
        These 5 come from {distinct} different digit{distinct === 1 ? '' : 's'}. Across its top 20, {n20 < 20 ? `only ${n20} were active, from ` : ''}{kinds} digit{kinds === 1 ? '' : 's'}:{' '}
        <span className="text-dim">
          {data.counts.map((c, d) => (c ? `${d}×${c}` : null)).filter(Boolean).join('  ')}
        </span>
      </div>
      <p className="text-xs leading-snug text-slate-400">
        Neurons do not have single clean meanings. Many respond to a mix of things, which researchers call <Term k="polysemantic" />, and explaining why is a real open problem.
        <Cite id="olah2020" /><Cite id="elhage2022" />
      </p>
      <NotShown>Whether the shared shape in these patches is the same stroke. Digit labels alone cannot tell you that; look at the patches.</NotShown>
    </div>
  );
}

// ---------- neighborhoods: units placed by how alike their responses are ----------
export function Neighborhood({ sel, onSelect }) {
  const [layer, setLayer] = useState('conv2');
  const cv = useRef(null);
  const data = useMemo(() => {
    const vecs = unitVectors(layer);
    const { M, dead } = cosineMatrix(vecs);
    const pos = mds2(M, dead.map((d) => !d));
    const thumbs = vecs.map((_, u) => (dead[u] ? null : featureVis(layer, centerUnit(layer, u))));
    return { M, dead, pos, thumbs, n: vecs.length };
  }, [layer]);
  const selUnit = sel && sel.layer === layer ? describe(sel.layer, sel.index).channel ?? sel.index : null;
  const W = 300, H = 220, T = layer === 'dense' ? 22 : 30;
  const pts = useMemo(() => {
    const ks = Object.keys(data.pos).map(Number);
    const xs = ks.map((k) => data.pos[k][0]), ys = ks.map((k) => data.pos[k][1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const out = {};
    ks.forEach((k) => (out[k] = [T / 2 + 6 + ((data.pos[k][0] - x0) / (x1 - x0 || 1)) * (W - T - 12), T / 2 + 6 + ((data.pos[k][1] - y0) / (y1 - y0 || 1)) * (H - T - 12)]));
    return out;
  }, [data, T]);

  useEffect(() => {
    const ctx = cv.current.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    const tmp = document.createElement('canvas');
    tmp.width = 8; tmp.height = 8;
    const tctx = tmp.getContext('2d');
    for (const [k, [x, y]] of Object.entries(pts)) {
      const img = tctx.createImageData(8, 8);
      data.thumbs[k].forEach((v, i) => { const c = Math.round(v * 255); img.data.set([c, c, c, 255], i * 4); });
      tctx.putImageData(img, 0, 0);
      ctx.drawImage(tmp, x - T / 2, y - T / 2, T, T);
      ctx.strokeStyle = Number(k) === selUnit ? '#ffb547' : '#1c2436';
      ctx.lineWidth = Number(k) === selUnit ? 2 : 1;
      ctx.strokeRect(x - T / 2, y - T / 2, T, T);
    }
  }, [pts, data, selUnit, T]);

  const nearest = selUnit != null && !data.dead[selUnit]
    ? Array.from(data.M[selUnit], (c, j) => [c, j]).filter(([c, j]) => j !== selUnit && !Number.isNaN(c)).sort((a, b) => b[0] - a[0]).slice(0, 3)
    : null;
  const unitName = layer === 'dense' ? 'neuron' : 'channel';
  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-100">Neighborhood</div>
        <div className="flex gap-1">
          {['conv1', 'conv2', 'dense'].map((l) => (
            <Btn key={l} on={layer === l} onClick={() => setLayer(l)}>{l === 'conv1' ? 'Conv 1' : l === 'conv2' ? 'Conv 2' : 'Dense'}</Btn>
          ))}
        </div>
      </div>
      <p className="text-xs leading-snug text-slate-400">
        Each square is one {unitName}, drawn as its feature visualization. {unitName === 'channel' ? 'Channels' : 'Neurons'} that respond to the same held-out digits sit close together
        (<Term k="cosine similarity" /> over 450 digits{layer !== 'dense' ? ', using the channel mean' : ''}, laid out with <Term k="MDS" />). Click one to fly the selection there.
      </p>
      <canvas
        ref={cv}
        width={W}
        height={H}
        className="w-full cursor-pointer rounded bg-[var(--pad)]"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - r.left) / r.width) * W, y = ((e.clientY - r.top) / r.height) * H;
          let best = null, bd = 1e9;
          for (const [k, [px, py]] of Object.entries(pts)) {
            const d = Math.hypot(px - x, py - y);
            if (d < bd) { bd = d; best = Number(k); }
          }
          if (best != null && bd < T) onSelect({ layer, index: centerUnit(layer, best) });
        }}
      />
      {nearest && (
        <div className="text-xs text-slate-300">
          Closest to {unitName} {selUnit}: {nearest.map(([c, j]) => `${j} (${c.toFixed(2)})`).join(', ')}
        </div>
      )}
      {data.dead.some(Boolean) && (
        <div className="text-xs text-dim">
          Never active on the held-out digits, so left off the map: {data.dead.map((d, i) => (d ? i : null)).filter((v) => v != null).join(', ')}
        </div>
      )}
      <NotShown>
        Whether these groups are edge or curve detectors. Large vision models have such families
        <Cite id="olah2020" /><Cite id="cammarata2020" />, but nobody has checked this 8x8 network for them, so the map makes no such claim.
      </NotShown>
    </div>
  );
}
